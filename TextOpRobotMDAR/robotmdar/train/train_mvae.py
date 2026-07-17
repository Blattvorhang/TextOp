import os
import torch
import torch.distributed as dist
import toolz
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import Dataset, VAE, Optimizer
from robotmdar.train.manager import MVAEManager, is_main_process, get_ddp_model


def ddp_setup():
    """Initialize DDP environment variables and process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def evaluate_distribution_match(normalized_data):
    """评估归一化后数据分布是否接近标准正态"""
    # 计算关键统计量
    actual_mean = normalized_data.mean(dim=0)  # 各特征均值
    actual_std = normalized_data.std(dim=0)  # 各特征标准差

    # 理想值对比
    perfect_mean = torch.zeros_like(actual_mean)  # 期望均值=0
    perfect_std = torch.ones_like(actual_std)  # 期望标准差=1

    # 计算偏差
    mean_error = (actual_mean - perfect_mean).abs().mean()
    std_error = (actual_std - perfect_std).abs().mean()

    return {
        'mean_error': mean_error.item(),
        'std_error': std_error.item(),
        'is_well_normalized': mean_error < 0.1 and std_error < 0.2
    }


def main(cfg: DictConfig):
    # Initialize DDP
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f'cuda:{local_rank}')

    logger.set(cfg)
    seed.set(cfg.seed + rank)

    # Override device in config for downstream components
    cfg.device = str(device)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    # Set rank/world_size on datasets for distributed data sharding
    train_data.rank = rank
    train_data.world_size = world_size
    val_data.rank = rank
    val_data.world_size = world_size

    vae: VAE = instantiate(cfg.vae)
    vae = vae.to(device)

    if world_size > 1:
        vae = torch.nn.parallel.DistributedDataParallel(
            vae, device_ids=[local_rank], output_device=local_rank
        )
        if is_main_process():
            print(f"[DDP] Using DistributedDataParallel with {world_size} GPUs")

    # Get the underlying model for accessing custom methods (encode/decode)
    # that are not proxied by DDP. Gradient sync still works because DDP hooks
    # operate on the backward pass, not forward.
    vae_raw = get_ddp_model(vae)

    optimizer: Optimizer = torch.optim.Adam(vae.parameters(), **cfg.train.opt)
    manager: MVAEManager = instantiate(cfg.train.manager)

    manager.hold_model(vae, optimizer, train_data)
    manager.rank = rank
    manager.world_size = world_size

    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)

    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    while manager:
        vae.train()
        batch = next(train_dataiter)

        prev_motion = None
        for pidx in range(num_primitive):
            manager.pre_step()
            motion, cond = batch[pidx]
            motion, cond = motion.to(device), cond.to(device)

            future_motion_gt = motion[:, -future_len:, :]
            gt_history = motion[:, :history_len, :]

            # 使用统一的history选择函数
            history_motion = manager.choose_history(gt_history, prev_motion,
                                                    history_len)

            latent, dist = vae_raw.encode(future_motion=future_motion_gt,
                                          history_motion=history_motion)
            future_motion_pred = vae_raw.decode(latent,
                                                history_motion,
                                                nfuture=future_len)  # [B, F, D]

            loss_dict, extras = manager.calc_loss(
                future_motion_gt,
                future_motion_pred,
                dist,
                history_motion=history_motion)
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()

            has_nan_grad = False
            for param in vae_raw.parameters():
                if param.grad is not None:
                    # 检查 NaN 和 Inf
                    if torch.isnan(param.grad).any() or torch.isinf(
                            param.grad).any():
                        has_nan_grad = True

            if not has_nan_grad:
                manager.grad_clip(vae)
                optimizer.step()

            prev_motion = future_motion_pred.detach()

            manager.post_step(is_eval=False,
                              loss_dict=toolz.valmap(
                                  lambda x: x.detach().cpu(), loss_dict),
                              extras=toolz.valmap(
                                  lambda x: x.detach().cpu()
                                  if isinstance(x, torch.Tensor) else x,
                                  extras))

        vae.eval()
        while manager.should_eval():
            batch = next(val_dataiter)
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                motion, cond = batch[pidx]
                motion, cond = motion.to(device), cond.to(device)

                future_motion_gt = motion[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                latent, dist = vae_raw.encode(future_motion=future_motion_gt,
                                              history_motion=history_motion)
                future_motion_pred = vae_raw.decode(latent,
                                                    history_motion,
                                                    nfuture=future_len)
                loss_dict, extras = manager.calc_loss(
                    future_motion_gt,
                    future_motion_pred,
                    dist,
                    history_motion=history_motion)
                manager.post_step(is_eval=True,
                                  loss_dict=toolz.valmap(
                                      lambda x: x.detach().cpu(), loss_dict),
                                  extras=toolz.valmap(
                                      lambda x: x.detach().cpu()
                                      if isinstance(x, torch.Tensor) else x,
                                      extras))
