import torch
import toolz
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import Dataset, VAE, Optimizer
from robotmdar.train.manager import MVAEManager


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
    logger.set(cfg)
    seed.set(cfg.seed)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    optimizer: Optimizer = torch.optim.Adam(vae.parameters(), **cfg.train.opt)
    manager: MVAEManager = instantiate(cfg.train.manager)

    manager.hold_model(vae, optimizer, train_data)
    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)

    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    # all_normalized = []

    # for i in range(100):
    #     batch = next(train_dataiter)
    #     for pidx in range(num_primitive):
    #         all_normalized.append(batch[pidx][0])
    # normalized_data = torch.cat(all_normalized)
    # dist_result = evaluate_distribution_match(normalized_data)
    # print(dist_result)
    # breakpoint()

    while manager:
        vae.train()
        batch = next(train_dataiter)

        prev_motion = None
        for pidx in range(num_primitive):
            manager.pre_step()
            motion = batch[pidx]['motion'].to(cfg.device)

            future_motion_gt = motion[:, -future_len:, :]
            gt_history = motion[:, :history_len, :]

            # 使用统一的history选择函数
            history_motion = manager.choose_history(gt_history, prev_motion,
                                                    history_len)

            latent, dist = vae.encode(future_motion=future_motion_gt,
                                      history_motion=history_motion)
            future_motion_pred = vae.decode(latent,
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
            for param in vae.parameters():
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
                motion = batch[pidx]['motion'].to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                latent, dist = vae.encode(future_motion=future_motion_gt,
                                          history_motion=history_motion)
                future_motion_pred = vae.decode(latent,
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
