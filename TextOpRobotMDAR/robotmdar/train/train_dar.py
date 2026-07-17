import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler

from robotmdar.train.manager import DARManager, is_main_process, get_ddp_model

USE_VAE = True


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


def main(cfg: DictConfig):
    # Initialize DDP
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f'cuda:{local_rank}')

    seed.set(cfg.seed + rank)
    logger.set(cfg)

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
    denoiser: Denoiser = instantiate(cfg.denoiser)
    vae = vae.to(device)
    denoiser = denoiser.to(device)

    if world_size > 1:
        denoiser = torch.nn.parallel.DistributedDataParallel(
            denoiser, device_ids=[local_rank], output_device=local_rank
        )
        # Note: VAE is frozen during DAR training, so it doesn't need DDP wrapping
        if is_main_process():
            print(f"[DDP] Using DistributedDataParallel with {world_size} GPUs")

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    optimizer: Optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.manager.learning_rate)

    manager: DARManager = instantiate(cfg.train.manager)

    manager.hold_model(vae, denoiser, optimizer, train_data)
    manager.rank = rank
    manager.world_size = world_size

    num_primitive: int = cfg.data.num_primitive
    future_len: int = cfg.data.future_len
    history_len: int = cfg.data.history_len

    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)

    # Training loop following train_mvae.py approach
    while manager:
        denoiser.train()
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

            # Sample timesteps
            batch_size = motion.shape[0]
            t, weights = schedule_sampler.sample(batch_size, device=device)

            if USE_VAE:
                # Encode using VAE
                latent_gt, _ = vae.encode(
                    future_motion=future_motion_gt,
                    history_motion=history_motion
                )  # [T=1, B, D]   latent_gt: (1, 512, 128)

                x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]
            else:
                latent_gt = None
                x_start = torch.cat((history_motion, future_motion_gt), dim=1)

            # Forward diffusion

            x_t = diffusion.q_sample(x_start=x_start,
                                     t=t,
                                     noise=torch.randn_like(x_start))

            # Denoise
            y = {
                'text_embedding':
                cond,  # cond is already the text_embedding tensor
                'history_motion_normalized': history_motion,
            }
            x_start_pred = denoiser(x_t=x_t,
                                    timesteps=diffusion._scale_timesteps(t),
                                    y=y)  # [B, T=1, D]
            # breakpoint()
            if USE_VAE:
                latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]

                # Decode
                future_motion_pred = vae.decode(
                    latent_pred, history_motion,
                    nfuture=future_len)  # [B, F, D], normalized
            else:
                latent_pred = None
                future_motion_pred = x_start_pred[:, -future_len:]

            # Calculate loss
            loss_dict, extras = manager.calc_loss(
                future_motion_gt,
                future_motion_pred,
                latent_gt,
                None,
                latent_pred,
                weights,
                history_motion=history_motion  # dist=None for DAR
            )
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            has_nan_grad = False
            raw_model = get_ddp_model(denoiser)
            for param in raw_model.parameters():
                if param.grad is not None:
                    # 检查 NaN 和 Inf
                    if torch.isnan(param.grad).any() or torch.isinf(
                            param.grad).any():
                        has_nan_grad = True

            if not has_nan_grad:
                manager.grad_clip(denoiser)
                optimizer.step()

            # 更新prev_motion，如果启用full sample则使用更高质量的采样
            if manager.should_use_full_sample():
                with torch.no_grad():
                    # 使用完整的DDPM采样循环来生成更高质量的rollout history
                    sample_fn = diffusion.p_sample_loop
                    x_start_full = sample_fn(
                        denoiser,
                        x_start.shape,
                        clip_denoised=False,
                        model_kwargs={'y':
                                      y},  # Wrap y in the expected structure
                        skip_timesteps=0,
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )
                    # 确保x_start_full是tensor并转换维度
                    if isinstance(x_start_full, torch.Tensor):
                        latent_full = x_start_full.permute(1, 0,
                                                           2)  # [T=1, B, D]
                    else:
                        # 如果返回的是其他格式，直接使用原始预测
                        latent_full = latent_pred
                    future_motion_full = vae.decode(latent_full,
                                                    history_motion,
                                                    nfuture=future_len)
                    prev_motion = torch.cat(
                        [history_motion, future_motion_full], dim=1).detach()
            else:
                prev_motion = torch.cat([history_motion, future_motion_pred],
                                        dim=1).detach()
            manager.post_step(
                is_eval=False,
                loss_dict={
                    k: v.detach().cpu()
                    for k, v in loss_dict.items()
                },
                extras={
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in extras.items()
                })

        # Validation loop
        denoiser.eval()
        while manager.should_eval():
            batch = next(val_dataiter)
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                motion, cond = batch[pidx]
                motion, cond = motion.to(device), cond.to(device)

                future_motion_gt = motion[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                with torch.no_grad():
                    t, weights = schedule_sampler.sample(motion.shape[0],
                                                         device=device)

                    if USE_VAE:
                        latent_gt, _ = vae.encode(
                            future_motion=future_motion_gt,
                            history_motion=history_motion)
                        # Forward diffusion
                        x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]
                    else:
                        latent_gt = None
                        x_start = torch.cat((history_motion, future_motion_gt),
                                            dim=1)

                    x_t = diffusion.q_sample(x_start=x_start,
                                             t=t,
                                             noise=torch.randn_like(x_start))

                    y = {
                        'text_embedding':
                        cond,  # cond is already the text_embedding tensor
                        'history_motion_normalized': history_motion,
                    }
                    x_start_pred = denoiser(
                        x_t=x_t, timesteps=diffusion._scale_timesteps(t), y=y)

                    if USE_VAE:
                        latent_pred = x_start_pred.permute(1, 0, 2)

                        future_motion_pred = vae.decode(latent_pred,
                                                        history_motion,
                                                        nfuture=future_len)

                    else:
                        latent_pred = None
                        future_motion_pred = x_start_pred[:, -future_len:]

                    loss_dict, extras = manager.calc_loss(
                        future_motion_gt,
                        future_motion_pred,
                        latent_gt,
                        None,
                        latent_pred,
                        weights,
                        history_motion=history_motion)

                manager.post_step(
                    is_eval=True,
                    loss_dict={
                        k: v.detach().cpu()
                        for k, v in loss_dict.items()
                    },
                    extras={
                        k:
                        v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in extras.items()
                    })
