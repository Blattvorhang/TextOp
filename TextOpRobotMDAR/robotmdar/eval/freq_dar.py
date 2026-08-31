"""
Frequency DAR Script - Model Inference Speed Benchmarking

This script measures the sequential inference speed of the DAR model by performing
autoregressive motion generation. It starts from text and zero history motion,
computes text embedding, runs through MVAE and denoiser models, performs full sampling,
and reconstructs future_motion_pred_dict and abs_pose in a continuous loop.

Usage:
- python eval/freq_dar.py --config-name=freq_dar
- The script will output timing statistics for each component and overall throughput
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
from hydra.utils import instantiate
from omegaconf import DictConfig
from collections import defaultdict

from robotmdar.dtype import seed, logger as dtype_logger
from loguru import logger
from robotmdar.dtype.motion import (motion_dict_to_abs_pose, get_zero_abs_pose,
                                    get_zero_feature_v3)
from robotmdar.dtype.abc import Dataset, VAE, Denoiser, Diffusion, SSampler
from robotmdar.train.manager import DARManager
from robotmdar.utils.dof_contract import configure_dof_contract
import clip
import torch_tensorrt


def get_text_embedding(text: str, clip_model,
                       device: str) -> Tuple[torch.Tensor, float]:
    """Encode text using CLIP model."""
    start_time = time.time()

    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1,
                                                              keepdim=True)

    timing = time.time() - start_time
    return text_embedding.float(), timing


def single_inference_step(prev_motion, abs_pose, text_embedding, vae, denoiser,
                          diffusion, val_data, history_len, future_len, cfg):
    """Perform a single inference step and return timing information."""
    step_start = time.time()
    timings = {}

    # 1. Prepare history motion
    history_motion = prev_motion[:, -history_len:, :]

    # 2. Generate latent representation
    latent_start = time.time()
    with torch.no_grad():
        batch_size = prev_motion.shape[0]
        latent_shape = (batch_size, 1, 128)  # [B, T=1, D]

        # Sample noise
        x_start_noise = torch.randn(latent_shape, device=cfg.device)

        # Prepare conditioning
        y = {
            'text_embedding': text_embedding,
            'history_motion_normalized': history_motion,
        }

        # Choose sampling method
        use_full_sample = getattr(cfg, 'use_full_sample', False)

        if use_full_sample:
            # Full DDPM sampling
            sample_fn = diffusion.p_sample_loop
            x_start_full = sample_fn(
                denoiser,
                x_start_noise.shape,
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            latent_pred = x_start_full.permute(1, 0, 2)  # [T=1, B, D]
        else:
            # Single-step denoising
            t = torch.zeros(batch_size, dtype=torch.int32, device=cfg.device)
            x_start_pred = denoiser(x_t=x_start_noise,
                                    timesteps=diffusion._scale_timesteps(t),
                                    y=y)
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]

    timings['denoiser_inference'] = time.time() - latent_start

    # 3. VAE decoding
    vae_start = time.time()
    with torch.no_grad():
        future_motion_pred = vae.decode(latent_pred,
                                        history_motion,
                                        nfuture=future_len)
    timings['vae_decode'] = time.time() - vae_start

    # 4. Motion reconstruction
    recon_start = time.time()
    full_motion = torch.cat([history_motion, future_motion_pred], dim=1)
    future_motion_pred_dict = val_data.reconstruct_motion(full_motion,
                                                          abs_pose=abs_pose,
                                                          ret_fk=False)
    timings['motion_reconstruction'] = time.time() - recon_start

    # 5. Update state for next iteration
    new_prev_motion = full_motion
    new_abs_pose = motion_dict_to_abs_pose(future_motion_pred_dict, idx=-2)

    timings['total_step'] = time.time() - step_start

    return new_prev_motion, new_abs_pose, future_motion_pred_dict, timings


def run_benchmark(vae,
                  denoiser,
                  diffusion,
                  val_data,
                  clip_model,
                  cfg,
                  num_iterations=100,
                  warmup_iterations=10):
    """Run the frequency benchmark."""
    logger.info(
        f"Starting benchmark with {warmup_iterations} warmup + {num_iterations} test iterations"
    )

    # Initialize state
    history_len = cfg.data.history_len
    future_len = cfg.data.future_len
    batch_size = 1

    # Initialize with zero motion
    zero_feature = get_zero_feature_v3().unsqueeze(0).expand(
        batch_size, history_len, -1).to(cfg.device)
    prev_motion = zero_feature
    abs_pose = get_zero_abs_pose((batch_size, ), device=cfg.device)

    # Get text embedding
    text_prompt = getattr(cfg, 'text_prompt', "walk forward")
    text_embedding, text_time = get_text_embedding(text_prompt, clip_model,
                                                   cfg.device)
    logger.info(
        f"Text embedding computed in {text_time*1000:.2f}ms for: '{text_prompt}'"
    )

    # Warmup phase
    logger.info("Warmup phase...")
    for i in range(warmup_iterations):
        prev_motion, abs_pose, _, _ = single_inference_step(
            prev_motion, abs_pose, text_embedding, vae, denoiser, diffusion,
            val_data, history_len, future_len, cfg)
        if (i + 1) % 5 == 0:
            logger.info(f"Warmup: {i + 1}/{warmup_iterations}")

    # Benchmark phase
    logger.info("Benchmark phase...")
    all_timings = defaultdict(list)
    start_time = time.time()

    for i in range(num_iterations):
        prev_motion, abs_pose, future_motion_pred_dict, timings = single_inference_step(
            prev_motion, abs_pose, text_embedding, vae, denoiser, diffusion,
            val_data, history_len, future_len, cfg)

        # Collect timings
        for component, timing in timings.items():
            all_timings[component].append(timing)

        if (i + 1) % 10 == 0:
            current_freq = (i + 1) / (time.time() - start_time)
            logger.info(
                f"Progress: {i + 1}/{num_iterations}, Current freq: {current_freq:.2f} Hz"
            )

    total_benchmark_time = time.time() - start_time

    # Display statistics
    display_statistics(all_timings, total_benchmark_time, num_iterations,
                       cfg.device)


def display_statistics(timings: Dict[str, List[float]], total_time: float,
                       num_iterations: int, device: str):
    """Display benchmark statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)

    # Overall frequency
    overall_freq = num_iterations / total_time
    logger.info(f"Overall Frequency: {overall_freq:.2f} Hz")
    logger.info(f"Average Step Time: {total_time/num_iterations*1000:.2f} ms")

    # Component-wise timing statistics
    logger.info("\nComponent Timing Statistics (ms):")
    logger.info("-" * 40)

    for component, times in timings.items():
        times_ms = [t * 1000 for t in times]
        mean_time = np.mean(times_ms)
        std_time = np.std(times_ms)
        min_time = np.min(times_ms)
        max_time = np.max(times_ms)

        logger.info(f"{component:20s}: {mean_time:6.2f} ± {std_time:5.2f} "
                    f"(min: {min_time:5.2f}, max: {max_time:6.2f})")

    # Percentage breakdown
    logger.info("\nTime Distribution:")
    logger.info("-" * 40)
    total_step_times = timings['total_step']
    avg_total = np.mean(total_step_times) * 1000

    for component, times in timings.items():
        if component != 'total_step':
            avg_component = np.mean(times) * 1000
            percentage = (avg_component / avg_total) * 100
            logger.info(f"{component:20s}: {percentage:5.1f}%")

    # Memory usage (if available)
    if torch.cuda.is_available() and 'cuda' in device:
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        logger.info(f"\nGPU Memory Usage:")
        logger.info(f"Allocated: {memory_allocated:.2f} GB")
        logger.info(f"Reserved:  {memory_reserved:.2f} GB")

    logger.info("=" * 60)


def main(cfg: DictConfig):
    """Main function to run the frequency benchmark."""
    configure_dof_contract(cfg)
    dtype_logger.set(cfg)
    seed.set(cfg.seed)

    logger.info("Loading models...")

    # Load CLIP model for text encoding
    clip_model, _ = clip.load("ViT-B/32", device=cfg.device)
    clip_model.eval()

    # Load dataset (for motion reconstruction)
    val_data: Dataset = instantiate(cfg.data.val)

    # Load VAE and Denoiser
    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    # Load diffusion model
    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    # Set models to eval mode
    vae.eval()
    denoiser.eval()

    # Load checkpoints
    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)
    vae_trt = torch.compile(vae, backend='tensorrt')
    denoiser_trt = torch.compile(denoiser, backend='tensorrt')

    logger.info("Models loaded successfully")

    # Run benchmark
    num_iterations = getattr(cfg, 'num_iterations', 100)
    warmup_iterations = getattr(cfg, 'warmup_iterations', 10)

    run_benchmark(vae_trt, denoiser_trt, diffusion, val_data, clip_model, cfg,
                  num_iterations, warmup_iterations)

    logger.info(f"Config: ")
    logger.info(f"use_full_sample: {cfg.use_full_sample}")
    # logger.info(f"guidance_scale: {cfg.guidance_scale}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None,
                config_path="../config",
                config_name="freq_dar")
    def hydra_main(cfg: DictConfig) -> None:
        main(cfg)

    hydra_main()
