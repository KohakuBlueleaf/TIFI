import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from diffusers import LMSDiscreteScheduler


def get_noise_noisy_latents_and_timesteps(
    noise_scheduler: LMSDiscreteScheduler, latents, blended_latents
):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0
    max_timestep = noise_scheduler.config.num_train_timesteps

    timesteps = torch.randint(
        min_timestep, max_timestep, (b_size,), device=latents.device
    )

    sigmas = noise_scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(latents.device)
    timesteps = timesteps.to(latents.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(latents.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = blended_latents + noise * sigma
    scale = 1 / (sigma**2 + 1**2) ** 0.5
    return noisy_samples * scale, noise + (blended_latents - latents) / sigma, timesteps
