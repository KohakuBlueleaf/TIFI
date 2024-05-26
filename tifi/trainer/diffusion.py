import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from diffusers import LMSDiscreteScheduler


def get_noise_noisy_latents_and_timesteps(
    noise_scheduler: LMSDiscreteScheduler, 
    latents, 
    blended_latents
):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0
    max_timestep = noise_scheduler.config.num_train_timesteps - 1

    timesteps = torch.randint(
        min_timestep, max_timestep, (b_size,), device=latents.device
    )

    sigmas = noise_scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(latents.device)
    timesteps = timesteps.to(latents.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    sigma_lt1 = sigma > 1
    while len(sigma.shape) < len(latents.shape):
        sigma = sigma.unsqueeze(-1)

    # Apply blended latent residual as noise
    # With larger sigma, we assume the stroke is more frome blended latent
    # With smaller sigma, we assume the stroke is from intermediate result
    blended_diff = blended_latents - latents
    blended_diff[sigma_lt1] /= sigma[sigma_lt1]**2
    blended_diff[~sigma_lt1] *= sigma[~sigma_lt1]
    noise = noise + blended_diff

    # Diffusion Forward process
    noisy_samples = latents + noise * sigma
    scale = 1 / (sigma**2 + 1) ** 0.5
    return noisy_samples * scale, noise, timesteps
