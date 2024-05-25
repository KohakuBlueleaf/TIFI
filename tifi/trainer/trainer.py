import os
from typing import *
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import pytorch_lightning as pl
from warmup_scheduler import GradualWarmupScheduler

from diffusers import LMSDiscreteScheduler
from library.sdxl_original_unet import SdxlUNet2DConditionModel

from library.sdxl_train_util import get_size_embeddings
from library.train_util import get_noise_noisy_latents_and_timesteps
from library.custom_train_functions import apply_snr_weight

from ..utils import instantiate
from .diffusion import get_noise_noisy_latents_and_timesteps


def prepare_scheduler_for_custom_training(
    noise_scheduler: LMSDiscreteScheduler, device
):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        *args,
        name="",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_scheduler: Optional[type[lr_sch.LRScheduler]] = lr_sch.CosineAnnealingLR,
        lr_sch_configs: dict[str, Any] = {
            "T_max": 100_000,
            "eta_min": 1e-7,
        },
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate(optimizer)
        self.opt_configs = opt_configs
        self.lr = lr
        self.lr_sch = instantiate(lr_scheduler)
        self.lr_sch_configs = lr_sch_configs
        self.use_warm_up = use_warm_up
        self.warm_up_period = warm_up_period

    def configure_optimizers(self):
        assert self.train_params is not None
        optimizer = self.optimizer(self.train_params, lr=self.lr, **self.opt_configs)

        lr_sch = None
        if self.lr_sch is not None:
            lr_sch = self.lr_sch(optimizer, **self.lr_sch_configs)

        if self.use_warm_up:
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 1, self.warm_up_period, lr_sch
            )
        else:
            lr_scheduler = lr_sch

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }


class SDXLTrainer(BaseTrainer):
    def __init__(
        self,
        sdxl_unet: SdxlUNet2DConditionModel = None,
        scheduler: LMSDiscreteScheduler = None,
        lycoris_model: nn.Module = None,
        *args,
        **kwargs,
    ):
        super(SDXLTrainer, self).__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["sdxl_unet", "scheduler", "lycoris_model"])
        prepare_scheduler_for_custom_training(
            scheduler, next(sdxl_unet.parameters()).device
        )
        self.sdxl_unet = sdxl_unet
        self.scheduler = scheduler
        self.lycoris_model = lycoris_model
        self.sdxl_unet.train()
        if lycoris_model is not None:
            self.lycoris_model.train()
            self.train_params = chain(
                self.lycoris_model.parameters(),
                (p for p in self.sdxl_unet.parameters() if p.requires_grad),
            )
        else:
            self.train_params = self.sdxl_unet.parameters()
        self.epoch = 0

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.sdxl_unet.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def training_step(self, batch, idx):
        x = batch["latents"]
        blended_x = batch["blended"]

        b, f, c, h, w = x.shape
        unet_batch_size = b * f

        ctx = batch["ctx"].repeat(unet_batch_size, 1, 1)
        embed = batch["embed"].repeat(unet_batch_size, 1)
        size_emb = (
            get_size_embeddings(
                torch.FloatTensor([[h * 8, w * 8]]),
                torch.FloatTensor([[0, 0]]),
                torch.FloatTensor([[h * 8, w * 8]]),
                "cuda",
            )
            .float()
            .to(x.device)
        ).repeat(unet_batch_size, 1)
        embed = torch.cat([embed, size_emb], dim=-1)
        noisy_latent, noise, timesteps = get_noise_noisy_latents_and_timesteps(
            self.scheduler, x, blended_x
        )

        noise_pred = self.sdxl_unet(noisy_latent, timesteps, ctx, embed)
        loss = F.mse_loss(noise_pred, noise)
        # Min-Snr-Gamma 5
        loss = apply_snr_weight(loss, timesteps, self.scheduler, 5, False)

        if self._trainer is not None:
            self.log("train/loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss
