import torch
import torch.utils.data as data
from diffusers import LMSDiscreteScheduler

torch.set_float32_matmul_precision("medium")

from library.train_util import replace_unet_modules
from library.sdxl_model_util import load_models_from_sdxl_checkpoint

from lycoris import create_lycoris, LycorisNetwork
from tifi.modules.animatediff.loader import load, inject
from tifi.trainer import SDXLTrainer
from tifi.trainer.data import LatentVideoDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


EPOCH = 10
GPUS = 1
BATCH_SIZE = 4
GRAD_ACC = 4


def load_model(model_file="./models/sdxl-1.0.safetensors"):
    ## Load Model
    unet = load_models_from_sdxl_checkpoint(
        "",
        model_file,
        "cpu",
        torch.bfloat16,
    )[3]
    mm = load("./models/mm_sdxl_hs.safetensors", "cpu")
    inject(unet, mm)

    ## Apply neede property
    unet.bfloat16()
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    replace_unet_modules(unet, False, True, False)

    ## Apply LyCORIS
    LycorisNetwork.apply_preset(
        {
            "enable_conv": True,
            "target_module": [
                "MotionModule",
                "Transformer2DModel",
                "ResnetBlock2D",
            ],
            "module_algo_map": {
                "VersatileAttention": {
                    "algo": "lora",
                    "dim": 32,
                    "alpha": 4,
                },
                "CrossAttention": {
                    "algo": "lora",
                    "dim": 16,
                    "alpha": 4,
                },
            },
        }
    )
    lycoris_network: LycorisNetwork = create_lycoris(
        unet, 1, 1, 1, factor=4, algo="lokr", full_matrix=True, train_norm=True
    )
    lycoris_network.apply_to()

    return unet, lycoris_network


def load_trainer(
    model, scheduler, lycoris_model=None, lr=5e-5, t_max=1000_000
) -> SDXLTrainer:
    return SDXLTrainer(
        model,
        scheduler,
        lycoris_model,
        name="TIFI-lyco",
        lr=lr,
        optimizer=torch.optim.AdamW,
        opt_configs={
            "weight_decay": 0.01,
            "betas": (0.9, 0.98),
        },
        lr_scheduler=None,
        lr_sch_configs=None,
        use_warm_up=True,
        warm_up_period=100,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    ## Load Model
    unet, lycoris_network = load_model()

    ## dataloader
    dataset = LatentVideoDataset()
    dataloader = data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    ## Trainer
    trainer_module = load_trainer(
        unet,
        LMSDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
        ),
        lycoris_network,
        5e-5,
        t_max=len(dataloader) * EPOCH,
    )

    ## Training
    logger = None
    logger = WandbLogger(
        name="TIFI-lyco-ft",
        project="TIFI-sdxl",
        offline=True,
    )
    trainer = pl.Trainer(
        precision="bf16",
        accelerator="gpu",
        # strategy="dp",
        devices=GPUS,
        max_epochs=EPOCH,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=GRAD_ACC,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_epochs=1),
        ],
        gradient_clip_val=1.0,
        # fast_dev_run=True,
    )
    trainer.fit(
        trainer_module,
        train_dataloaders=dataloader,
    )
