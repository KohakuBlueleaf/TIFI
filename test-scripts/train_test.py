from time import time_ns

import torch
from diffusers import LMSDiscreteScheduler

from library.train_util import replace_unet_modules
from library.sdxl_model_util import load_models_from_sdxl_checkpoint

from lycoris import create_lycoris, LycorisNetwork
from tifi.modules.animatediff.loader import load, inject
from tifi.utils.interpolation import frame_idx_map_gen, frame_index_gen
from tifi.trainer import SDXLTrainer


## Load Model
unet = load_models_from_sdxl_checkpoint(
    "",
    r"./models/sdxl-1.0.safetensors",
    "cpu",
    torch.float32,
)[3]
unet.bfloat16()
unet.requires_grad_(False)
unet.cuda()
unet.enable_gradient_checkpointing()
replace_unet_modules(unet, False, True, False)
mm = load("./models/mm_sdxl_hs.safetensors")
inject(unet, mm)
mm.bfloat16()

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
            },
            "CrossAttention": {
                "algo": "lora",
                "dim": 16,
            },
        },
    }
)
lycoris_network: LycorisNetwork = create_lycoris(
    unet, 1, 1, 1, factor=4, algo="lokr", full_matrix=True, train_norm=True
)
lycoris_network.apply_to()
lycoris_network.cuda()

print(sum(param.shape.numel() for param in unet.parameters()) / 1e6)
print(sum(param.shape.numel() for param in lycoris_network.parameters()) / 1e6)


## Test data
fid_map = frame_idx_map_gen(7)

test_data = torch.load(
    rf"dataset\choosed_septuplet\choosed_septuplet\sequences\00001_0029\latent-ctx-embed.pt"
)
latents = test_data["latents"]
ctx = test_data["ctx"].cuda()
embed = test_data["embed"].cuda()
x = latents[:7].cuda().unsqueeze(0)
blended_x = latents[frame_index_gen([0, 2, 4, 6], fid_map)].cuda().unsqueeze(0)

print(x.shape, blended_x.shape, ctx.shape, embed.shape)

## Trainer
trainer = SDXLTrainer(
    unet,
    LMSDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
    ),
    lycoris_network,
).cuda()


with torch.autocast("cuda", torch.bfloat16):
    loss = trainer.training_step(
        {
            "latents": x,
            "blended": blended_x,
            "ctx": ctx,
            "embed": embed,
        },
        0,
    )
    loss.backward()
    print(loss)

    t0 = time_ns()
    loss = trainer.training_step(
        {
            "latents": x,
            "blended": blended_x,
            "ctx": ctx,
            "embed": embed,
        },
        0,
    )
    loss.backward()
    torch.cuda.synchronize()
    t1 = time_ns()
    print((t1 - t0) / 1e6)
