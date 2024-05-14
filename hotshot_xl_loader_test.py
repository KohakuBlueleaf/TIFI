import torch

from library.sdxl_model_util import load_models_from_sdxl_checkpoint
from tifi.modules.animatediff.loader import load, inject


text_model1, text_model2, vae, unet, logit_scale, ckpt_info = (
    load_models_from_sdxl_checkpoint(
        "", "./models/sdxl-1.0.safetensors", "cpu", torch.float16
    )
)
mm = load("./models/mm_sdxl_hs.safetensors")
inject(unet, mm)

print(sum(param.shape.numel() for param in unet.parameters()) / 1e9)
