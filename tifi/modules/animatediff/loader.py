import os

from einops import rearrange

from library.sdxl_original_unet import SdxlUNet2DConditionModel, GroupNorm32

from tifi.logging import logger
from tifi.utils.model import read_state_dict
from tifi.modules.animatediff.motion_module import (
    MotionModuleType,
    MotionWrapper,
    VersatileAttention,
)


def load(model_name: str, device="cuda"):
    model_path = model_name
    if not os.path.isfile(model_path):
        raise RuntimeError("Please download models manually.")
    logger.info(f"Loading motion module {model_name} from {model_path}")
    model_hash = None
    mm_state_dict = read_state_dict(model_path)
    model_type = MotionModuleType.get_mm_type(mm_state_dict)
    logger.info(f"Guessed {model_name} architecture: {model_type}")
    mm_config = dict(mm_name=model_name, mm_hash=model_hash, mm_type=model_type)
    mm = MotionWrapper(**mm_config)
    mm.load_state_dict(mm_state_dict)
    mm.to(device).eval()
    return mm


def inject(unet: SdxlUNet2DConditionModel, mm: MotionWrapper):
    batch_size = 0
    all_versatile_attention: list[VersatileAttention] = []
    unet_original_forward = SdxlUNet2DConditionModel.forward

    def unet_forward_patch(self, x, *args, **kwargs):
        nonlocal batch_size
        B, F, C, H, W = x.shape
        batch_size = B
        for module in all_versatile_attention:
            module.video_length = F
        x = x.reshape(B * F, C, H, W)
        result = unet_original_forward(self, x, *args, **kwargs)
        result = result.reshape(B, F, C, H, W)
        return result

    SdxlUNet2DConditionModel.forward = unet_forward_patch

    if mm.enable_gn_hack():
        logger.info(f"Hacking GroupNorm32 forward function.")
        gn32_original_forward = GroupNorm32.forward

        def groupnorm32_mm_forward(self, x):
            nonlocal batch_size
            _, C, H, W = x.shape
            x = rearrange(x, "(b f) c h w -> b c f h w", b=batch_size)
            x = gn32_original_forward(self, x)
            x = rearrange(x, "b c f h w -> (b f) c h w", b=batch_size)
            return x

        GroupNorm32.orig_forward = gn32_original_forward
        GroupNorm32.forward = groupnorm32_mm_forward

    for module in mm.modules():
        if isinstance(module, VersatileAttention):
            all_versatile_attention.append(module)

    logger.info(f"Injecting motion module into UNet input blocks.")
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        mm_inject = getattr(mm.down_blocks[mm_idx0], "motion_modules")[mm_idx1]
        unet.input_blocks[unet_idx].append(mm_inject)

    logger.info(f"Injecting motion module into UNet output blocks.")
    for unet_idx in range(9):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        mm_inject = getattr(mm.up_blocks[mm_idx0], "motion_modules")[mm_idx1]
        if unet_idx % 3 == 2 and unet_idx != (8 if mm.is_xl else 11):
            unet.output_blocks[unet_idx].insert(
                len(unet.output_blocks[unet_idx]) - 1, mm_inject
            )
        else:
            unet.output_blocks[unet_idx].append(mm_inject)

    logger.info(f"Injection finished.")
