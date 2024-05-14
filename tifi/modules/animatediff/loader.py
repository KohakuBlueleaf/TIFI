import os

from einops import rearrange

from library.sdxl_original_unet import SdxlUNet2DConditionModel, GroupNorm32

from tifi.logging import logger
from tifi.utils.model import read_state_dict
from tifi.modules.animatediff.motion_module import MotionModuleType, MotionWrapper


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
    mm.half()
    return mm


def inject(unet: SdxlUNet2DConditionModel, mm: MotionWrapper):
    if mm.enable_gn_hack():
        logger.info(f"Hacking GroupNorm32 forward function.")
        gn32_original_forward = GroupNorm32.forward

        def groupnorm32_mm_forward(self, x):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
            x = gn32_original_forward(self, x)
            x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
            return x

        GroupNorm32.orig_forward = gn32_original_forward
        GroupNorm32.forward = groupnorm32_mm_forward

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
