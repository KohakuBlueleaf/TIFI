import torch
import torch.nn as nn
import torch.nn.functional as F

from library.train_util import replace_unet_modules
from library.sdxl_model_util import load_models_from_sdxl_checkpoint
from library.sdxl_train_util import get_size_embeddings

from lycoris import create_lycoris, LycorisNetwork
from tifi.modules.animatediff.loader import load, inject


## Load Model
text_model1, text_model2, vae, unet, logit_scale, ckpt_info = (
    load_models_from_sdxl_checkpoint(
        "", "./models/sdxl-1.0.safetensors", "cpu", torch.float16
    )
)
unet.requires_grad_(False)
unet.cuda()
unet.enable_gradient_checkpointing()
replace_unet_modules(unet, False, True, False)
mm = load("./models/mm_sdxl_hs.safetensors")
inject(unet, mm)

LycorisNetwork.apply_preset(
    {
        "enable_conv": True,
        "target_module": [
            "MotionModule",
            "Transformer2DModel",
            "ResnetBlock2D",
        ],
        "module_algo_map": {
            "TemporalTransformer3DModel": {
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


## Dummy Test
BATCH_SIZE = 1
FRAME_SIZE = 7

test_x = torch.randn(BATCH_SIZE, FRAME_SIZE, 4, 896 // 8, 512 // 8).cuda()
test_t = torch.randint(0, 1000, (BATCH_SIZE,)).repeat(FRAME_SIZE).cuda()
size_emb = (
    get_size_embeddings(
        torch.LongTensor([[896, 512]]),
        torch.LongTensor([[0, 0]]),
        torch.LongTensor([[896, 512]]),
        "cuda",
    )
    .float()
    .repeat(BATCH_SIZE * FRAME_SIZE, 1)
    .cuda()
)
test_emb = torch.randn(BATCH_SIZE, 1280).repeat(FRAME_SIZE, 1).cuda()
test_ctx = torch.randn(BATCH_SIZE, 77, 2048).repeat(FRAME_SIZE, 1, 1).cuda()
with torch.autocast("cuda"):
    # Simulate gradient accumulation
    test_output = unet(test_x, test_t, test_ctx, torch.concat([size_emb, test_emb], -1))
    loss = F.mse_loss(test_output, torch.zeros_like(test_output))
    loss.backward()
    test_output = unet(test_x, test_t, test_ctx, torch.concat([size_emb, test_emb], -1))
    loss = F.mse_loss(test_output, torch.zeros_like(test_output))
    loss.backward()
print(test_output.shape)

## Generating Test
