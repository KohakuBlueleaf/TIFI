import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from library.train_util import replace_unet_modules
from library.sdxl_model_util import load_models_from_sdxl_checkpoint
from library.sdxl_train_util import get_size_embeddings

from lycoris import create_lycoris, LycorisNetwork
from tifi.modules.animatediff.loader import load, inject
from diff_utils import load_tokenizers, encode_prompts


## Load Model
tokenizer, tokenizer_2 = load_tokenizers()
text_model1, text_model2, vae, unet, logit_scale, ckpt_info = (
    load_models_from_sdxl_checkpoint(
        "",
        r"G:\stable-diffusion-webui\models\Stable-diffusion\xl-epsilon\kohaku-xl-epsilon-rev1.safetensors",
        "cpu",
        torch.float16,
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
# lycoris_network: LycorisNetwork = create_lycoris(
#     unet, 1, 1, 1, factor=4, algo="lokr", full_matrix=True, train_norm=True
# )
# lycoris_network.apply_to()
# lycoris_network.cuda()

print(sum(param.shape.numel() for param in unet.parameters()) / 1e6)
# print(sum(param.shape.numel() for param in lycoris_network.parameters()) / 1e6)


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

torch.cuda.empty_cache()

## Generating Test
from diffusers import LMSDiscreteScheduler
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import sample_euler, get_sigmas_exponential


scheduler: LMSDiscreteScheduler = LMSDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
denoiser: DiscreteEpsDDPMDenoiser = DiscreteEpsDDPMDenoiser(
    unet, scheduler.alphas_cumprod, False
).cuda()


def sigmas(
    scheduler: LMSDiscreteScheduler,
    steps=16,
    sigma_schedule_function=get_sigmas_exponential,
    **kwargs,
):
    sigma_min = scheduler.sigmas[-2]
    sigma_max = scheduler.sigmas[0]
    sigmas = sigma_schedule_function(steps + 1, sigma_min, sigma_max, **kwargs)
    return torch.cat([sigmas[:-2], scheduler.sigmas.new_zeros([1])])


def cfg_wrapper(prompt, pooled, neg_prompt, neg_pooled, cfg=5.0):
    def denoise(x, t):
        b, f, c, h, w = x.shape
        batch_size = b * f
        size_emb = (
            get_size_embeddings(
                torch.FloatTensor([[h * 8, w * 8]]),
                torch.FloatTensor([[0, 0]]),
                torch.FloatTensor([[h * 8, w * 8]]),
                "cuda",
            )
            .float()
            .repeat(batch_size, 1)
            .to(x.device)
        )
        cond = denoiser(
            x,
            t,
            context=prompt.repeat(batch_size, 1, 1),
            y=torch.cat([pooled.repeat(batch_size, 1), size_emb], dim=-1),
        )
        uncond = denoiser(
            x,
            t,
            context=neg_prompt.repeat(batch_size, 1, 1),
            y=torch.cat([neg_pooled.repeat(batch_size, 1), size_emb], dim=-1),
        )
        return cfg * (cond - uncond) + uncond

    return denoise


prompt_embeds, negative_prompt_embeds, pooled_embeds2, neg_pooled_embeds2 = (
    encode_prompts(
        tokenizer,
        tokenizer_2,
        text_model1.cuda(),
        text_model2.cuda(),
        """1girl,

ciloranko, maccha (mochancc), lobelia (saclia), migolu, ask (askzy), wanke, jiu ye sang, rumoon, mizumi zumi,

loli, solo, dragon girl, dragon tail, dragon wings, dragon horns, white dress, long hair, side up, 
river, tree, forest, pointy ears, night, night sky, starry sky, pink hair, purple eyes, 
ponytail, wings, nature, star \(sky\), outdoors, ribbon, sky, tail, looking at viewer, hair ribbon, 
horns, from side, looking back, reflection, close-up, sitting, hair ornament, reflective water, water, 
mountainous horizon, scenery, bird, kimono, short kimono, off shoulder, black kimono, japanese clothes, 
dress, red ribbon, wide sleeves, bare shoulders, looking to the side, feathered wings, very long hair,

masterpiece, newest, absurdres, safe""",
        "low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, multiple views, multiple tails, multiple girls, multiple boys, nude, nipples, weibo username, multiple tails",
    )
)
text_model1.cpu()
text_model2.cpu()
torch.cuda.empty_cache()

sigma_schedule = sigmas(
    scheduler,
    steps=16,
    sigma_schedule_function=get_sigmas_exponential,
)
print(sigma_schedule)
denoise_func = cfg_wrapper(
    prompt_embeds, pooled_embeds2, negative_prompt_embeds, neg_pooled_embeds2, cfg=5.0
)
init_x = (
    torch.randn(BATCH_SIZE, FRAME_SIZE, 4, 512 // 8, 896 // 8).cuda()
    * sigma_schedule[0]
)

with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    result = sample_euler(
        denoise_func,
        init_x.cuda(),
        sigma_schedule.cuda(),
    )
    unet.cpu()
    torch.cuda.empty_cache()

    vae.cuda()
    vids = []
    for batch in result:
        vid = []
        decoded_frames = vae.decode(batch * (1 / 0.13025)).sample.float()
        decoded_frames = (decoded_frames / 2 + 0.5).clamp(0, 1).cpu()
        for decoded_frame in decoded_frames:
            decoded_frame_rgb = torch.permute(decoded_frame, (1, 2, 0))
            vid.append(
                Image.fromarray((decoded_frame_rgb.numpy() * 255).astype(np.uint8))
            )
        vids.append(vid)
    for fid, frame in enumerate(vids[0]):
        frame.save(f"vid/out/{fid}.png")
