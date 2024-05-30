import os
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from PIL import Image

from library.train_util import replace_unet_modules
from library.sdxl_model_util import load_models_from_sdxl_checkpoint
from library.sdxl_train_util import get_size_embeddings

from lycoris import create_lycoris, LycorisNetwork
from tifi.modules.animatediff.loader import load, inject
from tifi.utils.interpolation import (
    blend_frame_naive,
    blend_frame_optical_flow,
)
from tifi.utils.color_fixing import match_color
from tifi.utils.diff_utils import load_tokenizers, encode_prompts


## Load Model
tokenizer, tokenizer_2 = load_tokenizers()
text_model1, text_model2, vae, unet, logit_scale, ckpt_info = (
    load_models_from_sdxl_checkpoint(
        "",
        r"./models/sdxl-1.0.safetensors",
        "cpu",
        torch.float32,
    )
)
unet.bfloat16()
unet.requires_grad_(False)
unet.cuda()
# unet.enable_gradient_checkpointing()
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
# with torch.autocast("cuda"):
#     # Simulate gradient accumulation
#     test_output = unet(test_x, test_t, test_ctx, torch.concat([size_emb, test_emb], -1))
#     loss = F.mse_loss(test_output, torch.zeros_like(test_output))
#     loss.backward()
#     test_output = unet(test_x, test_t, test_ctx, torch.concat([size_emb, test_emb], -1))
#     loss = F.mse_loss(test_output, torch.zeros_like(test_output))
#     loss.backward()
# print(test_output.shape)

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


# prompt_embeds, negative_prompt_embeds, pooled_embeds2, neg_pooled_embeds2 = (
#     encode_prompts(
#         tokenizer,
#         tokenizer_2,
#         text_model1.cuda(),
#         text_model2.cuda(),
#         """1girl,

# ciloranko, maccha (mochancc), lobelia (saclia), migolu, ask (askzy), wanke, jiu ye sang, rumoon, mizumi zumi,

# loli, solo, dragon girl, dragon tail, dragon wings, dragon horns, white dress, long hair, side up,
# river, tree, forest, pointy ears, night, night sky, starry sky, pink hair, purple eyes,
# ponytail, wings, nature, star \(sky\), outdoors, ribbon, sky, tail, looking at viewer, hair ribbon,
# horns, from side, looking back, reflection, close-up, sitting, hair ornament, reflective water, water,
# mountainous horizon, scenery, bird, kimono, short kimono, off shoulder, black kimono, japanese clothes,
# dress, red ribbon, wide sleeves, bare shoulders, looking to the side, feathered wings, very long hair,

# masterpiece, newest, absurdres, safe""",
#         "low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, multiple views, multiple tails, multiple girls, multiple boys, nude, nipples, weibo username, multiple tails",
#     )
# )
# text_model1.cpu()
# text_model2.cpu()
# torch.cuda.empty_cache()

# sigma_schedule = sigmas(
#     scheduler,
#     steps=16,
#     sigma_schedule_function=get_sigmas_exponential,
# )
# print(sigma_schedule)
# denoise_func = cfg_wrapper(
#     prompt_embeds, pooled_embeds2, negative_prompt_embeds, neg_pooled_embeds2, cfg=5.0
# )
# init_x = (
#     torch.randn(BATCH_SIZE, FRAME_SIZE, 4, 512 // 8, 896 // 8).cuda()
#     * sigma_schedule[0]
# )

# with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
#     result = sample_euler(
#         denoise_func,
#         init_x.cuda(),
#         sigma_schedule.cuda(),
#     )
#     unet.cpu()
#     torch.cuda.empty_cache()

#     vae.cuda()
#     vids = []
#     for batch in result:
#         vid = []
#         decoded_frames = vae.decode(
#             batch * (1 / vae.config.scaling_factor)
#         ).sample.float()
#         decoded_frames = (decoded_frames / 2 + 0.5).clamp(0, 1).cpu()
#         for decoded_frame in decoded_frames:
#             decoded_frame_rgb = torch.permute(decoded_frame, (1, 2, 0))
#             vid.append(
#                 Image.fromarray((decoded_frame_rgb.numpy() * 255).astype(np.uint8))
#             )
#         vids.append(vid)
#     for fid, frame in enumerate(vids[0]):
#         frame.save(f"vid/out/{fid}.png")


## Temporal Inpainting test
def frame_encode(frames: list[torch.Tensor]):
    frames_latent = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for frame in frames:
            frame = frame.unsqueeze(0).cuda() * 2 - 1
            frame_latent = (
                vae.encode(frame).latent_dist.mode() * vae.config.scaling_factor
            )
            frames_latent.append(frame_latent)
    return torch.concat(frames_latent)


def cfg_wrapper_temporal_inpainting(
    prompt, pooled, neg_prompt, neg_pooled, reference_frames, cfg=5.0
):
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
        pred = cfg * (cond - uncond) + uncond
        for batch_idx in range(b):
            for frame_idx in range(f):
                if reference_frames[batch_idx][frame_idx] is not None:
                    pred[batch_idx, frame_idx] = reference_frames[batch_idx][
                        frame_idx
                    ].to(pred)
        return pred

    return denoise


def _psnr(gen, gt):
    # The in/out of VAE is [-1, 1], scale to [0, 1] than doing PSNR
    gen = gen.cpu().float()
    gt = gt.cpu().float()
    mse = F.mse_loss(gen, gt)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr


def psnr(results, ground_truths, references):
    total_psnr = 0
    total = 0
    for v1, v2, v3 in zip(results, ground_truths, references):
        for f1, f2, f3 in zip(v1, v2, v3):
            if f3 is not None:
                continue
            total += 1
            total_psnr += _psnr(f1, f2)
    return total_psnr / total


TEST_VID = "./vid/test/0008"
frame_files = os.listdir(TEST_VID)
frames = [
    [
        (
            to_tensor(
                Image.open(os.path.join(frame_folder, frame_file))
                .convert("RGB")
                .resize((896, 512))
            )
        )
        for frame_file in os.listdir(TEST_VID)
    ]
    for frame_folder in [TEST_VID]
]
ground_truths = deepcopy(frames)
# Use previous frames as reference
for vid in frames:
    vid[1] = blend_frame_optical_flow(vid[0], vid[2], 1)[0] / 255
    vid[3] = blend_frame_optical_flow(vid[2], vid[4], 1)[0] / 255
    vid[5] = blend_frame_optical_flow(vid[4], vid[6], 1)[0] / 255


vae.cuda()
frames_latent = torch.stack([frame_encode(vid) for vid in frames])
reference_frames = [
    [frames[i] if i in {0, 2, 4, 6} else None for i in range(len(frames))]
    for frames in frames_latent
]
vae.cpu()
torch.cuda.empty_cache()


prompt_embeds, negative_prompt_embeds, pooled_embeds2, neg_pooled_embeds2 = (
    encode_prompts(
        tokenizer,
        tokenizer_2,
        text_model1.cuda(),
        text_model2.cuda(),
        """The image shows two men sitting next to each other, engaged in a discussion. 
The man on the left, wearing a black shirt and glasses, 
is holding a camera or a similar device, explaining something about it. 
The man on the right, dressed in a white shirt, is attentively listening with his hands clasped.""",
        "",
    )
)
text_model1.cpu()
text_model2.cpu()
torch.cuda.empty_cache()


sigma_schedule = sigmas(
    scheduler,
    steps=16,
    sigma_schedule_function=get_sigmas_exponential,
).cuda()
sigma_schedule_inpainting = sigma_schedule[-10:]
print(sigma_schedule_inpainting)
denoise_func = cfg_wrapper_temporal_inpainting(
    prompt_embeds,
    pooled_embeds2,
    negative_prompt_embeds,
    neg_pooled_embeds2,
    reference_frames,
    cfg=5.0,
)
init_x = torch.randn_like(frames_latent).cuda() * sigma_schedule_inpainting[0]
init_x += frames_latent


with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    result = sample_euler(
        denoise_func,
        init_x.cuda(),
        sigma_schedule_inpainting,
    )
    # unet.cpu()
    # torch.cuda.empty_cache()

    vae.cuda()
    vids = []
    vids_tensor = []
    for batch, input_batch in tqdm(
        zip(result, frames), desc="decoding videos", leave=False
    ):
        vid = []
        decoded_frames = []
        for frame, inp in tqdm(
            zip(batch, input_batch), desc="decoding frames", leave=False
        ):
            current_frame = (
                (
                    vae.decode(
                        frame.unsqueeze(0) * (1 / vae.config.scaling_factor)
                    ).sample.float()
                    / 2
                    + 0.5
                )[0]
                .clamp(0, 1)
                .cpu()
            )

            decoded_frames.append(match_color(current_frame, inp))
        decoded_frames = torch.stack(decoded_frames)

        for decoded_frame in decoded_frames:
            decoded_frame_rgb = torch.permute(decoded_frame, (1, 2, 0))
            vid.append(
                Image.fromarray((decoded_frame_rgb.numpy() * 255).astype(np.uint8))
            )
        vids_tensor.append(decoded_frames)
        vids.append(vid)
    print("PSNR before TIFI", psnr(frames, ground_truths, reference_frames))
    print("PSNR after TIFI", psnr(vids_tensor, ground_truths, reference_frames))
    for vid, video in enumerate(vids):
        for fid, frame in enumerate(video):
            frame.save(f"vid/out/{vid}_{fid}.png")
