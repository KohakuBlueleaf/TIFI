import os
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from PIL import Image

from library.sdxl_model_util import load_models_from_sdxl_checkpoint
from tifi.utils.interpolation import (
    blend_frame_optical_flow,
)
from tifi.utils.diff_utils import load_tokenizers, encode_prompts_single

from llava_utils import caption_llava


DATASET_DIR = r"dataset\choosed_septuplet\choosed_septuplet\sequences"
ALL_VIDEOS = [os.path.join(DATASET_DIR, v) for v in os.listdir(DATASET_DIR)]


## Load Model
tokenizer, tokenizer_2 = load_tokenizers()
text_model1, text_model2, vae, _, _, _ = load_models_from_sdxl_checkpoint(
    "",
    r"./models/sdxl-1.0.safetensors",
    "cpu",
    torch.float32,
)
vae.cuda()
text_model1.cuda()
text_model2.cuda()


def encode_frames(frames: list[torch.Tensor]):
    frame_latents = []
    for frame in frames:
        frame = frame.unsqueeze(0).bfloat16().cuda() * 2 - 1
        frame_latent = vae.encode(frame).latent_dist.mode() * vae.config.scaling_factor
        frame_latents.append(frame_latent)
    return torch.concat(frame_latents)


@torch.no_grad()
@torch.autocast("cuda", torch.bfloat16)
def process_video(path):
    imgs = [
        to_tensor(
            Image.open(os.path.join(path, frame_file)).convert("RGB").resize((896, 512))
        )
        for frame_file in sorted(os.listdir(path))
    ]
    length = len(imgs)
    for i in range(length - 1):
        for j in range(i + 2, length):
            imgs.extend(blend_frame_optical_flow(imgs[i], imgs[j], j - i - 1))

    reference_frame = os.path.join(path, os.listdir(path)[0])

    caption = caption_llava(reference_frame)
    ctx, embed = encode_prompts_single(
        tokenizer, tokenizer_2, text_model1, text_model2, caption
    )
    return encode_frames(imgs), ctx, embed


idx = 7
length = 7
frame_idx_map = {}
for i in range(length - 1):
    for j in range(i + 2, length):
        for k in range(i + 1, j):
            frame_idx_map[(i, j, k)] = idx
            idx += 1


def frame_index_gen(gt_frames=[0, 2, 4, 6]):
    prev_gt = 0
    current_idx = 0
    all_idx = []
    for gt_idx in gt_frames:
        while current_idx < gt_idx:
            all_idx.append(frame_idx_map[(prev_gt, gt_idx, current_idx)])
            current_idx += 1
        all_idx.append(gt_idx)
        prev_gt = gt_idx
        current_idx += 1
    return all_idx


def test_run():
    latents, ctx, embed = process_video(ALL_VIDEOS[0])
    print(latents.shape)
    print(ctx.shape)
    print(embed.shape)
    print(latents[frame_index_gen([0, 2, 4, 6])].shape)


def main():
    for path in tqdm(ALL_VIDEOS, desc="Processing videos", smoothing=0.01):
        latents, ctx, embed = process_video(path)
        torch.save(
            {"latents": latents, "ctx": ctx, "embed": embed},
            os.path.join(path, "latent-ctx-embed.pt"),
        )


if __name__ == "__main__":
    # test_run()
    main()
