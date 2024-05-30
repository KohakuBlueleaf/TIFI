import os

import torch
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from PIL import Image

from library.sdxl_model_util import load_models_from_sdxl_checkpoint

import tifi.modules.image_caption.llava_utils as llava_utils
from tifi.utils.interpolation import (
    blend_frame_optical_flow,
)
from tifi.utils.diff_utils import load_tokenizers, encode_prompts_single


llava_utils.load_model(r"G:\nn_app\llama.cpp\models\llama3-llava-next-8b-gguf")

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

    caption = llava_utils.caption_llava(reference_frame)
    ctx, embed = encode_prompts_single(
        tokenizer, tokenizer_2, text_model1, text_model2, caption
    )
    return encode_frames(imgs), ctx, embed


def test_run():
    latents, ctx, embed = process_video(ALL_VIDEOS[0])
    print(latents.shape)
    print(ctx.shape)
    print(embed.shape)


def main():
    for path in tqdm(ALL_VIDEOS, desc="Processing videos", smoothing=0.01):
        latent_file = os.path.join(path, "latent-ctx-embed.pt")
        if os.path.exists(latent_file):
            continue
        latents, ctx, embed = process_video(path)
        torch.save(
            {"latents": latents, "ctx": ctx, "embed": embed},
            latent_file,
        )


if __name__ == "__main__":
    # test_run()
    main()
