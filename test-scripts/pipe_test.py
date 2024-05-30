import os, sys
from copy import deepcopy

sys.path.append(".")

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

from tifi.pipe.temporal_inpainting import TemporalInpainting
from tifi.modules.metrics.metrics import LPIPS, MSSSIM, SSIM, PSNR


lpips = LPIPS(net="vgg")
ssim = SSIM()
msssim = MSSSIM()
psnr = PSNR()


def make_video_tensor(frames: list[list[Image.Image]]):
    return [
        torch.stack([to_tensor(frame) for frame in vid]).permute(0, 2, 3, 1)
        for vid in frames
    ]


def compare(
    frames: list[list[Image.Image]],
    gens: list[list[Image.Image]],
    opticalflows: list[list[Image.Image]],
):
    total = 0
    total_lpips = 0
    total_ssim = 0
    total_msssim = 0
    total_psnr = 0
    total_lpips_ref = 0
    total_ssim_ref = 0
    total_msssim_ref = 0
    total_psnr_ref = 0
    for gt, gen, ref in zip(
        make_video_tensor(frames),
        make_video_tensor(gens),
        make_video_tensor(opticalflows),
    ):
        total += 1
        total_lpips += lpips.compute(gen, gt, data_range="0,1")
        total_ssim += ssim.compute(gen, gt, data_range="0,1")
        total_msssim += msssim.compute(gen, gt, data_range="0,1")
        total_psnr += psnr.compute(gen, gt, data_range="0,1")
        total_lpips_ref += lpips.compute(ref, gt, data_range="0,1")
        total_ssim_ref += ssim.compute(ref, gt, data_range="0,1")
        total_msssim_ref += msssim.compute(ref, gt, data_range="0,1")
        total_psnr_ref += psnr.compute(ref, gt, data_range="0,1")
    return (
        [
            total_lpips / total,
            total_ssim / total,
            total_msssim / total,
            total_psnr / total,
        ],
        [
            total_lpips_ref / total,
            total_ssim_ref / total,
            total_msssim_ref / total,
            total_psnr_ref / total,
        ],
    )


pipeline = TemporalInpainting(
    model_file="./models/sdxl-1.0.safetensors",
    motion_module_file="./models/mm_sdxl_hs.safetensors",
    lycoris_model_file="./models/animatediff-sdxl-ft-lycoris-epoch=10.pt",
    captioner_type="llava",
    captioner_config_path=os.path.abspath("./models"),
)

TEST_VID = "./vid/test/0024"
frame_files = os.listdir(TEST_VID)
frames_imgs = [
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
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    pipeline.vae.cuda()
    for vid in frames_imgs:
        for fid, frame in enumerate(vid):
            vid[fid] = to_pil_image(pipeline.vae_decode(pipeline.vae_encode(frame)))
    pipeline.vae.cpu()
ground_truth = deepcopy(frames_imgs)
for vid in frames_imgs:
    for fid, frame in enumerate(vid):
        vid[fid] = [vid[fid]]


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
for vid in frames:
    vid[1] = None
    vid[2] = None
    vid[4] = None
    vid[5] = None

vids_gen = pipeline(frames, steps=12, denoise_strength=1.0, cfg=5.0)
for vid, video in enumerate(vids_gen):
    for fid, frame in enumerate(video):
        frames_imgs[vid][fid].append(frame)


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
for vid in frames:
    vid[1] = None
    vid[2] = None
    vid[4] = None
    vid[5] = None
vids_opt = pipeline(frames, steps=2, denoise_strength=0.01, cfg=1.0)
for vid, video in enumerate(vids_opt):
    for fid, frame in enumerate(video):
        frames_imgs[vid][fid].append(frame)


for vid, video in enumerate(frames_imgs):
    video_seq_img = Image.new("RGB", (896 * len(video[0]), 512 * len(video)))
    for fid, frame in enumerate(video):
        for idx in range(len(frame)):
            video_seq_img.paste(frame[idx], (896 * idx, 512 * fid))
    video_seq_img.save(f"vid/out/{vid}.png")


ours, refs = compare(ground_truth, vids_gen, vids_opt)
print(
    f"Ours || Lpips: {ours[0]:5.3f}, ssim: {ours[1]:5.3f}, msssim: {ours[2]:5.3f}, psnr: {ours[3]:5.2f}"
)
print(
    f"Ref  || Lpips: {refs[0]:5.3f}, ssim: {refs[1]:5.3f}, msssim: {refs[2]:5.3f}, psnr: {refs[3]:5.2f}"
)
