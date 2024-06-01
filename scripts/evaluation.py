import os, sys
import random
from copy import deepcopy

sys.path.append(".")

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
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
    masks: list[bool]
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
    
    mask = torch.tensor(masks)
    
    for gt, gen, ref in zip(
        make_video_tensor(frames),
        make_video_tensor(gens),
        make_video_tensor(opticalflows),
    ):
        gt = gt[mask]
        gen = gen[mask]
        ref = ref[mask]
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

if os.path.isfile("./eval_out/results.txt"):
    os.remove("./eval_out/results.txt")

TEST_VIDS = r".\dataset\choosed_septuplet\test_sequences"
videos = os.listdir(TEST_VIDS)
for video_id in tqdm(videos, desc="Processing videos", smoothing=0.01):
    TEST_VID = os.path.join(TEST_VIDS, video_id)
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

    # Randomly drop some frames
    # But always drop the middle frame
    frame_drops = [False] * len(vid)
    drop_count = random.randint(1, len(frame_drops)-2)
    id_list = list(range(1, len(frame_drops) - 1))
    random.shuffle(id_list)
    removed_frames = sorted(id_list[:drop_count])
    for i in removed_frames:
        frame_drops[i] = True

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
        for fid, frame in enumerate(vid):
            if frame_drops[fid]:
                vid[fid] = None

    tifi_out, optical_flow_out = pipeline(frames, steps=12, denoise_strength=0.65, cfg=5.0)
    for vid, video in enumerate(tifi_out):
        for fid, frame in enumerate(video):
            frames_imgs[vid][fid].append(frame)
    for vid, video in enumerate(optical_flow_out):
        for fid, frame in enumerate(video):
            frames_imgs[vid][fid].append(frame)

    for vid, video in enumerate(frames_imgs):
        video_seq_img = Image.new("RGB", (896 * len(video[0]), 512 * len(video)))
        for fid, frame in enumerate(video):
            for idx in range(len(frame)):
                video_seq_img.paste(frame[idx], (896 * idx, 512 * fid))
        video_seq_img.save(f"eval_out/sample_img/{video_id}.png")

    ours, refs = compare(ground_truth, tifi_out, optical_flow_out, frame_drops)
    with open("eval_out/results.txt", "a") as f:
        f.write(f"{video_id}, removed frames: {removed_frames}\n")
        f.write(
            f"tifi || Lpips: {ours[0]:5.3f}, ssim: {ours[1]:5.3f}, msssim: {ours[2]:5.3f}, psnr: {ours[3]:5.2f}\n"
        )
        f.write(
            f"opti || Lpips: {refs[0]:5.3f}, ssim: {refs[1]:5.3f}, msssim: {refs[2]:5.3f}, psnr: {refs[3]:5.2f}\n"
        )
        f.write("\n")