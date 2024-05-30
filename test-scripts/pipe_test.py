import os, sys
sys.path.append(".")

from torchvision.transforms.functional import to_tensor
from PIL import Image

from tifi.pipe.temporal_inpainting import TemporalInpainting


pipeline = TemporalInpainting(
    model_file="./models/sdxl-1.0.safetensors",
    motion_module_file="./models/mm_sdxl_hs.safetensors",
    lycoris_model_file="./models/animatediff-sdxl-ft-lycoris-epoch=10.pt",
    captioner_type="llava",
    captioner_config_path=os.path.abspath("./models"),
)

TEST_VID = "./vid/test/0008"
frame_files = os.listdir(TEST_VID)
frames_imgs = [
    [
        [(
            Image.open(os.path.join(frame_folder, frame_file))
            .convert("RGB")
            .resize((896, 512))
        )]
        for frame_file in os.listdir(TEST_VID)
    ]
    for frame_folder in [TEST_VID]
]


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

vids = pipeline(frames, steps=12, denoise_strength=0.6, cfg=5.0)
for vid, video in enumerate(vids):
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
vids = pipeline(frames, steps=1, denoise_strength=0.01, cfg=1.0)
for vid, video in enumerate(vids):
    for fid, frame in enumerate(video):
        frames_imgs[vid][fid].append(frame)


for vid, video in enumerate(frames_imgs):
    video_seq_img = Image.new("RGB", (896 * len(video[0]), 512 * len(video)))
    for fid, frame in enumerate(video):
        for idx in range(len(frame)):
            video_seq_img.paste(frame[idx], (896 * idx, 512 * fid))
    video_seq_img.save(f"vid/out/{vid}.png")