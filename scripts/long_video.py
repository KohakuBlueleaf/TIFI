import os, sys

sys.path.append(".")

from torchvision.transforms.functional import to_tensor
from PIL import Image

from tifi.pipe.temporal_inpainting import TemporalInpainting


TEST_VID = "./vid/long-videos"
FORMAT = "720p30"
TEST_VID = os.path.join(TEST_VID, FORMAT)

if not os.path.exists(f"vid/out/{FORMAT}"):
    os.makedirs(f"vid/out/{FORMAT}")

if not os.path.exists(f"vid/out/{FORMAT}-optical"):
    os.makedirs(f"vid/out/{FORMAT}-optical")


pipeline = TemporalInpainting(
    model_file="./models/sdxl-1.0.safetensors",
    motion_module_file="./models/mm_sdxl_hs.safetensors",
    lycoris_model_file="./models/animatediff-sdxl-ft-lycoris-epoch=10.pt",
    captioner_type="llava",
    captioner_config_path=os.path.abspath("./models"),
)

frames = []

for frame_file in os.listdir(TEST_VID):
    frames.append(
        to_tensor(
            Image.open(os.path.join(TEST_VID, frame_file))
            .convert("RGB")
            .resize((896, 512))
        )
    )
    frames.append(None)
frames = frames[:-1]

tifi_out, optical_flow_out = pipeline(frames, steps=8, denoise_strength=0.5, cfg=7.0)

for idx, frame in enumerate(tifi_out[0]):
    frame.save(f"vid/out/{FORMAT}/{idx:04}.png")

for idx, frame in enumerate(optical_flow_out[0]):
    frame.save(f"vid/out/{FORMAT}-optical/{idx:04}.png")
