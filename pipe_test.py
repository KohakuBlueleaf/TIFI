import os
from copy import deepcopy
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tifi.pipe.temporal_inpainting import TemporalInpainting


pipeline = TemporalInpainting()

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
for vid in frames:
    vid[1] = None
    vid[2] = None
    vid[4] = None
    vid[5] = None
vids = pipeline(frames)
for vid, video in enumerate(vids):
    for fid, frame in enumerate(video):
        frame.save(f"vid/out/{vid}_{fid}.png")
