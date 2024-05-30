import os, sys

sys.path.append(".")
import time

from torchvision.transforms.functional import to_tensor
from PIL import Image

from tifi.modules.image_caption.image_caption import LlavaImageCaption


captioner = LlavaImageCaption(model_dir=os.path.abspath("./models"))

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
print(captioner.generate_caption(frames[0][0]))
captioner.offload()
time.sleep(2)
print(captioner.generate_caption(frames[0][1]))
