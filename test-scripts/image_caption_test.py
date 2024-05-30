import argparse

import resource_paths
from minigpt4.common.config import Config

from tifi.modules.image_caption.image_caption import (
    BullshitImageCaption,
    MiniGPT4ImageCaption,
)
from PIL import Image
import numpy as np
import torch
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor


def tensor_to_pil_image(tnsr: torch.Tensor):
    # tnsr: in range[0, 1] and (C, H, W)
    tnsr *= 255
    tnsr = tnsr.to(torch.uint8).permute(1, 2, 0)

    return Image.fromarray(tnsr.numpy(), mode="RGB")


image_caption_gen = MiniGPT4ImageCaption(
    gpu_id=0,
    cfg_path="./config/minigpt4_llama2_eval.yaml",
    model_cfg_path="./config/minigpt4_llama2.yaml",
)

# Another implementation of ImageCaption
# image_caption_gen = BullshitImageCaption()

# Replace image_path to the path of the image to be tested
image_path = "<The path of the image to be tested>"
print("Testing on the image from {}".format(image_path))
answer = image_caption_gen.generate_caption(image_path)
print("Answer is:\n{} (type: {})".format(answer, type(answer)))

image_path = "<The path of the image to be tested>"
print("Testing on the image from {}".format(image_path))
answer = image_caption_gen.generate_caption(image_path)
print("Answer is:\n{} (type: {})".format(answer, type(answer)))

# Create a tensor whose range is [0, 1] and (C, H, W)
image_path = "<The path of the image to be tested>"
image_tnsr = torch.tensor(np.array(Image.open(image_path))) / 255
image_tnsr = image_tnsr.permute(2, 0, 1)
answer = image_caption_gen.generate_caption(tensor_to_pil_image(image_tnsr))
print("Answer is:\n{} (type: {})".format(answer, type(answer)))

## command:
## python image_caption_test.py
