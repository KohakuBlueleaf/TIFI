import argparse

import resource_paths
from minigpt4.common.config import Config

from tifi.modules.image_caption.image_caption import (BullshitImageCaption,
                                                      MiniGPT4ImageCaption)

image_caption_gen = MiniGPT4ImageCaption(
    gpu_id=0,
    cfg_path='./config/minigpt4_llama2_eval.yaml',
    model_cfg_path='./config/minigpt4_llama2.yaml',
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

## command:
## python image_caption_test.py