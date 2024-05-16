import argparse

import resource_paths
from minigpt4.common.config import Config

from tifi.modules.image_caption.image_caption import (BullshitImageCaption,
                                                      MiniGPT4ImageCaption)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Demo")
#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#     args = parser.parse_args()
#     return args

# args = parse_args()
# cfg = Config(args)

image_caption_gen = MiniGPT4ImageCaption(
    gpu_id=0,
    cfg_path='/content/minigpt4_llama2_eval.yaml',
    model_cfg_path='/content/minigpt4_llama2.yaml',
)
# image_caption_gen = BullshitImageCaption()


image_path = "/content/daylily-flower-and-buds-sharp.jpg"
print("Testing on the image from {}".format(image_path))
answer = image_caption_gen.generate_caption(image_path)
print("Answer is:\n{} (type: {})".format(answer, type(answer)))

image_path = "/content/low-res-72dpi.jpg"
print("Testing on the image from {}".format(image_path))
answer = image_caption_gen.generate_caption(image_path)
print("Answer is:\n{} (type: {})".format(answer, type(answer)))

## command:
## python image_caption_test.py --gpu-id 0 --cfg-path minigpt4_llama2_eval.yaml

## Example configuration file of minigpt4_llama2_eval.yaml:
# model:
#   arch: minigpt4
#   model_type: pretrain_llama2
#   max_txt_len: 160
#   end_sym: "</s>"
#   low_resource: True
#   prompt_template: '[INST] {} [/INST] '
#   ckpt: 'The path of checkpoint of llama2-7b'


# datasets:
#   cc_sbu_align:
#     vis_processor:
#       train:
#         name: "blip2_image_eval"
#         image_size: 224
#     text_processor:
#       train:
#         name: "blip_caption"

# run:
#   task: image_text_pretrain

## Example configuration file of minigpt4_llama2.yaml: (It is just to override the default configuration)
# model:
#   llama_model: "meta-llama/Llama-2-7b-chat-hf"