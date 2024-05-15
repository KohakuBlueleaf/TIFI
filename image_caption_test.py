import argparse

from minigpt4.common.config import Config

import resource_paths
from tifi.modules.image_caption.image_caption import (BullshitImageCaption,
                                                      MiniGPT4ImageCaption)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config(args)

image_caption_gen = MiniGPT4ImageCaption(args.gpu_id, cfg)
# image_caption_gen = BullshitImageCaption()


image_path = "./daylily-flower-and-buds-sharp.jpg"
print("Testing on the image from {}".format(image_path))
print("Answer is:\n{}".format(image_caption_gen.generate_caption(image_path)))

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