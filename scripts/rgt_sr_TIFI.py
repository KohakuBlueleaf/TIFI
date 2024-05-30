import os

import torch
from tqdm import tqdm
from PIL import Image
from imagesize import imagesize

from hakuir.image_super_resolution import ImageSuperRes


SCALE = 2
model: ImageSuperRes = None


def load_model():
    global model
    model = ImageSuperRes()
    model.load_model(f"RGT_x2")
    model.tile_overlap = 8
    model.model.bfloat16()


def super_resolution(image_path):
    width, height = imagesize.get(image_path)
    if width >= 896 or height >= 512:
        return None
    img = Image.open(image_path).convert("RGB")
    return model.upscale(img, SCALE, 64, dtype=torch.bfloat16)


def main(root="."):
    # Use os.walk to go through all files
    for root, dirs, files in tqdm(list(os.walk(root))):
        for file in tqdm(files, leave=False):
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                output = super_resolution(image_path)
                if output is None:
                    continue
                try:
                    output.save(image_path)
                except KeyboardInterrupt:
                    # Ensure the output is saved before exiting
                    output.save(image_path)
                    raise KeyboardInterrupt


if __name__ == "__main__":
    load_model()
    main("./dataset/choosed_septuplet/choosed_septuplet")
