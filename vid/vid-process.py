import os
import shutil

import ffmpeg
from PIL import Image
from tqdm import tqdm


OUTPUT_FOLDER = "./long-videos"
VID = "For-Test-{}.mp4"
FORMATS = [
    "720p10",
    "720p20",
    "720p30",
    "720p60",
]


for format in FORMATS:
    if os.path.exists(os.path.join(OUTPUT_FOLDER, format)):
        shutil.rmtree(os.path.join(OUTPUT_FOLDER, format))
    os.makedirs(os.path.join(OUTPUT_FOLDER, format))
    stream = ffmpeg.input(VID.format(format))
    stream = stream.output(os.path.join(OUTPUT_FOLDER, format, "%04d.png"))

    ffmpeg.run(stream)


# os.walk through output folder and convert png to webp
for root, dirs, files in tqdm(os.walk(OUTPUT_FOLDER)):
    for file in tqdm(files):
        if file.endswith(".png"):
            img = Image.open(os.path.join(root, file))
            img.save(os.path.join(root, file.replace(".png", ".webp")), "webp", quality=90)
            os.remove(os.path.join(root, file))