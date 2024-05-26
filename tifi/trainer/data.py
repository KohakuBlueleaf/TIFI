import os
from random import random

import torch
import torch.utils.data as data

from tifi.utils.interpolation import frame_idx_map_gen, frame_index_gen


class LatentVideoDataset(data.Dataset):
    def __init__(self, latents_folder="dataset/latent_set", frames=7):
        self.files = [
            os.path.join(latents_folder, f)
            for f in os.listdir(latents_folder)
            if f.endswith(".pt")
        ]
        self.fid_map = frame_idx_map_gen(frames)
        self.frames = frames

    def __getitem__(self, index):
        entry = torch.load(self.files[index])
        latents = entry["latents"]
        ctx = entry["ctx"].squeeze(0)
        embed = entry["embed"].squeeze(0)
        x = latents[:7]
        choosed_frames = [0]
        for i in range(1, self.frames - 1):
            if random() > 0.5:
                choosed_frames.append(i)
        choosed_frames.append(self.frames - 1)
        blended_x = latents[frame_index_gen(choosed_frames, self.fid_map)]

        return x, blended_x, ctx, embed

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    dataset = LatentVideoDataset()
    print(len(dataset))
    print(
        dataset[0][0].shape,
        dataset[0][1].shape,
        dataset[0][2].shape,
        dataset[0][3].shape,
    )

    loader = data.DataLoader(dataset, batch_size=4, shuffle=True)
    for x, blended_x, ctx, embed in loader:
        print(x.shape, blended_x.shape, ctx.shape, embed.shape)
        break
