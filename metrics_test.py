import torch
import numpy as np
from tifi.modules.metrics.metrics import Metrics
from PIL import Image
import cv2

###
# pip install pytorch_msssim
# pip install lpips
###

def add_noise(img: torch.Tensor, mag: float=0.05):
    noise = torch.ones_like(img) * mag * (img.max() - img.min())
    noise[torch.rand(size=noise.shape) > 0.5] *= -1

    noisy_img = img + noise
    noisy_img[noisy_img >= 255] = 255
    noisy_img[noisy_img <= 0] = 0
    return noisy_img

def create_videos(noise_mag: float=0.05):
    frames = []
    noisy_frames = []
    
    for i in range(7):
        frame = torch.tensor(np.array(Image.open("./vid/out/0_{}.png".format(i))))
        noisy_frame = add_noise(frame, noise_mag)
    
        frames.append(frame)
        noisy_frames.append(noisy_frame)
        
        # For checking
        # noisy_frame = torch.stack([noisy_frame[:, :, 2], noisy_frame[:, :, 1], noisy_frame[:, :, 0]], dim=-1)
        # cv2.imwrite("../playground/noisy_frames/0_{}.png".format(i), np.array(noisy_frame))
    
    return torch.stack(frames, dim=0), torch.stack(noisy_frames, dim=0)

if __name__ == '__main__':
    # Create videos
    vid1, vid2 = create_videos(0.05)
    
    metric_names = ["psnr", "ssim", "msssim", "lpips"]
    metric_kwargs = [{}, {}, {}, {"net": "alex"}]
    
    metrics = [Metrics.from_name(name, **kwargs) for name, kwargs in zip(metric_names, metric_kwargs)]
    
    vid1, vid2 = vid1.to(torch.double), vid2.to(torch.double)
    
    # Range [0,255]
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1, vid2, data_range="0,255")
        print("{} score is {}".format(name, score))
    
    print()
    
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1, vid2, data_range=None)
        print("{} score is {}".format(name, score))
    
    print()
    
    # Range [0, 1]
    vid1_0_1, vid2_0_1 = vid1 / 255, vid2 / 255
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1_0_1, vid2_0_1, data_range="0,1")
        print("{} score is {}".format(name, score))
    
    print()
    
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1_0_1, vid2_0_1, data_range=None)
        print("{} score is {}".format(name, score))
    
    print()
    
    # Range [-1, 1]
    vid1_neg1_1, vid2_neg1_1 = vid1 * 2 / 255 - 1, vid2 * 2 / 255 - 1
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1_neg1_1, vid2_neg1_1, data_range="-1,1")
        print("{} score is {}".format(name, score))
    
    print()
    
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1_neg1_1, vid2_neg1_1)
        print("{} score is {}".format(name, score))
        
    print()
    
    for name, metric in zip(metric_names, metrics):
        score = metric.compute(vid1_neg1_1, vid2_neg1_1, data_range=None)
        print("{} score is {}".format(name, score))