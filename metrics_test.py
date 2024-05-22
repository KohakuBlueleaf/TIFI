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
    
    # PSNR
    metric = Metrics.from_name("psnr")
    psnr_score = metric.compute(vid1, vid2)
    
    print("PSNR score is {}".format(psnr_score))
    
    # SSIM
    metric = Metrics.from_name("ssim")
    ssim_score = metric.compute(vid1, vid2)
    
    print("SSIM score is {}".format(ssim_score))
    
    # MSSSIM
    metric = Metrics.from_name("msssim")
    msssim_score = metric.compute(vid1, vid2)
    
    print("MS-SSIM score is {}".format(msssim_score))
    
    # LPIPS
    metric = Metrics.from_name("lpips", net="alex")
    lpips_score = metric.compute(vid1, vid2)
    
    print("LPIPS score is {}".format(lpips_score))