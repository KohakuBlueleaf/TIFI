import cv2
import numpy as np
import torch
from PIL import Image


def match_color(source, target):
    source = (
        (source.permute(1, 2, 0).float() * 255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.float32)
    )
    target = (
        (target.permute(1, 2, 0).float() * 255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.float32)
    )

    # Use wavelet colorfix method to match original low frequency data at first
    source[:, :, 0] = wavelet_colorfix(source[:, :, 0], target[:, :, 0])
    source[:, :, 1] = wavelet_colorfix(source[:, :, 1], target[:, :, 1])
    source[:, :, 2] = wavelet_colorfix(source[:, :, 2], target[:, :, 2])
    output = source
    output_tensor = torch.from_numpy(output.clip(0, 255)) / 255
    return output_tensor.permute(2, 0, 1)


def wavelet_colorfix(inp, target):
    inp_high, _ = wavelet_decomposition(inp, 5)
    _, target_low = wavelet_decomposition(target, 5)
    output = inp_high + target_low
    return output


def wavelet_decomposition(inp, levels):
    high_freq = np.zeros_like(inp)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def wavelet_blur(inp, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(inp, (kernel_size, kernel_size), 0)
    return output


def color_styling(inp, saturation=1.2, contrast=1.1):
    output = inp.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    output[:, :, 1] = output[:, :, 1] * saturation
    output[:, :, 2] = output[:, :, 2] * contrast - (contrast - 1)
    output = np.clip(output, 0, 1)
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return output


def kmeans_color_quant(inp, colors=32):
    img = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_quant = img_pil.quantize(colors, 1, kmeans=colors).convert("RGB")
    return cv2.cvtColor(np.array(img_quant), cv2.COLOR_RGB2BGR)
