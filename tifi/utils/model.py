import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open


def read_state_dict(file_name):
    try:
        state_dict = torch.load(file_name, map_location="cpu")
    except Exception as e:
        state_dict = {}
        with safe_open(file_name, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict
