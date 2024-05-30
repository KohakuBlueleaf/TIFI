from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim, ms_ssim


class Metrics:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self, vid1: torch.Tensor, vid2: torch.Tensor, data_range: str = "-1,1"):
        """
        vid1: The first video, whose shape is (B, H, W, C)
        vid2: The second video, whose shape is (B, H, W, C)

        return the value of the metric
        """

    @abstractmethod
    def _to_valid(self, video: torch.Tensor, data_range: str = None):
        """
        Transform the video to valid form for "compute"
        """

    def _to_target(self, video: torch.Tensor, target: str, data_range: str = None):
        """target and data_range should be in ["0,255", "0,1", "-1,1"]"""
        video = video.to(torch.double)

        video_max = torch.max(video).item()
        video_min = torch.min(video).item()

        # Transform to [0, 255] first
        if data_range == "0,1" or (
            data_range == None and video_max <= 1 and video_min >= 0
        ):
            video_inter = video * 255
        elif data_range == "0,255" or (
            data_range == None and video_max <= 255 and video_min >= 0
        ):
            video_inter = video
        elif data_range == "-1,1" or (
            data_range == None and video_max <= 1 and video_min >= -1
        ):
            video_inter = (video + 1) * 255 / 2
        else:
            raise Exception("unknown form of video")

        # Transform to target form
        if target == "0,255":
            return video_inter
        elif target == "0,1":
            return video_inter / 255
        elif target == "-1,1":
            return video_inter * 2 / 255 - 1
        else:
            raise Exception("unknown target name: {}".format(target))

    def _check_shapes(self, vid1: torch.Tensor, vid2: torch.Tensor):
        # Get the shapes and check if they are in the same shape
        B, H, W, C = vid1.shape
        B2, H2, W2, C2 = vid2.shape

        # If not, raise exception
        if B != B2 or H != H2 or W != W2 or C != C2:
            raise Exception("The shapes of provided videos are different")

    @staticmethod
    def from_name(name: str, **init_kwargs):
        if name == "psnr":
            return PSNR(**init_kwargs)
        elif name == "ssim":
            return SSIM(**init_kwargs)
        elif name == "msssim":
            return MSSSIM(**init_kwargs)
        elif name == "lpips":
            return LPIPS(**init_kwargs)
        else:
            raise Exception("Unknown matric name {}.".format(name))


class PSNR(Metrics):
    def compute(self, vid1: torch.Tensor, vid2: torch.Tensor, data_range: str = "-1,1"):
        # Check shapes
        self._check_shapes(vid1, vid2)

        # Get shapes
        B, H, W, C = vid1.shape

        # Transform it to the valid form
        vid1 = self._to_valid(vid1, data_range)
        vid2 = self._to_valid(vid2, data_range)

        # Compute the MSE for each
        vid_diff_2 = F.mse_loss(vid1, vid2, reduction="none")

        # Compute the mean square errors for each frame
        mse = vid_diff_2.mean(dim=(1, 2, 3))

        # Prevent divide by zero problem
        psnr = 10 * torch.log10((255 * 255) / (mse + torch.finfo(mse.dtype).eps))
        psnr_avg = torch.mean(psnr)
        return psnr_avg.item()

    def _to_valid(self, video: torch.Tensor, data_range: str = None):
        return self._to_target(video, "0,255", data_range=data_range).float()


class SSIM(Metrics):
    def compute(self, vid1: torch.Tensor, vid2: torch.Tensor, data_range: str = "-1,1"):
        # Check shapes
        self._check_shapes(vid1, vid2)

        # Transform to valid form
        vid1 = self._to_valid(vid1, data_range)
        vid2 = self._to_valid(vid2, data_range)

        # Transpose the tensor into (B, C, H, W)
        vid1_tnsr = vid1.permute(0, 3, 1, 2)
        vid2_tnsr = vid2.permute(0, 3, 1, 2)

        # Compute the ssim score
        ssim_score = ssim(
            vid1_tnsr, vid2_tnsr, data_range=255, size_average=True
        ).item()

        return ssim_score

    def _to_valid(self, video: torch.Tensor, data_range: str = None):
        return self._to_target(video, "0,255", data_range=data_range).float()


class MSSSIM(Metrics):
    def compute(self, vid1: torch.Tensor, vid2: torch.Tensor, data_range: str = "-1,1"):
        # Check shapes
        self._check_shapes(vid1, vid2)

        # Transform to valid form
        vid1 = self._to_valid(vid1, data_range)
        vid2 = self._to_valid(vid2, data_range)

        # Transpose the tensor into (B, C, H, W)
        vid1_tnsr = vid1.permute(0, 3, 1, 2)
        vid2_tnsr = vid2.permute(0, 3, 1, 2)

        # Compute the ms-ssim score
        msssim_score = ms_ssim(
            vid1_tnsr, vid2_tnsr, data_range=255, size_average=True
        ).item()

        return msssim_score

    def _to_valid(self, video: torch.Tensor, data_range: str = None):
        return self._to_target(video, "0,255", data_range=data_range).float()


class LPIPS(Metrics):
    def __init__(self, net="alex") -> None:
        super().__init__()

        # Create loss function by "net"
        self.loss_fn = lpips.LPIPS(net=net)

    def compute(self, vid1: torch.Tensor, vid2: torch.Tensor, data_range: str = "-1,1"):
        # Check shapes
        self._check_shapes(vid1, vid2)

        # Transform to valid form
        vid1 = self._to_valid(vid1, data_range)
        vid2 = self._to_valid(vid2, data_range)

        # Transpose the tensor into (B, C, H, W)
        vid1_tnsr = vid1.permute(0, 3, 1, 2)
        vid2_tnsr = vid2.permute(0, 3, 1, 2)

        # Compute the ms-ssim score
        with torch.no_grad():
            lpips_score = self.loss_fn(vid1_tnsr, vid2_tnsr)

        # Compute the average
        lpips_score_avg = torch.mean(lpips_score)

        return lpips_score_avg.item()

    def _to_valid(self, video: torch.Tensor, data_range: str = None):
        return self._to_target(video, "-1,1", data_range=data_range).float()
