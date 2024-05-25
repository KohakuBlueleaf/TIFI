import cv2
import numpy as np
import torch


def calculate_optical_flow_between_frames(
    frame_1: np.ndarray, frame_2: np.ndarray
) -> np.ndarray:
    """Calculate optical flow between two frames.

    Args:
        frame_1: The first frame as a NumPy array.
        frame_2: The second frame as a NumPy array.

    Returns:
        The optical flow as a NumPy array.
    """
    frame_1_gray, frame_2_gray = cv2.cvtColor(
        frame_1, cv2.COLOR_BGR2GRAY
    ), cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.calcOpticalFlowFarneback(
        frame_1_gray,
        frame_2_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=10,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    return optical_flow


def blend_frame_optical_flow(
    frame_1: torch.Tensor, frame_2: torch.Tensor, num_frames: int
):
    frame_1 = (frame_1.permute(1, 2, 0) * 255).numpy()[:, :, ::-1].astype(np.uint8)
    frame_2 = (frame_2.permute(1, 2, 0) * 255).numpy()[:, :, ::-1].astype(np.uint8)
    resultant_frames = []
    resultant_frames.append(frame_1)
    optical_flow = calculate_optical_flow_between_frames(frame_1, frame_2)
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames + 1):
        alpha = frame_num / (num_frames + 1)
        flow = -1 * alpha * optical_flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        interpolated_frame = cv2.remap(frame_1, flow, None, cv2.INTER_LINEAR)
        resultant_frames.append(cv2.cvtColor(interpolated_frame, cv2.COLOR_BGR2RGB))
    # convert BGR numpy image to RGB tensor image
    resultant_frames = [
        torch.from_numpy(frame).permute(2, 0, 1) / 255 for frame in resultant_frames
    ]
    return resultant_frames[1:]


def blend_frame_naive(frame_1: torch.Tensor, frame_2: torch.Tensor, num_frames: int):
    result = []
    for frame_num in range(num_frames):
        alpha = (frame_num + 1) / (num_frames + 1)
        frame = (1 - alpha) * frame_1 + alpha * frame_2
        result.append(frame)
    return result


def frame_idx_map_gen(length=7):
    idx = length
    frame_idx_map = {}
    for i in range(length - 1):
        for j in range(i + 2, length):
            for k in range(i + 1, j):
                frame_idx_map[(i, j, k)] = idx
                idx += 1
    return frame_idx_map


def frame_index_gen(gt_frames=[0, 2, 4, 6], frame_idx_map=frame_idx_map_gen(7)):
    prev_gt = 0
    current_idx = 0
    all_idx = []
    for gt_idx in gt_frames:
        while current_idx < gt_idx:
            all_idx.append(frame_idx_map[(prev_gt, gt_idx, current_idx)])
            current_idx += 1
        all_idx.append(gt_idx)
        prev_gt = gt_idx
        current_idx += 1
    return all_idx
