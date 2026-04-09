"""
video_processing.py
Handles video loading, frame extraction, and optical flow computation.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoMetadata:
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    is_short: bool       # < 10 seconds
    has_onset: bool      # heuristic: motion starts within first 20% of video
    has_offset: bool     # heuristic: motion ends within last 20% of video


@dataclass
class VideoData:
    metadata: VideoMetadata
    motion_magnitudes: np.ndarray   # shape (N,) – per-frame optical flow magnitude
    flow_angles: np.ndarray         # shape (N,) – dominant flow angle per frame (radians)
    timestamps: np.ndarray          # shape (N,) – time in seconds
    quality_flags: dict = field(default_factory=dict)


def extract_video_data(video_path: str, max_frames: int = 300) -> Optional[VideoData]:
    """
    Load a video and compute per-frame dense optical flow.

    Parameters
    ----------
    video_path  : Path to the video file.
    max_frames  : Maximum number of frames to analyse (stride is applied automatically).

    Returns
    -------
    VideoData on success, None if the video cannot be opened or has fewer than 2 frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps              = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration         = total_frames_raw / fps if fps > 0 else 0.0

    # Stride so we analyse at most max_frames
    stride = max(1, total_frames_raw // max_frames)

    frames_gray: list  = []
    frame_indices: list = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            frames_gray.append(gray)
            frame_indices.append(idx)
        idx += 1
    cap.release()

    if len(frames_gray) < 2:
        return None

    # Dense optical flow (Farneback)
    motion_magnitudes: list = []
    flow_angles: list       = []

    prev = frames_gray[0]
    for curr in frames_gray[1:]:
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(float(np.mean(mag)))

        # Circular-mean dominant angle, weighted by local magnitude
        weights      = mag.flatten() + 1e-6
        angles_flat  = ang.flatten()
        sin_mean     = np.average(np.sin(angles_flat), weights=weights)
        cos_mean     = np.average(np.cos(angles_flat), weights=weights)
        flow_angles.append(float(np.arctan2(sin_mean, cos_mean)))

        prev = curr

    motion_magnitudes_arr = np.array(motion_magnitudes)
    flow_angles_arr       = np.array(flow_angles)
    timestamps_arr        = np.array([frame_indices[i + 1] / fps for i in range(len(motion_magnitudes_arr))])

    # Onset / offset heuristics
    threshold   = np.percentile(motion_magnitudes_arr, 50)
    above       = motion_magnitudes_arr > threshold
    n           = len(above)
    first_idx   = int(np.argmax(above)) if above.any() else n - 1
    last_idx    = int(n - 1 - np.argmax(above[::-1])) if above.any() else 0
    has_onset   = first_idx / n < 0.20
    has_offset  = last_idx  / n > 0.80

    metadata = VideoMetadata(
        fps=fps,
        total_frames=total_frames_raw,
        duration_seconds=duration,
        width=width,
        height=height,
        is_short=duration < 10.0,
        has_onset=has_onset,
        has_offset=has_offset,
    )

    quality_flags = {
        "short_video":     duration < 10.0,
        "missing_onset":   not has_onset,
        "missing_offset":  not has_offset,
        "low_resolution":  width < 240 or height < 240,
        "very_low_motion": float(np.mean(motion_magnitudes_arr)) < 0.3,
    }

    return VideoData(
        metadata=metadata,
        motion_magnitudes=motion_magnitudes_arr,
        flow_angles=flow_angles_arr,
        timestamps=timestamps_arr,
        quality_flags=quality_flags,
    )
