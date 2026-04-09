"""
motion_features.py
Extracts interpretable motor features from optical flow data,
following the Global Dynamic Impression (GDI) framework.
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from video_processing import VideoData


@dataclass
class MotionFeatures:
    # ── A. Motor Dynamism ───────────────────────────────────────────────
    mean_motion: float          # average optical flow magnitude
    motion_evolution: float     # normalised linear slope (+ = increasing, – = decreasing)
    motion_variability: float   # std of motion over time

    # ── B. Direction Variability ────────────────────────────────────────
    angle_variability: float    # circular std of dominant flow angles (radians)
    angle_changes: float        # fraction of frames with large angle shifts

    # ── C. Frequency / Rhythm Variability ──────────────────────────────
    dominant_frequency: float   # Hz of dominant motion oscillation
    frequency_stationarity: float  # 1 = steady (PNES), 0 = changing (ES)

    # ── D. Temporal Evolution ───────────────────────────────────────────
    fade_in_score: float        # 0–1: gradual ramp-up detected
    fade_out_score: float       # 0–1: gradual wind-down detected
    burst_repeat_score: float   # 0–1: repetitive burst-arrest-burst pattern

    # ── E. Burst Similarity ─────────────────────────────────────────────
    burst_similarity: float     # 0–1: how similar consecutive bursts are

    # ── F. Eye State ────────────────────────────────────────────────────
    eye_state: str              # "open" | "closed" | "unavailable"

    # ── Raw signals for plotting ────────────────────────────────────────
    motion_signal: np.ndarray
    timestamps: np.ndarray
    variability_signal: np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _linear_trend(arr: np.ndarray) -> float:
    """Normalised linear slope (slope / mean absolute value)."""
    if len(arr) < 2:
        return 0.0
    x     = np.arange(len(arr), dtype=float)
    slope = np.polyfit(x, arr, 1)[0]
    return float(slope / (np.mean(np.abs(arr)) + 1e-6))


def _circular_std(angles: np.ndarray) -> float:
    """Circular standard deviation of angles (radians). 0 = fixed, ~1.8 = fully random."""
    if len(angles) < 2:
        return 0.0
    R = np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2)
    return float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-9, 1.0))))


def _angle_change_fraction(angles: np.ndarray, threshold_rad: float = np.pi / 4) -> float:
    """Fraction of consecutive frame-pairs where the angle changes by more than threshold."""
    if len(angles) < 2:
        return 0.0
    diffs = np.abs(np.diff(np.unwrap(angles)))
    return float(np.mean(diffs > threshold_rad))


def _dominant_frequency(motion_1d: np.ndarray, effective_fps: float) -> float:
    """Dominant oscillation frequency via Welch PSD (skips DC component)."""
    if len(motion_1d) < 8:
        return 0.0
    freqs, psd = signal.welch(motion_1d, fs=effective_fps, nperseg=min(len(motion_1d), 64))
    if len(freqs) < 2:
        return 0.0
    return float(freqs[np.argmax(psd[1:]) + 1])


def _frequency_stationarity(motion: np.ndarray, effective_fps: float) -> float:
    """
    Compare dominant frequency in the first and second halves of the signal.
    Returns 1.0 if frequencies match (PNES-like), 0.0 if very different (ES-like).
    """
    n = len(motion)
    if n < 16:
        return 0.5
    mid = n // 2
    f1  = _dominant_frequency(motion[:mid],  effective_fps)
    f2  = _dominant_frequency(motion[mid:],  effective_fps)
    denom = max(f1, f2, 1e-3)
    return float(1.0 - min(abs(f1 - f2) / denom, 1.0))


def _fade_score(motion: np.ndarray, region: str = "in") -> float:
    """
    Detect gradual fade-in (ES build-up) or fade-out (ES decline).
    Returns 0–1.
    """
    n = len(motion)
    if n < 10:
        return 0.0
    q = max(n // 4, 2)
    if region == "in":
        early, late = motion[:q], motion[q: 2 * q]
    else:
        early, late = motion[-2 * q:-q], motion[-q:]

    mean_early = np.mean(early) + 1e-6
    mean_late  = np.mean(late)  + 1e-6
    ratio      = (mean_late / mean_early) if region == "in" else (mean_early / mean_late)
    return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))


def _detect_bursts(motion: np.ndarray) -> list:
    """
    Simple threshold-based burst detection.
    Returns list of (start_idx, end_idx) tuples.
    """
    threshold = np.percentile(motion, 60)
    in_burst  = False
    bursts    = []
    start     = 0
    for i, m in enumerate(motion):
        if not in_burst and m >= threshold:
            in_burst = True
            start    = i
        elif in_burst and m < threshold:
            in_burst = False
            bursts.append((start, i))
    if in_burst:
        bursts.append((start, len(motion) - 1))
    return bursts


def _burst_repeat_score(motion: np.ndarray) -> float:
    """Score for burst-arrest-burst pattern (PNES indicator)."""
    n = len(_detect_bursts(motion))
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.6
    if n == 1:
        return 0.3
    return 0.0


def _burst_similarity(motion: np.ndarray) -> float:
    """
    Compare motion envelopes of consecutive bursts.
    High similarity → PNES; low similarity → ES.
    Returns 0.5 (neutral) if fewer than 2 bursts are found.
    """
    bursts = _detect_bursts(motion)
    if len(bursts) < 2:
        return 0.5

    target_len = 20
    profiles   = []
    for s, e in bursts:
        seg = motion[s: e + 1]
        if len(seg) < 2:
            continue
        resampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(seg)),
            seg,
        )
        rng = resampled.max() - resampled.min() + 1e-6
        profiles.append((resampled - resampled.min()) / rng)

    if len(profiles) < 2:
        return 0.5

    correlations = [
        float(np.corrcoef(profiles[i], profiles[i + 1])[0, 1])
        for i in range(len(profiles) - 1)
    ]
    return float(np.clip((np.mean(correlations) + 1.0) / 2.0, 0.0, 1.0))


def _rolling_variability(motion: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling standard deviation of the motion signal."""
    result = np.zeros(len(motion))
    half   = window // 2
    for i in range(len(motion)):
        lo         = max(0, i - half)
        hi         = min(len(motion), i + half + 1)
        result[i]  = float(np.std(motion[lo:hi]))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(video_data: VideoData, eye_state: str = "unavailable") -> MotionFeatures:
    """
    Derive all GDI-based features from a VideoData object.

    Parameters
    ----------
    video_data : Output of video_processing.extract_video_data().
    eye_state  : "open" | "closed" | "unavailable"

    Returns
    -------
    MotionFeatures dataclass.
    """
    motion     = video_data.motion_magnitudes
    angles     = video_data.flow_angles
    timestamps = video_data.timestamps

    effective_fps = len(motion) / max(video_data.metadata.duration_seconds, 1.0)
    variability   = _rolling_variability(motion)

    return MotionFeatures(
        # A
        mean_motion=float(np.mean(motion)),
        motion_evolution=_linear_trend(motion),
        motion_variability=float(np.std(motion)),
        # B
        angle_variability=_circular_std(angles),
        angle_changes=_angle_change_fraction(angles),
        # C
        dominant_frequency=_dominant_frequency(motion, effective_fps),
        frequency_stationarity=_frequency_stationarity(motion, effective_fps),
        # D
        fade_in_score=_fade_score(motion, "in"),
        fade_out_score=_fade_score(motion, "out"),
        burst_repeat_score=_burst_repeat_score(motion),
        # E
        burst_similarity=_burst_similarity(motion),
        # F
        eye_state=eye_state,
        # Raw
        motion_signal=motion,
        timestamps=timestamps,
        variability_signal=variability,
    )
