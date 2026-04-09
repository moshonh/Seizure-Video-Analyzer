"""
motion_features.py
Extracts GDI-based motor features from optical flow data.

Design is grounded in two observed clinical profiles:

  PNES profile (Graph 1):
    - Motion near baseline for most of recording
    - Single isolated burst on a quiet background
    - Variability spike only around that burst, then returns to near zero
    - Pattern: focal burst on quiet background

  ES profile (Graph 2):
    - Motion gradually escalates across the recording (fade-in)
    - Sustained high variability throughout the active period
    - No prolonged quiet baseline after onset
    - Pattern: sustained, escalating, persistent activity
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from video_processing import VideoData


@dataclass
class MotionFeatures:
    # ── A. Baseline ratio ────────────────────────────────────────────────
    # Fraction of signal spent near baseline (low motion).
    # PNES: high (most of recording is quiet).  ES: low (sustained activity).
    baseline_fraction: float

    # ── B. Burst isolation ───────────────────────────────────────────────
    # How isolated / focal the peak activity is.
    # PNES: high (single burst stands out sharply).  ES: low (broad activity).
    burst_isolation: float

    # ── C. Temporal escalation (fade-in) ────────────────────────────────
    # Degree to which motion increases steadily from start to end.
    # ES: high.  PNES: low.
    temporal_escalation: float

    # ── D. Sustained variability ─────────────────────────────────────────
    # Mean variability during active (above-baseline) segments.
    # ES: high (variability persists).  PNES: lower (burst is sharper).
    sustained_variability: float

    # ── E. Active fraction ────────────────────────────────────────────────
    # Proportion of recording spent in active (above-baseline) motion.
    # ES: high.  PNES: low (only the burst).
    active_fraction: float

    # ── F. Post-burst quiet ───────────────────────────────────────────────
    # How quickly motion returns to baseline after the peak.
    # PNES: high (sharp return).  ES: low (activity lingers).
    post_burst_quiet: float

    # ── G. Direction variability ─────────────────────────────────────────
    # Circular std of dominant flow angles.
    # ES: higher.  PNES: lower.
    angle_variability: float

    # ── H. Eye state ─────────────────────────────────────────────────────
    eye_state: str   # "open" | "closed" | "unavailable"

    # ── Raw signals for plotting ─────────────────────────────────────────
    motion_signal: np.ndarray
    timestamps: np.ndarray
    variability_signal: np.ndarray

    # ── Derived intermediate values (for report transparency) ────────────
    baseline_threshold: float
    peak_motion: float
    mean_motion: float


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_variability(motion: np.ndarray, window: int = 15) -> np.ndarray:
    """Rolling standard deviation of the motion signal."""
    n      = len(motion)
    result = np.zeros(n, dtype=float)
    half   = window // 2
    for i in range(n):
        lo        = max(0, i - half)
        hi        = min(n, i + half + 1)
        result[i] = float(np.std(motion[lo:hi]))
    return result


def _baseline_threshold(motion: np.ndarray) -> float:
    """
    Threshold separating baseline from active motion.
    Uses the 30th percentile – robust even when activity is brief.
    """
    return float(np.percentile(motion, 30))


def _baseline_fraction(motion: np.ndarray, threshold: float) -> float:
    """Fraction of frames at or below the baseline threshold."""
    return float(np.mean(motion <= threshold))


def _active_fraction(motion: np.ndarray, threshold: float) -> float:
    """Fraction of frames clearly above baseline."""
    return float(np.mean(motion > threshold))


def _burst_isolation(motion: np.ndarray, threshold: float) -> float:
    """
    How isolated / focal the peak activity is.

    Computed as:
        peak_width_fraction = fraction of frames within 50% of peak value
        isolation = 1 - peak_width_fraction

    A single narrow burst → small peak_width_fraction → high isolation (PNES).
    Broad sustained activity → large peak_width_fraction → low isolation (ES).
    """
    peak = float(np.max(motion))
    half_peak_threshold = threshold + 0.5 * (peak - threshold)
    peak_width_fraction = float(np.mean(motion >= half_peak_threshold))
    return float(np.clip(1.0 - peak_width_fraction, 0.0, 1.0))


def _temporal_escalation(motion: np.ndarray) -> float:
    """
    Degree to which motion steadily increases from beginning to end.

    Method: compare mean of first third vs mean of last third, normalised
    by overall mean. Also checks for monotonic trend using linear regression.

    ES: last third >> first third (escalation).
    PNES: last third ≈ first third or lower.
    Returns 0–1.
    """
    n = len(motion)
    if n < 9:
        return 0.0
    third = n // 3
    mean_first = float(np.mean(motion[:third])) + 1e-6
    mean_last  = float(np.mean(motion[-third:])) + 1e-6
    ratio      = mean_last / mean_first

    # Linear slope, normalised
    x     = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, motion, 1)[0])
    slope_norm = float(np.clip(slope / (np.mean(motion) + 1e-6) * 10, 0.0, 1.0))

    # Combine ratio evidence and slope evidence
    ratio_score = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))
    return float(np.clip(0.5 * ratio_score + 0.5 * slope_norm, 0.0, 1.0))


def _sustained_variability(motion: np.ndarray,
                            variability: np.ndarray,
                            threshold: float) -> float:
    """
    Mean rolling variability during active frames only (above baseline).

    ES: variability stays high throughout activity.
    PNES: variability spikes briefly then drops.
    Returns raw mean variability value (normalised by peak motion for comparability).
    """
    active_mask = motion > threshold
    if not active_mask.any():
        return 0.0
    active_var = variability[active_mask]
    raw = float(np.mean(active_var))
    peak = float(np.max(motion)) + 1e-6
    return float(np.clip(raw / peak, 0.0, 2.0))   # 0–2, higher = more sustained ES-like


def _post_burst_quiet(motion: np.ndarray, threshold: float) -> float:
    """
    How quickly and completely the signal returns to baseline after peak activity.

    Looks at the last 25% of the recording.
    PNES: motion returns to near-baseline (post-burst quiet is high).
    ES: motion stays elevated (post-burst quiet is low).
    Returns 0–1.
    """
    n = len(motion)
    if n < 8:
        return 0.5
    tail_start = max(0, int(n * 0.75))
    tail        = motion[tail_start:]
    tail_mean   = float(np.mean(tail))
    peak        = float(np.max(motion)) + 1e-6
    # Quiet = tail is close to threshold, not close to peak
    quiet_score = 1.0 - float(np.clip((tail_mean - threshold) / (peak - threshold + 1e-6), 0.0, 1.0))
    return float(np.clip(quiet_score, 0.0, 1.0))


def _circular_std(angles: np.ndarray) -> float:
    """Circular standard deviation of angles (radians)."""
    if len(angles) < 2:
        return 0.0
    R = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))
    return float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-9, 1.0))))


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
    """
    motion     = video_data.motion_magnitudes
    angles     = video_data.flow_angles
    timestamps = video_data.timestamps

    variability = _rolling_variability(motion, window=15)
    threshold   = _baseline_threshold(motion)
    peak_motion = float(np.max(motion))

    return MotionFeatures(
        # A
        baseline_fraction    = _baseline_fraction(motion, threshold),
        # B
        burst_isolation      = _burst_isolation(motion, threshold),
        # C
        temporal_escalation  = _temporal_escalation(motion),
        # D
        sustained_variability = _sustained_variability(motion, variability, threshold),
        # E
        active_fraction      = _active_fraction(motion, threshold),
        # F
        post_burst_quiet     = _post_burst_quiet(motion, threshold),
        # G
        angle_variability    = _circular_std(angles),
        # H
        eye_state            = eye_state,
        # Raw
        motion_signal        = motion,
        timestamps           = timestamps,
        variability_signal   = variability,
        # Intermediates
        baseline_threshold   = threshold,
        peak_motion          = peak_motion,
        mean_motion          = float(np.mean(motion)),
    )
