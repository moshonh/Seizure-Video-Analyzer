"""
motion_features.py
GDI-based motor feature extraction.

Core clinical model (revised):
─────────────────────────────────────────────────────────────────────────────
The key distinction is NOT "isolated burst vs sustained activity".

The key distinction IS:
  "Temporally stereotyped movement  vs  dynamically evolving movement"

PNES:
  - Repetitive, internally consistent movement cycles
  - Limited evolution in rhythm, direction, and amplitude
  - High inter-segment similarity (stereotypy)

ES:
  - Evolving motor pattern across the event
  - Changing rhythm, direction, and amplitude over time
  - Low inter-segment similarity (transformation)

Temporal evolution is MORE important than motion quantity.
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
from video_processing import VideoData


# ──────────────────────────────────────────────────────────────────────────────
# Feature dataclass
# Fields marked PRIMARY drive classification.
# Fields marked SECONDARY are supportive / contextual.
# app.py reads: mean_motion, variability_signal, motion_derivative,
#               dynamic_index, stability_index, rhythm_irregularity,
#               temporal_evolution, regional_asynchrony
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionFeatures:

    # ── PRIMARY: Temporal evolution ──────────────────────────────────────────
    # Composite score: how much rhythm, direction, and amplitude change over time.
    # High = evolving = ES.  Low = stable = PNES.
    temporal_evolution: float

    # ── PRIMARY: Movement stereotypy ─────────────────────────────────────────
    # Mean cross-correlation between consecutive signal segments.
    # High = stereotyped = PNES.  Low = evolving = ES.
    movement_stereotypy: float

    # ── PRIMARY: Rhythm irregularity ─────────────────────────────────────────
    # Coefficient of variation of inter-peak intervals.
    # High = irregular = ES.  Low = regular = PNES.
    rhythm_irregularity: float

    # ── PRIMARY: Direction variability ───────────────────────────────────────
    # Circular std of dominant flow angles across the recording.
    # High = changing direction = ES.  Low = fixed direction = PNES.
    angle_variability: float

    # ── PRIMARY: Amplitude variability over time ──────────────────────────────
    # Std of per-segment mean motion amplitude.
    # High = amplitude changes = ES.  Low = consistent = PNES.
    amplitude_variability_over_time: float

    # ── PRIMARY: Vector change (early vs late direction) ─────────────────────
    # Circular distance between dominant direction in first and last third.
    # High = direction shifted = ES.  Low = stable direction = PNES.
    vector_change: float

    # ── PRIMARY: Temporal distribution stability ──────────────────────────────
    # How stable the SHAPE of motion/variability distributions remains across time windows.
    # Per-window normalisation removes pure scale effects, so this feature reflects
    # stable pattern structure rather than simple amplitude growth.
    # High = statistically stable over time = PNES.
    # Low  = distribution changes over time = ES.
    temporal_distribution_stability: float

    # ── PRIMARY: Variance drift ────────────────────────────────────────────────
    # Progressive increase in amplitude / variance over time.
    # High = non-stationary growth or variance drift = ES.
    # Low  = no progressive drift = PNES.
    variance_drift: float

    # ── SECONDARY: Baseline / activity structure ──────────────────────────────
    baseline_fraction: float      # fraction of frames near baseline
    burst_isolation: float        # how focal/narrow the peak activity is
    post_burst_quiet: float       # does motion return to baseline at end
    active_fraction: float        # fraction of frames above baseline
    sustained_variability: float  # mean variability during active frames
    temporal_escalation: float    # does motion increase from start to end

    # ── Eye state (manual) ────────────────────────────────────────────────────
    eye_state: str   # "open" | "closed" | "unavailable"

    # ── Regional asynchrony ───────────────────────────────────────────────────
    regional_asynchrony: float    # std of motion across ROI bands

    # ── Raw signals (for plotting and JSON export) ────────────────────────────
    motion_signal: np.ndarray
    timestamps: np.ndarray
    variability_signal: np.ndarray

    # ── Scalar GDI indices (displayed in app metrics) ─────────────────────────
    baseline_threshold: float
    peak_motion: float
    mean_motion: float
    motion_derivative: float
    dynamic_index: float
    stability_index: float


# ──────────────────────────────────────────────────────────────────────────────
# Low-level signal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_variability(motion: np.ndarray, window: int = 15) -> np.ndarray:
    """Rolling standard deviation."""
    n      = len(motion)
    result = np.zeros(n, dtype=float)
    half   = window // 2
    for i in range(n):
        lo        = max(0, i - half)
        hi        = min(n, i + half + 1)
        result[i] = float(np.std(motion[lo:hi]))
    return result


def _baseline_threshold(motion: np.ndarray) -> float:
    """30th-percentile baseline threshold – robust to brief events."""
    return float(np.percentile(motion, 30))


def _circular_std(angles: np.ndarray) -> float:
    """Circular standard deviation (radians). 0 = fixed, ~1.8 = random."""
    if len(angles) < 2:
        return 0.0
    R = float(np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    return float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-9, 1.0))))


def _segment_angles(angles: np.ndarray, n_segs: int = 4) -> np.ndarray:
    """Dominant circular-mean angle per segment."""
    n      = len(angles)
    size   = max(1, n // n_segs)
    result = []
    for i in range(n_segs):
        seg = angles[i * size: (i + 1) * size]
        if len(seg) == 0:
            result.append(0.0)
            continue
        sin_m = float(np.mean(np.sin(seg)))
        cos_m = float(np.mean(np.cos(seg)))
        result.append(float(np.arctan2(sin_m, cos_m)))
    return np.array(result)


# ──────────────────────────────────────────────────────────────────────────────
# PRIMARY feature extractors
# ──────────────────────────────────────────────────────────────────────────────

def _temporal_evolution(motion: np.ndarray,
                        variability: np.ndarray,
                        angles: np.ndarray,
                        n_segs: int = 4) -> float:
    """
    Composite temporal evolution score.

    Combines three independent axes of change across time segments:
      1. Variability change  – does the local variability level shift?
      2. Amplitude change    – does the mean motion amplitude shift?
      3. Direction change    – does the dominant movement direction shift?

    Each axis contributes equally.  Returns 0–1.
    High = evolving (ES).  Low = stable (PNES).
    """
    n = len(motion)
    if n < n_segs * 3:
        return 0.0

    size = n // n_segs
    seg_var = np.array([float(np.mean(variability[i*size:(i+1)*size])) for i in range(n_segs)])
    seg_amp = np.array([float(np.mean(motion[i*size:(i+1)*size]))       for i in range(n_segs)])
    seg_dir = _segment_angles(angles, n_segs)

    # 1. Variability axis: std / mean
    var_cv = float(np.std(seg_var)) / (float(np.mean(seg_var)) + 1e-6)
    var_score = float(np.clip(var_cv / 1.0, 0.0, 1.0))

    # 2. Amplitude axis: std / mean
    amp_cv = float(np.std(seg_amp)) / (float(np.mean(seg_amp)) + 1e-6)
    amp_score = float(np.clip(amp_cv / 1.0, 0.0, 1.0))

    # 3. Direction axis: circular std of per-segment dominant angles
    dir_std = _circular_std(seg_dir)
    dir_score = float(np.clip(dir_std / 1.2, 0.0, 1.0))

    return float(np.clip((var_score + amp_score + dir_score) / 3.0, 0.0, 1.0))


def _movement_stereotypy(motion: np.ndarray, n_segs: int = 6) -> float:
    """
    Mean Pearson correlation between consecutive normalised motion segments.

    High correlation → segments look alike → stereotyped → PNES.
    Low correlation  → segments differ     → evolving    → ES.

    Returns 0–1 (0 = maximally evolving, 1 = perfectly stereotyped).
    Neutral 0.5 when fewer than 2 valid segments.
    """
    n = len(motion)
    if n < n_segs * 4:
        return 0.5

    size = n // n_segs
    profiles = []
    for i in range(n_segs):
        seg = motion[i * size: (i + 1) * size].copy()
        rng = seg.max() - seg.min()
        if rng < 1e-9:
            continue
        profiles.append((seg - seg.min()) / rng)

    if len(profiles) < 2:
        return 0.5

    corrs = []
    for i in range(len(profiles) - 1):
        # Pearson on segments of the same length (guaranteed by size slicing)
        a, b = profiles[i], profiles[i + 1]
        min_len = min(len(a), len(b))
        c = float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])
        if not np.isnan(c):
            corrs.append(c)

    if not corrs:
        return 0.5

    mean_corr = float(np.mean(corrs))
    # Map [-1, 1] → [0, 1]
    return float(np.clip((mean_corr + 1.0) / 2.0, 0.0, 1.0))


def _rhythm_irregularity(motion: np.ndarray, effective_fps: float) -> float:
    """
    Coefficient of variation of inter-peak intervals.
    High = irregular rhythm = ES.  Low = regular = PNES.
    Returns 0.0 safely when fewer than 2 peaks are detected.
    """
    if len(motion) < 6:
        return 0.0
    min_dist = max(2, int(effective_fps * 0.4))
    try:
        peaks, _ = find_peaks(
            motion,
            distance=min_dist,
            prominence=float(np.std(motion)) * 0.25,
        )
    except Exception:
        return 0.0
    if len(peaks) < 2:
        return 0.0
    intervals = np.diff(peaks.astype(float))
    mean_iv   = float(np.mean(intervals))
    if mean_iv < 1e-6:
        return 0.0
    return float(np.clip(float(np.std(intervals)) / mean_iv, 0.0, 3.0))


def _amplitude_variability_over_time(motion: np.ndarray, n_segs: int = 6) -> float:
    """
    Std of per-segment mean amplitude, normalised by overall mean.
    High = amplitude changes across event = ES.
    Low  = consistent amplitude = PNES.
    Returns 0–1.
    """
    n = len(motion)
    if n < n_segs * 3:
        return 0.0
    size      = n // n_segs
    seg_means = np.array([float(np.mean(motion[i*size:(i+1)*size])) for i in range(n_segs)])
    overall   = float(np.mean(seg_means)) + 1e-6
    return float(np.clip(float(np.std(seg_means)) / overall, 0.0, 1.0))


def _vector_change(angles: np.ndarray) -> float:
    """
    Circular angular distance between the dominant direction in the first
    and last third of the recording, normalised to [0, 1].
    High = direction shifted = ES.  Low = direction stable = PNES.
    """
    n = len(angles)
    if n < 6:
        return 0.0
    third = n // 3

    def _mean_angle(a: np.ndarray) -> float:
        return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

    ang_early = _mean_angle(angles[:third])
    ang_late  = _mean_angle(angles[-third:])
    diff      = abs(ang_late - ang_early)
    diff      = min(diff, 2 * np.pi - diff)  # wrap to [0, π]
    return float(np.clip(diff / np.pi, 0.0, 1.0))


def _variance_drift(motion: np.ndarray, variability: np.ndarray, n_windows: int = 6) -> float:
    """
    Estimate whether amplitude and local variance increase progressively over time.

    This feature is designed to capture non-stationary growth:
    a signal may remain "noisy", yet still become progressively stronger and more variable.
    That pattern should support ES rather than PNES.

    Method:
      - split into time windows
      - compute mean motion and mean local variability per window
      - compute correlation of each with window index
      - positive drift in either dimension increases the score

    Returns 0–1.
    """
    n = len(motion)
    if n < n_windows * 5:
        return 0.0

    win = max(1, n // n_windows)
    seg_amp = []
    seg_var = []
    for i in range(n_windows):
        seg_m = motion[i * win:(i + 1) * win]
        seg_v = variability[i * win:(i + 1) * win]
        if len(seg_m) < 3 or len(seg_v) < 3:
            continue
        seg_amp.append(float(np.mean(seg_m)))
        seg_var.append(float(np.mean(seg_v)))

    if len(seg_amp) < 3:
        return 0.0

    x = np.arange(len(seg_amp), dtype=float)

    def _safe_corr(y: np.ndarray) -> float:
        if len(y) < 3 or np.std(y) < 1e-9:
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        if np.isnan(c):
            return 0.0
        return float(c)

    amp_corr = max(0.0, _safe_corr(np.asarray(seg_amp, dtype=float)))
    var_corr = max(0.0, _safe_corr(np.asarray(seg_var, dtype=float)))

    # Growth magnitude relative to baseline
    amp_growth = max(0.0, (seg_amp[-1] - seg_amp[0]) / (abs(seg_amp[0]) + 1e-6))
    var_growth = max(0.0, (seg_var[-1] - seg_var[0]) / (abs(seg_var[0]) + 1e-6))
    amp_growth = float(np.clip(amp_growth / 2.0, 0.0, 1.0))
    var_growth = float(np.clip(var_growth / 2.0, 0.0, 1.0))

    score = 0.35 * amp_corr + 0.35 * var_corr + 0.15 * amp_growth + 0.15 * var_growth
    return float(np.clip(score, 0.0, 1.0))



def _normalize_hist(x: np.ndarray, bins: int = 12, value_range=(0.0, 1.0)) -> np.ndarray:
    hist, _ = np.histogram(x, bins=bins, range=value_range, density=False)
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-9)
    return hist


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _window_shape_normalize(x: np.ndarray) -> np.ndarray:
    """
    Per-window normalisation to isolate SHAPE from scale.
    This prevents pure amplitude drift from being mistaken for stable structure.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    rng = hi - lo
    if rng < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / rng


def _temporal_distribution_stability(
    motion: np.ndarray,
    variability: np.ndarray,
    n_windows: int = 6,
    bins: int = 12,
) -> float:
    """
    Estimate whether the SHAPE of the signal remains stable over time.

    Important:
    - each window is normalised internally before histogram comparison
    - therefore this feature reflects structural similarity, not simple scale growth
    - amplitude/variance growth is handled separately by variance_drift

    High stability = windows have similar normalised motion/variability distributions → PNES.
    Low stability  = distributions change shape over time → ES.
    Returns 0–1.
    """
    n = len(motion)
    if n < n_windows * 5:
        return 0.5

    win = max(1, n // n_windows)
    motion_hists = []
    var_hists = []

    for i in range(n_windows):
        seg_m = motion[i * win:(i + 1) * win]
        seg_v = variability[i * win:(i + 1) * win]
        if len(seg_m) < 3 or len(seg_v) < 3:
            continue
        seg_mn = _window_shape_normalize(seg_m)
        seg_vn = _window_shape_normalize(seg_v)
        motion_hists.append(_normalize_hist(seg_mn, bins=bins, value_range=(0.0, 1.0)))
        var_hists.append(_normalize_hist(seg_vn, bins=bins, value_range=(0.0, 1.0)))

    if len(motion_hists) < 2:
        return 0.5

    dists = []
    for i in range(len(motion_hists) - 1):
        d_m = _js_divergence(motion_hists[i], motion_hists[i + 1])
        d_v = _js_divergence(var_hists[i], var_hists[i + 1])
        dists.append(0.5 * (d_m + d_v))

    mean_dist = float(np.mean(dists)) if dists else 0.25
    stability = 1.0 - float(np.clip(mean_dist / 0.5, 0.0, 1.0))
    return float(np.clip(stability, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# SECONDARY feature extractors (kept from previous version)
# ──────────────────────────────────────────────────────────────────────────────

def _baseline_fraction(motion: np.ndarray, threshold: float) -> float:
    return float(np.mean(motion <= threshold))


def _active_fraction(motion: np.ndarray, threshold: float) -> float:
    return float(np.mean(motion > threshold))


def _burst_isolation(motion: np.ndarray, threshold: float) -> float:
    peak = float(np.max(motion))
    half_peak = threshold + 0.5 * (peak - threshold)
    return float(np.clip(1.0 - float(np.mean(motion >= half_peak)), 0.0, 1.0))


def _temporal_escalation(motion: np.ndarray) -> float:
    n = len(motion)
    if n < 9:
        return 0.0
    third      = n // 3
    mean_first = float(np.mean(motion[:third])) + 1e-6
    mean_last  = float(np.mean(motion[-third:])) + 1e-6
    ratio      = mean_last / mean_first
    x          = np.arange(n, dtype=float)
    slope      = float(np.polyfit(x, motion, 1)[0])
    slope_norm = float(np.clip(slope / (float(np.mean(motion)) + 1e-6) * 10, 0.0, 1.0))
    ratio_score = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))
    return float(np.clip(0.5 * ratio_score + 0.5 * slope_norm, 0.0, 1.0))


def _sustained_variability(motion: np.ndarray,
                            variability: np.ndarray,
                            threshold: float) -> float:
    active_mask = motion > threshold
    if not active_mask.any():
        return 0.0
    raw  = float(np.mean(variability[active_mask]))
    peak = float(np.max(motion)) + 1e-6
    return float(np.clip(raw / peak, 0.0, 2.0))


def _post_burst_quiet(motion: np.ndarray, threshold: float) -> float:
    n = len(motion)
    if n < 8:
        return 0.5
    tail      = motion[max(0, int(n * 0.75)):]
    tail_mean = float(np.mean(tail))
    peak      = float(np.max(motion)) + 1e-6
    return float(np.clip(1.0 - (tail_mean - threshold) / (peak - threshold + 1e-6), 0.0, 1.0))


def _regional_asynchrony(frames_gray: list) -> float:
    """
    Per-frame std of optical-flow magnitude across upper/middle/lower bands.
    Returns 0.0 on any failure.
    """
    if len(frames_gray) < 2:
        return 0.0
    try:
        diffs = []
        prev  = frames_gray[0]
        h     = prev.shape[0]
        t1, t2 = h // 3, 2 * h // 3
        for curr in frames_gray[1:]:
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=2, winsize=13,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            rm  = np.array([
                float(np.mean(mag[:t1])),
                float(np.mean(mag[t1:t2])),
                float(np.mean(mag[t2:])),
            ])
            diffs.append(float(np.std(rm)))
            prev = curr
        return float(np.mean(diffs)) if diffs else 0.0
    except Exception:
        return 0.0


def _collect_region_frames(video_data: VideoData, max_frames: int = 100) -> list:
    """Small grayscale sample for regional-asynchrony analysis."""
    path = getattr(video_data, "_source_path", None)
    if not path:
        return []
    try:
        cap    = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        stride = max(1, total // max_frames)
        frames = []
        idx    = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 120))
                frames.append(gray)
            idx += 1
        cap.release()
        return frames
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(video_data: VideoData, eye_state: str = "unavailable") -> MotionFeatures:
    """
    Extract all GDI features from a VideoData object.

    Parameters
    ----------
    video_data : output of video_processing.extract_video_data()
    eye_state  : "open" | "closed" | "unavailable"
    """
    motion     = video_data.motion_magnitudes
    angles     = video_data.flow_angles
    timestamps = video_data.timestamps

    variability   = _rolling_variability(motion, window=15)
    threshold     = _baseline_threshold(motion)
    peak_motion   = float(np.max(motion))
    effective_fps = len(motion) / max(video_data.metadata.duration_seconds, 1.0)

    # GDI scalar indices
    _motion_deriv  = float(np.mean(np.abs(np.diff(motion)))) if len(motion) >= 2 else 0.0
    _var_mean      = float(np.mean(variability))
    _dynamic_idx   = _var_mean + _motion_deriv
    _stability_idx = 1.0 / (_dynamic_idx + 1e-6)

    # Regional frames (best-effort)
    _region_frames = _collect_region_frames(video_data)

    return MotionFeatures(
        # ── PRIMARY ──────────────────────────────────────────────────────────
        temporal_evolution           = _temporal_evolution(motion, variability, angles),
        movement_stereotypy          = _movement_stereotypy(motion),
        rhythm_irregularity          = _rhythm_irregularity(motion, effective_fps),
        angle_variability            = _circular_std(angles),
        amplitude_variability_over_time = _amplitude_variability_over_time(motion),
        vector_change                = _vector_change(angles),
        temporal_distribution_stability = _temporal_distribution_stability(motion, variability),
        variance_drift               = _variance_drift(motion, variability),
        # ── SECONDARY ────────────────────────────────────────────────────────
        baseline_fraction            = _baseline_fraction(motion, threshold),
        burst_isolation              = _burst_isolation(motion, threshold),
        post_burst_quiet             = _post_burst_quiet(motion, threshold),
        active_fraction              = _active_fraction(motion, threshold),
        sustained_variability        = _sustained_variability(motion, variability, threshold),
        temporal_escalation          = _temporal_escalation(motion),
        # ── Eye state ────────────────────────────────────────────────────────
        eye_state                    = eye_state,
        # ── Regional ─────────────────────────────────────────────────────────
        regional_asynchrony          = _regional_asynchrony(_region_frames),
        # ── Raw signals ──────────────────────────────────────────────────────
        motion_signal                = motion,
        timestamps                   = timestamps,
        variability_signal           = variability,
        # ── Scalar indices ───────────────────────────────────────────────────
        baseline_threshold           = threshold,
        peak_motion                  = peak_motion,
        mean_motion                  = float(np.mean(motion)),
        motion_derivative            = _motion_deriv,
        dynamic_index                = _dynamic_idx,
        stability_index              = _stability_idx,
    )
