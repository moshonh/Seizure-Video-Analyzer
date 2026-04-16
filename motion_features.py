
"""
motion_features.py
Simplified and more robust GDI-oriented motion feature extraction.

Main design change:
- focus on EVENT SHAPE over time
- explicitly model burst/island structure versus continuously evolving activity
- remove scipy dependency
"""

import cv2
import numpy as np
from dataclasses import dataclass
from video_processing import VideoData


@dataclass
class MotionFeatures:
    # Primary evolution / stability features
    temporal_evolution: float
    movement_stereotypy: float
    rhythm_irregularity: float
    angle_variability: float
    amplitude_variability_over_time: float
    vector_change: float
    temporal_distribution_stability: float
    variance_drift: float

    # New episode-shape features
    episode_count: int
    episode_coverage: float
    long_quiet_fraction: float
    largest_episode_fraction: float
    burst_recurrence: float
    active_fragmentation: float
    late_intensification: float

    # Secondary legacy features
    baseline_fraction: float
    burst_isolation: float
    post_burst_quiet: float
    active_fraction: float
    sustained_variability: float
    temporal_escalation: float

    eye_state: str
    regional_asynchrony: float

    motion_signal: np.ndarray
    timestamps: np.ndarray
    variability_signal: np.ndarray

    baseline_threshold: float
    peak_motion: float
    mean_motion: float
    motion_derivative: float
    dynamic_index: float
    stability_index: float
    gdi_evolution_ratio: float


def _rolling_variability(motion: np.ndarray, window: int = 15) -> np.ndarray:
    n = len(motion)
    out = np.zeros(n, dtype=float)
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.std(motion[lo:hi]))
    return out


def _baseline_threshold(motion: np.ndarray) -> float:
    q25 = float(np.percentile(motion, 25))
    q60 = float(np.percentile(motion, 60))
    return q25 + 0.20 * max(0.0, q60 - q25)


def _high_activity_threshold(motion: np.ndarray, base: float) -> float:
    peak = float(np.max(motion))
    return base + 0.25 * max(0.0, peak - base)


def _circular_std(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 0.0
    R = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))
    return float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-9, 1.0))))


def _segment_angles(angles: np.ndarray, n_segs: int = 4) -> np.ndarray:
    n = len(angles)
    size = max(1, n // n_segs)
    result = []
    for i in range(n_segs):
        seg = angles[i * size: (i + 1) * size]
        if len(seg) == 0:
            result.append(0.0)
            continue
        result.append(float(np.arctan2(np.mean(np.sin(seg)), np.mean(np.cos(seg)))))
    return np.asarray(result, dtype=float)


def _smooth_signal(x: np.ndarray, window: int = 9) -> np.ndarray:
    if len(x) < 3:
        return x.astype(float)
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


def _find_peaks_simple(signal: np.ndarray, min_distance: int = 3, min_height: float = 0.0) -> np.ndarray:
    if len(signal) < 3:
        return np.array([], dtype=int)
    candidates = []
    last_idx = -10**9
    for i in range(1, len(signal) - 1):
        if signal[i] >= min_height and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last_idx < min_distance:
                if candidates and signal[i] > signal[candidates[-1]]:
                    candidates[-1] = i
                    last_idx = i
            else:
                candidates.append(i)
                last_idx = i
    return np.asarray(candidates, dtype=int)


def _rhythm_irregularity(motion: np.ndarray, effective_fps: float) -> float:
    if len(motion) < 8:
        return 0.0
    sm = _smooth_signal(motion, 7)
    prominence_floor = float(np.percentile(sm, 75))
    peaks = _find_peaks_simple(sm, min_distance=max(2, int(effective_fps * 0.35)), min_height=prominence_floor)
    if len(peaks) < 2:
        return 0.0
    intervals = np.diff(peaks.astype(float))
    mean_iv = float(np.mean(intervals))
    if mean_iv < 1e-6:
        return 0.0
    return float(np.clip(float(np.std(intervals)) / mean_iv, 0.0, 3.0))


def _window_shape_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    rng = hi - lo
    if rng < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / rng


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


def _temporal_distribution_stability(motion: np.ndarray, variability: np.ndarray, n_windows: int = 6, bins: int = 12) -> float:
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
        motion_hists.append(_normalize_hist(_window_shape_normalize(seg_m), bins=bins))
        var_hists.append(_normalize_hist(_window_shape_normalize(seg_v), bins=bins))
    if len(motion_hists) < 2:
        return 0.5
    dists = []
    for i in range(len(motion_hists) - 1):
        dists.append(0.5 * (_js_divergence(motion_hists[i], motion_hists[i + 1]) + _js_divergence(var_hists[i], var_hists[i + 1])))
    mean_dist = float(np.mean(dists)) if dists else 0.25
    return float(np.clip(1.0 - mean_dist / 0.5, 0.0, 1.0))


def _variance_drift(motion: np.ndarray, variability: np.ndarray, n_windows: int = 6) -> float:
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
    amp_growth = max(0.0, (seg_amp[-1] - seg_amp[0]) / (abs(seg_amp[0]) + 1e-6))
    var_growth = max(0.0, (seg_var[-1] - seg_var[0]) / (abs(seg_var[0]) + 1e-6))
    amp_growth = float(np.clip(amp_growth / 2.0, 0.0, 1.0))
    var_growth = float(np.clip(var_growth / 2.0, 0.0, 1.0))
    return float(np.clip(0.35 * amp_corr + 0.35 * var_corr + 0.15 * amp_growth + 0.15 * var_growth, 0.0, 1.0))


def _temporal_evolution(motion: np.ndarray, variability: np.ndarray, angles: np.ndarray, n_segs: int = 5) -> float:
    n = len(motion)
    if n < n_segs * 4:
        return 0.0
    size = max(1, n // n_segs)
    seg_var = np.array([float(np.mean(variability[i * size:(i + 1) * size])) for i in range(n_segs)], dtype=float)
    seg_amp = np.array([float(np.mean(motion[i * size:(i + 1) * size])) for i in range(n_segs)], dtype=float)
    seg_dir = _segment_angles(angles, n_segs)

    var_cv = float(np.std(seg_var)) / (float(np.mean(seg_var)) + 1e-6)
    amp_cv = float(np.std(seg_amp)) / (float(np.mean(seg_amp)) + 1e-6)
    dir_std = _circular_std(seg_dir)

    var_score = float(np.clip(var_cv / 0.9, 0.0, 1.0))
    amp_score = float(np.clip(amp_cv / 0.9, 0.0, 1.0))
    dir_score = float(np.clip(dir_std / 1.1, 0.0, 1.0))
    return float(np.clip((var_score + amp_score + dir_score) / 3.0, 0.0, 1.0))


def _movement_stereotypy(motion: np.ndarray, active_mask: np.ndarray, n_segs: int = 6) -> float:
    n = len(motion)
    if n < n_segs * 4:
        return 0.5
    size = max(1, n // n_segs)
    profiles = []
    for i in range(n_segs):
        seg = motion[i * size:(i + 1) * size].astype(float)
        act = active_mask[i * size:(i + 1) * size]
        if len(seg) < 4 or np.mean(act) < 0.15:
            continue
        seg = _smooth_signal(seg, 5)
        seg = _window_shape_normalize(seg)
        profiles.append(seg)
    if len(profiles) < 2:
        return 0.5
    corrs = []
    for i in range(len(profiles) - 1):
        a, b = profiles[i], profiles[i + 1]
        m = min(len(a), len(b))
        if m < 4:
            continue
        c = float(np.corrcoef(a[:m], b[:m])[0, 1])
        if not np.isnan(c):
            corrs.append(c)
    if not corrs:
        return 0.5
    return float(np.clip((float(np.mean(corrs)) + 1.0) / 2.0, 0.0, 1.0))


def _vector_change(angles: np.ndarray) -> float:
    n = len(angles)
    if n < 6:
        return 0.0

    def _mean_angle(a: np.ndarray) -> float:
        return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

    third = max(1, n // 3)
    early = _mean_angle(angles[:third])
    late = _mean_angle(angles[-third:])
    diff = abs(late - early)
    diff = min(diff, 2 * np.pi - diff)
    return float(np.clip(diff / np.pi, 0.0, 1.0))


def _baseline_fraction(motion: np.ndarray, threshold: float) -> float:
    return float(np.mean(motion <= threshold))


def _active_fraction(motion: np.ndarray, threshold: float) -> float:
    return float(np.mean(motion > threshold))


def _burst_isolation(motion: np.ndarray, threshold: float) -> float:
    peak = float(np.max(motion))
    half_peak = threshold + 0.5 * max(0.0, peak - threshold)
    return float(np.clip(1.0 - float(np.mean(motion >= half_peak)), 0.0, 1.0))


def _temporal_escalation(motion: np.ndarray) -> float:
    n = len(motion)
    if n < 9:
        return 0.0
    third = max(1, n // 3)
    mean_first = float(np.mean(motion[:third])) + 1e-6
    mean_last = float(np.mean(motion[-third:])) + 1e-6
    ratio = mean_last / mean_first
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, motion, 1)[0])
    slope_norm = float(np.clip(slope / (float(np.mean(motion)) + 1e-6) * 10, 0.0, 1.0))
    ratio_score = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))
    return float(np.clip(0.5 * ratio_score + 0.5 * slope_norm, 0.0, 1.0))


def _sustained_variability(motion: np.ndarray, variability: np.ndarray, threshold: float) -> float:
    active_mask = motion > threshold
    if not np.any(active_mask):
        return 0.0
    raw = float(np.mean(variability[active_mask]))
    peak = float(np.max(motion)) + 1e-6
    return float(np.clip(raw / peak, 0.0, 2.0))


def _post_burst_quiet(motion: np.ndarray, threshold: float) -> float:
    n = len(motion)
    if n < 8:
        return 0.5
    tail = motion[max(0, int(n * 0.75)):]
    tail_mean = float(np.mean(tail))
    peak = float(np.max(motion)) + 1e-6
    return float(np.clip(1.0 - (tail_mean - threshold) / (peak - threshold + 1e-6), 0.0, 1.0))


def _extract_activity_episodes(motion: np.ndarray, active_mask: np.ndarray, min_len: int = 3, bridge_gap: int = 2):
    n = len(motion)
    mask = active_mask.astype(bool).copy()
    if n == 0:
        return []
    # bridge tiny gaps
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        gap = j - i
        if 0 < gap <= bridge_gap and i > 0 and j < n and mask[i - 1] and mask[j]:
            mask[i:j] = True
        i = j

    episodes = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            seg = motion[i:j]
            episodes.append({
                "start": i,
                "end": j,
                "len": j - i,
                "mean": float(np.mean(seg)),
                "peak": float(np.max(seg)),
                "profile": _window_shape_normalize(_smooth_signal(seg, 5)),
            })
        i = j
    return episodes


def _episode_features(motion: np.ndarray, active_threshold: float):
    active_mask = motion > active_threshold
    episodes = _extract_activity_episodes(motion, active_mask)
    n = len(motion)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.5, 0.0, active_mask

    coverage = float(np.mean(active_mask))
    quiet_runs = []
    i = 0
    while i < n:
        if active_mask[i]:
            i += 1
            continue
        j = i
        while j < n and not active_mask[j]:
            j += 1
        quiet_runs.append(j - i)
        i = j

    long_quiet_fraction = float(sum(q for q in quiet_runs if q >= max(4, int(0.06 * n))) / max(1, n))
    largest_episode_fraction = 0.0
    if episodes:
        largest_episode_fraction = float(max(ep["len"] for ep in episodes) / max(1, np.sum(active_mask)))

    if len(episodes) >= 2:
        sims = []
        for i in range(len(episodes) - 1):
            a = episodes[i]["profile"]
            b = episodes[i + 1]["profile"]
            m = min(len(a), len(b))
            if m >= 4:
                c = float(np.corrcoef(a[:m], b[:m])[0, 1])
                if not np.isnan(c):
                    sims.append(c)
        burst_recurrence = float(np.clip((float(np.mean(sims)) + 1.0) / 2.0, 0.0, 1.0)) if sims else 0.5
    else:
        burst_recurrence = 0.5

    active_fragmentation = 0.0
    if np.sum(active_mask) > 0:
        transitions = np.sum(active_mask[1:] != active_mask[:-1])
        active_fragmentation = float(np.clip(transitions / max(1, len(motion) // 6), 0.0, 1.0))

    return len(episodes), coverage, long_quiet_fraction, largest_episode_fraction, burst_recurrence, active_fragmentation, active_mask


def _late_intensification(motion: np.ndarray) -> float:
    n = len(motion)
    if n < 12:
        return 0.0
    third = max(1, n // 3)
    first = float(np.mean(motion[:third])) + 1e-6
    last = float(np.mean(motion[-third:])) + 1e-6
    ratio = last / first
    return float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))


def _regional_asynchrony(frames_gray: list) -> float:
    if len(frames_gray) < 2:
        return 0.0
    try:
        diffs = []
        prev = frames_gray[0]
        h = prev.shape[0]
        t1, t2 = h // 3, 2 * h // 3
        for curr in frames_gray[1:]:
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=2, winsize=13,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            rm = np.array([
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
    path = getattr(video_data, "_source_path", None)
    if not path:
        return []
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        stride = max(1, total // max_frames)
        frames = []
        idx = 0
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



def _calculate_gdi_evolution(motion: np.ndarray) -> float:
    """
    Calculates the ratio of variance between the last 25% and first 25% 
    of the active motion burst (GDI Evolution Principle).
    """
    if len(motion) < 20: return 1.0
    threshold = np.max(motion) * 0.3
    active_idx = np.where(motion > threshold)[0]
    if len(active_idx) < 10: return 1.0
    
    burst_data = motion[active_idx[0]:active_idx[-1]]
    n = len(burst_data)
    first_q = burst_data[:max(1, n//4)]
    last_q = burst_data[-max(1, n//4):]
    
    v_first = np.var(first_q) + 1e-10
    v_last = np.var(last_q) + 1e-10
    return float(v_last / v_first)

def extract_features(video_data: VideoData, eye_state: str = "unavailable") -> MotionFeatures:
    motion = np.asarray(video_data.motion_magnitudes, dtype=float)
    angles = np.asarray(video_data.flow_angles, dtype=float)
    timestamps = np.asarray(video_data.timestamps, dtype=float)

    variability = _rolling_variability(motion, window=15)
    threshold = _baseline_threshold(motion)
    active_threshold = _high_activity_threshold(motion, threshold)
    peak_motion = float(np.max(motion)) if len(motion) else 0.0
    effective_fps = len(motion) / max(getattr(video_data.metadata, "duration_seconds", 1.0), 1.0)

    (
        episode_count,
        episode_coverage,
        long_quiet_fraction,
        largest_episode_fraction,
        burst_recurrence,
        active_fragmentation,
        active_mask,
    ) = _episode_features(motion, active_threshold)

    motion_deriv = float(np.mean(np.abs(np.diff(motion)))) if len(motion) >= 2 else 0.0
    var_mean = float(np.mean(variability)) if len(variability) else 0.0
    dynamic_idx = var_mean + motion_deriv
    stability_idx = 1.0 / (dynamic_idx + 1e-6)

    region_frames = _collect_region_frames(video_data)
    evolution_ratio = _calculate_gdi_evolution(motion)

    return MotionFeatures(
        temporal_evolution=_temporal_evolution(motion, variability, angles),
        movement_stereotypy=_movement_stereotypy(motion, active_mask),
        rhythm_irregularity=_rhythm_irregularity(motion, effective_fps),
        angle_variability=_circular_std(angles),
        amplitude_variability_over_time=float(np.clip(np.std([float(np.mean(motion[i:i + max(1, len(motion)//6)])) for i in range(0, len(motion), max(1, len(motion)//6))]) / (float(np.mean(motion)) + 1e-6), 0.0, 1.0)) if len(motion) else 0.0,
        vector_change=_vector_change(angles),
        temporal_distribution_stability=_temporal_distribution_stability(motion, variability),
        variance_drift=_variance_drift(motion, variability),

        episode_count=episode_count,
        episode_coverage=episode_coverage,
        long_quiet_fraction=long_quiet_fraction,
        largest_episode_fraction=largest_episode_fraction,
        burst_recurrence=burst_recurrence,
        active_fragmentation=active_fragmentation,
        late_intensification=_late_intensification(motion),

        baseline_fraction=_baseline_fraction(motion, threshold),
        burst_isolation=_burst_isolation(motion, threshold),
        post_burst_quiet=_post_burst_quiet(motion, threshold),
        active_fraction=_active_fraction(motion, threshold),
        sustained_variability=_sustained_variability(motion, variability, threshold),
        temporal_escalation=_temporal_escalation(motion),

        eye_state=eye_state,
        regional_asynchrony=_regional_asynchrony(region_frames),

        motion_signal=motion,
        timestamps=timestamps,
        variability_signal=variability,

        baseline_threshold=threshold,
        peak_motion=peak_motion,
        mean_motion=float(np.mean(motion)) if len(motion) else 0.0,
        motion_derivative=motion_deriv,
        dynamic_index=dynamic_idx,
        stability_index=stability_idx,
        gdi_evolution_ratio=evolution_ratio,
    )
