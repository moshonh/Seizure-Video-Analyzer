
"""
motion_features.py
Burst- and evolution-oriented GDI feature extraction.

Clinical framing used here
--------------------------
PNES-like motor events tend to show:
- bursty / island structure
- abrupt start-stop episodes
- relatively similar repeated bursts
- relatively stable direction / rhythm within bursts

ES-like motor events tend to show:
- stronger temporal evolution
- fade-in / fade-out or progressive build-up
- changing direction / frequency / amplitude over time
- one more continuous evolving block rather than repeated similar bursts

This module keeps the public interface expected by the Streamlit app,
but shifts the internal features toward that distinction.
"""

from dataclasses import dataclass
import cv2
import numpy as np
from video_processing import VideoData


@dataclass
class MotionFeatures:
    # Core evolution / stability features used by app + scoring
    temporal_evolution: float
    movement_stereotypy: float
    rhythm_irregularity: float
    angle_variability: float
    amplitude_variability_over_time: float
    vector_change: float
    temporal_distribution_stability: float
    variance_drift: float

    # Burst-structure features
    episode_count: int
    episode_coverage: float
    long_quiet_fraction: float
    largest_episode_fraction: float
    burst_recurrence: float
    active_fragmentation: float
    late_intensification: float
    onset_abruptness: float
    offset_abruptness: float
    within_burst_frequency_consistency: float
    within_burst_amplitude_consistency: float
    direction_drift: float
    frequency_drift: float

    # Legacy / supportive
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


def _smooth_signal(x: np.ndarray, window: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


def _rolling_variability(motion: np.ndarray, window: int = 15) -> np.ndarray:
    n = len(motion)
    out = np.zeros(n, dtype=float)
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.std(motion[lo:hi]))
    return out


def _robust_thresholds(motion: np.ndarray) -> tuple[float, float]:
    motion = np.asarray(motion, dtype=float)
    med = float(np.median(motion))
    mad = float(np.median(np.abs(motion - med))) + 1e-9
    q20 = float(np.percentile(motion, 20))
    q60 = float(np.percentile(motion, 60))
    q85 = float(np.percentile(motion, 85))
    baseline = min(med + 0.2 * mad, q20 + 0.15 * max(0.0, q60 - q20))
    active = max(med + 2.2 * mad, q60, baseline + 0.18 * max(0.0, q85 - baseline))
    return float(baseline), float(active)


def _circular_std(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 0.0
    R = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))
    return float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-9, 1.0))))


def _mean_angle(a: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def _segment_active(active: np.ndarray, merge_gap: int = 3, min_len: int = 4) -> list[tuple[int, int]]:
    active = np.asarray(active, dtype=bool).copy()
    n = len(active)
    i = 0
    while i < n:
        if not active[i]:
            j = i
            while j < n and not active[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= merge_gap:
                active[i:j] = True
            i = j
        else:
            i += 1

    episodes = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if (j - i) >= min_len:
                episodes.append((i, j))
            i = j
        else:
            i += 1
    return episodes


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    rng = hi - lo
    if rng < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / rng


def _find_peaks_simple(signal: np.ndarray, min_distance: int = 3, min_height: float = 0.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return np.array([], dtype=int)
    peaks = []
    last = -10**9
    for i in range(1, len(signal) - 1):
        if signal[i] >= min_height and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last < min_distance:
                if peaks and signal[i] > signal[peaks[-1]]:
                    peaks[-1] = i
                    last = i
            else:
                peaks.append(i)
                last = i
    return np.asarray(peaks, dtype=int)


def _burst_recurrence(signal: np.ndarray, episodes: list[tuple[int, int]]) -> float:
    profiles = []
    for i, j in episodes:
        seg = signal[i:j]
        if len(seg) < 5:
            continue
        z = np.interp(np.linspace(0, len(seg) - 1, 32), np.arange(len(seg)), seg)
        z = _normalize_01(z)
        profiles.append(z)
    if len(profiles) < 2:
        return 0.5
    sims = []
    for a in range(len(profiles)):
        for b in range(a + 1, len(profiles)):
            c = float(np.corrcoef(profiles[a], profiles[b])[0, 1])
            if not np.isnan(c):
                sims.append(c)
    if not sims:
        return 0.5
    return float(np.clip((float(np.mean(sims)) + 1.0) / 2.0, 0.0, 1.0))


def _movement_stereotypy(signal: np.ndarray, n_segs: int = 6) -> float:
    n = len(signal)
    if n < n_segs * 4:
        return 0.5
    size = max(1, n // n_segs)
    profiles = []
    for i in range(n_segs):
        seg = signal[i * size:(i + 1) * size]
        if len(seg) < 4:
            continue
        z = _normalize_01(seg)
        profiles.append(z)
    if len(profiles) < 2:
        return 0.5
    corrs = []
    for i in range(len(profiles) - 1):
        a, b = profiles[i], profiles[i + 1]
        m = min(len(a), len(b))
        c = float(np.corrcoef(a[:m], b[:m])[0, 1])
        if not np.isnan(c):
            corrs.append(c)
    if not corrs:
        return 0.5
    return float(np.clip((float(np.mean(corrs)) + 1.0) / 2.0, 0.0, 1.0))


def _onset_offset_abruptness(signal: np.ndarray, baseline: float, episodes: list[tuple[int, int]]) -> tuple[float, float]:
    if not episodes:
        return 0.0, 0.0
    onset_scores = []
    offset_scores = []
    for i, j in episodes:
        seg = signal[i:j]
        if len(seg) < 4:
            continue
        k = max(2, min(len(seg) // 4, 8))
        peak = float(np.max(seg))
        scale = max(peak - baseline, 1e-6)
        onset_jump = (float(np.mean(seg[1:k + 1])) - float(seg[0])) / scale
        offset_drop = (float(seg[-1]) - float(np.mean(seg[-k - 1:-1]))) / scale
        onset_scores.append(np.clip(onset_jump * 1.5, 0.0, 1.0))
        offset_scores.append(np.clip((-offset_drop) * 1.5, 0.0, 1.0))
    if not onset_scores:
        return 0.0, 0.0
    return float(np.mean(onset_scores)), float(np.mean(offset_scores))


def _within_burst_consistency(signal: np.ndarray, effective_fps: float, episodes: list[tuple[int, int]]) -> tuple[float, float]:
    freq_cons = []
    amp_cons = []
    for i, j in episodes:
        seg = signal[i:j]
        if len(seg) < 6:
            continue
        local_floor = float(np.percentile(seg, 65))
        peaks = _find_peaks_simple(seg, min_distance=max(2, int(effective_fps * 0.20)), min_height=local_floor)
        if len(peaks) >= 3:
            intervals = np.diff(peaks.astype(float))
            cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
            freq_cons.append(float(np.clip(1.0 - cv / 1.0, 0.0, 1.0)))
        amp_cv = float(np.std(seg) / (np.mean(seg) + 1e-9))
        amp_cons.append(float(np.clip(1.0 - amp_cv / 0.8, 0.0, 1.0)))
    if not freq_cons:
        freq_cons = [0.5]
    if not amp_cons:
        amp_cons = [0.5]
    return float(np.mean(freq_cons)), float(np.mean(amp_cons))


def _frequency_drift(signal: np.ndarray, effective_fps: float, n_windows: int = 5) -> float:
    n = len(signal)
    if n < n_windows * 6:
        return 0.0
    win = max(4, n // n_windows)
    freqs = []
    for i in range(n_windows):
        seg = signal[i * win:(i + 1) * win]
        if len(seg) < 6:
            continue
        floor = float(np.percentile(seg, 70))
        peaks = _find_peaks_simple(seg, min_distance=max(2, int(effective_fps * 0.20)), min_height=floor)
        if len(peaks) >= 2:
            iv = np.diff(peaks.astype(float))
            freqs.append(float(1.0 / (np.mean(iv) + 1e-9)))
        else:
            freqs.append(0.0)
    if len(freqs) < 3:
        return 0.0
    freqs = np.asarray(freqs, dtype=float)
    return float(np.clip(np.std(freqs) / (np.mean(np.abs(freqs)) + 1e-9), 0.0, 1.0))


def _direction_drift(angles: np.ndarray, n_windows: int = 5) -> float:
    n = len(angles)
    if n < n_windows * 6:
        return 0.0
    win = max(4, n // n_windows)
    mean_dirs = []
    for i in range(n_windows):
        seg = angles[i * win:(i + 1) * win]
        if len(seg) == 0:
            continue
        mean_dirs.append(_mean_angle(seg))
    if len(mean_dirs) < 3:
        return 0.0
    mean_dirs = np.asarray(mean_dirs, dtype=float)
    return float(np.clip(_circular_std(mean_dirs) / 1.2, 0.0, 1.0))


def _temporal_distribution_stability(motion: np.ndarray, variability: np.ndarray, n_windows: int = 6, bins: int = 12) -> float:
    n = len(motion)
    if n < n_windows * 5:
        return 0.5
    win = max(1, n // n_windows)
    hists = []
    for i in range(n_windows):
        m = motion[i * win:(i + 1) * win]
        v = variability[i * win:(i + 1) * win]
        if len(m) < 3 or len(v) < 3:
            continue
        mz = _normalize_01(m)
        vz = _normalize_01(v)
        hm, _ = np.histogram(mz, bins=bins, range=(0.0, 1.0), density=False)
        hv, _ = np.histogram(vz, bins=bins, range=(0.0, 1.0), density=False)
        h = np.concatenate([hm, hv]).astype(float)
        h /= (np.sum(h) + 1e-9)
        hists.append(h)
    if len(hists) < 2:
        return 0.5
    dists = []
    for i in range(len(hists) - 1):
        a = hists[i]
        b = hists[i + 1]
        dists.append(float(np.sum(np.abs(a - b))) / 2.0)
    return float(np.clip(1.0 - np.mean(dists), 0.0, 1.0))


def _variance_drift(motion: np.ndarray, variability: np.ndarray, n_windows: int = 6) -> float:
    n = len(motion)
    if n < n_windows * 5:
        return 0.0
    win = max(1, n // n_windows)
    amps = []
    vars_ = []
    for i in range(n_windows):
        m = motion[i * win:(i + 1) * win]
        v = variability[i * win:(i + 1) * win]
        if len(m) < 3:
            continue
        amps.append(float(np.mean(m)))
        vars_.append(float(np.mean(v)))
    if len(amps) < 3:
        return 0.0
    x = np.arange(len(amps), dtype=float)

    def _corr(y):
        y = np.asarray(y, dtype=float)
        if np.std(y) < 1e-9:
            return 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return 0.0 if np.isnan(c) else max(0.0, c)

    amp_corr = _corr(amps)
    var_corr = _corr(vars_)
    amp_growth = max(0.0, (amps[-1] - amps[0]) / (abs(amps[0]) + 1e-9))
    var_growth = max(0.0, (vars_[-1] - vars_[0]) / (abs(vars_[0]) + 1e-9))
    return float(np.clip(0.35 * amp_corr + 0.35 * var_corr + 0.15 * min(1.0, amp_growth / 2.0) + 0.15 * min(1.0, var_growth / 2.0), 0.0, 1.0))


def _temporal_evolution(motion: np.ndarray, variability: np.ndarray, angles: np.ndarray) -> float:
    amp_drift = _variance_drift(motion, variability)
    dir_drift = _direction_drift(angles)
    freq_drift = _frequency_drift(motion, effective_fps=max(len(motion) / max(len(motion), 1), 1))
    return float(np.clip(0.45 * amp_drift + 0.30 * dir_drift + 0.25 * freq_drift, 0.0, 1.0))


def _regional_asynchrony(frames_gray: list) -> float:
    if len(frames_gray) < 2:
        return 0.0
    try:
        diffs = []
        prev = frames_gray[0]
        h = prev.shape[0]
        t1, t2 = h // 3, 2 * h // 3
        for curr in frames_gray[1:]:
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 2, 13, 2, 5, 1.1, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            rm = np.array([float(np.mean(mag[:t1])), float(np.mean(mag[t1:t2])), float(np.mean(mag[t2:]))])
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


def extract_features(video_data: VideoData, eye_state: str = "unavailable") -> MotionFeatures:
    motion = np.asarray(video_data.motion_magnitudes, dtype=float)
    angles = np.asarray(video_data.flow_angles, dtype=float)
    timestamps = np.asarray(video_data.timestamps, dtype=float)

    sm_motion = _smooth_signal(motion, 7)
    variability = _rolling_variability(sm_motion, window=15)
    baseline_threshold, active_threshold = _robust_thresholds(sm_motion)

    n = len(sm_motion)
    effective_fps = len(sm_motion) / max(float(video_data.metadata.duration_seconds), 1.0)

    active = sm_motion > active_threshold
    episodes = _segment_active(
        active,
        merge_gap=max(2, min(8, int(0.02 * n))),
        min_len=max(3, int(0.01 * n)),
    )

    episode_count = len(episodes)
    active_fraction = float(np.mean(active)) if n else 0.0
    baseline_fraction = float(np.mean(sm_motion <= baseline_threshold)) if n else 0.0
    episode_coverage = float(sum(j - i for i, j in episodes) / max(n, 1))
    largest_episode_fraction = float(max((j - i for i, j in episodes), default=0) / max(n, 1))
    active_fragmentation = float(np.clip((episode_count - 1) / max(episode_count, 1), 0.0, 1.0))

    # quiet gaps
    quiet_total = 0
    long_quiet = 0
    quiet_min = max(4, int(0.03 * n))
    if episodes:
        gaps = [(0, episodes[0][0])]
        gaps += [(b, c) for (_, b), (c, _) in zip(episodes, episodes[1:])]
        gaps += [(episodes[-1][1], n)]
    else:
        gaps = [(0, n)]
    for a, b in gaps:
        gl = max(0, b - a)
        quiet_total += gl
        if gl >= quiet_min:
            long_quiet += gl
    long_quiet_fraction = float(long_quiet / max(n, 1))

    # burst recurrence + within-burst consistency
    burst_recurrence = _burst_recurrence(sm_motion, episodes)
    freq_cons, amp_cons = _within_burst_consistency(sm_motion, effective_fps, episodes)

    # drift features
    direction_drift = _direction_drift(angles)
    frequency_drift = _frequency_drift(sm_motion, effective_fps)
    variance_drift = _variance_drift(sm_motion, variability)

    # late intensification: late-third stronger than early-third
    thirds = np.array_split(sm_motion, 3)
    if len(thirds) == 3 and len(thirds[0]) and len(thirds[2]):
        early = float(np.mean(thirds[0]))
        late = float(np.mean(thirds[2]))
        late_intensification = float(np.clip((late - early) / (early + 1e-9), 0.0, 2.0) / 2.0)
        vector_change = float(np.clip(abs(_mean_angle(angles[-len(angles)//3:]) - _mean_angle(angles[:len(angles)//3])) / np.pi, 0.0, 1.0)) if len(angles) >= 6 else 0.0
    else:
        late_intensification = 0.0
        vector_change = 0.0

    onset_abruptness, offset_abruptness = _onset_offset_abruptness(sm_motion, baseline_threshold, episodes)

    # global descriptive features
    movement_stereotypy = float(np.clip(0.60 * _movement_stereotypy(sm_motion) + 0.40 * burst_recurrence, 0.0, 1.0))
    rhythm_irregularity = float(np.clip(1.0 - freq_cons + 0.35 * frequency_drift, 0.0, 1.5))
    amplitude_variability_over_time = float(np.clip(1.0 - amp_cons + 0.50 * variance_drift, 0.0, 1.0))
    angle_variability = _circular_std(angles)
    temporal_distribution_stability = _temporal_distribution_stability(sm_motion, variability)
    temporal_evolution = float(np.clip(0.40 * variance_drift + 0.35 * direction_drift + 0.25 * frequency_drift, 0.0, 1.0))

    # supportive legacy metrics
    peak_motion = float(np.max(sm_motion)) if n else 0.0
    mean_motion = float(np.mean(sm_motion)) if n else 0.0
    motion_derivative = float(np.mean(np.abs(np.diff(sm_motion)))) if n >= 2 else 0.0
    dynamic_index = float(np.mean(variability) + motion_derivative) if n else 0.0
    stability_index = float(1.0 / (dynamic_index + 1e-6))
    burst_isolation = float(np.clip(1.0 - episode_coverage, 0.0, 1.0))
    tail = sm_motion[max(0, int(0.75 * n)):] if n else np.array([], dtype=float)
    post_burst_quiet = float(np.clip(1.0 - max(0.0, float(np.mean(tail)) - baseline_threshold) / (peak_motion - baseline_threshold + 1e-6), 0.0, 1.0)) if len(tail) else 0.5
    active_mask = sm_motion > active_threshold
    sustained_variability = float(np.mean(variability[active_mask])) / (peak_motion + 1e-6) if np.any(active_mask) else 0.0
    temporal_escalation = late_intensification

    region_frames = _collect_region_frames(video_data)
    regional_asynchrony = _regional_asynchrony(region_frames)

    return MotionFeatures(
        temporal_evolution=temporal_evolution,
        movement_stereotypy=movement_stereotypy,
        rhythm_irregularity=rhythm_irregularity,
        angle_variability=angle_variability,
        amplitude_variability_over_time=amplitude_variability_over_time,
        vector_change=vector_change,
        temporal_distribution_stability=temporal_distribution_stability,
        variance_drift=variance_drift,
        episode_count=episode_count,
        episode_coverage=episode_coverage,
        long_quiet_fraction=long_quiet_fraction,
        largest_episode_fraction=largest_episode_fraction,
        burst_recurrence=burst_recurrence,
        active_fragmentation=active_fragmentation,
        late_intensification=late_intensification,
        onset_abruptness=onset_abruptness,
        offset_abruptness=offset_abruptness,
        within_burst_frequency_consistency=freq_cons,
        within_burst_amplitude_consistency=amp_cons,
        direction_drift=direction_drift,
        frequency_drift=frequency_drift,
        baseline_fraction=baseline_fraction,
        burst_isolation=burst_isolation,
        post_burst_quiet=post_burst_quiet,
        active_fraction=active_fraction,
        sustained_variability=sustained_variability,
        temporal_escalation=temporal_escalation,
        eye_state=eye_state,
        regional_asynchrony=regional_asynchrony,
        motion_signal=sm_motion,
        timestamps=timestamps,
        variability_signal=variability,
        baseline_threshold=baseline_threshold,
        peak_motion=peak_motion,
        mean_motion=mean_motion,
        motion_derivative=motion_derivative,
        dynamic_index=dynamic_index,
        stability_index=stability_index,
    )
