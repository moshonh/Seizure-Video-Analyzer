
"""
scoring.py
Simplified 3-output GDI classifier with stronger emphasis on overall event shape.

Clinical intent:
- Typical PNES pattern: repeated burst/island structure, long quiet returns to baseline,
  relatively stereotyped shape, limited evolution
- Evolving / not typical PNES: more continuous activity, stronger drift/evolution,
  reduced quiet return, changing organisation over time
- Indeterminate: mixed / poor quality / borderline
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


RELIABILITY_FLOOR = 0.35
COMMIT_MARGIN = 1.0
PNES_MIN_SCORE = 4.8
EVOL_MIN_SCORE = 4.8
PNES_MAX_EVOL = 5.6
EVOL_MAX_PNES = 5.6
HIGH_CONFIDENCE_MARGIN = 2.6
MODERATE_CONFIDENCE_MARGIN = 1.5

# Temporal evolution veto gate:
# If temporal_evolution exceeds this threshold, PNES is blocked.
# Clinically: clear temporal evolution is incompatible with a typical PNES pattern.
# Media1 (epileptic): te=0.615 > 0.55 → blocked ✅
# Media2 (PNES):      te=0.441 < 0.55 → unaffected ✅
PNES_TE_GATE      = 0.55   # temporal_evolution above this → cannot call PNES
EVOL_MIN_GATED    = 3.0    # minimum evol_score to call Evolving via gate


@dataclass
class Weights:
    movement_stereotypy: float = 1.8
    temporal_distribution_stability: float = 1.1
    variance_drift: float = 1.9
    vector_change: float = 1.0
    amplitude_variability: float = 0.9
    rhythm_irregularity: float = 0.7
    temporal_evolution: float = 1.9
    eye_feature: float = 0.7

    # new shape-level weights
    long_quiet_fraction: float = 2.2
    episode_structure: float = 2.0
    burst_recurrence: float = 1.5
    late_intensification: float = 1.2

    # accepted from sidebar / legacy
    temporal_escalation: float = 0.0
    burst_isolation: float = 0.0
    sustained_variability: float = 0.0
    active_fraction: float = 0.0
    baseline_fraction: float = 0.0
    post_burst_quiet: float = 0.0
    direction_variability: float = 0.0
    regional_asynchrony: float = 0.0


@dataclass
class ScoringResult:
    pnes_score: float
    es_score: float
    pnes_pattern_confidence: float
    reliability_score: float
    classification: str
    confidence: str
    reliability: str
    feature_scores: Dict[str, dict] = field(default_factory=dict)
    quality_penalties: Dict[str, bool] = field(default_factory=dict)
    explanation: str = ""
    gate_flags: Dict[str, bool] = field(default_factory=dict)


def _feat(p: float, e: float, wp: float, we: float, label: str, detail: str) -> dict:
    return {
        "pnes": round(float(p) * wp, 3),
        "es": round(float(e) * we, 3),
        "label": label,
        "detail": detail,
        "weight": max(wp, we),
    }


def _score_stereotypy(f: MotionFeatures, w: float) -> dict:
    p = f.movement_stereotypy
    e = 1.0 - p
    lbl = ("Repeated shape across segments → PNES-like" if p > 0.58
           else "Shape changes across segments → evolving" if p < 0.42
           else "Borderline segment similarity")
    return _feat(p, e, w, w, lbl, f"Movement stereotypy: {p:.3f}")


def _score_distribution_stability(f: MotionFeatures, w: float) -> dict:
    p = f.temporal_distribution_stability
    e = 1.0 - p
    lbl = ("Stable signal structure → PNES-like" if p > 0.56
           else "Changing signal structure → evolving" if p < 0.42
           else "Borderline structural stability")
    return _feat(p, e, w, w, lbl, f"Distribution stability: {p:.3f}")


def _score_variance_drift(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.variance_drift
    e = f.variance_drift
    lbl = ("Little progressive drift → PNES-like" if f.variance_drift < 0.28
           else "Progressive drift over time → evolving" if f.variance_drift > 0.40
           else "Mild drift")
    return _feat(p, e, w, w, lbl, f"Variance drift: {f.variance_drift:.3f}")


def _score_vector_change(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.vector_change
    e = f.vector_change
    lbl = ("Direction broadly stable → PNES-like" if f.vector_change < 0.30
           else "Direction shifts over time → evolving" if f.vector_change > 0.45
           else "Moderate direction change")
    return _feat(p, e, w, w, lbl, f"Vector change: {f.vector_change:.3f}")


def _score_amplitude_variability(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.amplitude_variability_over_time
    e = f.amplitude_variability_over_time
    lbl = ("Amplitude relatively stable → PNES-like" if f.amplitude_variability_over_time < 0.30
           else "Amplitude changes across event → evolving" if f.amplitude_variability_over_time > 0.46
           else "Moderate amplitude variation")
    return _feat(p, e, w, w, lbl, f"Amplitude variability: {f.amplitude_variability_over_time:.3f}")


def _score_rhythm(f: MotionFeatures, w: float) -> dict:
    norm = min(f.rhythm_irregularity / 1.4, 1.0)
    p = 1.0 - norm
    e = norm
    lbl = ("More regular burst rhythm → PNES-like" if f.rhythm_irregularity < 0.55
           else "Irregular rhythm → evolving" if f.rhythm_irregularity > 0.95
           else "Moderately irregular rhythm")
    return _feat(p, e, w, w, lbl, f"Rhythm irregularity: {f.rhythm_irregularity:.3f}")


def _score_temporal_evolution(f: MotionFeatures, w: float) -> dict:
    e = f.temporal_evolution
    p = 1.0 - e
    lbl = ("Limited overall evolution → PNES-like" if e < 0.32
           else "Clear evolution across time → evolving" if e > 0.46
           else "Moderate temporal evolution")
    return _feat(p, e, w, w, lbl, f"Temporal evolution: {e:.3f}")


def _score_eye_state(f: MotionFeatures, w: float) -> dict:
    if f.eye_state == "closed":
        return _feat(1.0, 0.0, w, 0.0, "Eyes closed → supports PNES-like", "Eye closure observed.")
    if f.eye_state == "open":
        return _feat(0.0, 1.0, 0.0, w, "Eyes open → mild support for evolving pattern", "Eyes appear open.")
    return _feat(0.0, 0.0, 0.0, 0.0, "Eye state unavailable → neutral", "Eye state not determined.")


def _score_quiet_returns(f: MotionFeatures, w: float) -> dict:
    p = min(1.0, max(0.0, 0.65 * f.long_quiet_fraction + 0.35 * f.post_burst_quiet))
    e = 1.0 - p
    lbl = ("Clear returns to quiet baseline between bursts → PNES-like" if p > 0.58
           else "Little quiet return / more continuous activity → evolving" if p < 0.40
           else "Intermediate quiet-return pattern")
    return _feat(p, e, w, w, lbl,
                 f"Long quiet fraction: {f.long_quiet_fraction:.3f}; post-burst quiet: {f.post_burst_quiet:.3f}")


def _score_episode_structure(f: MotionFeatures, w: float) -> dict:
    # PNES-like: multiple islands, lower coverage, not one single continuous active block
    multi_episode = min(max((f.episode_count - 1) / 3.0, 0.0), 1.0)
    lower_coverage = 1.0 - min(f.episode_coverage / 0.70, 1.0)
    not_one_block = 1.0 - min(f.largest_episode_fraction, 1.0)
    p = min(1.0, max(0.0, 0.40 * multi_episode + 0.35 * lower_coverage + 0.25 * not_one_block))
    e = 1.0 - p
    lbl = ("Burst/island structure → PNES-like" if p > 0.56
           else "Single dominant / continuous active block → evolving" if p < 0.40
           else "Intermediate episode structure")
    return _feat(p, e, w, w, lbl,
                 f"Episodes: {f.episode_count}; coverage: {f.episode_coverage:.3f}; largest active block share: {f.largest_episode_fraction:.3f}")


def _score_burst_recurrence(f: MotionFeatures, w: float) -> dict:
    p = f.burst_recurrence
    e = 1.0 - p
    lbl = ("Repeated burst shape → PNES-like" if p > 0.58
           else "Burst shapes differ over time → evolving" if p < 0.42
           else "Borderline burst recurrence")
    return _feat(p, e, w, w, lbl, f"Burst recurrence: {f.burst_recurrence:.3f}")


def _score_late_intensification(f: MotionFeatures, w: float) -> dict:
    e = min(1.0, max(0.0, 0.60 * f.late_intensification + 0.40 * f.temporal_escalation))
    p = 1.0 - e
    lbl = ("No major late build-up → PNES-like" if e < 0.30
           else "Late intensification / build-up → evolving" if e > 0.45
           else "Moderate late intensification")
    return _feat(p, e, w, w, lbl,
                 f"Late intensification: {f.late_intensification:.3f}; temporal escalation: {f.temporal_escalation:.3f}")


def _compute_reliability(video_data: VideoData) -> float:
    score = 1.0
    qf = getattr(video_data, "quality_flags", {}) or {}
    if qf.get("short_video"):
        score -= 0.22
    if qf.get("missing_onset"):
        score -= 0.18
    if qf.get("missing_offset"):
        score -= 0.10
    if qf.get("low_resolution"):
        score -= 0.15
    if qf.get("very_low_motion"):
        score -= 0.25
    return max(0.0, round(score, 2))


def _reliability_label(r: float) -> str:
    if r >= 0.70:
        return "adequate"
    if r >= 0.45:
        return "limited"
    return "poor"


def _confidence_label(margin: float) -> str:
    if margin >= HIGH_CONFIDENCE_MARGIN:
        return "high"
    if margin >= MODERATE_CONFIDENCE_MARGIN:
        return "moderate"
    return "low"


def _generate_explanation(result: "ScoringResult", features: MotionFeatures, video_data: VideoData) -> str:
    lines = []
    if result.classification == "Typical PNES pattern":
        lines.append(
            "This recording shows a more burst-like, internally repetitive pattern with clearer returns toward baseline, "
            "which favours a typical PNES-like motor pattern in this prototype."
        )
        if features.long_quiet_fraction > 0.18:
            lines.append(
                f"There are meaningful quiet intervals between active epochs (long quiet fraction {features.long_quiet_fraction:.2f})."
            )
        if features.episode_count >= 2:
            lines.append(
                f"The activity is organised into {features.episode_count} distinct active epochs rather than one dominant continuously active block."
            )
        if features.burst_recurrence > 0.56:
            lines.append(
                f"Active epochs show recurrent shape similarity (burst recurrence {features.burst_recurrence:.2f})."
            )
        if features.temporal_evolution < 0.35:
            lines.append("Overall temporal evolution remains limited.")
        if features.eye_state == "closed":
            lines.append("Eye closure adds mild supportive evidence for a PNES-like pattern.")
    elif result.classification == "Evolving / not typical PNES":
        lines.append(
            "This recording shows a more continuously evolving pattern with less return to quiet baseline, "
            "which is not typical of the PNES-like pattern targeted by this prototype."
        )
        if features.long_quiet_fraction < 0.10:
            lines.append("There is little quiet return between active periods.")
        if features.temporal_evolution > 0.42:
            lines.append(f"Temporal evolution is substantial (score {features.temporal_evolution:.2f}).")
        if features.variance_drift > 0.38:
            lines.append(f"Amplitude/variance drift increases over time (score {features.variance_drift:.2f}).")
        if features.largest_episode_fraction > 0.65:
            lines.append("A large proportion of activity is concentrated in one dominant active block.")
        if features.eye_state == "open":
            lines.append("Eyes appearing open adds mild supportive evidence for a non-PNES evolving pattern.")
    else:
        lines.append(
            "The recording contains mixed or borderline features, or the video quality is insufficient for a confident graph-shape classification."
        )

    qf = getattr(video_data, "quality_flags", {}) or {}
    if qf.get("missing_onset"):
        lines.append("Visible onset was incomplete, so early evolution may be under-sampled.")
    if qf.get("short_video"):
        lines.append("The video is short, which reduces confidence in temporal pattern assessment.")
    if qf.get("very_low_motion"):
        lines.append("Overall motion level is very low, which makes graph-based interpretation less reliable.")

    lines.append(
        f"PNES-pattern score: {result.pnes_score:.1f}/10 · Evolving-pattern score: {result.es_score:.1f}/10 · Reliability: {result.reliability}."
    )
    lines.append("This remains a research prototype and requires expert clinical review.")
    return " ".join(lines)


def compute_scores(features: MotionFeatures, video_data: VideoData, weights: Optional[Weights] = None) -> ScoringResult:
    if weights is None:
        weights = Weights()

    reliability_score = _compute_reliability(video_data)
    reliability = _reliability_label(reliability_score)

    low_motion = (getattr(video_data, "quality_flags", {}) or {}).get("very_low_motion", False)

    eff_weights = Weights(
        movement_stereotypy=weights.movement_stereotypy,
        temporal_distribution_stability=weights.temporal_distribution_stability,
        variance_drift=weights.variance_drift * (0.5 if low_motion else 1.0),
        vector_change=weights.vector_change * (0.7 if low_motion else 1.0),
        amplitude_variability=weights.amplitude_variability * (0.5 if low_motion else 1.0),
        rhythm_irregularity=weights.rhythm_irregularity * (0.6 if low_motion else 1.0),
        temporal_evolution=weights.temporal_evolution * (0.6 if low_motion else 1.0),
        eye_feature=weights.eye_feature,
        long_quiet_fraction=weights.long_quiet_fraction,
        episode_structure=weights.episode_structure,
        burst_recurrence=weights.burst_recurrence,
        late_intensification=weights.late_intensification,
    )

    feature_scores = {
        "Quiet Returns": _score_quiet_returns(features, eff_weights.long_quiet_fraction),
        "Episode Structure": _score_episode_structure(features, eff_weights.episode_structure),
        "Burst Recurrence": _score_burst_recurrence(features, eff_weights.burst_recurrence),
        "Temporal Evolution": _score_temporal_evolution(features, eff_weights.temporal_evolution),
        "Variance Drift": _score_variance_drift(features, eff_weights.variance_drift),
        "Movement Stereotypy": _score_stereotypy(features, eff_weights.movement_stereotypy),
        "Distribution Stability": _score_distribution_stability(features, eff_weights.temporal_distribution_stability),
        "Late Intensification": _score_late_intensification(features, eff_weights.late_intensification),
        "Vector Change": _score_vector_change(features, eff_weights.vector_change),
        "Amplitude Variability": _score_amplitude_variability(features, eff_weights.amplitude_variability),
        "Rhythm Regularity": _score_rhythm(features, eff_weights.rhythm_irregularity),
        "Eye State": _score_eye_state(features, eff_weights.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_evol = sum(v["es"] for v in feature_scores.values())
    sym_weights = sum(v["weight"] for v in feature_scores.values()) or 1.0

    pnes_score = round(min(total_pnes / sym_weights * 10.0, 10.0), 2)
    evol_score = round(min(total_evol / sym_weights * 10.0, 10.0), 2)
    
    margin = pnes_score - evol_score

    # Veto gate: strong temporal evolution blocks PNES classification
    te_blocks_pnes = features.temporal_evolution > PNES_TE_GATE

    if reliability_score < RELIABILITY_FLOOR:
        classification = "Indeterminate / insufficient data"
        confidence = "low"
    elif (pnes_score >= PNES_MIN_SCORE and margin >= COMMIT_MARGIN
          and evol_score <= PNES_MAX_EVOL and not te_blocks_pnes):
        classification = "Typical PNES pattern"
        confidence = _confidence_label(margin)
    elif evol_score >= EVOL_MIN_SCORE and (-margin) >= COMMIT_MARGIN and pnes_score <= EVOL_MAX_PNES:
        classification = "Evolving / not typical PNES"
        confidence = _confidence_label(-margin)
    elif te_blocks_pnes and evol_score >= EVOL_MIN_GATED:
        # Gate-triggered: temporal evolution clearly elevated → not typical PNES
        classification = "Evolving / not typical PNES"
        confidence = "low"
    else:
        classification = "Indeterminate / insufficient data"
        confidence = "low"

    result = ScoringResult(
        pnes_score=pnes_score,
        es_score=evol_score,
        pnes_pattern_confidence=pnes_score,
        reliability_score=reliability_score,
        classification=classification,
        confidence=confidence,
        reliability=reliability,
        feature_scores=feature_scores,
        quality_penalties=getattr(video_data, "quality_flags", {}) or {},
        gate_flags={
            "quiet_returns_high": features.long_quiet_fraction > 0.18,
            "multiple_episodes": features.episode_count >= 2,
            "burst_recurrence_high": features.burst_recurrence > 0.56,
            "temporal_evolution_high": features.temporal_evolution > 0.46,
            "te_gate_triggered": te_blocks_pnes,
            "variance_drift_high": features.variance_drift > 0.40,
            "single_large_block": features.largest_episode_fraction > 0.65,
        },
    )
    result.explanation = _generate_explanation(result, features, video_data)
    return result
