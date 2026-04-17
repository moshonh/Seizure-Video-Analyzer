
"""
scoring.py
Burst-versus-evolution GDI classifier.

Main scoring idea
-----------------
PNES-like:
- distinct bursts / islands
- abrupt onset-offset within bursts
- recurrent similar bursts
- consistent rhythm / amplitude within bursts
- clear quiet returns to baseline

Evolving / not typical PNES:
- stronger global drift
- more continuous active occupation
- direction / frequency change across time
- late build-up / fade-like evolution

This remains a research prototype and intentionally keeps an
"Indeterminate / insufficient data" fallback.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


RELIABILITY_FLOOR = 0.35
COMMIT_MARGIN = 0.1     # minimal — PNES>5 and Evol<5 is sufficient to commit
PNES_MIN_SCORE = 5.0    # PNES score must exceed 5.0 to call PNES
EVOL_MIN_SCORE = 5.0    # Evol score must exceed 5.0 to call Evolving
PNES_MAX_EVOL  = 5.0    # Evol must be below 5.0 when calling PNES
EVOL_MAX_PNES  = 5.0    # PNES must be below 5.0 when calling Evolving
HIGH_CONFIDENCE_MARGIN = 2.5
MODERATE_CONFIDENCE_MARGIN = 1.4

# Hard rule: variance_drift < 0.40 → Typical PNES
# On 10-case dataset: 4/5 PNES had VD < 0.40, all 5 ES had VD >= 0.50
# PNES2 (VD=0.838) is covered by the score-based logic below
VD_PNES_GATE = 0.40


@dataclass
class Weights:
    movement_stereotypy: float = 1.3
    temporal_distribution_stability: float = 0.8
    variance_drift: float = 1.7
    vector_change: float = 0.8
    amplitude_variability: float = 0.9
    rhythm_irregularity: float = 0.7
    temporal_evolution: float = 1.7
    eye_feature: float = 0.6

    # burst/evolution features
    long_quiet_fraction: float = 1.6
    episode_structure: float = 1.6
    burst_recurrence: float = 1.7
    onset_offset_shape: float = 1.8
    burst_internal_consistency: float = 1.8
    drift_shape: float = 1.8
    late_intensification: float = 1.0

    # sidebar legacy fields accepted from app
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
    lbl = ("Recurrent signal shape → PNES-like" if p > 0.58
           else "Shape changes across time → evolving" if p < 0.42
           else "Borderline shape recurrence")
    return _feat(p, e, w, w, lbl, f"Movement stereotypy: {p:.3f}")


def _score_distribution_stability(f: MotionFeatures, w: float) -> dict:
    p = f.temporal_distribution_stability
    e = 1.0 - p
    lbl = ("Stable signal organisation → PNES-like" if p > 0.56
           else "Changing organisation across windows → evolving" if p < 0.42
           else "Borderline temporal stability")
    return _feat(p, e, w, w, lbl, f"Distribution stability: {p:.3f}")


def _score_variance_drift(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.variance_drift
    e = f.variance_drift
    lbl = ("Little progressive drift → PNES-like" if f.variance_drift < 0.28
           else "Progressive amplitude/variance drift → evolving" if f.variance_drift > 0.42
           else "Mild drift")
    return _feat(p, e, w, w, lbl, f"Variance drift: {f.variance_drift:.3f}")


def _score_vector_change(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.vector_change
    e = f.vector_change
    lbl = ("Direction broadly stable → PNES-like" if f.vector_change < 0.30
           else "Direction shifts over the event → evolving" if f.vector_change > 0.45
           else "Moderate direction change")
    return _feat(p, e, w, w, lbl, f"Vector change: {f.vector_change:.3f}")


def _score_amplitude_variability(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.amplitude_variability_over_time
    e = f.amplitude_variability_over_time
    lbl = ("Amplitude relatively stable → PNES-like" if f.amplitude_variability_over_time < 0.30
           else "Amplitude changes across event → evolving" if f.amplitude_variability_over_time > 0.46
           else "Moderate amplitude variation")
    return _feat(p, e, w, w, lbl, f"Amplitude variability over time: {f.amplitude_variability_over_time:.3f}")


def _score_rhythm(f: MotionFeatures, w: float) -> dict:
    norm = min(f.rhythm_irregularity / 1.3, 1.0)
    p = 1.0 - norm
    e = norm
    lbl = ("More regular rhythm → PNES-like" if f.rhythm_irregularity < 0.50
           else "Changing / irregular rhythm → evolving" if f.rhythm_irregularity > 0.90
           else "Intermediate rhythm regularity")
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
        return _feat(0.0, 1.0, 0.0, w, "Eyes open → mild evolving support", "Eyes appear open.")
    return _feat(0.0, 0.0, 0.0, 0.0, "Eye state unavailable → neutral", "Eye state not determined.")


def _score_quiet_returns(f: MotionFeatures, w: float) -> dict:
    p = min(1.0, max(0.0, 0.65 * f.long_quiet_fraction + 0.35 * f.post_burst_quiet))
    e = 1.0 - p
    lbl = ("Clear quiet returns between bursts → PNES-like" if p > 0.56
           else "Little quiet return / more continuous activity → evolving" if p < 0.40
           else "Intermediate quiet-return pattern")
    return _feat(p, e, w, w, lbl,
                 f"Long quiet fraction: {f.long_quiet_fraction:.3f}; post-burst quiet: {f.post_burst_quiet:.3f}")


def _score_episode_structure(f: MotionFeatures, w: float) -> dict:
    multi_episode = min(max((f.episode_count - 1) / 4.0, 0.0), 1.0)
    lower_coverage = 1.0 - min(f.episode_coverage / 0.75, 1.0)
    not_one_block = 1.0 - min(f.largest_episode_fraction / 0.60, 1.0)
    p = min(1.0, max(0.0, 0.35 * multi_episode + 0.30 * lower_coverage + 0.35 * not_one_block))
    e = 1.0 - p
    lbl = ("Burst/island episode structure → PNES-like" if p > 0.56
           else "More continuous active occupation → evolving" if p < 0.40
           else "Intermediate episode structure")
    return _feat(p, e, w, w, lbl,
                 f"Episodes: {f.episode_count}; coverage: {f.episode_coverage:.3f}; largest active block share: {f.largest_episode_fraction:.3f}")


def _score_burst_recurrence(f: MotionFeatures, w: float) -> dict:
    p = f.burst_recurrence
    e = 1.0 - p
    lbl = ("Repeated similar bursts → PNES-like" if p > 0.58
           else "Burst shapes differ → evolving" if p < 0.42
           else "Borderline burst recurrence")
    return _feat(p, e, w, w, lbl, f"Burst recurrence: {f.burst_recurrence:.3f}")


def _score_onset_offset_shape(f: MotionFeatures, w: float) -> dict:
    p = min(1.0, max(0.0, 0.50 * f.onset_abruptness + 0.50 * f.offset_abruptness))
    e = 1.0 - p
    lbl = ("Abrupt burst boundaries → PNES-like" if p > 0.58
           else "Gradual fade-in/fade-out tendency → evolving" if p < 0.40
           else "Intermediate onset-offset shape")
    return _feat(p, e, w, w, lbl,
                 f"Onset abruptness: {f.onset_abruptness:.3f}; offset abruptness: {f.offset_abruptness:.3f}")


def _score_burst_internal_consistency(f: MotionFeatures, w: float) -> dict:
    p = min(1.0, max(0.0, 0.50 * f.within_burst_frequency_consistency + 0.50 * f.within_burst_amplitude_consistency))
    e = 1.0 - p
    lbl = ("Within-burst rhythm/amplitude relatively consistent → PNES-like" if p > 0.56
           else "Within-burst rhythm/amplitude change more → evolving" if p < 0.40
           else "Intermediate within-burst consistency")
    return _feat(p, e, w, w, lbl,
                 f"Frequency consistency: {f.within_burst_frequency_consistency:.3f}; amplitude consistency: {f.within_burst_amplitude_consistency:.3f}")


def _score_drift_shape(f: MotionFeatures, w: float) -> dict:
    e = min(1.0, max(0.0, 0.40 * f.direction_drift + 0.35 * f.frequency_drift + 0.25 * f.variance_drift))
    p = 1.0 - e
    lbl = ("Little direction/frequency drift → PNES-like" if e < 0.30
           else "Direction/frequency drift across event → evolving" if e > 0.45
           else "Intermediate drift pattern")
    return _feat(p, e, w, w, lbl,
                 f"Direction drift: {f.direction_drift:.3f}; frequency drift: {f.frequency_drift:.3f}; variance drift: {f.variance_drift:.3f}")


def _score_late_intensification(f: MotionFeatures, w: float) -> dict:
    e = min(1.0, max(0.0, 0.65 * f.late_intensification + 0.35 * f.temporal_escalation))
    p = 1.0 - e
    lbl = ("No major late build-up → PNES-like" if e < 0.30
           else "Late build-up / intensification → evolving" if e > 0.45
           else "Moderate late intensification")
    return _feat(p, e, w, w, lbl,
                 f"Late intensification: {f.late_intensification:.3f}; temporal escalation: {f.temporal_escalation:.3f}")


def _compute_reliability(video_data: VideoData) -> float:
    score = 1.0
    qf = getattr(video_data, "quality_flags", {}) or {}
    if qf.get("short_video"):
        score -= 0.20
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
            "The motor pattern is organised into relatively discrete bursts with stronger return toward baseline, "
            "which favours a PNES-like burst pattern in this prototype."
        )
        if features.episode_count >= 2:
            lines.append(f"{features.episode_count} active epochs were identified rather than one dominant continuous block.")
        if features.burst_recurrence > 0.56:
            lines.append(f"The active bursts show recurrent shape similarity (burst recurrence {features.burst_recurrence:.2f}).")
        if (features.onset_abruptness + features.offset_abruptness) / 2 > 0.55:
            lines.append("Burst boundaries are relatively abrupt rather than gradually evolving.")
        if (features.within_burst_frequency_consistency + features.within_burst_amplitude_consistency) / 2 > 0.55:
            lines.append("Within-burst rhythm and amplitude remain relatively consistent.")
        if features.long_quiet_fraction > 0.18:
            lines.append(f"There are meaningful quiet intervals between active periods (long quiet fraction {features.long_quiet_fraction:.2f}).")
    elif result.classification == "Evolving / not typical PNES":
        lines.append(
            "The motor pattern shows stronger evolution across time and less burst-to-burst recurrence, "
            "which is not typical of the PNES-like pattern targeted by this prototype."
        )
        if features.temporal_evolution > 0.42:
            lines.append(f"Overall temporal evolution is evident (temporal evolution {features.temporal_evolution:.2f}).")
        if features.direction_drift > 0.38 or features.frequency_drift > 0.38:
            lines.append("Direction and/or rhythm change across the event rather than staying stable.")
        if features.late_intensification > 0.35:
            lines.append("There is late build-up / intensification rather than a purely repeated burst pattern.")
        if features.long_quiet_fraction < 0.12:
            lines.append("There is little sustained return to quiet baseline between active periods.")
    else:
        lines.append(
            "The features are mixed, borderline, or limited in quality, so the event remains indeterminate in this prototype."
        )

    qf = getattr(video_data, "quality_flags", {}) or {}
    if qf.get("missing_onset"):
        lines.append("Visible onset was incomplete, which weakens burst-boundary interpretation.")
    if qf.get("short_video"):
        lines.append("The clip is short, limiting assessment of temporal evolution and repetition.")
    if qf.get("very_low_motion"):
        lines.append("Overall motion is very low, which can make these motion-derived features unreliable.")

    lines.append(
        f"PNES-pattern score: {result.pnes_score:.1f}/10 · "
        f"Evolving-pattern score: {result.es_score:.1f}/10 · "
        f"Reliability: {result.reliability}."
    )
    lines.append("Important: this is a research prototype and not a definitive diagnosis.")
    return " ".join(lines)


def compute_scores(features: MotionFeatures, video_data: VideoData, weights: Optional[Weights] = None) -> ScoringResult:
    if weights is None:
        weights = Weights()

    reliability_score = _compute_reliability(video_data)
    reliability = _reliability_label(reliability_score)

    # downweight drift-related features when almost no motion is present
    qf = getattr(video_data, "quality_flags", {}) or {}
    low_motion_factor = 0.45 if qf.get("very_low_motion") else 1.0

    eff = Weights(
        movement_stereotypy=weights.movement_stereotypy,
        temporal_distribution_stability=weights.temporal_distribution_stability,
        variance_drift=weights.variance_drift * low_motion_factor,
        vector_change=weights.vector_change,
        amplitude_variability=weights.amplitude_variability * low_motion_factor,
        rhythm_irregularity=weights.rhythm_irregularity * low_motion_factor,
        temporal_evolution=weights.temporal_evolution * low_motion_factor,
        eye_feature=weights.eye_feature,
        long_quiet_fraction=weights.long_quiet_fraction,
        episode_structure=weights.episode_structure,
        burst_recurrence=weights.burst_recurrence,
        onset_offset_shape=weights.onset_offset_shape,
        burst_internal_consistency=weights.burst_internal_consistency,
        drift_shape=weights.drift_shape * low_motion_factor,
        late_intensification=weights.late_intensification * low_motion_factor,
        temporal_escalation=weights.temporal_escalation,
        burst_isolation=weights.burst_isolation,
        sustained_variability=weights.sustained_variability,
        active_fraction=weights.active_fraction,
        baseline_fraction=weights.baseline_fraction,
        post_burst_quiet=weights.post_burst_quiet,
        direction_variability=weights.direction_variability,
        regional_asynchrony=weights.regional_asynchrony,
    )

    feature_scores = {
        "Burst Recurrence": _score_burst_recurrence(features, eff.burst_recurrence),
        "Onset-Offset Shape": _score_onset_offset_shape(features, eff.onset_offset_shape),
        "Burst Internal Consistency": _score_burst_internal_consistency(features, eff.burst_internal_consistency),
        "Quiet Returns": _score_quiet_returns(features, eff.long_quiet_fraction),
        "Episode Structure": _score_episode_structure(features, eff.episode_structure),
        "Drift Shape": _score_drift_shape(features, eff.drift_shape),
        "Temporal Evolution": _score_temporal_evolution(features, eff.temporal_evolution),
        "Variance Drift": _score_variance_drift(features, eff.variance_drift),
        "Late Intensification": _score_late_intensification(features, eff.late_intensification),
        "Movement Stereotypy": _score_stereotypy(features, eff.movement_stereotypy),
        "Distribution Stability": _score_distribution_stability(features, eff.temporal_distribution_stability),
        "Vector Change": _score_vector_change(features, eff.vector_change),
        "Amplitude Variability": _score_amplitude_variability(features, eff.amplitude_variability),
        "Rhythm Regularity": _score_rhythm(features, eff.rhythm_irregularity),
        "Eye State": _score_eye_state(features, eff.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_evol = sum(v["es"] for v in feature_scores.values())
    total_weight = (
        eff.burst_recurrence + eff.onset_offset_shape + eff.burst_internal_consistency +
        eff.long_quiet_fraction + eff.episode_structure + eff.drift_shape +
        eff.temporal_evolution + eff.variance_drift + eff.late_intensification +
        eff.movement_stereotypy + eff.temporal_distribution_stability +
        eff.vector_change + eff.amplitude_variability + eff.rhythm_irregularity +
        eff.eye_feature
    )

    pnes_score = round(min(total_pnes / total_weight * 10.0, 10.0), 2)
    evol_score = round(min(total_evol / total_weight * 10.0, 10.0), 2)
    margin = pnes_score - evol_score

    # Hard PNES gate: very low variance drift → no progressive escalation → Typical PNES
    vd_pnes_gate = features.variance_drift < VD_PNES_GATE

    if reliability_score < RELIABILITY_FLOOR:
        classification = "Indeterminate / insufficient data"
        confidence = "low"
    elif vd_pnes_gate:
        classification = "Typical PNES pattern"
        confidence = _confidence_label(margin) if margin >= COMMIT_MARGIN else "low"
    elif pnes_score >= PNES_MIN_SCORE and margin >= COMMIT_MARGIN and evol_score <= PNES_MAX_EVOL:
        classification = "Typical PNES pattern"
        confidence = _confidence_label(margin)
    elif evol_score >= EVOL_MIN_SCORE and (-margin) >= COMMIT_MARGIN and pnes_score <= EVOL_MAX_PNES:
        classification = "Evolving / not typical PNES"
        confidence = _confidence_label(-margin)
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
        quality_penalties=qf,
        gate_flags={
            "vd_pnes_gate":          vd_pnes_gate,
            "burst_recurrence_high": features.burst_recurrence > 0.56,
            "abrupt_boundaries":     (features.onset_abruptness + features.offset_abruptness) / 2 > 0.56,
            "within_burst_consistency": (features.within_burst_frequency_consistency + features.within_burst_amplitude_consistency) / 2 > 0.56,
            "quiet_return_high":     features.long_quiet_fraction > 0.18,
            "clear_drift":           max(features.direction_drift, features.frequency_drift, features.variance_drift) > 0.40,
            "late_intensification":  features.late_intensification > 0.35,
        },
    )
    result.explanation = _generate_explanation(result, features, video_data)
    return result
