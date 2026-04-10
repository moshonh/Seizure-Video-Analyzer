"""
scoring.py
GDI-based scoring engine.

Core clinical model (revised):
─────────────────────────────────────────────────────────────────────────────
The dominant question is:
  "Does the movement pattern remain similar over time, or does it evolve?"

PNES:
  - High movement_stereotypy (repetitive cycles)
  - Low temporal_evolution
  - Regular rhythm
  - Consistent direction and amplitude

ES:
  - Low movement_stereotypy (changing cycles)
  - High temporal_evolution
  - Irregular or changing rhythm
  - Shifting direction and amplitude

Motion quantity (burst size, active fraction) is SECONDARY.
─────────────────────────────────────────────────────────────────────────────

Weights hierarchy:
  PRIMARY   (weight 1.5–2.5): temporal_evolution, movement_stereotypy,
                               rhythm_irregularity, direction_variability,
                               amplitude_variability, vector_change
  SECONDARY (weight 0.3–0.6): burst_isolation, baseline_fraction,
                               post_burst_quiet, active_fraction,
                               temporal_escalation
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


# ─────────────────────────────────────────────────────────────────────────────
# Weights — adjust defaults here without touching logic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weights:
    # PRIMARY — drive the classification
    temporal_escalation:   float = 0.4   # kept for backward compat with app.py slider
    burst_isolation:       float = 0.4   # secondary; kept for slider
    sustained_variability: float = 0.5   # secondary; kept for slider
    active_fraction:       float = 0.4   # secondary; kept for slider
    baseline_fraction:     float = 0.4   # secondary; kept for slider
    post_burst_quiet:      float = 0.4   # secondary; kept for slider
    direction_variability: float = 1.8   # PRIMARY
    rhythm_irregularity:   float = 1.6   # PRIMARY
    temporal_evolution:    float = 2.5   # PRIMARY — highest weight
    regional_asynchrony:   float = 0.5   # secondary
    eye_feature:           float = 0.8   # manual annotation

    # New primaries — not exposed as sliders yet, but used in scoring
    movement_stereotypy:         float = 2.2   # PRIMARY (inverse ES)
    amplitude_variability:       float = 1.6   # PRIMARY
    vector_change:               float = 1.4   # PRIMARY


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    pnes_score:        float
    es_score:          float
    reliability_score: float
    classification:    str    # "ES" | "PNES" | "Indeterminate"
    confidence:        str    # "low" | "moderate" | "high"
    reliability:       str    # "poor" | "limited" | "adequate"
    feature_scores:    Dict[str, dict] = field(default_factory=dict)
    quality_penalties: Dict[str, bool] = field(default_factory=dict)
    explanation:       str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature scoring functions
# Each returns: {pnes, es, label, detail, weight}
# ─────────────────────────────────────────────────────────────────────────────

def _score_temporal_evolution(f: MotionFeatures, weight: float) -> dict:
    """
    HIGH temporal_evolution → pattern evolves over time → ES.
    LOW  temporal_evolution → pattern stays consistent → PNES.
    """
    es_raw   = f.temporal_evolution
    pnes_raw = 1.0 - es_raw
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Evolving pattern → ES" if es_raw > pnes_raw else "Consistent pattern → PNES",
        "detail": f"Temporal evolution (variability + amplitude + direction): {f.temporal_evolution:.3f}",
        "weight": weight,
    }


def _score_movement_stereotypy(f: MotionFeatures, weight: float) -> dict:
    """
    HIGH stereotypy → repetitive cycles → PNES.
    LOW  stereotypy → changing cycles   → ES.
    """
    pnes_raw = f.movement_stereotypy
    es_raw   = 1.0 - f.movement_stereotypy
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Stereotyped cycles → PNES" if pnes_raw > es_raw else "Changing cycles → ES",
        "detail": f"Movement stereotypy (inter-segment correlation): {f.movement_stereotypy:.3f}",
        "weight": weight,
    }


def _score_rhythm_irregularity(f: MotionFeatures, weight: float) -> dict:
    """
    HIGH rhythm_irregularity → irregular inter-peak intervals → ES.
    LOW  rhythm_irregularity → regular rhythm → PNES.
    rhythm_irregularity is CV (0–3); normalise at 1.5.
    """
    norm     = float(min(f.rhythm_irregularity / 1.5, 1.0))
    es_raw   = norm
    pnes_raw = 1.0 - norm
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Irregular rhythm → ES" if es_raw > pnes_raw else "Regular rhythm → PNES",
        "detail": f"Rhythm irregularity (CV of inter-peak intervals): {f.rhythm_irregularity:.3f}",
        "weight": weight,
    }


def _score_direction_variability(f: MotionFeatures, weight: float) -> dict:
    """
    HIGH angle_variability → changing direction → ES.
    LOW  angle_variability → fixed direction   → PNES.
    Circular std; normalise at 1.2 rad.
    """
    norm     = float(min(f.angle_variability / 1.2, 1.0))
    es_raw   = norm
    pnes_raw = 1.0 - norm
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Changing direction → ES" if es_raw > pnes_raw else "Fixed direction → PNES",
        "detail": f"Direction variability (circular std): {f.angle_variability:.3f} rad",
        "weight": weight,
    }


def _score_amplitude_variability(f: MotionFeatures, weight: float) -> dict:
    """
    HIGH amplitude_variability_over_time → amplitude changes → ES.
    LOW  → consistent amplitude → PNES.
    """
    es_raw   = f.amplitude_variability_over_time
    pnes_raw = 1.0 - es_raw
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Changing amplitude → ES" if es_raw > pnes_raw else "Consistent amplitude → PNES",
        "detail": f"Amplitude variability over time: {f.amplitude_variability_over_time:.3f}",
        "weight": weight,
    }


def _score_vector_change(f: MotionFeatures, weight: float) -> dict:
    """
    HIGH vector_change → direction shifted from early to late → ES.
    LOW  → direction stable → PNES.
    """
    es_raw   = f.vector_change
    pnes_raw = 1.0 - es_raw
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Direction shift early→late → ES" if es_raw > pnes_raw else "Stable direction → PNES",
        "detail": f"Vector change (early vs late dominant direction): {f.vector_change:.3f}",
        "weight": weight,
    }


# ── Secondary features ────────────────────────────────────────────────────────

def _score_burst_isolation(f: MotionFeatures, weight: float) -> dict:
    """Secondary: focal burst → mild PNES support; but should NOT dominate."""
    pnes_raw = f.burst_isolation
    es_raw   = 1.0 - f.burst_isolation
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Focal activity distribution → mild PNES" if pnes_raw > es_raw else "Broad activity distribution → mild ES",
        "detail": f"Burst isolation: {f.burst_isolation:.3f}",
        "weight": weight,
    }


def _score_baseline_fraction(f: MotionFeatures, weight: float) -> dict:
    """Secondary: long quiet periods → mild PNES support."""
    pnes_raw = f.baseline_fraction
    es_raw   = 1.0 - f.baseline_fraction
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Extended quiet periods → mild PNES" if pnes_raw > es_raw else "Little quiet time → mild ES",
        "detail": f"Baseline fraction: {f.baseline_fraction:.2f}",
        "weight": weight,
    }


def _score_post_burst_quiet(f: MotionFeatures, weight: float) -> dict:
    """Secondary: return to baseline → mild PNES support."""
    pnes_raw = f.post_burst_quiet
    es_raw   = 1.0 - f.post_burst_quiet
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Returns to baseline → mild PNES" if pnes_raw > es_raw else "Sustained end-activity → mild ES",
        "detail": f"Post-activity quiet: {f.post_burst_quiet:.3f}",
        "weight": weight,
    }


def _score_active_fraction(f: MotionFeatures, weight: float) -> dict:
    """Secondary: prolonged activity → mild ES support."""
    es_raw   = f.active_fraction
    pnes_raw = 1.0 - f.active_fraction
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Prolonged active phase → mild ES" if es_raw > pnes_raw else "Brief active phase → mild PNES",
        "detail": f"Active fraction: {f.active_fraction:.2f}",
        "weight": weight,
    }


def _score_temporal_escalation(f: MotionFeatures, weight: float) -> dict:
    """Secondary: motion increases start→end → mild ES support."""
    es_raw   = f.temporal_escalation
    pnes_raw = 1.0 - es_raw
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Motion escalation → mild ES" if es_raw > pnes_raw else "No escalation → mild PNES",
        "detail": f"Temporal escalation: {f.temporal_escalation:.3f}",
        "weight": weight,
    }


def _score_regional_asynchrony(f: MotionFeatures, weight: float) -> dict:
    """Secondary: different body regions move differently → mild ES support."""
    norm     = float(min(f.regional_asynchrony / 0.3, 1.0))
    es_raw   = norm
    pnes_raw = 1.0 - norm
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Regional asynchrony → mild ES" if es_raw > pnes_raw else "Regional synchrony → mild PNES",
        "detail": f"Regional asynchrony: {f.regional_asynchrony:.4f}",
        "weight": weight,
    }


def _score_eye_state(f: MotionFeatures, weight: float) -> dict:
    if f.eye_state == "closed":
        return {"pnes": round(weight, 3), "es": 0.0,
                "label": "Eyes closed → supports PNES",
                "detail": "Eye closure observed.", "weight": weight}
    if f.eye_state == "open":
        return {"pnes": 0.0, "es": round(weight, 3),
                "label": "Eyes open → supports ES",
                "detail": "Eyes appear open.", "weight": weight}
    return {"pnes": 0.0, "es": 0.0,
            "label": "Eye state unavailable → neutral",
            "detail": "Eye state not determined.", "weight": weight}


# ─────────────────────────────────────────────────────────────────────────────
# Reliability
# ─────────────────────────────────────────────────────────────────────────────

def _compute_reliability(video_data: VideoData) -> float:
    score = 1.0
    qf    = video_data.quality_flags
    if qf.get("short_video"):     score -= 0.25
    if qf.get("missing_onset"):   score -= 0.20
    if qf.get("missing_offset"):  score -= 0.10
    if qf.get("low_resolution"):  score -= 0.15
    if qf.get("very_low_motion"): score -= 0.20
    return max(0.0, round(score, 2))


def _reliability_label(r: float) -> str:
    if r >= 0.70:
        return "adequate"
    if r >= 0.40:
        return "limited"
    return "poor"


def _confidence_label(pnes: float, es: float, reliability: float) -> str:
    total           = pnes + es + 1e-6
    relative_margin = abs(pnes - es) / total
    if relative_margin >= 0.30 and reliability >= 0.60:
        return "high"
    if relative_margin >= 0.15 and reliability >= 0.35:
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Explanation generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_explanation(result: "ScoringResult",
                           features: MotionFeatures,
                           video_data: VideoData) -> str:
    cls   = result.classification
    lines = []

    if cls == "PNES":
        lines.append(
            "The movement pattern in this recording appears relatively stereotyped and internally "
            "consistent over time, which is more consistent with a psychogenic nonepileptic seizure "
            "(PNES) according to the GDI framework."
        )
        if features.movement_stereotypy > 0.60:
            lines.append(
                f"Consecutive movement cycles show high similarity to one another "
                f"(stereotypy index: {features.movement_stereotypy:.2f}), "
                f"suggesting a repetitive pattern with limited evolution."
            )
        if features.temporal_evolution < 0.35:
            lines.append(
                "The rhythm, direction, and amplitude of movement remain largely consistent "
                "across the early, middle, and late portions of the recording."
            )
        if features.rhythm_irregularity < 0.5:
            lines.append(
                f"The rhythm of movement is relatively regular and consistent over time "
                f"(irregularity index: {features.rhythm_irregularity:.2f}), "
                f"consistent with a stereotyped, repetitive motor pattern."
            )
        if features.vector_change < 0.25:
            lines.append(
                "The dominant movement direction shows little change from the beginning "
                "to the end of the recording."
            )

    elif cls == "ES":
        lines.append(
            "The movement pattern in this recording shows dynamic temporal evolution, "
            "which is more consistent with an epileptic seizure (ES) according to the GDI framework."
        )
        if features.temporal_evolution > 0.50:
            lines.append(
                f"The pattern of motion changes across the recording — rhythm, direction, and amplitude "
                f"all shift over time, reflecting a dynamic temporal progression "
                f"(temporal evolution score: {features.temporal_evolution:.2f})."
            )
        if features.movement_stereotypy < 0.45:
            lines.append(
                f"Movement cycles are dissimilar to one another "
                f"(stereotypy index: {features.movement_stereotypy:.2f}), "
                f"indicating an evolving rather than repetitive pattern."
            )
        if features.rhythm_irregularity > 0.8:
            lines.append(
                f"The rhythm of movement is irregular "
                f"(irregularity index: {features.rhythm_irregularity:.2f}), "
                f"suggesting dynamic changes in motor activity over time."
            )
        if features.vector_change > 0.35:
            lines.append(
                f"The dominant direction of movement shifts from the early to the late portion "
                f"of the recording (vector change: {features.vector_change:.2f}), "
                f"consistent with the spreading or evolving character of epileptic motor activity."
            )
        if features.amplitude_variability_over_time > 0.35:
            lines.append(
                "Motion amplitude changes across the event, "
                "reflecting the dynamic progression typical of an epileptic seizure."
            )
    else:
        lines.append(
            "The movement features in this recording do not clearly favour either ES or PNES. "
            "Temporal evolution, stereotypy, and rhythm measures do not provide a consistent pattern, "
            "and classification is indeterminate."
        )

    # Eye state
    if features.eye_state == "closed":
        lines.append("Eye closure was noted, adding mild additional support for PNES.")
    elif features.eye_state == "open":
        lines.append("Eyes appeared open, adding mild additional support for ES.")

    # Quality notes
    qf = video_data.quality_flags
    if qf.get("missing_onset"):
        lines.append(
            "The event onset was not fully captured; temporal evolution features "
            "may underestimate the actual degree of change."
        )
    if qf.get("short_video"):
        lines.append(
            "The recording is brief. Stereotypy and evolution features require "
            "sufficient duration to be reliable."
        )
    if qf.get("very_low_motion"):
        lines.append(
            "Overall motion is very low. This may be a non-motor event, or "
            "the patient may be partially outside the selected ROI."
        )

    lines.append(
        f"PNES support: {result.pnes_score:.1f}/10 · "
        f"ES support: {result.es_score:.1f}/10 · "
        f"Reliability: {result.reliability}."
    )
    lines.append(
        "IMPORTANT: This is a research prototype. All results require expert clinical "
        "review and, when appropriate, video-EEG and full clinical context."
    )
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(
    features: MotionFeatures,
    video_data: VideoData,
    weights: Optional[Weights] = None,
) -> ScoringResult:
    """Apply GDI scoring and return a ScoringResult."""
    if weights is None:
        weights = Weights()

    feature_scores = {
        # ── PRIMARY ────────────────────────────────────────────────────────
        "Temporal Evolution":    _score_temporal_evolution(features,   weights.temporal_evolution),
        "Movement Stereotypy":   _score_movement_stereotypy(features,  weights.movement_stereotypy),
        "Rhythm Irregularity":   _score_rhythm_irregularity(features,  weights.rhythm_irregularity),
        "Direction Variability": _score_direction_variability(features, weights.direction_variability),
        "Amplitude Variability": _score_amplitude_variability(features, weights.amplitude_variability),
        "Vector Change":         _score_vector_change(features,        weights.vector_change),
        # ── SECONDARY ──────────────────────────────────────────────────────
        "Burst Isolation":       _score_burst_isolation(features,      weights.burst_isolation),
        "Baseline Fraction":     _score_baseline_fraction(features,    weights.baseline_fraction),
        "Post-activity Quiet":   _score_post_burst_quiet(features,     weights.post_burst_quiet),
        "Active Fraction":       _score_active_fraction(features,      weights.active_fraction),
        "Motion Escalation":     _score_temporal_escalation(features,  weights.temporal_escalation),
        "Regional Asynchrony":   _score_regional_asynchrony(features,  weights.regional_asynchrony),
        # ── Manual annotation ──────────────────────────────────────────────
        "Eye State":             _score_eye_state(features,            weights.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_es   = sum(v["es"]   for v in feature_scores.values())

    max_possible = (
        weights.temporal_evolution
        + weights.movement_stereotypy
        + weights.rhythm_irregularity
        + weights.direction_variability
        + weights.amplitude_variability
        + weights.vector_change
        + weights.burst_isolation
        + weights.baseline_fraction
        + weights.post_burst_quiet
        + weights.active_fraction
        + weights.temporal_escalation
        + weights.regional_asynchrony
        + weights.eye_feature
    )

    pnes_score = round(min(total_pnes / max_possible * 10.0, 10.0), 2)
    es_score   = round(min(total_es   / max_possible * 10.0, 10.0), 2)

    reliability_score = _compute_reliability(video_data)
    reliability       = _reliability_label(reliability_score)

    total           = pnes_score + es_score + 1e-6
    relative_margin = abs(pnes_score - es_score) / total

    if reliability_score < 0.30 or relative_margin < 0.10:
        classification = "Indeterminate"
    elif pnes_score > es_score:
        classification = "PNES"
    else:
        classification = "ES"

    confidence = _confidence_label(pnes_score, es_score, reliability_score)

    result = ScoringResult(
        pnes_score=pnes_score,
        es_score=es_score,
        reliability_score=reliability_score,
        classification=classification,
        confidence=confidence,
        reliability=reliability,
        feature_scores=feature_scores,
        quality_penalties=video_data.quality_flags,
    )
    result.explanation = _generate_explanation(result, features, video_data)
    return result
