"""
scoring.py
GDI-based scoring engine grounded in two observed clinical profiles:

  PNES (Graph 1): single isolated burst on a quiet background.
    Key signals: high baseline_fraction, high burst_isolation,
                 high post_burst_quiet, low active_fraction,
                 low temporal_escalation.

  ES (Graph 2): sustained, escalating, persistent activity.
    Key signals: high temporal_escalation, high sustained_variability,
                 high active_fraction, low burst_isolation,
                 low post_burst_quiet.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


# ─────────────────────────────────────────────────────────────────────────────
# Weights  ← adjust here to change feature influence
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weights:
    temporal_escalation:   float = 2.0   # strongest ES indicator (Graph 2)
    burst_isolation:       float = 1.8   # strongest PNES indicator (Graph 1)
    sustained_variability: float = 1.6   # high → ES
    active_fraction:       float = 1.4   # high → ES
    baseline_fraction:     float = 1.4   # high → PNES
    post_burst_quiet:      float = 1.2   # high → PNES
    direction_variability: float = 0.8   # high → ES (supporting feature)
    eye_feature:           float = 0.8   # open → ES, closed → PNES


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

def _score_temporal_escalation(f: MotionFeatures, weight: float) -> dict:
    """
    Steady increase in motion from start to end → ES.
    No escalation (flat or single burst) → PNES.
    """
    es_raw   = f.temporal_escalation
    pnes_raw = 1.0 - es_raw
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Sustained escalation → ES" if es_raw > pnes_raw else "No escalation → PNES",
        "detail": f"Temporal escalation score: {f.temporal_escalation:.3f} "
                  f"(first-third mean: {f.mean_motion:.2f}, "
                  f"peak: {f.peak_motion:.2f})",
        "weight": weight,
    }


def _score_burst_isolation(f: MotionFeatures, weight: float) -> dict:
    """
    Single isolated burst on quiet background → PNES.
    Broad / sustained activity → ES.
    """
    pnes_raw = f.burst_isolation
    es_raw   = 1.0 - f.burst_isolation
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Isolated focal burst → PNES" if pnes_raw > es_raw else "Broad sustained activity → ES",
        "detail": f"Burst isolation score: {f.burst_isolation:.3f} "
                  f"(baseline threshold: {f.baseline_threshold:.2f})",
        "weight": weight,
    }


def _score_sustained_variability(f: MotionFeatures, weight: float) -> dict:
    """
    High variability during active frames → ES (activity is rich and variable).
    Low variability during active frames → PNES (burst is stereotyped).
    sustained_variability is normalised to ~0–1 range for scoring.
    """
    # Normalise: values above 0.5 lean ES, below 0.5 lean PNES
    norm     = float(min(f.sustained_variability / 0.5, 1.0))
    es_raw   = norm
    pnes_raw = 1.0 - norm
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Sustained high variability → ES" if es_raw > pnes_raw else "Low sustained variability → PNES",
        "detail": f"Mean variability during active frames: {f.sustained_variability:.3f}",
        "weight": weight,
    }


def _score_active_fraction(f: MotionFeatures, weight: float) -> dict:
    """
    Long proportion of recording spent active → ES (sustained seizure).
    Short active proportion (brief burst) → PNES.
    """
    es_raw   = f.active_fraction
    pnes_raw = 1.0 - f.active_fraction
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Prolonged active phase → ES" if es_raw > pnes_raw else "Brief active phase → PNES",
        "detail": f"Active fraction: {f.active_fraction:.2f} "
                  f"(baseline fraction: {f.baseline_fraction:.2f})",
        "weight": weight,
    }


def _score_baseline_fraction(f: MotionFeatures, weight: float) -> dict:
    """
    Most of recording spent near baseline → PNES (long quiet periods).
    Little time near baseline → ES (sustained activity).
    """
    pnes_raw = f.baseline_fraction
    es_raw   = 1.0 - f.baseline_fraction
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Long quiet baseline → PNES" if pnes_raw > es_raw else "Little baseline time → ES",
        "detail": f"Baseline fraction: {f.baseline_fraction:.2f}",
        "weight": weight,
    }


def _score_post_burst_quiet(f: MotionFeatures, weight: float) -> dict:
    """
    Motion returns to near-baseline in the last 25% of recording → PNES.
    Motion stays elevated throughout → ES.
    """
    pnes_raw = f.post_burst_quiet
    es_raw   = 1.0 - f.post_burst_quiet
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Returns to baseline post-burst → PNES" if pnes_raw > es_raw else "Sustained end-activity → ES",
        "detail": f"Post-burst quiet score: {f.post_burst_quiet:.3f}",
        "weight": weight,
    }


def _score_direction_variability(f: MotionFeatures, weight: float) -> dict:
    """
    Changing direction → ES.  Fixed direction → PNES.
    """
    # Normalise circular std: 0 = fixed, ~1.8 = fully random
    norm     = float(min(f.angle_variability / 1.2, 1.0))
    es_raw   = norm
    pnes_raw = 1.0 - norm
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Changing direction → ES" if es_raw > pnes_raw else "Fixed direction → PNES",
        "detail": f"Circular std of flow angles: {f.angle_variability:.3f} rad",
        "weight": weight,
    }


def _score_eye_state(f: MotionFeatures, weight: float) -> dict:
    """Eyes closed → PNES; open → ES; unavailable → neutral."""
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
            "This video shows a pattern consistent with PNES according to the GDI framework: "
            "most of the recording is spent at low baseline motion, with a relatively isolated, "
            "focal burst of activity that does not show the gradual build-up and sustained escalation "
            "typical of an epileptic seizure."
        )
        # Specific evidence
        if features.baseline_fraction > 0.55:
            lines.append(
                f"The recording spends {features.baseline_fraction * 100:.0f}% of its duration "
                f"near the motion baseline, consistent with the prolonged quiet periods seen in PNES."
            )
        if features.burst_isolation > 0.65:
            lines.append(
                "The peak activity is notably isolated and narrow, resembling the focal burst-on-quiet-background "
                "pattern observed in PNES."
            )
        if features.post_burst_quiet > 0.60:
            lines.append(
                "Motion returns toward baseline toward the end of the recording, consistent with "
                "the post-burst quiet typically seen after a PNES event."
            )

    elif cls == "ES":
        lines.append(
            "This video shows a pattern consistent with ES according to the GDI framework: "
            "motion activity escalates over time, variability remains high throughout the active period, "
            "and the recording does not show the long quiet baseline or isolated focal burst "
            "characteristic of PNES."
        )
        if features.temporal_escalation > 0.40:
            lines.append(
                f"A clear temporal escalation was detected: motion intensity increases steadily "
                f"from the beginning toward the end of the recording, consistent with the evolving "
                f"dynamics of an epileptic seizure."
            )
        if features.sustained_variability > 0.30:
            lines.append(
                "Variability during active frames remains elevated, suggesting ongoing irregular "
                "motor activity rather than a single stereotyped burst."
            )
        if features.active_fraction > 0.45:
            lines.append(
                f"The recording spends {features.active_fraction * 100:.0f}% of its duration "
                f"in above-baseline activity, consistent with the sustained nature of epileptic motor seizures."
            )
    else:
        lines.append(
            "The motor features in this video do not clearly favour either ES or PNES. "
            "The evidence is mixed or the recording quality is insufficient for a confident GDI-based classification."
        )

    # Eye state
    if features.eye_state == "closed":
        lines.append("Eye closure was noted, which adds mild additional support for PNES.")
    elif features.eye_state == "open":
        lines.append("Eyes appeared open, which adds mild additional support for ES.")

    # Quality warnings
    qf = video_data.quality_flags
    if qf.get("missing_onset"):
        lines.append("The event onset was not fully captured, which may reduce the reliability of temporal features.")
    if qf.get("short_video"):
        lines.append("The video is brief; longer recordings provide more reliable temporal pattern analysis.")
    if qf.get("very_low_motion"):
        lines.append(
            "Very little overall motion was detected. This may indicate a non-motor event, "
            "or the patient may be partially outside the ROI."
        )

    lines.append(
        f"PNES support: {result.pnes_score:.1f}/10 · "
        f"ES support: {result.es_score:.1f}/10 · "
        f"Reliability: {result.reliability}."
    )
    lines.append(
        "IMPORTANT: This is a research prototype. All results require expert clinical review "
        "and, when appropriate, video-EEG and full clinical context."
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
    if weights is None:
        weights = Weights()

    feature_scores = {
        "Temporal Escalation":   _score_temporal_escalation(features,   weights.temporal_escalation),
        "Burst Isolation":       _score_burst_isolation(features,       weights.burst_isolation),
        "Sustained Variability": _score_sustained_variability(features, weights.sustained_variability),
        "Active Fraction":       _score_active_fraction(features,       weights.active_fraction),
        "Baseline Fraction":     _score_baseline_fraction(features,     weights.baseline_fraction),
        "Post-burst Quiet":      _score_post_burst_quiet(features,      weights.post_burst_quiet),
        "Direction Variability": _score_direction_variability(features, weights.direction_variability),
        "Eye State":             _score_eye_state(features,             weights.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_es   = sum(v["es"]   for v in feature_scores.values())

    max_possible = (
        weights.temporal_escalation
        + weights.burst_isolation
        + weights.sustained_variability
        + weights.active_fraction
        + weights.baseline_fraction
        + weights.post_burst_quiet
        + weights.direction_variability
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
