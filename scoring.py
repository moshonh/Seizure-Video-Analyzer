"""
scoring.py
Rule-based scoring engine based on the Global Dynamic Impression (GDI) framework.
Maps extracted motor features to PNES / ES / Indeterminate classification.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weights:
    motor_dynamism:       float = 1.5
    direction_variability: float = 1.2
    frequency_variability: float = 1.0
    temporal_evolution:   float = 1.2
    burst_pattern:        float = 1.0
    eye_feature:          float = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    pnes_score:       float          # 0–10
    es_score:         float          # 0–10
    reliability_score: float         # 0–1
    classification:   str            # "ES" | "PNES" | "Indeterminate"
    confidence:       str            # "low" | "moderate" | "high"
    reliability:      str            # "poor" | "limited" | "adequate"
    feature_scores:   Dict[str, dict] = field(default_factory=dict)
    quality_penalties: Dict[str, bool] = field(default_factory=dict)
    explanation:      str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature scoring
# Each function returns a dict: {pnes, es, label, detail, weight}
# ─────────────────────────────────────────────────────────────────────────────

def _score_motor_dynamism(f: MotionFeatures, weight: float) -> dict:
    """Low variability + flat evolution → PNES; evolving → ES."""
    evol      = abs(f.motion_evolution)
    var_norm  = min(f.motion_variability / (f.mean_motion + 1e-6), 2.0) / 2.0

    pnes_raw  = (1.0 - var_norm) * (1.0 - min(evol * 2.0, 1.0))
    es_raw    = var_norm * 0.5 + min(evol * 2.0, 1.0) * 0.5

    direction = ("increasing" if f.motion_evolution > 0.01
                 else "decreasing" if f.motion_evolution < -0.01
                 else "stable")

    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Low dynamism → PNES" if pnes_raw >= es_raw else "Evolving dynamics → ES",
        "detail": f"Motion trend: {direction}, relative variability: {var_norm:.2f}",
        "weight": weight,
    }


def _score_direction(f: MotionFeatures, weight: float) -> dict:
    """Fixed direction → PNES; changing direction → ES."""
    circ_norm  = min(f.angle_variability / 1.5, 1.0)
    es_raw     = circ_norm * 0.6 + f.angle_changes * 0.4
    pnes_raw   = 1.0 - es_raw

    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Fixed direction → PNES" if pnes_raw >= es_raw else "Changing direction → ES",
        "detail": (f"Circular std of angles: {f.angle_variability:.2f} rad, "
                   f"large-shift fraction: {f.angle_changes:.2f}"),
        "weight": weight,
    }


def _score_frequency(f: MotionFeatures, weight: float) -> dict:
    """Constant rhythm → PNES; evolving rhythm → ES."""
    pnes_raw = f.frequency_stationarity
    es_raw   = 1.0 - f.frequency_stationarity

    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Constant rhythm → PNES" if pnes_raw >= es_raw else "Evolving rhythm → ES",
        "detail": (f"Rhythm stationarity: {f.frequency_stationarity:.2f}, "
                   f"dominant freq: {f.dominant_frequency:.2f} Hz"),
        "weight": weight,
    }


def _score_temporal_evolution(f: MotionFeatures, weight: float) -> dict:
    """Fade-in + fade-out → ES; burst-arrest-burst → PNES."""
    es_raw   = (f.fade_in_score + f.fade_out_score) / 2.0
    pnes_raw = f.burst_repeat_score
    total    = es_raw + pnes_raw + 1e-6

    return {
        "pnes":   round((pnes_raw / total) * weight, 3),
        "es":     round((es_raw   / total) * weight, 3),
        "label":  "Burst-arrest pattern → PNES" if pnes_raw >= es_raw else "Fade in/out → ES",
        "detail": (f"Fade-in: {f.fade_in_score:.2f}, Fade-out: {f.fade_out_score:.2f}, "
                   f"Burst-repeat: {f.burst_repeat_score:.2f}"),
        "weight": weight,
    }


def _score_burst_similarity(f: MotionFeatures, weight: float) -> dict:
    """Highly repetitive bursts → PNES; variable bursts → ES."""
    pnes_raw = f.burst_similarity
    es_raw   = 1.0 - f.burst_similarity

    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     round(es_raw   * weight, 3),
        "label":  "Repetitive bursts → PNES" if pnes_raw >= es_raw else "Variable bursts → ES",
        "detail": f"Burst profile similarity: {f.burst_similarity:.2f}",
        "weight": weight,
    }


def _score_eye_state(f: MotionFeatures, weight: float) -> dict:
    """Eyes closed → PNES; eyes open → ES; unavailable → neutral (0 / 0)."""
    if f.eye_state == "closed":
        return {
            "pnes": round(weight, 3), "es": 0.0,
            "label": "Eyes closed → supports PNES",
            "detail": "Eye closure observed in visible segments.",
            "weight": weight,
        }
    if f.eye_state == "open":
        return {
            "pnes": 0.0, "es": round(weight, 3),
            "label": "Eyes open → supports ES",
            "detail": "Eyes appear open in visible segments.",
            "weight": weight,
        }
    return {
        "pnes": 0.0, "es": 0.0,
        "label": "Eye state unavailable → neutral",
        "detail": "Eye state could not be determined from this video.",
        "weight": weight,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reliability
# ─────────────────────────────────────────────────────────────────────────────

def _compute_reliability(video_data: VideoData) -> float:
    """0–1 reliability score based on video quality flags."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────

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

def _generate_explanation(
    result: ScoringResult,
    features: MotionFeatures,
    video_data: VideoData,
) -> str:
    cls = result.classification
    lines = []

    if cls == "PNES":
        lines.append(
            "This video shows relatively repetitive motor patterns with limited change in direction "
            "and rhythm over time, which is more consistent with a psychogenic nonepileptic seizure "
            "(PNES) according to the GDI framework."
        )
    elif cls == "ES":
        lines.append(
            "This video displays evolving motor activity with changing direction and rhythm over time, "
            "which is more consistent with an epileptic seizure (ES) according to the GDI framework."
        )
    else:
        lines.append(
            "The motor features in this video do not clearly favour either an epileptic seizure (ES) "
            "or a psychogenic nonepileptic seizure (PNES). The evidence is mixed or insufficient for "
            "a confident classification."
        )

    if features.eye_state == "closed":
        lines.append("Eye closure was observed during visible segments, further supporting PNES.")
    elif features.eye_state == "open":
        lines.append("Eyes appeared open during visible segments, which adds mild support for ES.")

    qf = video_data.quality_flags
    if qf.get("missing_onset"):    lines.append("The seizure onset was not fully captured, reducing confidence.")
    if qf.get("missing_offset"):   lines.append("The event end is not visible, limiting assessment of the fade-out phase.")
    if qf.get("short_video"):      lines.append("The video is brief, restricting temporal pattern analysis.")
    if qf.get("very_low_motion"):  lines.append("Very little overall motion was detected; this may be a non-motor event or the subject is partially out of frame.")

    lines.append(
        f"PNES support score: {result.pnes_score:.1f}/10 · "
        f"ES support score: {result.es_score:.1f}/10 · "
        f"Reliability: {result.reliability}."
    )
    lines.append(
        "IMPORTANT: This is a research prototype. Results require expert clinical review and, "
        "when appropriate, video-EEG and full clinical context."
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
    """
    Apply GDI-based scoring to extracted features.

    Returns
    -------
    ScoringResult with classification, confidence, reliability, and plain-language explanation.
    """
    if weights is None:
        weights = Weights()

    feature_scores = {
        "Motor Dynamism":       _score_motor_dynamism(features,    weights.motor_dynamism),
        "Direction Variability": _score_direction(features,        weights.direction_variability),
        "Frequency Variability": _score_frequency(features,        weights.frequency_variability),
        "Temporal Evolution":   _score_temporal_evolution(features, weights.temporal_evolution),
        "Burst Similarity":     _score_burst_similarity(features,  weights.burst_pattern),
        "Eye State":            _score_eye_state(features,         weights.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_es   = sum(v["es"]   for v in feature_scores.values())

    max_possible = (
        weights.motor_dynamism
        + weights.direction_variability
        + weights.frequency_variability
        + weights.temporal_evolution
        + weights.burst_pattern
        + weights.eye_feature
    )

    pnes_score = round(min(total_pnes / max_possible * 10.0, 10.0), 2)
    es_score   = round(min(total_es   / max_possible * 10.0, 10.0), 2)

    reliability_score = _compute_reliability(video_data)
    reliability       = _reliability_label(reliability_score)

    # Classification decision
    total          = pnes_score + es_score + 1e-6
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
