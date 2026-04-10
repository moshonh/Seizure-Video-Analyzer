"""
scoring.py
3-Output GDI Motor Pattern Classifier  —  GDI Framework

Output categories:
  1. "Typical PNES pattern"            — stereotyped, stable, internally consistent
  2. "Evolving / not typical PNES"     — clear drift, evolution, or changing organisation
  3. "Indeterminate / insufficient data" — poor quality, short clip, or genuinely mixed

Design philosophy:
  - Does NOT label videos as "ES"
  - "Typical PNES" requires positive evidence of stereotypy + stability
  - "Evolving" requires positive evidence of drift / change
  - "Indeterminate" is the fallback for mixed, borderline, or poor-quality cases
  - Thresholds are LOOSER than the previous high-specificity PNES-only detector
  - Both scores (pnes_score, evol_score) are computed; the clearer one wins

Scoring approach:
  Two independent continuous scores (0–10):
    pnes_score  — evidence FOR typical PNES (stereotypy, stability, low drift)
    evol_score  — evidence FOR evolving pattern (drift, evolution, direction change)

  Classification gate:
    - pnes_score dominant, evol low  → "Typical PNES pattern"
    - evol_score dominant, pnes low  → "Evolving / not typical PNES"
    - otherwise (including poor reliability) → "Indeterminate / insufficient data"

Thresholds  ← all in one block at the top for easy tuning
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  ← adjust here
# ─────────────────────────────────────────────────────────────────────────────

RELIABILITY_FLOOR          = 0.30   # below → Indeterminate regardless
COMMIT_MARGIN              = 0.8    # minimum pnes-evol margin to commit (lowered for sensitivity)
PNES_MIN_SCORE             = 4.5    # pnes_score must reach this to call PNES
EVOL_MIN_SCORE             = 4.0    # evol_score must reach this to call Evolving
PNES_MAX_EVOL              = 5.9    # evol must stay below this when calling PNES (relaxed)
EVOL_MAX_PNES              = 5.9    # pnes must stay below this when calling Evolving (relaxed)
HIGH_CONFIDENCE_MARGIN     = 3.0
MODERATE_CONFIDENCE_MARGIN = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weights:
    # Primary features
    movement_stereotypy:             float = 2.5   # PRIMARY: stereotypy
    temporal_distribution_stability: float = 2.0   # PRIMARY: distribution shape
    variance_drift:                  float = 3.0   # RAISED: strongest Evolving signal
    vector_change:                   float = 1.2   # direction shift
    amplitude_variability:           float = 0.6   # LOWERED: noisy in short clips
    rhythm_irregularity:             float = 0.4   # LOWERED: unreliable in short clips
    temporal_evolution:              float = 2.5   # PRIMARY: composite evolution
    eye_feature:                     float = 0.8   # manual annotation

    # Legacy sidebar fields — accepted but not used in scoring
    temporal_escalation:   float = 0.0
    burst_isolation:       float = 0.0
    sustained_variability: float = 0.0
    active_fraction:       float = 0.0
    baseline_fraction:     float = 0.0
    post_burst_quiet:      float = 0.0
    direction_variability: float = 0.0
    regional_asynchrony:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    pnes_score:              float
    es_score:                float          # = evolving pattern score
    pnes_pattern_confidence: float          # alias for pnes_score
    reliability_score:       float
    classification:          str
    confidence:              str
    reliability:             str
    feature_scores:          Dict[str, dict] = field(default_factory=dict)
    quality_penalties:       Dict[str, bool] = field(default_factory=dict)
    explanation:             str = ""
    gate_flags:              Dict[str, bool] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _feat(p: float, e: float, wp: float, we: float, label: str, detail: str) -> dict:
    return {
        "pnes":   round(p * wp, 3),
        "es":     round(e * we, 3),
        "label":  label,
        "detail": detail,
        "weight": max(wp, we),
    }


def _score_stereotypy(f: MotionFeatures, w: float) -> dict:
    p = f.movement_stereotypy
    e = 1.0 - p
    lbl = ("Stereotyped cycles → PNES-like" if p > 0.55
           else "Low stereotypy → Evolving" if p < 0.42
           else "Borderline stereotypy")
    return _feat(p, e, w, w, lbl,
                 f"Inter-segment correlation: {p:.3f}")


def _score_distribution_stability(f: MotionFeatures, w: float) -> dict:
    p = f.temporal_distribution_stability
    e = 1.0 - p
    lbl = ("Stable distribution → PNES-like" if p > 0.55
           else "Unstable distribution → Evolving" if p < 0.42
           else "Borderline stability")
    return _feat(p, e, w, w, lbl,
                 f"Distribution stability (JS): {p:.3f}")


def _score_variance_drift(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.variance_drift
    e = f.variance_drift
    lbl = ("No progressive drift → PNES-like" if f.variance_drift < 0.25
           else "Progressive drift → Evolving" if f.variance_drift > 0.40
           else "Mild drift — borderline")
    return _feat(p, e, w, w, lbl,
                 f"Variance drift: {f.variance_drift:.3f}")


def _score_vector_change(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.vector_change
    e = f.vector_change
    lbl = ("Stable direction → PNES-like" if f.vector_change < 0.28
           else "Direction shift → Evolving" if f.vector_change > 0.42
           else "Mild direction change — borderline")
    return _feat(p, e, w, w, lbl,
                 f"Vector change (early→late): {f.vector_change:.3f}")


def _score_amplitude_variability(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.amplitude_variability_over_time
    e = f.amplitude_variability_over_time
    lbl = ("Consistent amplitude → PNES-like" if f.amplitude_variability_over_time < 0.28
           else "Amplitude variability → Evolving" if f.amplitude_variability_over_time > 0.42
           else "Moderate amplitude variation — borderline")
    return _feat(p, e, w, w, lbl,
                 f"Amplitude variability over time: {f.amplitude_variability_over_time:.3f}")


def _score_rhythm(f: MotionFeatures, w: float) -> dict:
    norm = float(min(f.rhythm_irregularity / 1.5, 1.0))
    p, e = 1.0 - norm, norm
    lbl = ("Regular rhythm → PNES-like" if f.rhythm_irregularity < 0.6
           else "Irregular rhythm → Evolving" if f.rhythm_irregularity > 1.0
           else "Moderate rhythm irregularity — borderline")
    return _feat(p, e, w, w, lbl,
                 f"Rhythm irregularity (CV): {f.rhythm_irregularity:.3f}")


def _score_temporal_evolution(f: MotionFeatures, w: float) -> dict:
    e = f.temporal_evolution
    p = 1.0 - e
    lbl = ("Limited evolution → PNES-like" if e < 0.30
           else "Clear evolution → Evolving" if e > 0.45
           else "Moderate temporal evolution — borderline")
    return _feat(p, e, w, w, lbl,
                 f"Temporal evolution (variability+amplitude+direction): {e:.3f}")


def _score_eye_state(f: MotionFeatures, w: float) -> dict:
    if f.eye_state == "closed":
        return _feat(1.0, 0.0, w, 0.0,
                     "Eyes closed → supports PNES-like", "Eye closure observed.")
    if f.eye_state == "open":
        return _feat(0.0, 1.0, 0.0, w,
                     "Eyes open → mild support for evolving pattern", "Eyes appear open.")
    return _feat(0.0, 0.0, 0.0, 0.0,
                 "Eye state unavailable → neutral", "Eye state not determined.")


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


def _confidence_label(margin: float) -> str:
    if margin >= HIGH_CONFIDENCE_MARGIN:
        return "high"
    if margin >= MODERATE_CONFIDENCE_MARGIN:
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

    if cls == "Typical PNES pattern":
        lines.append(
            "The movement pattern in this recording appears relatively stereotyped and "
            "internally consistent over time, which is more consistent with a typical "
            "PNES-like motor pattern according to the GDI framework."
        )
        if features.movement_stereotypy > 0.55:
            lines.append(
                f"Movement cycles remain similar to one another across the recording "
                f"(stereotypy: {features.movement_stereotypy:.2f}), suggesting a "
                f"repetitive, consistent pattern with limited evolution."
            )
        if features.temporal_distribution_stability > 0.55:
            lines.append(
                f"The statistical structure of the motion pattern remains stable across "
                f"time windows (distribution stability: "
                f"{features.temporal_distribution_stability:.2f})."
            )
        if features.temporal_evolution < 0.35:
            lines.append(
                "The rhythm, direction, and amplitude show limited change from the "
                "beginning to the end of the recording."
            )
        if features.variance_drift < 0.25:
            lines.append(
                "No meaningful progressive drift in amplitude or variance was detected."
            )
        if features.eye_state == "closed":
            lines.append(
                "Eye closure was observed, providing additional support for a PNES-like pattern."
            )

    elif cls == "Evolving / not typical PNES":
        lines.append(
            "The movement pattern in this recording shows meaningful temporal evolution "
            "or dynamic change over time, which is not consistent with a typical "
            "PNES-like motor pattern according to the GDI framework."
        )
        if features.temporal_evolution > 0.42:
            lines.append(
                f"The pattern of motion changes across the recording — rhythm, direction, "
                f"and/or amplitude shift over time "
                f"(temporal evolution: {features.temporal_evolution:.2f})."
            )
        if features.movement_stereotypy < 0.45:
            lines.append(
                f"Movement cycles are not sufficiently similar to one another "
                f"(stereotypy: {features.movement_stereotypy:.2f}), indicating a "
                f"transforming rather than repetitive pattern."
            )
        if features.variance_drift > 0.38:
            lines.append(
                f"A progressive increase in amplitude or variance was detected "
                f"(variance drift: {features.variance_drift:.2f})."
            )
        if features.vector_change > 0.40:
            lines.append(
                f"The dominant direction of movement shifts from the early to the late "
                f"portion of the recording (vector change: {features.vector_change:.2f})."
            )
        if features.temporal_distribution_stability < 0.45:
            lines.append(
                f"The statistical distribution of the motion pattern changes across time "
                f"windows (stability: {features.temporal_distribution_stability:.2f})."
            )
        if features.eye_state == "open":
            lines.append(
                "Eyes appeared open, providing mild additional support for an "
                "evolving, non-PNES pattern."
            )

    else:
        lines.append(
            "The movement features in this recording do not clearly favour either a "
            "typical PNES-like pattern or a clearly evolving pattern. "
            "The evidence is mixed, borderline, or the recording quality is insufficient "
            "for a confident GDI-based classification."
        )

    # Quality notes
    qf = video_data.quality_flags
    if qf.get("missing_onset"):
        lines.append(
            "The event onset was not fully captured; temporal features may not fully "
            "reflect the complete motor evolution of the event."
        )
    if qf.get("short_video"):
        lines.append(
            "The recording is brief — reliable stereotypy and evolution assessment "
            "requires sufficient event duration."
        )
    if qf.get("very_low_motion"):
        lines.append(
            "Very little overall motion was detected. This may be a non-motor event "
            "or the patient may be partially outside the selected ROI."
        )

    lines.append(
        f"PNES-pattern score: {result.pnes_score:.1f}/10 · "
        f"Evolving-pattern score: {result.es_score:.1f}/10 · "
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
    if weights is None:
        weights = Weights()

    reliability_score = _compute_reliability(video_data)
    reliability       = _reliability_label(reliability_score)

    # ── Feature scores ────────────────────────────────────────────────────────
    feature_scores = {
        "Movement Stereotypy":    _score_stereotypy(features,             weights.movement_stereotypy),
        "Distribution Stability": _score_distribution_stability(features, weights.temporal_distribution_stability),
        "Variance Drift":         _score_variance_drift(features,         weights.variance_drift),
        "Vector Change":          _score_vector_change(features,          weights.vector_change),
        "Amplitude Variability":  _score_amplitude_variability(features,  weights.amplitude_variability),
        "Rhythm Regularity":      _score_rhythm(features,                 weights.rhythm_irregularity),
        "Temporal Evolution":     _score_temporal_evolution(features,     weights.temporal_evolution),
        "Eye State":              _score_eye_state(features,              weights.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_evol = sum(v["es"]   for v in feature_scores.values())

    sym_weights = (
        weights.movement_stereotypy
        + weights.temporal_distribution_stability
        + weights.variance_drift
        + weights.vector_change
        + weights.amplitude_variability
        + weights.rhythm_irregularity
        + weights.temporal_evolution
        + weights.eye_feature
    )

    pnes_score = round(min(total_pnes / sym_weights * 10.0, 10.0), 2)
    evol_score = round(min(total_evol / sym_weights * 10.0, 10.0), 2)
    margin     = pnes_score - evol_score   # positive = PNES-leaning

    # ── Classification ────────────────────────────────────────────────────────
    if reliability_score < RELIABILITY_FLOOR:
        classification = "Indeterminate / insufficient data"
        confidence     = "low"

    elif (pnes_score >= PNES_MIN_SCORE
          and margin >= COMMIT_MARGIN
          and evol_score <= PNES_MAX_EVOL):
        classification = "Typical PNES pattern"
        confidence     = _confidence_label(margin)

    elif (evol_score >= EVOL_MIN_SCORE
          and (-margin) >= COMMIT_MARGIN
          and pnes_score <= EVOL_MAX_PNES):
        classification = "Evolving / not typical PNES"
        confidence     = _confidence_label(-margin)

    else:
        classification = "Indeterminate / insufficient data"
        confidence     = "low"

    result = ScoringResult(
        pnes_score=pnes_score,
        es_score=evol_score,
        pnes_pattern_confidence=pnes_score,
        reliability_score=reliability_score,
        classification=classification,
        confidence=confidence,
        reliability=reliability,
        feature_scores=feature_scores,
        quality_penalties=video_data.quality_flags,
        gate_flags={
            "high_stereotypy":     features.movement_stereotypy > 0.55,
            "stable_distribution": features.temporal_distribution_stability > 0.55,
            "low_drift":           features.variance_drift < 0.25,
            "stable_direction":    features.vector_change < 0.28,
            "low_amplitude_var":   features.amplitude_variability_over_time < 0.28,
            "clear_evolution":     features.temporal_evolution > 0.42,
            "variance_drift_high": features.variance_drift > 0.40,
            "vector_change_high":  features.vector_change > 0.42,
        },
    )
    result.explanation = _generate_explanation(result, features, video_data)
    return result
