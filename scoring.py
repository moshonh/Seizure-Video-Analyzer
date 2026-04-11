"""
scoring.py
3-Output GDI Motor Pattern Classifier  —  GDI Framework

Output categories:
  1. "Typical PNES pattern"              — stereotyped, stable, burst on quiet background
  2. "Evolving / not typical PNES"       — clear drift, evolution, or changing organisation
  3. "Indeterminate / insufficient data" — mixed, borderline, or poor quality

Classification logic:
  Two continuous scores are computed (0–10):
    pnes_score  — evidence FOR typical PNES
    evol_score  — evidence FOR evolving pattern

  PNES call requires:
    - pnes_score >= PNES_MIN_SCORE (5.0)
    - margin (pnes - evol) >= PNES_COMMIT_MARGIN (2.2)  ← conservative
    - evol_score <= PNES_MAX_EVOL (5.9)
    - temporal_evolution <= PNES_TE_GATE (0.50)         ← veto gate
    - variance_drift <= PNES_VD_GATE (0.45)             ← veto gate

  Evolving call (two paths):
    Path A — score dominance: evol leads pnes by EVOL_COMMIT_MARGIN (0.8)
    Path B — gate triggered:  te > 0.50 or vd > 0.45, plus evol >= 3.5

  Otherwise → Indeterminate

Thresholds  ← adjust at the top of this file
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  ← all adjustable here
# ─────────────────────────────────────────────────────────────────────────────

RELIABILITY_FLOOR          = 0.30   # below this → Indeterminate regardless

# PNES call: requires strong, unambiguous evidence
PNES_COMMIT_MARGIN         = 2.2    # pnes must lead evol by at least this
PNES_MIN_SCORE             = 5.0    # minimum pnes_score to call PNES
PNES_MAX_EVOL              = 5.9    # evol must stay below this when calling PNES

# PNES veto gates — if either fires, PNES is blocked regardless of scores
PNES_TE_GATE               = 0.50   # temporal_evolution above this → cannot call PNES
PNES_VD_GATE               = 0.45   # variance_drift above this → cannot call PNES

# Evolving call thresholds
EVOL_COMMIT_MARGIN         = 0.8    # evol must lead pnes by this (standard path)
EVOL_MIN_SCORE             = 4.0    # minimum evol_score (standard path)
EVOL_MAX_PNES              = 5.9    # pnes must stay below this when calling Evolving
EVOL_MIN_GATED             = 3.5    # minimum evol_score when called via gate

HIGH_CONFIDENCE_MARGIN     = 3.0
MODERATE_CONFIDENCE_MARGIN = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Weights — controls feature importance; sidebar sliders pass values in here
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weights:
    # Primary features (used in scoring)
    movement_stereotypy:             float = 2.5   # high → PNES
    temporal_distribution_stability: float = 2.0   # high → PNES
    variance_drift:                  float = 3.0   # high → Evolving (strongest signal)
    vector_change:                   float = 1.2   # high → Evolving
    amplitude_variability:           float = 0.6   # high → Evolving (noisy in short clips)
    rhythm_irregularity:             float = 0.4   # high → Evolving (noisy in short clips)
    temporal_evolution:              float = 2.5   # high → Evolving
    eye_feature:                     float = 0.8   # closed → PNES; open → Evolving

    # Legacy sidebar fields — accepted but not used in scoring logic
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
    pnes_score:              float          # 0–10: evidence for typical PNES
    es_score:                float          # 0–10: evidence for evolving pattern
    pnes_pattern_confidence: float          # alias = pnes_score
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
# Each returns {pnes, es, label, detail, weight}
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
    return _feat(p, e, w, w, lbl, f"Inter-segment correlation: {p:.3f}")


def _score_distribution_stability(f: MotionFeatures, w: float) -> dict:
    p = f.temporal_distribution_stability
    e = 1.0 - p
    lbl = ("Stable distribution → PNES-like" if p > 0.55
           else "Unstable distribution → Evolving" if p < 0.42
           else "Borderline stability")
    return _feat(p, e, w, w, lbl, f"Distribution stability (JS): {p:.3f}")


def _score_variance_drift(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.variance_drift
    e = f.variance_drift
    lbl = ("No progressive drift → PNES-like" if f.variance_drift < 0.25
           else "Progressive drift → Evolving" if f.variance_drift > 0.40
           else "Mild drift — borderline")
    return _feat(p, e, w, w, lbl, f"Variance drift: {f.variance_drift:.3f}")


def _score_vector_change(f: MotionFeatures, w: float) -> dict:
    p = 1.0 - f.vector_change
    e = f.vector_change
    lbl = ("Stable direction → PNES-like" if f.vector_change < 0.28
           else "Direction shift → Evolving" if f.vector_change > 0.42
           else "Mild direction change — borderline")
    return _feat(p, e, w, w, lbl, f"Vector change (early→late): {f.vector_change:.3f}")


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
    return _feat(p, e, w, w, lbl, f"Rhythm irregularity (CV): {f.rhythm_irregularity:.3f}")


def _score_temporal_evolution(f: MotionFeatures, w: float) -> dict:
    e = f.temporal_evolution
    p = 1.0 - e
    lbl = ("Limited evolution → PNES-like" if e < 0.30
           else "Clear evolution → Evolving" if e > 0.45
           else "Moderate temporal evolution — borderline")
    return _feat(p, e, w, w, lbl,
                 f"Temporal evolution (max of variability/amplitude/direction axes): {e:.3f}")


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
    if r >= 0.70: return "adequate"
    if r >= 0.40: return "limited"
    return "poor"


def _confidence_label(margin: float) -> str:
    if margin >= HIGH_CONFIDENCE_MARGIN:     return "high"
    if margin >= MODERATE_CONFIDENCE_MARGIN: return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Explanation generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_explanation(result: "ScoringResult",
                           features: MotionFeatures,
                           video_data: VideoData) -> str:
    cls   = result.classification
    gates = result.gate_flags
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
        if gates.get("te_gate_triggered"):
            lines.append(
                f"Temporal evolution is clearly elevated "
                f"(score: {features.temporal_evolution:.2f}), indicating that the rhythm, "
                f"direction, and/or amplitude of movement change substantially across the recording."
            )
        if gates.get("vd_gate_triggered"):
            lines.append(
                f"A progressive increase in motion amplitude or variance was detected "
                f"(variance drift: {features.variance_drift:.2f}), inconsistent with a "
                f"stable, stereotyped PNES-like pattern."
            )
        if features.temporal_evolution > 0.42 and not gates.get("te_gate_triggered"):
            lines.append(
                f"The pattern of motion shows moderate temporal evolution "
                f"(score: {features.temporal_evolution:.2f}) in rhythm, direction, or amplitude."
            )
        if features.movement_stereotypy < 0.45:
            lines.append(
                f"Movement cycles are not sufficiently similar to one another "
                f"(stereotypy: {features.movement_stereotypy:.2f}), indicating a "
                f"transforming rather than repetitive pattern."
            )
        if features.variance_drift > 0.38 and not gates.get("vd_gate_triggered"):
            lines.append(
                f"Mild progressive drift in amplitude or variance was detected "
                f"(variance drift: {features.variance_drift:.2f})."
            )
        if features.vector_change > 0.40:
            lines.append(
                f"The dominant direction of movement shifts from the early to the late "
                f"portion of the recording (vector change: {features.vector_change:.2f})."
            )
        if features.eye_state == "open":
            lines.append(
                "Eyes appeared open, providing mild additional support for a non-PNES pattern."
            )

    else:
        lines.append(
            "The movement features in this recording do not clearly favour either a "
            "typical PNES-like pattern or a clearly evolving pattern. "
            "The evidence is mixed, borderline, or the recording quality is insufficient "
            "for a confident GDI-based classification."
        )

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
            "or the patient may be partially outside the selected ROI. "
            "Rhythm and amplitude features are less reliable in this context."
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

    # ── Low-motion quality adjustment ────────────────────────────────────────
    # When very_low_motion is flagged, rhythm, amplitude, and temporal_evolution
    # are computed on near-zero noise and become unreliable.
    very_low = video_data.quality_flags.get("very_low_motion", False)
    eff_weights = Weights(
        movement_stereotypy             = weights.movement_stereotypy,
        temporal_distribution_stability = weights.temporal_distribution_stability,
        variance_drift                  = weights.variance_drift,
        vector_change                   = weights.vector_change,
        amplitude_variability           = weights.amplitude_variability * (0.4 if very_low else 1.0),
        rhythm_irregularity             = weights.rhythm_irregularity   * (0.4 if very_low else 1.0),
        temporal_evolution              = weights.temporal_evolution     * (0.5 if very_low else 1.0),
        eye_feature                     = weights.eye_feature,
        # legacy
        temporal_escalation=weights.temporal_escalation,
        burst_isolation=weights.burst_isolation,
        sustained_variability=weights.sustained_variability,
        active_fraction=weights.active_fraction,
        baseline_fraction=weights.baseline_fraction,
        post_burst_quiet=weights.post_burst_quiet,
        direction_variability=weights.direction_variability,
        regional_asynchrony=weights.regional_asynchrony,
    )

    # ── Feature scores ────────────────────────────────────────────────────────
    feature_scores = {
        "Movement Stereotypy":    _score_stereotypy(features,             eff_weights.movement_stereotypy),
        "Distribution Stability": _score_distribution_stability(features, eff_weights.temporal_distribution_stability),
        "Variance Drift":         _score_variance_drift(features,         eff_weights.variance_drift),
        "Vector Change":          _score_vector_change(features,          eff_weights.vector_change),
        "Amplitude Variability":  _score_amplitude_variability(features,  eff_weights.amplitude_variability),
        "Rhythm Regularity":      _score_rhythm(features,                 eff_weights.rhythm_irregularity),
        "Temporal Evolution":     _score_temporal_evolution(features,     eff_weights.temporal_evolution),
        "Eye State":              _score_eye_state(features,              eff_weights.eye_feature),
    }

    total_pnes = sum(v["pnes"] for v in feature_scores.values())
    total_evol = sum(v["es"]   for v in feature_scores.values())

    sym_weights = (
        eff_weights.movement_stereotypy
        + eff_weights.temporal_distribution_stability
        + eff_weights.variance_drift
        + eff_weights.vector_change
        + eff_weights.amplitude_variability
        + eff_weights.rhythm_irregularity
        + eff_weights.temporal_evolution
        + eff_weights.eye_feature
    )

    pnes_score = round(min(total_pnes / sym_weights * 10.0, 10.0), 2)
    evol_score = round(min(total_evol / sym_weights * 10.0, 10.0), 2)
    margin     = pnes_score - evol_score   # positive = PNES-leaning

    # ── PNES veto gates ───────────────────────────────────────────────────────
    # Use the ORIGINAL (non-low-motion-adjusted) feature values for gate decisions,
    # because the gates are clinical rules, not noise-compensation.
    te_blocks_pnes = features.temporal_evolution > PNES_TE_GATE
    vd_blocks_pnes = features.variance_drift     > PNES_VD_GATE

    # ── Classification ────────────────────────────────────────────────────────
    if reliability_score < RELIABILITY_FLOOR:
        classification = "Indeterminate / insufficient data"
        confidence     = "low"

    elif (pnes_score >= PNES_MIN_SCORE
          and margin >= PNES_COMMIT_MARGIN
          and evol_score <= PNES_MAX_EVOL
          and not te_blocks_pnes
          and not vd_blocks_pnes):
        # Strong, unambiguous PNES-like pattern
        classification = "Typical PNES pattern"
        confidence     = _confidence_label(margin)

    elif (evol_score >= EVOL_MIN_SCORE
          and (-margin) >= EVOL_COMMIT_MARGIN
          and pnes_score <= EVOL_MAX_PNES):
        # Standard Evolving call via score dominance
        classification = "Evolving / not typical PNES"
        confidence     = _confidence_label(-margin)

    elif (te_blocks_pnes or vd_blocks_pnes) and evol_score >= EVOL_MIN_GATED:
        # Gate-triggered: temporal evolution or variance drift clearly elevated
        classification = "Evolving / not typical PNES"
        confidence     = "low"

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
            "te_gate_triggered":   te_blocks_pnes,
            "vd_gate_triggered":   vd_blocks_pnes,
        },
    )
    result.explanation = _generate_explanation(result, features, video_data)
    return result
