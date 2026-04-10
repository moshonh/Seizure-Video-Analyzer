"""
scoring.py
High-Confidence PNES Pattern Detector  —  GDI Framework

Clinical goal (revised):
─────────────────────────────────────────────────────────────────────────────
This module no longer asks "ES vs PNES?"

It asks ONE question:
  "Does this video show a high-confidence, specifically PNES-like motor pattern?"

Output categories:
  1. High-confidence PNES pattern
  2. Not confidently PNES
  3. Insufficient / poor-quality video

Rationale:
  PNES-typical motor patterns may be identified with relatively high specificity.
  The heterogeneous ES category is NOT modelled here — the tool abstains
  whenever the pattern is not strongly and specifically PNES-like.

Decision philosophy:
  - HIGH SPECIFICITY over high sensitivity
  - Abstain (Not confidently PNES) when in any doubt
  - Only output "High-confidence PNES pattern" when multiple
    PNES-supportive features are clearly present together

PNES-supportive features (must be JOINTLY present):
  - high movement_stereotypy
  - high temporal_distribution_stability
  - low temporal_evolution
  - low variance_drift
  - low vector_change
  - low amplitude_variability_over_time
  - stable rhythm (low rhythm_irregularity)
  - eye closure if visible

Anti-PNES features (any one can trigger abstention):
  - strong temporal evolution
  - clear variance drift
  - directional drift (vector_change)
  - low stereotypy
  - evolving distribution
  - eyes open (weak but additive)
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from motion_features import MotionFeatures
from video_processing import VideoData


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  ← adjust here without touching logic
# ─────────────────────────────────────────────────────────────────────────────

# PNES-supportive thresholds (feature must EXCEED these to support PNES)
PNES_STEREOTYPY_MIN          = 0.62   # movement_stereotypy
PNES_DISTRIBUTION_STABILITY_MIN = 0.60  # temporal_distribution_stability

# Anti-PNES thresholds (feature must stay BELOW these for PNES to remain possible)
PNES_EVOLUTION_MAX           = 0.38   # temporal_evolution
PNES_VARIANCE_DRIFT_MAX      = 0.30   # variance_drift
PNES_VECTOR_CHANGE_MAX       = 0.35   # vector_change
PNES_AMPLITUDE_VAR_MAX       = 0.35   # amplitude_variability_over_time
PNES_RHYTHM_IRR_MAX          = 0.90   # rhythm_irregularity

# Reliability floor below which we return "Insufficient video"
RELIABILITY_FLOOR            = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Weights for the continuous PNES confidence score (internal use)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weights:
    # These are kept for backward compatibility with app.py sliders.
    # Only the weights used in _pnes_confidence_score() matter for scoring;
    # the rest are accepted but not used in the final classification.
    temporal_escalation:   float = 0.0    # not used in PNES detector
    burst_isolation:       float = 0.0    # not used
    sustained_variability: float = 0.0    # not used
    active_fraction:       float = 0.0    # not used
    baseline_fraction:     float = 0.0    # not used
    post_burst_quiet:      float = 0.0    # not used
    direction_variability: float = 0.0    # folded into anti-PNES check
    rhythm_irregularity:   float = 1.0    # contributes to confidence score
    temporal_evolution:    float = 0.0    # used as anti-PNES gate
    regional_asynchrony:   float = 0.0    # not used
    eye_feature:           float = 0.8    # contributes if visible

    # Primary PNES features — drive the confidence score
    movement_stereotypy:         float = 2.5
    temporal_distribution_stability: float = 2.0
    variance_drift:              float = 1.5   # inverse: low drift → PNES
    vector_change:               float = 1.2   # inverse: low change → PNES
    amplitude_variability:       float = 1.2   # inverse: low variability → PNES


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    # Primary outputs
    pnes_pattern_confidence: float      # 0–10: how PNES-like is the pattern?
    classification:          str        # "High-confidence PNES pattern" |
                                        # "Not confidently PNES" |
                                        # "Insufficient / poor-quality video"
    confidence:              str        # "low" | "moderate" | "high"
    reliability:             str        # "poor" | "limited" | "adequate"
    reliability_score:       float      # 0–1

    # Kept for backward compat with app.py / reporting.py / JSON export
    pnes_score:   float = 0.0    # = pnes_pattern_confidence (alias)
    es_score:     float = 0.0    # always 0 — not used in this detector

    feature_scores:    Dict[str, dict] = field(default_factory=dict)
    quality_penalties: Dict[str, bool] = field(default_factory=dict)
    explanation:       str = ""

    # Gate flags (for transparency)
    gate_flags: Dict[str, bool] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature evidence functions
# Each returns: {pnes, es, label, detail, weight}
# "es" column is always 0 in this detector — kept for table compatibility.
# ─────────────────────────────────────────────────────────────────────────────

def _feat_stereotypy(f: MotionFeatures, weight: float) -> dict:
    pnes_raw = f.movement_stereotypy
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     0.0,
        "label":  "Stereotyped cycles → supports PNES" if pnes_raw >= PNES_STEREOTYPY_MIN
                  else "Insufficient stereotypy",
        "detail": f"Inter-segment correlation: {f.movement_stereotypy:.3f} "
                  f"(threshold ≥ {PNES_STEREOTYPY_MIN})",
        "weight": weight,
    }


def _feat_distribution_stability(f: MotionFeatures, weight: float) -> dict:
    pnes_raw = f.temporal_distribution_stability
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     0.0,
        "label":  "Stable pattern distribution → supports PNES"
                  if pnes_raw >= PNES_DISTRIBUTION_STABILITY_MIN
                  else "Unstable distribution",
        "detail": f"Distribution stability (JS): {f.temporal_distribution_stability:.3f} "
                  f"(threshold ≥ {PNES_DISTRIBUTION_STABILITY_MIN})",
        "weight": weight,
    }


def _feat_variance_drift(f: MotionFeatures, weight: float) -> dict:
    # Low drift → PNES; high drift → anti-PNES
    pnes_raw = 1.0 - f.variance_drift
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     0.0,
        "label":  "No progressive drift → supports PNES"
                  if f.variance_drift <= PNES_VARIANCE_DRIFT_MAX
                  else f"Progressive drift detected (anti-PNES gate triggered)",
        "detail": f"Variance drift: {f.variance_drift:.3f} "
                  f"(threshold ≤ {PNES_VARIANCE_DRIFT_MAX})",
        "weight": weight,
    }


def _feat_vector_change(f: MotionFeatures, weight: float) -> dict:
    pnes_raw = 1.0 - f.vector_change
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     0.0,
        "label":  "Stable direction → supports PNES"
                  if f.vector_change <= PNES_VECTOR_CHANGE_MAX
                  else "Direction shift detected (anti-PNES)",
        "detail": f"Vector change (early→late): {f.vector_change:.3f} "
                  f"(threshold ≤ {PNES_VECTOR_CHANGE_MAX})",
        "weight": weight,
    }


def _feat_amplitude_variability(f: MotionFeatures, weight: float) -> dict:
    pnes_raw = 1.0 - f.amplitude_variability_over_time
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     0.0,
        "label":  "Consistent amplitude → supports PNES"
                  if f.amplitude_variability_over_time <= PNES_AMPLITUDE_VAR_MAX
                  else "Amplitude variability (anti-PNES)",
        "detail": f"Amplitude variability: {f.amplitude_variability_over_time:.3f} "
                  f"(threshold ≤ {PNES_AMPLITUDE_VAR_MAX})",
        "weight": weight,
    }


def _feat_rhythm(f: MotionFeatures, weight: float) -> dict:
    # Low rhythm_irregularity → PNES
    norm     = float(min(f.rhythm_irregularity / 1.5, 1.0))
    pnes_raw = 1.0 - norm
    return {
        "pnes":   round(pnes_raw * weight, 3),
        "es":     0.0,
        "label":  "Regular rhythm → supports PNES"
                  if f.rhythm_irregularity <= PNES_RHYTHM_IRR_MAX
                  else "Irregular rhythm (anti-PNES)",
        "detail": f"Rhythm irregularity (CV): {f.rhythm_irregularity:.3f} "
                  f"(threshold ≤ {PNES_RHYTHM_IRR_MAX})",
        "weight": weight,
    }


def _feat_eye_state(f: MotionFeatures, weight: float) -> dict:
    if f.eye_state == "closed":
        return {
            "pnes":   round(weight, 3), "es": 0.0,
            "label":  "Eyes closed → supports PNES",
            "detail": "Eye closure observed.", "weight": weight,
        }
    if f.eye_state == "open":
        return {
            "pnes":   0.0, "es": 0.0,
            "label":  "Eyes open → mild anti-PNES signal",
            "detail": "Eyes appear open (additive anti-PNES factor).", "weight": weight,
        }
    return {
        "pnes":   0.0, "es": 0.0,
        "label":  "Eye state unavailable → neutral",
        "detail": "Eye state not determined.", "weight": weight,
    }


def _feat_temporal_evolution(f: MotionFeatures) -> dict:
    """Gate-only feature — not added to score, but shown in table."""
    triggered = f.temporal_evolution > PNES_EVOLUTION_MAX
    return {
        "pnes":   0.0, "es": 0.0,
        "label":  f"Temporal evolution gate: {'⚠️ TRIGGERED (anti-PNES)' if triggered else '✅ clear'}",
        "detail": f"Temporal evolution: {f.temporal_evolution:.3f} "
                  f"(anti-PNES if > {PNES_EVOLUTION_MAX})",
        "weight": 0.0,
    }


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


# ─────────────────────────────────────────────────────────────────────────────
# Continuous PNES confidence score (internal)
# ─────────────────────────────────────────────────────────────────────────────

def _pnes_confidence_score(features: MotionFeatures, weights: Weights) -> float:
    """
    Weighted sum of PNES-supportive signals, normalised to 0–10.
    This is NOT the classification — it feeds into the final gate logic.
    """
    total_pnes = (
        features.movement_stereotypy             * weights.movement_stereotypy
        + features.temporal_distribution_stability * weights.temporal_distribution_stability
        + (1.0 - features.variance_drift)          * weights.variance_drift
        + (1.0 - features.vector_change)           * weights.vector_change
        + (1.0 - features.amplitude_variability_over_time) * weights.amplitude_variability
        + (1.0 - min(features.rhythm_irregularity / 1.5, 1.0)) * weights.rhythm_irregularity
        + (weights.eye_feature if features.eye_state == "closed" else 0.0)
    )

    max_possible = (
        weights.movement_stereotypy
        + weights.temporal_distribution_stability
        + weights.variance_drift
        + weights.vector_change
        + weights.amplitude_variability
        + weights.rhythm_irregularity
        + weights.eye_feature
    )

    return round(min(total_pnes / max_possible * 10.0, 10.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Gate logic — all gates must be clear for PNES classification
# ─────────────────────────────────────────────────────────────────────────────

def _compute_gate_flags(features: MotionFeatures) -> Dict[str, bool]:
    """
    Returns a dict of gate name → True if gate is TRIGGERED (anti-PNES).
    Classification is PNES only when ALL gates are False.
    """
    return {
        "temporal_evolution_high":      features.temporal_evolution > PNES_EVOLUTION_MAX,
        "variance_drift_high":          features.variance_drift > PNES_VARIANCE_DRIFT_MAX,
        "vector_change_high":           features.vector_change > PNES_VECTOR_CHANGE_MAX,
        "amplitude_variability_high":   features.amplitude_variability_over_time > PNES_AMPLITUDE_VAR_MAX,
        "rhythm_too_irregular":         features.rhythm_irregularity > PNES_RHYTHM_IRR_MAX,
        "stereotypy_too_low":           features.movement_stereotypy < PNES_STEREOTYPY_MIN,
        "distribution_unstable":        features.temporal_distribution_stability < PNES_DISTRIBUTION_STABILITY_MIN,
        "eyes_open":                    features.eye_state == "open",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Explanation generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_explanation(
    result: "ScoringResult",
    features: MotionFeatures,
    video_data: VideoData,
) -> str:
    cls   = result.classification
    gates = result.gate_flags
    lines = []

    if cls == "High-confidence PNES pattern":
        lines.append(
            "The movement pattern in this recording shows a relatively stereotyped and "
            "statistically stable temporal organisation across the event, which supports a "
            "high-confidence PNES-like motor pattern according to the GDI framework."
        )
        if features.movement_stereotypy >= PNES_STEREOTYPY_MIN:
            lines.append(
                f"Movement cycles remain similar to one another throughout the recording "
                f"(stereotypy: {features.movement_stereotypy:.2f}), consistent with a "
                f"repetitive, internally consistent pattern."
            )
        if features.temporal_distribution_stability >= PNES_DISTRIBUTION_STABILITY_MIN:
            lines.append(
                f"The statistical structure of the motion pattern remains stable across time "
                f"windows (distribution stability: {features.temporal_distribution_stability:.2f}), "
                f"indicating limited change in the nature of the movement over time."
            )
        if features.temporal_evolution <= PNES_EVOLUTION_MAX:
            lines.append(
                "The rhythm, direction, and amplitude of movement show limited change "
                "from the beginning to the end of the recording."
            )
        if features.eye_state == "closed":
            lines.append("Eye closure was observed, providing additional support for a PNES-like pattern.")

    elif cls == "Not confidently PNES":
        lines.append(
            "The movement pattern in this recording does not show a sufficiently specific "
            "or consistent PNES-like signature. The tool does not classify this video as "
            "high-confidence PNES."
        )
        # Explain which gates were triggered
        triggered = [k for k, v in gates.items() if v]
        if "temporal_evolution_high" in triggered:
            lines.append(
                f"The motor pattern shows meaningful temporal evolution — the rhythm, direction, "
                f"and/or amplitude change across the recording "
                f"(evolution score: {features.temporal_evolution:.2f}), "
                f"which argues against a stereotyped PNES-like pattern."
            )
        if "stereotypy_too_low" in triggered:
            lines.append(
                f"Movement cycles are not sufficiently similar to one another "
                f"(stereotypy: {features.movement_stereotypy:.2f}), "
                f"suggesting the motor pattern is not consistently repetitive."
            )
        if "variance_drift_high" in triggered:
            lines.append(
                f"A progressive increase in amplitude or variance was detected over time "
                f"(variance drift: {features.variance_drift:.2f}), "
                f"which is inconsistent with a stable, stereotyped PNES-like pattern."
            )
        if "vector_change_high" in triggered:
            lines.append(
                f"The dominant direction of movement shifts from the early to the late portion "
                f"of the recording (vector change: {features.vector_change:.2f})."
            )
        if "amplitude_variability_high" in triggered:
            lines.append(
                f"Motion amplitude varies considerably across the recording "
                f"(amplitude variability: {features.amplitude_variability_over_time:.2f})."
            )
        if "distribution_unstable" in triggered:
            lines.append(
                f"The statistical distribution of the motion pattern changes across time windows "
                f"(stability: {features.temporal_distribution_stability:.2f}), "
                f"suggesting the pattern is not internally consistent."
            )
        if "eyes_open" in triggered:
            lines.append(
                "Eyes appeared open during the event, which adds a mild anti-PNES signal."
            )
        if not triggered:
            lines.append(
                "The individual features do not show strong red flags, but the combination "
                "does not meet the conservative threshold for high-confidence PNES classification."
            )

    else:  # Insufficient / poor-quality video
        lines.append(
            "The available video is too limited for confident pattern recognition. "
            "The recording quality, duration, or completeness is insufficient to support "
            "a reliable GDI-based assessment."
        )

    # Quality notes
    qf = video_data.quality_flags
    if qf.get("missing_onset"):
        lines.append(
            "The event onset was not fully captured; temporal features may underestimate "
            "the actual degree of evolution or stereotypy."
        )
    if qf.get("short_video"):
        lines.append(
            "The recording is brief. Reliable stereotypy and evolution assessment "
            "requires sufficient event duration."
        )
    if qf.get("very_low_motion"):
        lines.append(
            "Very little overall motion was detected. This may be a non-motor event "
            "or the patient may be partially outside the selected ROI."
        )

    lines.append(
        f"PNES-pattern confidence: {result.pnes_pattern_confidence:.1f}/10 · "
        f"Reliability: {result.reliability}."
    )
    lines.append(
        "IMPORTANT: This is a research prototype designed to identify typical PNES-like motor patterns "
        "with high specificity. It does not attempt to classify all epileptic seizure types. "
        "All results require expert clinical review and, when appropriate, video-EEG and "
        "full clinical context."
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
    High-confidence PNES pattern detector.

    Returns a ScoringResult with one of three classifications:
      - "High-confidence PNES pattern"
      - "Not confidently PNES"
      - "Insufficient / poor-quality video"
    """
    if weights is None:
        weights = Weights()

    reliability_score = _compute_reliability(video_data)
    reliability       = _reliability_label(reliability_score)

    # ── Step 1: Insufficient video check ─────────────────────────────────────
    if reliability_score < RELIABILITY_FLOOR:
        result = ScoringResult(
            pnes_pattern_confidence=0.0,
            pnes_score=0.0,
            es_score=0.0,
            classification="Insufficient / poor-quality video",
            confidence="low",
            reliability=reliability,
            reliability_score=reliability_score,
            quality_penalties=video_data.quality_flags,
            gate_flags={},
        )
        result.explanation = _generate_explanation(result, features, video_data)
        return result

    # ── Step 2: Feature scores (for table display) ────────────────────────────
    feature_scores = {
        "Movement Stereotypy":        _feat_stereotypy(features,             weights.movement_stereotypy),
        "Distribution Stability":     _feat_distribution_stability(features, weights.temporal_distribution_stability),
        "Variance Drift (inverse)":   _feat_variance_drift(features,         weights.variance_drift),
        "Vector Change (inverse)":    _feat_vector_change(features,          weights.vector_change),
        "Amplitude Variability (inv)":_feat_amplitude_variability(features,  weights.amplitude_variability),
        "Rhythm Regularity":          _feat_rhythm(features,                 weights.rhythm_irregularity),
        "Eye State":                  _feat_eye_state(features,              weights.eye_feature),
        "Temporal Evolution (gate)":  _feat_temporal_evolution(features),
    }

    # ── Step 3: Continuous confidence score ───────────────────────────────────
    pnes_confidence = _pnes_confidence_score(features, weights)

    # ── Step 4: Gate-based classification ────────────────────────────────────
    gate_flags   = _compute_gate_flags(features)
    any_gate_triggered = any(gate_flags.values())

    if not any_gate_triggered and pnes_confidence >= 5.5:
        classification = "High-confidence PNES pattern"
        # Confidence in the positive call
        n_gates_clear = sum(1 for v in gate_flags.values() if not v)
        if pnes_confidence >= 7.5 and n_gates_clear == len(gate_flags):
            confidence = "high"
        elif pnes_confidence >= 6.0:
            confidence = "moderate"
        else:
            confidence = "low"
    else:
        classification = "Not confidently PNES"
        confidence     = "low"

    result = ScoringResult(
        pnes_pattern_confidence=pnes_confidence,
        pnes_score=pnes_confidence,        # alias for backward compat
        es_score=0.0,
        classification=classification,
        confidence=confidence,
        reliability=reliability,
        reliability_score=reliability_score,
        feature_scores=feature_scores,
        quality_penalties=video_data.quality_flags,
        gate_flags=gate_flags,
    )
    result.explanation = _generate_explanation(result, features, video_data)
    return result
