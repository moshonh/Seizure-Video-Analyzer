"""
app.py
Seizure Video Analyzer – Research & Educational Prototype
Based on the Global Dynamic Impression (GDI) framework.
NOT a medical device. Does not provide a definitive diagnosis.
"""

import os
import io
import json
import math
import tempfile
import traceback
from dataclasses import asdict, is_dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from video_processing import extract_video_data
from motion_features import extract_features
from scoring import compute_scores, Weights
from reporting import generate_html_report
from utils import (
    save_uploaded_file,
    is_valid_video,
    format_duration,
    classification_emoji,
    confidence_colour,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config – must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Seizure Video Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .disclaimer-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.95rem;
        margin-bottom: 16px;
    }
    .soft-info-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.93rem;
        margin-bottom: 12px;
    }
    .result-card-pnes {
        border: 3px solid #f59e0b;
        border-radius: 10px;
        padding: 20px;
        background: #fffbeb;
    }
    .result-card-es {
        border: 3px solid #3b82f6;
        border-radius: 10px;
        padding: 20px;
        background: #eff6ff;
    }
    .result-card-indeterminate {
        border: 3px solid #6b7280;
        border-radius: 10px;
        padding: 20px;
        background: #f9fafb;
    }
    .explanation-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.98rem;
    }
    .warning-box {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.93rem;
        margin-bottom: 10px;
    }
    h1 { margin-bottom: 0px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return default if value is None else int(value)
    except Exception:
        return default


def label_for_display(classification: str) -> str:
    return {
        "PNES": "More consistent with PNES",
        "ES": "More consistent with ES",
        "Indeterminate": "Indeterminate / insufficient data",
    }.get(classification, "Indeterminate / insufficient data")


def card_class_for_result(classification: str) -> str:
    return {
        "PNES": "result-card-pnes",
        "ES": "result-card-es",
        "Indeterminate": "result-card-indeterminate",
    }.get(classification, "result-card-indeterminate")


def normalize_text_bool(value: bool) -> str:
    return "Yes" if value else "No"


def build_video_warnings(video_data) -> list:
    warnings = []
    metadata = getattr(video_data, "metadata", None)
    if metadata is None:
        warnings.append("Video metadata could not be fully read. Reliability may be reduced.")
        return warnings

    duration = safe_float(getattr(metadata, "duration_seconds", None))
    fps = safe_float(getattr(metadata, "fps", None))
    width = safe_int(getattr(metadata, "width", None))
    height = safe_int(getattr(metadata, "height", None))

    if 0 < duration < 8:
        warnings.append("Very short video: limited event sampling may reduce reliability.")
    elif 0 < duration < 15:
        warnings.append("Short video: incomplete evolution may reduce reliability.")
    if 0 < fps < 10:
        warnings.append("Low frame rate detected: motion dynamics and rhythm estimates may be less reliable.")
    if width and height and (width < 320 or height < 240):
        warnings.append("Low resolution detected: subtle motor or eye features may be missed.")

    return warnings


def object_to_dict(obj):
    """Recursively convert dataclasses / numpy types to JSON-safe primitives."""
    def convert(value):
        if is_dataclass(value):
            return {k: convert(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, float):
            return None if (math.isnan(value) or math.isinf(value)) else value
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if hasattr(value, "__dict__"):
            return {k: convert(v) for k, v in vars(value).items() if not k.startswith("_")}
        return str(value)
    return convert(obj)


def build_json_export(result, features, video_data, uploaded_filename, manual_annotations, warnings) -> str:
    payload = {
        "video_filename": uploaded_filename,
        "result": object_to_dict(result),
        "features": object_to_dict(features),
        "video_metadata": object_to_dict(getattr(video_data, "metadata", {})),
        "manual_annotations": manual_annotations,
        "warnings": warnings,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_feature_dataframe(result) -> pd.DataFrame:
    feature_scores = getattr(result, "feature_scores", {}) or {}
    rows = [
        {
            "Feature": name,
            "PNES score": safe_float(scores.get("pnes", 0.0)),
            "ES score": safe_float(scores.get("es", 0.0)),
            "Interpretation": scores.get("label", ""),
            "Detail": scores.get("detail", ""),
        }
        for name, scores in feature_scores.items()
    ]
    return pd.DataFrame(rows)


def apply_manual_annotation_notes(base_explanation: str, manual_annotations: dict) -> str:
    notes = []
    if not manual_annotations.get("onset_visible"):
        notes.append("The visible seizure onset was not confirmed, which reduces interpretive confidence.")
    if not manual_annotations.get("offset_visible"):
        notes.append("The visible seizure offset was not confirmed, limiting temporal evolution assessment.")
    if not manual_annotations.get("face_clearly_visible"):
        notes.append("The face was not clearly visible, so facial/eye-based interpretation is limited.")
    if not manual_annotations.get("eyes_reliably_visible"):
        notes.append("Eye state was not reliably visible during enough of the event.")
    if manual_annotations.get("free_text_notes"):
        notes.append(f"Expert note: {manual_annotations['free_text_notes']}")
    return (base_explanation.strip() + " " + " ".join(notes)).strip() if notes else base_explanation


# ─────────────────────────────────────────────────────────────────────────────
# Header & Disclaimers
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Seizure Video Analyzer")
st.markdown("**Research & Educational Prototype** · GDI-based assessment for motor seizure videos")

st.markdown("""
<div class="disclaimer-box">
<b>⚠️ Important Notice</b><br>
This application is a <b>research and educational prototype</b> for motor seizure video assessment.
It is <b>not a medical device</b> and does <b>not provide a definitive diagnosis</b>.
Final clinical interpretation requires expert review and, when appropriate, video-EEG and full clinical context.<br><br>
• Applicable mainly to <b>motor events</b> &nbsp;|&nbsp;
• Not validated for unsupervised clinical use &nbsp;|&nbsp;
• Do <b>not</b> use as sole basis for treatment decisions
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="soft-info-box">
This version prioritises <b>transparency</b> and <b>interpretability</b>.
It provides a GDI-based supportive assessment using motion features from video and optional manual expert annotations.
When evidence is mixed, incomplete, or the video is limited, the preferred output is <b>Indeterminate / insufficient data</b>.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.caption("Adjust feature weights. Evolution and stability-over-time features should usually matter more than burst shape alone.")

    w_escalation  = st.slider("Temporal Escalation",   0.5, 3.0, 2.0, 0.1,
                              help="Secondary feature: steady increase over time gives mild ES support.")
    w_isolation   = st.slider("Burst Isolation",        0.5, 3.0, 1.8, 0.1,
                              help="Secondary feature only: a focal burst can mildly support PNES but should not dominate the decision.")
    w_sust_var    = st.slider("Sustained Variability",  0.5, 3.0, 1.6, 0.1,
                              help="Secondary supportive feature: persistent variability during active periods gives mild ES support.")
    w_active      = st.slider("Active Fraction",        0.5, 3.0, 1.4, 0.1,
                              help="Secondary feature: prolonged active phase gives only mild ES support.")
    w_baseline    = st.slider("Baseline Fraction",      0.5, 3.0, 1.4, 0.1,
                              help="Secondary feature: long quiet periods mildly support PNES.")
    w_postquiet   = st.slider("Post-burst Quiet",       0.5, 3.0, 1.2, 0.1,
                              help="Secondary feature: return toward baseline mildly supports PNES.")
    w_direction   = st.slider("Direction Variability",  0.0, 2.0, 0.8, 0.1,
                              help="Primary feature: changing dominant movement direction supports ES.")
    w_rhythm      = st.slider("Rhythm Irregularity",    0.0, 2.0, 1.0, 0.1,
                              help="Primary feature: rhythm irregularity supports ES, but irregularity alone is not enough.")
    w_temp_evol   = st.slider("Temporal Evolution",     0.0, 2.0, 1.0, 0.1,
                              help="Primary feature: changing rhythm, direction, and amplitude across time supports ES.")
    w_regional    = st.slider("Regional Asynchrony",    0.0, 2.0, 0.6, 0.1,
                              help="Secondary feature: different ROI bands moving differently gives mild ES support.")
    w_eye         = st.slider("Eye Feature Weight",     0.0, 2.0, 0.8, 0.1)

    st.divider()
    st.subheader("👁 Manual Eye Annotation")
    st.caption(
        "Eye state is not automatically detected in this version. "
        "Use manual expert annotation only if visibility is adequate."
    )
    eye_detection_mode = st.selectbox(
        "Manual eye-state annotation",
        [
            "Unavailable / not reliable",
            "Eyes closed (manual annotation)",
            "Eyes open (manual annotation)",
        ],
        help="Use only if the eyes are sufficiently visible during the event.",
    )

    st.divider()
    st.subheader("🛠 Advanced")
    max_frames = st.slider("Maximum frames to analyse", 100, 600, 450, 25)
    show_technical_details = st.checkbox("Show technical details", value=False)

    st.divider()
    st.caption(
        "**About**\n"
        "This prototype implements a GDI-based framework for distinguishing ES from PNES "
        "using video-derived motor features."
    )

eye_map = {
    "Unavailable / not reliable":       "unavailable",
    "Eyes closed (manual annotation)":  "closed",
    "Eyes open (manual annotation)":    "open",
}
eye_state = eye_map[eye_detection_mode]

weights = Weights(
    temporal_escalation=w_escalation,
    burst_isolation=w_isolation,
    sustained_variability=w_sust_var,
    active_fraction=w_active,
    baseline_fraction=w_baseline,
    post_burst_quiet=w_postquiet,
    direction_variability=w_direction,
    rhythm_irregularity=w_rhythm,
    temporal_evolution=w_temp_evol,
    regional_asynchrony=w_regional,
    eye_feature=w_eye,
)

# ─────────────────────────────────────────────────────────────────────────────
# Upload & How It Works
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Upload a seizure video",
        type=["mp4", "mov", "avi", "mkv"],
        help="For best results, use a video with visible motor activity and as much event evolution as possible.",
    )
    if uploaded_file:
        if not is_valid_video(uploaded_file):
            st.error("Unsupported file format. Please upload MP4, MOV, AVI, or MKV.")
        else:
            st.video(uploaded_file)
            st.caption(f"📁 {uploaded_file.name} · {uploaded_file.size / 1024:.1f} KB")

with right_col:
    st.subheader("ℹ️ How It Works")
    st.markdown("""
This prototype extracts motion features from the uploaded video and applies an interpretable GDI-based scoring approach.

| Feature | May favour PNES | May favour ES |
|---------|-----------------|---------------|
| Distribution stability | Similar shape/statistics over time | Shape/statistics drift across windows |
| Movement stereotypy | Repetitive / internally similar | Less similar / transforming |
| Movement direction | More fixed over time | More changing over time |
| Rhythm / frequency | More stable | More variable or drifting |
| Temporal evolution | Limited change | Dynamic progression |
| Eye state* | Closed | Open |

&#42; Eye state is **manual expert annotation** in this version, not automatic detection.
""")

with st.expander("Clinical rationale"):
    st.markdown("""
The app is based on the **Global Dynamic Impression (GDI)** principle:
motor PNES may still look noisy or irregular, but often remains the same kind of noisy over time,
while motor ES is more likely to show genuine change in direction, rhythm, amplitude, distribution, or progressive variance drift across the clip.

These are supportive features, **not absolute diagnostic rules**.
Mixed videos, poor-quality videos, partial recordings, or non-motor events should be interpreted cautiously.
""")

# ─────────────────────────────────────────────────────────────────────────────
# ROI Selection (slider-based, shown only when a video is uploaded)
# ─────────────────────────────────────────────────────────────────────────────
roi = None  # default – no ROI until video is uploaded

if uploaded_file and is_valid_video(uploaded_file):
    st.divider()
    st.subheader("🎯 ROI Selection")
    st.markdown(
        "Use the sliders to position a rectangle around the patient. "
        "The red rectangle preview updates immediately. "
        "The selected ROI will be used for the entire analysis."
    )

    # Load first frame from the uploaded file
    _tmp_suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_tmp_suffix)
    _tmp.write(uploaded_file.getbuffer())
    _tmp.flush()
    _tmp_path_roi = _tmp.name
    _tmp.close()

    _cap = cv2.VideoCapture(_tmp_path_roi)
    _roi_ok, _first_frame = _cap.read()
    _cap.release()
    try:
        os.unlink(_tmp_path_roi)
    except OSError:
        pass

    if not _roi_ok or _first_frame is None:
        st.warning("Could not read the first frame for ROI preview. Full frame will be used.")
        roi = None  # will be resolved in the analysis block
    else:
        _fh, _fw = _first_frame.shape[:2]

        _slider_col, _preview_col = st.columns([1, 2], gap="large")
        with _slider_col:
            _roi_x = st.slider("ROI X (left edge)",  0, max(0, _fw - 10), 0,    step=1, key="roi_x")
            _roi_y = st.slider("ROI Y (top edge)",   0, max(0, _fh - 10), 0,    step=1, key="roi_y")
            _roi_w = st.slider("ROI Width",          10, _fw,             _fw,  step=1, key="roi_w")
            _roi_h = st.slider("ROI Height",         10, _fh,             _fh,  step=1, key="roi_h")

            _x2 = min(_roi_x + _roi_w, _fw)
            _y2 = min(_roi_y + _roi_h, _fh)
            _w_clamped = _x2 - _roi_x
            _h_clamped = _y2 - _roi_y
            st.caption(f"x1={_roi_x}, y1={_roi_y}, x2={_x2}, y2={_y2} · {_w_clamped}×{_h_clamped} px")

        with _preview_col:
            _preview = _first_frame.copy()
            cv2.rectangle(_preview, (_roi_x, _roi_y), (_x2, _y2), (220, 38, 38), 3)
            _preview_rgb = cv2.cvtColor(_preview, cv2.COLOR_BGR2RGB)
            _preview_pil = Image.fromarray(_preview_rgb)
            _disp_w = min(700, _fw)
            _disp_scale = _disp_w / _fw
            _preview_resized = _preview_pil.resize(
                (_disp_w, int(_fh * _disp_scale)), Image.LANCZOS
            )
            st.image(_preview_resized, caption="ROI preview (red rectangle)",
                     use_container_width=False)

        if _w_clamped >= 10 and _h_clamped >= 10:
            roi = (_roi_x, _roi_y, _x2, _y2)
            st.success(
                f"✅ ROI ready — x1={_roi_x}, y1={_roi_y}, x2={_x2}, y2={_y2} "
                f"({_w_clamped}×{_h_clamped} px)"
            )
        else:
            st.warning("ROI too small — adjust sliders so Width and Height are at least 10 px.")

# ─────────────────────────────────────────────────────────────────────────────
# Manual Expert Annotations
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Manual Expert Override / Annotation"):
    ma1, ma2 = st.columns(2)
    with ma1:
        onset_visible       = st.checkbox("Visible event onset captured", value=True)
        offset_visible      = st.checkbox("Visible event offset captured", value=True)
        face_clearly_visible = st.checkbox("Face clearly visible",         value=False)
    with ma2:
        eyes_reliably_visible = st.checkbox("Eyes reliably visible",       value=False)
        use_manual_note       = st.checkbox("Add expert free-text note",   value=False)

    free_text_notes = ""
    if use_manual_note:
        free_text_notes = st.text_area(
            "Expert notes",
            placeholder="Example: apparent asymmetric arm involvement; eye visibility brief; onset partially off-camera",
            height=100,
        )

manual_annotations = {
    "onset_visible":         onset_visible,
    "offset_visible":        offset_visible,
    "face_clearly_visible":  face_clearly_visible,
    "eyes_reliably_visible": eyes_reliably_visible,
    "manual_eye_state":      eye_state,
    "free_text_notes":       free_text_notes.strip(),
}

# ─────────────────────────────────────────────────────────────────────────────
# Analyse button
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
analyze_clicked = st.button(
    "🔍 Analyze Video",
    type="primary",
    disabled=(uploaded_file is None or roi is None),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file and analyze_clicked:
    tmp_path = None
    try:
        with st.spinner("Saving uploaded video…"):
            tmp_path = save_uploaded_file(uploaded_file)

        progress = st.progress(0, text="Reading video…")
        video_data = extract_video_data(tmp_path, max_frames=max_frames, roi=roi)

        if video_data is None:
            st.error(
                "❌ The video could not be processed. "
                "It may be corrupt, too short, or encoded with an unsupported codec."
            )
            st.stop()

        metadata = getattr(video_data, "metadata", None)
        pre_warnings = build_video_warnings(video_data)

        progress.progress(35, text="Extracting motion features…")
        features = extract_features(video_data, eye_state=eye_state)

        progress.progress(65, text="Computing GDI-based scores…")
        result = compute_scores(features, video_data, weights=weights)

        progress.progress(82, text="Preparing explanations and exports…")

        # Enrich explanation with manual annotation notes
        result.explanation = apply_manual_annotation_notes(result.explanation, manual_annotations)

        if eye_state != "unavailable" and not manual_annotations["eyes_reliably_visible"]:
            result.explanation += (
                " Manual eye-state annotation was provided despite limited confirmed eye visibility; "
                "this should be interpreted with caution."
            )
        if eye_state != "unavailable" and not manual_annotations["face_clearly_visible"]:
            result.explanation += (
                " Because the face was not clearly visible, any eye-based support should be considered weak."
            )

        html_report = generate_html_report(result, features, video_filename=uploaded_file.name)
        json_export = build_json_export(
            result=result,
            features=features,
            video_data=video_data,
            uploaded_filename=uploaded_file.name,
            manual_annotations=manual_annotations,
            warnings=pre_warnings,
        )

        progress.progress(100, text="Done.")
        progress.empty()

        # ─────────────────────────────────────────────────────────────────
        # Results section
        # ─────────────────────────────────────────────────────────────────
        st.divider()
        st.header("📊 Analysis Results")

        raw_classification = getattr(result, "classification", "Indeterminate")
        display_label = label_for_display(raw_classification)
        card_class    = card_class_for_result(raw_classification)
        emoji         = classification_emoji(raw_classification)
        confidence    = getattr(result, "confidence", "low")
        reliability   = getattr(result, "reliability", "limited")
        conf_color    = confidence_colour(confidence)
        pnes_score    = safe_float(getattr(result, "pnes_score", 0.0))
        es_score      = safe_float(getattr(result, "es_score", 0.0))

        st.markdown(f"""
<div class="{card_class}">
  <h2 style="margin:0">{emoji} {display_label}</h2>
  <p style="margin:8px 0 0 0">
    Confidence: <b style="color:{conf_color}">{str(confidence).upper()}</b>
    &nbsp;|&nbsp;
    Reliability: <b>{str(reliability).upper()}</b>
    &nbsp;|&nbsp;
    PNES evidence: <b>{pnes_score:.1f}/10</b>
    &nbsp;|&nbsp;
    ES evidence: <b>{es_score:.1f}/10</b>
  </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            f'<div class="explanation-box">'
            f'{getattr(result, "explanation", "No explanation available.")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if pre_warnings:
            st.markdown("### ⚠️ Video limitations")
            for warning in pre_warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)

        # ── Video info ────────────────────────────────────────────────────
        with st.expander("📹 Video information", expanded=False):
            if metadata is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Duration",   format_duration(safe_float(getattr(metadata, "duration_seconds", 0.0))))
                c2.metric("FPS",        f"{safe_float(getattr(metadata, 'fps', 0.0)):.1f}")
                c3.metric("Resolution", f"{safe_int(getattr(metadata, 'width', 0))}×{safe_int(getattr(metadata, 'height', 0))}")
                c4.metric("Frames analysed", len(getattr(features, "motion_signal", [])))
            else:
                st.info("Video metadata not available.")

            st.markdown("#### Manual expert annotations")
            ann_df = pd.DataFrame([
                {"Field": "Visible onset captured",    "Value": normalize_text_bool(manual_annotations["onset_visible"])},
                {"Field": "Visible offset captured",   "Value": normalize_text_bool(manual_annotations["offset_visible"])},
                {"Field": "Face clearly visible",      "Value": normalize_text_bool(manual_annotations["face_clearly_visible"])},
                {"Field": "Eyes reliably visible",     "Value": normalize_text_bool(manual_annotations["eyes_reliably_visible"])},
                {"Field": "Manual eye-state",          "Value": manual_annotations["manual_eye_state"]},
                {"Field": "Free-text note",            "Value": manual_annotations["free_text_notes"] or "—"},
            ])
            st.dataframe(ann_df, use_container_width=True, hide_index=True)

        # ── Motion graphs ─────────────────────────────────────────────────
        st.subheader("📈 Motion graphs")
        timestamps        = np.asarray(getattr(features, "timestamps",        []))
        motion_signal     = np.asarray(getattr(features, "motion_signal",     []))
        variability_signal = np.asarray(getattr(features, "variability_signal", []))

        gc1, gc2 = st.columns(2)
        with gc1:
            if timestamps.size > 0 and motion_signal.size > 0:
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.plot(timestamps, motion_signal, linewidth=1.4, color="#2563eb")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Motion magnitude")
                ax.set_title("Optical Flow Magnitude Over Time")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Motion magnitude signal unavailable.")

        with gc2:
            if timestamps.size > 0 and variability_signal.size > 0:
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.plot(timestamps, variability_signal, linewidth=1.4, color="#dc2626")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Local variability")
                ax.set_title("Rolling Motion Variability")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Variability signal unavailable.")

        # ── Numeric metrics ──────────────────────────────────────────────────
        st.subheader("📐 Numeric metrics")
        _f = features   # shorthand
        nm1, nm2, nm3, nm4 = st.columns(4)
        nm1.metric("Motion Mean",        f"{_f.mean_motion:.4f}")
        nm2.metric("Variability Mean",   f"{float(np.mean(_f.variability_signal)):.4f}")
        nm3.metric("Motion Derivative",  f"{_f.motion_derivative:.4f}")
        nm4.metric("Dynamic Index",      f"{_f.dynamic_index:.4f}")

        nm5, nm6, nm7 = st.columns(3)
        nm5.metric("Stability Index",    f"{_f.stability_index:.2f}")
        nm6.metric("Rhythm Irregularity",f"{_f.rhythm_irregularity:.3f}",
                   help="CV of inter-peak intervals. Higher values support ES, but irregularity alone is not enough.")
        nm7.metric("Temporal Evolution", f"{_f.temporal_evolution:.3f}",
                   help="Composite change in variability, amplitude, and direction across time. Higher = more evolving = ES.")

        nm8, nm9, nm10 = st.columns(3)
        nm8.metric("Distribution Stability", f"{_f.temporal_distribution_stability:.3f}",
                   help="How similar the normalised motion/variability distributions remain across windows. Higher = same kind of pattern over time = PNES.")
        nm9.metric("Variance Drift", f"{_f.variance_drift:.3f}",
                   help="Progressive increase in amplitude and/or local variance across time windows. Higher = non-stationary growth = ES.")
        nm10.metric("Regional Asynchrony",f"{_f.regional_asynchrony:.4f}",
                   help="Mean std of motion across upper/mid/lower bands. High = asynchronous = mild ES.")

        # ── Feature table ─────────────────────────────────────────────────
        st.subheader("📋 Feature summary")
        df_features = build_feature_dataframe(result)
        if not df_features.empty:
            st.dataframe(df_features, use_container_width=True, hide_index=True)
        else:
            st.info("Feature-level scoring details were not available.")

        # ── Score bar chart ───────────────────────────────────────────────
        st.subheader("📊 Feature score comparison")
        if not df_features.empty:
            names     = df_features["Feature"].tolist()
            pnes_vals = df_features["PNES score"].tolist()
            es_vals   = df_features["ES score"].tolist()

            x     = np.arange(len(names))
            bwidth = 0.36
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.bar(x - bwidth / 2, pnes_vals, bwidth, label="PNES", color="#f59e0b", alpha=0.85)
            ax.bar(x + bwidth / 2, es_vals,   bwidth, label="ES",   color="#3b82f6", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
            ax.set_ylabel("Weighted score")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Feature comparison chart unavailable.")

        # ── Quality flags ─────────────────────────────────────────────────
        st.subheader("🔍 Video quality assessment")
        qf = getattr(result, "quality_penalties", {}) or {}
        if qf:
            qcols = st.columns(len(qf))
            for i, (flag, triggered) in enumerate(qf.items()):
                icon  = "⚠️" if triggered else "✅"
                label = str(flag).replace("_", " ").title()
                qcols[i].metric(label, icon)
        else:
            st.info("No structured quality flags were generated by the scoring module.")

        # ── Technical details ─────────────────────────────────────────────
        if show_technical_details:
            with st.expander("🧪 Technical details", expanded=False):
                st.markdown("**Raw result object**")
                st.json(object_to_dict(result))
                st.markdown("**Raw feature object**")
                st.json(object_to_dict(features))

        # ── Downloads ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("⬇️ Download outputs")
        dl1, dl2 = st.columns(2)

        stem = uploaded_file.name.rsplit(".", 1)[0]
        with dl1:
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report.encode("utf-8"),
                file_name=f"seizure_analysis_{stem}.html",
                mime="text/html",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                label="🧾 Download JSON Features",
                data=json_export.encode("utf-8"),
                file_name=f"seizure_analysis_{stem}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption(
            "The exported files are intended for research documentation and review. "
            "All outputs require expert clinical interpretation."
        )

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

elif not uploaded_file:
    st.info("👆 Upload a video above to begin analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Seizure Video Analyzer · Research Prototype · Not a Medical Device · "
    "GDI-based supportive assessment for motor seizure videos · "
    "All outputs require expert clinical review."
)
