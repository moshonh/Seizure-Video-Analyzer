"""
app.py
Main Streamlit application for the Seizure Video Analyzer.
Research and educational prototype – NOT a medical device.
"""

import os
import json
import math
import traceback
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

st.set_page_config(
    page_title="Seizure Video Analyzer",
    page_icon="🧠",
    layout="wide",
)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def label_for_display(classification: str) -> str:
    mapping = {
        "PNES": "More consistent with PNES",
        "ES": "More consistent with ES",
        "Indeterminate": "Indeterminate / insufficient data",
    }
    return mapping.get(classification, "Indeterminate / insufficient data")


def card_class_for_result(classification: str) -> str:
    mapping = {
        "PNES": "result-card-pnes",
        "ES": "result-card-es",
        "Indeterminate": "result-card-indeterminate",
    }
    return mapping.get(classification, "result-card-indeterminate")


def result_emoji(classification: str) -> str:
    return classification_emoji(classification)


def normalize_text_bool(value: bool) -> str:
    return "Yes" if value else "No"


def build_video_warnings(video_data):
    warnings = []
    metadata = getattr(video_data, "metadata", None)

    if metadata is None:
        warnings.append("Video metadata could not be fully read. Reliability may be reduced.")
        return warnings

    duration = safe_float(getattr(metadata, "duration_seconds", None), 0.0)
    fps = safe_float(getattr(metadata, "fps", None), 0.0)
    width = safe_int(getattr(metadata, "width", None), 0)
    height = safe_int(getattr(metadata, "height", None), 0)

    if duration and duration < 8:
        warnings.append("Very short video: limited event sampling may reduce reliability.")
    elif duration and duration < 15:
        warnings.append("Short video: incomplete evolution may reduce reliability.")

    if fps and fps < 10:
        warnings.append("Low frame rate detected: motion dynamics and rhythm estimates may be less reliable.")

    if width and height and (width < 320 or height < 240):
        warnings.append("Low resolution detected: subtle motor or eye features may be missed.")

    return warnings


def object_to_dict(obj):
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

        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value

        if hasattr(value, "__dict__"):
            return {k: convert(v) for k, v in vars(value).items() if not k.startswith("_")}

        return str(value)

    return convert(obj)


def build_json_export(result, features, video_data, uploaded_filename, manual_annotations, warnings):
    payload = {
        "video_filename": uploaded_filename,
        "result": object_to_dict(result),
        "features": object_to_dict(features),
        "video_metadata": object_to_dict(getattr(video_data, "metadata", {})),
        "manual_annotations": manual_annotations,
        "warnings": warnings,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_feature_dataframe(result):
    feature_scores = getattr(result, "feature_scores", {}) or {}
    rows = []
    for name, scores in feature_scores.items():
        rows.append(
            {
                "Feature": name,
                "PNES score": safe_float(scores.get("pnes", 0.0)),
                "ES score": safe_float(scores.get("es", 0.0)),
                "Interpretation": scores.get("label", ""),
                "Detail": scores.get("detail", ""),
            }
        )
    return pd.DataFrame(rows)


def apply_manual_annotation_notes(base_explanation: str, manual_annotations: dict) -> str:
    notes = []

    if manual_annotations.get("onset_visible") is False:
        notes.append("The visible seizure onset was not confirmed, which reduces interpretive confidence.")

    if manual_annotations.get("offset_visible") is False:
        notes.append("The visible seizure offset was not confirmed, which reduces confidence in temporal evolution assessment.")

    if manual_annotations.get("face_clearly_visible") is False:
        notes.append("The face was not clearly visible, so facial/eye-based interpretation is limited.")

    if manual_annotations.get("eyes_reliably_visible") is False:
        notes.append("Eye state was not reliably visible during enough of the event.")

    if manual_annotations.get("free_text_notes"):
        notes.append(f"Expert note: {manual_annotations['free_text_notes']}")

    if notes:
        return base_explanation.strip() + " " + " ".join(notes)

    return base_explanation


st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.markdown("# 🧠 Seizure Video Analyzer")
st.markdown("**Research & Educational Prototype** · GDI-based assessment for motor seizure videos")

st.markdown(
    """
<div class="disclaimer-box">
<b>⚠️ Important Notice</b><br>
This application is a <b>research and educational prototype</b> for motor seizure video assessment.
It is <b>not a medical device</b> and does <b>not provide a definitive diagnosis</b>.
Final clinical interpretation requires expert review and, when appropriate, video-EEG and full clinical context.<br><br>
• Applicable mainly to <b>motor events</b> &nbsp;|&nbsp;
• Not validated for unsupervised clinical use &nbsp;|&nbsp;
• Do <b>not</b> use as sole basis for treatment decisions
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="soft-info-box">
This version prioritizes <b>transparency</b> and <b>interpretability</b>.
It provides a GDI-based supportive assessment using motion features from video and optional manual expert annotations.
When evidence is mixed, incomplete, or the video is limited, the preferred output is <b>Indeterminate / insufficient data</b>.
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.caption("Adjust feature weights. Higher values increase influence on the final score.")

    w_motion = st.slider("Motor Dynamism", 0.5, 3.0, 1.5, 0.1)
    w_direction = st.slider("Direction Variability", 0.5, 3.0, 1.2, 0.1)
    w_frequency = st.slider("Frequency Variability", 0.5, 3.0, 1.0, 0.1)
    w_temporal = st.slider("Temporal Evolution", 0.5, 3.0, 1.2, 0.1)
    w_burst = st.slider("Burst Pattern", 0.5, 3.0, 1.0, 0.1)
    w_eye = st.slider("Eye Feature Weight", 0.0, 2.0, 0.8, 0.1)

    st.divider()
    st.subheader("👁 Manual Eye Annotation")
    st.caption("Eye state is not automatically detected in this version. Use manual expert annotation only if visibility is adequate.")

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
    max_frames = st.slider("Maximum frames to analyse", 100, 600, 300, 25)
    show_technical_details = st.checkbox("Show technical details", value=False)

    st.divider()
    st.caption(
        "**About**\n"
        "This prototype implements a Global Dynamic Impression (GDI)-based framework for distinguishing ES from PNES using video-derived motor features."
    )

eye_map = {
    "Unavailable / not reliable": "unavailable",
    "Eyes closed (manual annotation)": "closed",
    "Eyes open (manual annotation)": "open",
}
eye_state = eye_map[eye_detection_mode]

weights = Weights(
    motor_dynamism=w_motion,
    direction_variability=w_direction,
    frequency_variability=w_frequency,
    temporal_evolution=w_temporal,
    burst_pattern=w_burst,
    eye_feature=w_eye,
)

st.divider()
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Upload a seizure video",
        type=["mp4", "mov", "avi", "mkv"],
        help="For best results, use a video with visible motor activity and as much of the event evolution as possible.",
    )

    if uploaded_file:
        if not is_valid_video(uploaded_file):
            st.error("Unsupported file format. Please upload MP4, MOV, AVI, or MKV.")
        else:
            st.video(uploaded_file)
            st.caption(f"📁 {uploaded_file.name} · {uploaded_file.size / 1024:.1f} KB")

with right_col:
    st.subheader("ℹ️ How It Works")
    st.markdown(
        """
This prototype extracts motion features from the uploaded video and applies an interpretable GDI-based scoring approach.

| Feature | Pattern that may favor PNES | Pattern that may favor ES |
|---------|------------------------------|---------------------------|
| Motor dynamism | Lower / more fixed | More evolving |
| Movement direction | Repetitive / relatively fixed | More changing |
| Rhythm / frequency | More constant | More variable |
| Temporal pattern | Burst–arrest–burst repetition | Fade-in / fade-out evolution |
| Eye state* | Closed may support PNES | Open may support ES |

\* In this version, eye state is **manual expert annotation**, not automatic detection.
"""
    )

with st.expander("Clinical rationale"):
    st.markdown(
        """
The app is based on the Global Dynamic Impression (GDI) idea:
motor PNES often appears more repetitive and less dynamically evolving over time, while motor ES may show greater change in movement direction, rhythm, and temporal evolution.

These are supportive features, not absolute diagnostic rules.
Mixed videos, poor-quality videos, partial recordings, or non-motor events should be interpreted cautiously.
"""
    )

with st.expander("Manual Expert Override / Annotation"):
    ma1, ma2 = st.columns(2)

    with ma1:
        onset_visible = st.checkbox("Visible event onset captured", value=True)
        offset_visible = st.checkbox("Visible event offset captured", value=True)
        face_clearly_visible = st.checkbox("Face clearly visible", value=False)

    with ma2:
        eyes_reliably_visible = st.checkbox("Eyes reliably visible", value=False)
        use_manual_note = st.checkbox("Add expert free-text note", value=False)

    free_text_notes = ""
    if use_manual_note:
        free_text_notes = st.text_area(
            "Expert notes",
            placeholder="Example: apparent asymmetric arm involvement; eye visibility brief; onset partially off-camera",
            height=100,
        )

manual_annotations = {
    "onset_visible": onset_visible,
    "offset_visible": offset_visible,
    "face_clearly_visible": face_clearly_visible,
    "eyes_reliably_visible": eyes_reliably_visible,
    "manual_eye_state": eye_state,
    "free_text_notes": free_text_notes.strip(),
}

st.divider()
analyze_clicked = st.button(
    "🔍 Analyze Video",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True,
)

if uploaded_file and analyze_clicked:
    tmp_path = None

    try:
        with st.spinner("Saving uploaded video..."):
            tmp_path = save_uploaded_file(uploaded_file)

        progress = st.progress(0, text="Reading video...")
        video_data = extract_video_data(tmp_path, max_frames=max_frames)

        if video_data is None:
            st.error(
                "❌ The video could not be processed. It may be corrupt, too short, or encoded with an unsupported codec."
            )
            st.stop()

        metadata = getattr(video_data, "metadata", None)
        pre_warnings = build_video_warnings(video_data)

        progress.progress(35, text="Extracting motion features...")
        features = extract_features(video_data, eye_state=eye_state)

        progress.progress(65, text="Computing GDI-based scores...")
        result = compute_scores(features, video_data, weights=weights)

        progress.progress(82, text="Preparing explanations and exports...")

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

        st.divider()
        st.header("📊 Analysis Results")

        raw_classification = getattr(result, "classification", "Indeterminate")
        display_label = label_for_display(raw_classification)
        card_class = card_class_for_result(raw_classification)
        emoji = result_emoji(raw_classification)
        confidence = getattr(result, "confidence", "low")
        reliability = getattr(result, "reliability", "limited")
        conf_color = confidence_colour(confidence)

        pnes_score = safe_float(getattr(result, "pnes_score", 0.0))
        es_score = safe_float(getattr(result, "es_score", 0.0))

        st.markdown(
            f"""
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
""",
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown(
            f'<div class="explanation-box">{getattr(result, "explanation", "No explanation available.")}</div>',
            unsafe_allow_html=True,
        )

        if pre_warnings:
            st.markdown("### ⚠️ Video limitations")
            for warning in pre_warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)

        with st.expander("📹 Video information", expanded=False):
            if metadata is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Duration", format_duration(safe_float(getattr(metadata, "duration_seconds", 0.0))))
                c2.metric("FPS", f"{safe_float(getattr(metadata, 'fps', 0.0)):.1f}")
                c3.metric(
                    "Resolution",
                    f"{safe_int(getattr(metadata, 'width', 0))}×{safe_int(getattr(metadata, 'height', 0))}",
                )
                motion_signal = getattr(features, "motion_signal", [])
                c4.metric("Frames analysed", len(motion_signal))
            else:
                st.info("Video metadata not available.")

            st.markdown("#### Manual expert annotations")
            ann_df = pd.DataFrame(
                [
                    {"Field": "Visible onset captured", "Value": normalize_text_bool(manual_annotations["onset_visible"])},
                    {"Field": "Visible offset captured", "Value": normalize_text_bool(manual_annotations["offset_visible"])},
                    {"Field": "Face clearly visible", "Value": normalize_text_bool(manual_annotations["face_clearly_visible"])},
                    {"Field": "Eyes reliably visible", "Value": normalize_text_bool(manual_annotations["eyes_reliably_visible"])},
                    {"Field": "Manual eye-state annotation", "Value": manual_annotations["manual_eye_state"]},
                    {"Field": "Free-text note", "Value": manual_annotations["free_text_notes"] or "—"},
                ]
            )
            st.dataframe(ann_df, use_container_width=True, hide_index=True)

        st.subheader("📈 Motion graphs")
        gc1, gc2 = st.columns(2)

        timestamps = getattr(features, "timestamps", [])
        motion_signal = getattr(features, "motion_signal", [])
        variability_signal = getattr(features, "variability_signal", [])

        with gc1:
            if len(timestamps) > 0 and len(motion_signal) > 0:
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.plot(timestamps, motion_signal, linewidth=1.4)
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
            if len(timestamps) > 0 and len(variability_signal) > 0:
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.plot(timestamps, variability_signal, linewidth=1.4)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Local variability")
                ax.set_title("Rolling Motion Variability")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Variability signal unavailable.")

        st.subheader("📋 Feature summary")
        df_features = build_feature_dataframe(result)

        if not df_features.empty:
            st.dataframe(df_features, use_container_width=True, hide_index=True)
        else:
            st.info("Feature-level scoring details were not available.")

        st.subheader("📊 Feature score comparison")
        if not df_features.empty:
            names = df_features["Feature"].tolist()
            pnes_vals = df_features["PNES score"].tolist()
            es_vals = df_features["ES score"].tolist()

            x = np.arange(len(names))
            width = 0.36
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.bar(x - width / 2, pnes_vals, width, label="PNES", alpha=0.85)
            ax.bar(x + width / 2, es_vals, width, label="ES", alpha=0.85)
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

        st.subheader("🔍 Video quality assessment")
        qf = getattr(result, "quality_penalties", {}) or {}

        if qf:
            qcols = st.columns(len(qf))
            for i, (flag, triggered) in enumerate(qf.items()):
                icon = "⚠️" if triggered else "✅"
                label = str(flag).replace("_", " ").title()
                qcols[i].metric(label, icon)
        else:
            st.info("No structured quality flags were generated by the scoring module.")

        if show_technical_details:
            with st.expander("🧪 Technical details", expanded=False):
                st.markdown("**Raw result object**")
                st.json(object_to_dict(result))
                st.markdown("**Raw feature object**")
                st.json(object_to_dict(features))

        st.divider()
        st.subheader("⬇️ Download outputs")

        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report.encode("utf-8"),
                file_name=f"seizure_analysis_{uploaded_file.name.rsplit('.', 1)[0]}.html",
                mime="text/html",
                use_container_width=True,
            )

        with dl2:
            st.download_button(
                label="🧾 Download JSON Features",
                data=json_export.encode("utf-8"),
                file_name=f"seizure_analysis_{uploaded_file.name.rsplit('.', 1)[0]}.json",
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

st.divider()
st.caption(
    "Seizure Video Analyzer · Research Prototype · Not a Medical Device · "
    "GDI-based supportive assessment for motor seizure videos · "
    "All outputs require expert clinical review."
)