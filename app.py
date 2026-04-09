"""
app.py  –  Seizure Video Analyzer  (GDI Framework)
============================================================
Research & Educational Prototype — NOT a medical device.

Required packages:
    pip install streamlit opencv-python-headless numpy pandas matplotlib scipy streamlit-drawable-canvas Pillow

Run:
    streamlit run app.py
"""

# ════════════════════════════════════════════════════════════
#  REQUIRED PACKAGES
#  pip install streamlit opencv-python-headless numpy pandas
#              matplotlib scipy streamlit-drawable-canvas Pillow
# ════════════════════════════════════════════════════════════

import io
import os
import math
import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ════════════════════════════════════════════════════════════
#  THRESHOLD CONSTANTS  ← adjust here without touching logic
# ════════════════════════════════════════════════════════════

THRESHOLD_ES    = 0.7   # dynamic_index above this  → supports ES
THRESHOLD_PNES  = 0.4   # dynamic_index below this  → supports PNES
ROLLING_WINDOW  = 15    # frames for rolling std variability

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Seizure Video Analyzer · GDI Framework",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
    .block-container { padding-top: 2rem; max-width: 1180px; }

    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 1.2rem;
    }
    .result-card {
        border: 2px solid #d1cfc8;
        border-radius: 8px;
        padding: 18px 22px;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .result-es   { border-color: #1d4ed8; background: #eff6ff; }
    .result-pnes { border-color: #b45309; background: #fffbeb; }
    .result-ind  { border-color: #6b7280; background: #f9fafb; }
    .result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        margin: 0 0 6px 0;
    }
    .result-es   .result-label { color: #1d4ed8; }
    .result-pnes .result-label { color: #b45309; }
    .result-ind  .result-label { color: #6b7280; }
    .metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
    .metric-chip {
        background: #f3f4f6;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.81rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #6b7280;
        margin: 1.6rem 0 0.6rem 0;
        border-bottom: 1px solid #d1cfc8;
        padding-bottom: 4px;
    }
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 11px 15px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .warn-box {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        padding: 11px 15px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CORE ANALYSIS FUNCTIONS
# ════════════════════════════════════════════════════════════

def load_first_frame(video_path: str):
    """Return the first readable BGR frame, or None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def compute_motion_signal(video_path: str, roi: tuple) -> dict:
    """
    Compute per-frame optical flow magnitude over the ENTIRE video,
    restricted to the user-selected ROI. No frame limit is applied.

    Parameters
    ----------
    video_path : path to the video file
    roi        : (x1, y1, x2, y2) in original-frame pixel coordinates

    Returns
    -------
    dict with:
        motion_signal    – np.ndarray, mean optical flow magnitude per frame pair
        timestamps       – np.ndarray, time in seconds
        fps              – float
        total_frames     – int
        frames_processed – int
    """
    x1, y1, x2, y2 = roi

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")

    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion_signal: list = []
    timestamps:    list = []
    prev_gray          = None
    frame_idx          = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and gray.shape == prev_gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_signal.append(float(np.mean(mag)))
            timestamps.append(frame_idx / fps)

        prev_gray  = gray
        frame_idx += 1

    cap.release()

    if len(motion_signal) < 2:
        raise ValueError(
            f"Too few usable frames ({len(motion_signal)}). "
            "Check your ROI selection and video file."
        )

    return {
        "motion_signal":    np.array(motion_signal, dtype=float),
        "timestamps":       np.array(timestamps,    dtype=float),
        "fps":              fps,
        "total_frames":     total_frames,
        "frames_processed": len(motion_signal),
    }


def compute_rolling_variability(motion_signal: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Rolling standard deviation of motion_signal."""
    n      = len(motion_signal)
    result = np.zeros(n, dtype=float)
    half   = window // 2
    for i in range(n):
        lo        = max(0, i - half)
        hi        = min(n, i + half + 1)
        result[i] = float(np.std(motion_signal[lo:hi]))
    return result


def compute_gdi_indices(motion_signal: np.ndarray, variability_signal: np.ndarray) -> dict:
    """
    GDI Dynamic Index and Stability Index.

    Exact formulas (do not modify without clinical review):
        motion_derivative = mean(abs(diff(motion_signal)))
        variability_mean  = mean(variability_signal)
        dynamic_index     = variability_mean + motion_derivative
        stability_index   = 1 / (dynamic_index + 1e-6)
    """
    if len(motion_signal) < 2:
        return {
            "motion_mean":       float(np.mean(motion_signal)),
            "variability_mean":  float(np.mean(variability_signal)),
            "motion_derivative": 0.0,
            "dynamic_index":     0.0,
            "stability_index":   1e6,
        }

    motion_derivative = float(np.mean(np.abs(np.diff(motion_signal))))
    variability_mean  = float(np.mean(variability_signal))
    dynamic_index     = variability_mean + motion_derivative
    stability_index   = 1.0 / (dynamic_index + 1e-6)

    return {
        "motion_mean":       float(np.mean(motion_signal)),
        "variability_mean":  variability_mean,
        "motion_derivative": motion_derivative,
        "dynamic_index":     dynamic_index,
        "stability_index":   stability_index,
    }


def interpret_dynamic_index(dynamic_index: float) -> tuple:
    """
    Return (label, card_class, detail_text).
    ← Adjust THRESHOLD_ES and THRESHOLD_PNES at the top of this file.
    """
    if dynamic_index > THRESHOLD_ES:
        return (
            "Pattern appears dynamic and evolving — supports ES",
            "result-es",
            "High variability and frame-to-frame change detected, consistent with the GDI "
            "concept of epileptic dynamics.",
        )
    if dynamic_index < THRESHOLD_PNES:
        return (
            "Pattern appears relatively stable — supports PNES",
            "result-pnes",
            "Low variability and stable motion over time, consistent with the GDI concept "
            "of psychogenic stability.",
        )
    return (
        "Borderline dynamic pattern — indeterminate",
        "result-ind",
        "Motion dynamics fall between the defined thresholds. Clinical judgment and "
        "additional context are required.",
    )


def compute_combined_impression(dynamic_index: float, eye_state: str) -> dict:
    """
    Rule-based combined clinical impression.
    Returns ES score, PNES score, overall label, and card class.
    """
    es_score   = 0.0
    pnes_score = 0.0

    if dynamic_index > THRESHOLD_ES:
        es_score += 2.0
    elif dynamic_index < THRESHOLD_PNES:
        pnes_score += 2.0
    else:
        es_score   += 0.5
        pnes_score += 0.5

    if eye_state == "Eyes open":
        es_score   += 1.0
    elif eye_state == "Eyes closed":
        pnes_score += 1.0

    margin = abs(es_score - pnes_score)
    if margin < 0.5:
        return {"es_score": es_score, "pnes_score": pnes_score,
                "overall": "Indeterminate / mixed",  "card_class": "result-ind"}
    if es_score > pnes_score:
        return {"es_score": es_score, "pnes_score": pnes_score,
                "overall": "More supportive of ES",  "card_class": "result-es"}
    return     {"es_score": es_score, "pnes_score": pnes_score,
                "overall": "More supportive of PNES","card_class": "result-pnes"}


# ════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════

def plot_motion(timestamps: np.ndarray, motion_signal: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(timestamps, motion_signal, color="#1d4ed8", linewidth=1.1, alpha=0.9)
    ax.fill_between(timestamps, motion_signal, alpha=0.08, color="#1d4ed8")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Mean magnitude", fontsize=9)
    ax.set_title("Optical Flow Magnitude Over Time", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_variability(timestamps: np.ndarray, variability_signal: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(timestamps, variability_signal, color="#b45309", linewidth=1.1, alpha=0.9)
    ax.fill_between(timestamps, variability_signal, alpha=0.08, color="#b45309")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Local std", fontsize=9)
    ax.set_title("Rolling Motion Variability", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def frame_to_pil(bgr_frame) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


# ════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════

def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def warn_box(text: str):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def result_card(label: str, card_class: str, detail: str, chips: list = None):
    chip_html = ""
    if chips:
        chip_html = (
            '<div class="metric-row">'
            + "".join(f'<span class="metric-chip">{c}</span>' for c in chips)
            + "</div>"
        )
    st.markdown(
        f'<div class="result-card {card_class}">'
        f'  <div class="result-label">{label}</div>'
        f'  <p style="margin:4px 0 8px 0;font-size:0.9rem;color:#374151">{detail}</p>'
        f"  {chip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────
    st.markdown("# 🧠 Seizure Video Analyzer")
    st.markdown("**GDI Framework** · Research & Educational Prototype")

    st.markdown("""
<div class="disclaimer">
<strong>⚠️ Research / Educational Use Only</strong> — This application is a prototype for
motor seizure video assessment. It is <strong>not a medical device</strong> and does
<strong>not provide a definitive diagnosis</strong>. Final clinical interpretation requires
expert review, video-EEG, and full clinical context.<br>
• Applicable mainly to <strong>motor events</strong> &nbsp;·&nbsp;
Not validated for unsupervised clinical use &nbsp;·&nbsp;
Do not use as the sole basis for treatment decisions.
</div>
""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        eye_state = st.selectbox(
            "Eye state during event",
            ["Unknown", "Eyes open", "Eyes closed"],
            help="Eyes open → mild ES support · Eyes closed → mild PNES support",
        )

        st.divider()
        st.caption(
            "**Thresholds** (edit at top of `app.py`)\n"
            f"- ES: `dynamic_index > {THRESHOLD_ES}`\n"
            f"- PNES: `dynamic_index < {THRESHOLD_PNES}`\n"
            f"- Rolling window: `{ROLLING_WINDOW}` frames"
        )
        st.divider()
        st.caption(
            "**GDI Concept**\n"
            "PNES → more stable over time.\n"
            "ES → more dynamic and evolving.\n\n"
            "High variability + high derivative = ES\n"
            "Low variability + low derivative = PNES"
        )

    # ── 1. Upload ─────────────────────────────────────────
    section("1 · Upload Video")
    uploaded = st.file_uploader(
        "Upload a seizure video (MP4, MOV, AVI, MKV)",
        type=["mp4", "mov", "avi", "mkv"],
        help="Static video-EEG camera assumed. The entire video will be analysed.",
    )

    if not uploaded:
        info_box("Upload a video above to begin.")
        return

    suffix   = os.path.splitext(uploaded.name)[-1].lower()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded.getbuffer())
    tmp_file.flush()
    tmp_path = tmp_file.name
    tmp_file.close()

    st.caption(f"📁 {uploaded.name} · {uploaded.size / 1024:.1f} KB")

    # ── 2. First frame ────────────────────────────────────
    section("2 · First Frame Preview")
    try:
        first_frame = load_first_frame(tmp_path)
    except Exception as exc:
        st.error(f"Could not open video: {exc}")
        return

    if first_frame is None:
        st.error(
            "Failed to read the first frame. "
            "The video may be corrupt or in an unsupported codec."
        )
        return

    frame_h, frame_w = first_frame.shape[:2]

    # ── 3. ROI Selection ──────────────────────────────────
    section("3 · Draw ROI Around Patient")
    info_box(
        "Draw a <strong>single rectangle</strong> around the patient on the frame below. "
        "The same ROI will be used for all frames. "
        "Only the area inside the rectangle will be analysed."
    )

    CANVAS_MAX_W = 900

    # Convert BGR → RGB → PIL  (st_canvas requires an RGB PIL Image)
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Scale down to fit the canvas width (never upscale)
    scale    = min(1.0, CANVAS_MAX_W / frame_w)
    canvas_w = int(frame_w * scale)
    canvas_h = int(frame_h * scale)

    # Resize with high-quality resampling
    bg_image = frame_pil.resize((canvas_w, canvas_h), Image.LANCZOS)

    canvas_result = st_canvas(
        fill_color="rgba(29, 78, 216, 0.07)",
        stroke_width=2,
        stroke_color="#1d4ed8",
        background_image=bg_image,   # RGB PIL Image, already resized
        update_streamlit=True,
        width=canvas_w,
        height=canvas_h,
        drawing_mode="rect",
        key="roi_canvas",
    )

    # Parse ROI
    roi = None
    if (
        canvas_result.json_data is not None
        and len(canvas_result.json_data.get("objects", [])) > 0
    ):
        obj = canvas_result.json_data["objects"][-1]
        if obj.get("type") == "rect":
            cx = float(obj.get("left",   0))
            cy = float(obj.get("top",    0))
            cw = float(obj.get("width",  0))
            ch = float(obj.get("height", 0))

            x1 = max(0,        int(cx          / scale))
            y1 = max(0,        int(cy          / scale))
            x2 = min(frame_w,  int((cx + cw)   / scale))
            y2 = min(frame_h,  int((cy + ch)   / scale))

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                roi = (x1, y1, x2, y2)
                st.success(
                    f"✅ ROI selected — x1={x1}, y1={y1}, x2={x2}, y2={y2} "
                    f"({x2 - x1} × {y2 - y1} px)"
                )
            else:
                warn_box("ROI is too small (< 10 px in one dimension). Please draw a larger rectangle.")

    if roi is None:
        warn_box(
            "No valid ROI selected. "
            "Draw a rectangle around the patient before clicking Analyze."
        )

    # ── Analyze button ────────────────────────────────────
    st.divider()
    analyze_clicked = st.button(
        "🔍 Analyze Video",
        type="primary",
        disabled=(roi is None),
        use_container_width=True,
    )

    if not analyze_clicked:
        return

    if roi is None:
        st.error("Please select a ROI before analysing.")
        return

    # ── Run pipeline ──────────────────────────────────────
    cleanup_path = tmp_path
    try:
        progress = st.progress(0, text="Reading video and computing optical flow…")

        result_data = compute_motion_signal(tmp_path, roi)
        progress.progress(60, text="Computing variability and GDI indices…")

        motion_signal      = result_data["motion_signal"]
        timestamps         = result_data["timestamps"]
        fps                = result_data["fps"]
        total_frames       = result_data["total_frames"]
        frames_processed   = result_data["frames_processed"]

        variability_signal = compute_rolling_variability(motion_signal, window=ROLLING_WINDOW)
        indices            = compute_gdi_indices(motion_signal, variability_signal)
        impression         = compute_combined_impression(indices["dynamic_index"], eye_state)
        dyn_label, dyn_cls, dyn_detail = interpret_dynamic_index(indices["dynamic_index"])

        progress.progress(100, text="Done.")
        progress.empty()

    except ValueError as exc:
        st.error(f"Analysis failed: {exc}")
        return
    except RuntimeError as exc:
        st.error(f"Video read error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return
    finally:
        try:
            os.unlink(cleanup_path)
        except OSError:
            pass

    # ════════════════════════════════════════════════════════
    #  RESULTS
    # ════════════════════════════════════════════════════════
    st.divider()
    section("4 · Combined Clinical Impression")

    result_card(
        label=impression["overall"],
        card_class=impression["card_class"],
        detail=(
            f"ES support score: {impression['es_score']:.1f} · "
            f"PNES support score: {impression['pnes_score']:.1f} · "
            f"Eye state: {eye_state}"
        ),
        chips=[
            f"Dynamic index: {indices['dynamic_index']:.4f}",
            f"Stability index: {indices['stability_index']:.2f}",
            f"Frames analysed: {frames_processed}",
        ],
    )

    section("5 · Motion Pattern Interpretation")
    result_card(label=dyn_label, card_class=dyn_cls, detail=dyn_detail)

    section("6 · Video Information")
    duration_s       = frames_processed / fps if fps > 0 else 0
    vi1, vi2, vi3, vi4 = st.columns(4)
    vi1.metric("Duration analysed", f"{duration_s:.1f} s")
    vi2.metric("FPS",               f"{fps:.1f}")
    vi3.metric("Total frames",      total_frames)
    vi4.metric("Frames processed",  frames_processed)

    section("7 · Motion Graphs")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        fig = plot_motion(timestamps, motion_signal)
        st.pyplot(fig)
        plt.close(fig)
    with gcol2:
        fig = plot_variability(timestamps, variability_signal)
        st.pyplot(fig)
        plt.close(fig)

    section("8 · Numeric Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Motion Mean",       f"{indices['motion_mean']:.4f}")
    m2.metric("Variability Mean",  f"{indices['variability_mean']:.4f}")
    m3.metric("Motion Derivative", f"{indices['motion_derivative']:.4f}")
    m4.metric("Dynamic Index",     f"{indices['dynamic_index']:.4f}")
    m5.metric("Stability Index",   f"{indices['stability_index']:.2f}")

    section("9 · Score Breakdown")
    score_df = pd.DataFrame([
        {
            "Factor":       "Dynamic Index",
            "Value":        f"{indices['dynamic_index']:.4f}",
            "ES weight":    "+2.0" if indices["dynamic_index"] > THRESHOLD_ES
                            else ("0"   if indices["dynamic_index"] < THRESHOLD_PNES else "+0.5"),
            "PNES weight":  "+2.0" if indices["dynamic_index"] < THRESHOLD_PNES
                            else ("0"   if indices["dynamic_index"] > THRESHOLD_ES  else "+0.5"),
            "Notes": (
                f"Above ES threshold ({THRESHOLD_ES})"     if indices["dynamic_index"] > THRESHOLD_ES
                else f"Below PNES threshold ({THRESHOLD_PNES})" if indices["dynamic_index"] < THRESHOLD_PNES
                else "Between thresholds"
            ),
        },
        {
            "Factor":      "Eye State",
            "Value":       eye_state,
            "ES weight":   "+1.0" if eye_state == "Eyes open"   else "0",
            "PNES weight": "+1.0" if eye_state == "Eyes closed" else "0",
            "Notes": (
                "Eyes open → mild ES support"         if eye_state == "Eyes open"
                else "Eyes closed → mild PNES support" if eye_state == "Eyes closed"
                else "Unknown — no contribution"
            ),
        },
    ])
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "⚠️ All outputs are for research and educational purposes only. "
        "Results require expert clinical review and, when appropriate, "
        "video-EEG and full clinical context. Not a medical device."
    )


main()
