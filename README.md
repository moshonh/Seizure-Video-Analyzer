# 🧠 Seizure Video Analyzer

**Research & Educational Prototype** — Motor Seizure Video Assessment  
Based on the Global Dynamic Impression (GDI) Framework

---

## ⚠️ Disclaimer

> This application is a **research and educational prototype** for motor seizure video assessment.  
> It is **not a medical device** and does **not provide a definitive diagnosis**.  
> Final clinical interpretation requires expert review and, when appropriate,
> video-EEG and full clinical context.
>
> - Applicable mainly to **motor events**
> - Not validated for unsupervised clinical use
> - Do **not** use as the sole basis for treatment decisions

---

## Overview

The app analyses uploaded seizure videos and classifies the motor event as:

| Label | Description |
|-------|-------------|
| **More consistent with ES** | Epileptic Seizure |
| **More consistent with PNES** | Psychogenic Nonepileptic Seizure |
| **Indeterminate / insufficient data** | Mixed or insufficient evidence |

Classification is based on the **Global Dynamic Impression (GDI)** framework — an interpretable,
rule-based scoring approach that quantifies motor pattern evolution over time.

---

## Project Structure

```
seizure-video-analyzer/
├── app.py                 # Main Streamlit application
├── video_processing.py   # Frame extraction & dense optical flow
├── motion_features.py    # GDI feature extraction
├── scoring.py            # Rule-based scoring engine
├── reporting.py          # HTML report generation
├── utils.py              # Shared utilities
├── requirements.txt      # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml       # Streamlit theme & upload-size config
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/seizure-video-analyzer.git
cd seizure-video-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Use `opencv-python` instead of `opencv-python-headless` if you need
> a local display (e.g., running `cv2.imshow`). For servers / Streamlit Cloud, keep
> `opencv-python-headless`.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

> Streamlit Cloud uses `requirements.txt` automatically — no extra configuration needed.

---

## Features

### Video Processing
- Dense optical flow (Farneback algorithm) for per-frame motion magnitude and direction

### GDI Feature Set

| Feature | PNES indicator | ES indicator |
|---------|----------------|--------------|
| Motor Dynamism | Low, stable | Evolving over time |
| Direction Variability | Fixed angle | Changing angle |
| Frequency Variability | Constant rhythm | Changing rhythm |
| Temporal Evolution | Burst–arrest–burst | Fade-in / fade-out |
| Burst Similarity | High (repetitive) | Low (variable) |
| Eye State | Closed | Open |

### Outputs
- **Classification** — ES / PNES / Indeterminate
- **Confidence** — low / moderate / high
- **Reliability** — poor / limited / adequate (penalised for short videos, missing onset/offset, etc.)
- **Feature summary table** with per-feature weighted scores
- **Motion graphs** (optical flow magnitude + rolling variability over time)
- **Feature comparison bar chart**
- **Plain-language explanation**
- **Downloadable HTML report**
- **Downloadable JSON feature export**

### Sidebar Controls
- Adjustable weights for all 6 features
- Manual eye-state annotation (open / closed / unavailable)
- Maximum frames to analyse (100–600)
- Toggle technical details view

### Manual Expert Annotation Panel
- Confirm onset / offset visibility
- Confirm face / eye visibility
- Free-text expert notes (appended to explanation)

---

## Clinical Background

The **Global Dynamic Impression (GDI)** principle:

**PNES characteristics:**
- Repetitive, stereotyped movements
- Fixed movement direction and frequency
- Burst–arrest–burst pattern

**ES characteristics:**
- Evolving motor activity over time
- Changing direction and rhythm
- Gradual build-up (fade-in) and decline (fade-out)

---

## Limitations

1. This is an **MVP** — accuracy has not been clinically validated.
2. Optical flow is sensitive to camera motion and poor lighting.
3. Eye state is **manually entered**; automated detection is not implemented.
4. Intended for **motor seizures only** — not validated for absence, focal aware, or non-motor events.
5. Short videos (< 10 s) substantially reduce reliability.
6. Results are **never sufficient alone** for a clinical decision.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `opencv-python-headless` | Video I/O, optical flow |
| `numpy` | Numerical computation |
| `pandas` | Feature tables |
| `matplotlib` | Plotting |
| `scipy` | Welch PSD for frequency analysis |
| `Pillow` | Image utilities |

---

## License

For research and educational use only. Not for clinical deployment.
