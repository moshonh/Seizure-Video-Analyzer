"""
reporting.py
Generates a self-contained HTML report summarising the seizure video analysis.
"""

import base64
import io
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scoring import ScoringResult
from motion_features import MotionFeatures


DISCLAIMER = (
    "This application is a research and educational prototype for motor seizure video assessment. "
    "It is not a medical device and does not provide a definitive diagnosis. "
    "Final clinical interpretation requires expert review and, when appropriate, "
    "video-EEG and full clinical context."
)

LIMITATIONS = [
    "Applicable mainly to motor events.",
    "Not validated for unsupervised clinical use.",
    "Do not use as sole basis for treatment decisions.",
    "Eye-state detection is manual in this version; automated detection is not implemented.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _motion_figure_b64(features: MotionFeatures) -> str:
    t   = np.asarray(features.timestamps)
    mag = np.asarray(features.motion_signal)
    var = np.asarray(features.variability_signal)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    axes[0].plot(t, mag, color="#2563eb", linewidth=1.2)
    axes[0].set_ylabel("Motion magnitude")
    axes[0].set_title("Optical Flow Magnitude over Time")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, var, color="#dc2626", linewidth=1.2)
    axes[1].set_ylabel("Local variability")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Rolling Motion Variability")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _score_figure_b64(result: ScoringResult) -> str:
    names     = list(result.feature_scores.keys())
    pnes_vals = [result.feature_scores[k]["pnes"] for k in names]
    es_vals   = [result.feature_scores[k]["es"]   for k in names]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, pnes_vals, width, label="PNES", color="#f59e0b", alpha=0.85)
    ax.bar(x + width / 2, es_vals,   width, label="ES",   color="#3b82f6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Weighted score")
    ax.set_title("Feature-level Score Breakdown")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    result: ScoringResult,
    features: MotionFeatures,
    video_filename: str = "unknown.mp4",
) -> str:
    """Return a self-contained HTML string suitable for browser download."""

    ts         = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    motion_img = _motion_figure_b64(features)
    score_img  = _score_figure_b64(result)

    colour_map = {"PNES": "#f59e0b", "ES": "#3b82f6", "Indeterminate": "#6b7280"}
    cls_colour = colour_map.get(result.classification, "#6b7280")

    display_map = {
        "PNES":          "More consistent with PNES",
        "ES":            "More consistent with ES",
        "Indeterminate": "Indeterminate / insufficient data",
    }
    display_label = display_map.get(result.classification, result.classification)

    # Feature table rows
    feature_rows = "".join(
        f"<tr>"
        f"<td>{name}</td>"
        f"<td>{scores['pnes']:.2f}</td>"
        f"<td>{scores['es']:.2f}</td>"
        f"<td>{scores['label']}</td>"
        f"<td style='font-size:0.85em;color:#555'>{scores['detail']}</td>"
        f"</tr>"
        for name, scores in result.feature_scores.items()
    )

    quality_rows = "".join(
        f"<tr><td>{'⚠️' if triggered else '✅'} {flag.replace('_',' ').title()}</td></tr>"
        for flag, triggered in result.quality_penalties.items()
    )

    lim_items = "".join(f"<li>{lim}</li>" for lim in LIMITATIONS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Seizure Video Analysis Report</title>
<style>
  body      {{ font-family: Georgia, serif; max-width: 880px; margin: 40px auto; color: #1a1a1a; line-height: 1.65; padding: 0 16px; }}
  h1        {{ font-size: 1.65rem; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
  h2        {{ font-size: 1.2rem; margin-top: 28px; color: #333; }}
  .disclaimer {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; }}
  .result-card {{ border: 2px solid {cls_colour}; border-radius: 8px; padding: 20px; margin: 20px 0; }}
  .result-label {{ font-size: 1.8rem; font-weight: bold; color: {cls_colour}; }}
  .badge    {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; background: #f3f4f6; margin-right: 8px; margin-top: 6px; }}
  table     {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th        {{ background: #f3f4f6; text-align: left; padding: 8px; }}
  td        {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  img       {{ max-width: 100%; border-radius: 6px; margin: 12px 0; }}
  .explanation {{ background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; }}
  footer    {{ margin-top: 40px; font-size: 0.8rem; color: #888; border-top: 1px solid #eee; padding-top: 12px; }}
</style>
</head>
<body>

<h1>🧠 Seizure Video Analysis Report</h1>
<p><strong>File:</strong> {video_filename} &nbsp;|&nbsp; <strong>Generated:</strong> {ts}</p>

<div class="disclaimer">
  <strong>⚠️ Research / Educational Use Only</strong><br>
  {DISCLAIMER}
</div>

<div class="result-card">
  <div class="result-label">{display_label}</div>
  <div style="margin-top:8px;">
    <span class="badge">Confidence: <strong>{result.confidence}</strong></span>
    <span class="badge">Reliability: <strong>{result.reliability}</strong></span>
    <span class="badge">PNES evidence: <strong>{result.pnes_score:.1f}/10</strong></span>
    <span class="badge">ES evidence: <strong>{result.es_score:.1f}/10</strong></span>
  </div>
</div>

<h2>Explanation</h2>
<div class="explanation">{result.explanation}</div>

<h2>Feature Score Breakdown</h2>
<img src="data:image/png;base64,{score_img}" alt="Score chart">
<table>
  <thead>
    <tr>
      <th>Feature</th><th>PNES score</th><th>ES score</th>
      <th>Interpretation</th><th>Detail</th>
    </tr>
  </thead>
  <tbody>{feature_rows}</tbody>
</table>

<h2>Motion Over Time</h2>
<img src="data:image/png;base64,{motion_img}" alt="Motion chart">

<h2>Video Quality Flags</h2>
<table><tbody>{quality_rows}</tbody></table>

<h2>Limitations</h2>
<ul>{lim_items}</ul>

<footer>
  Generated by Seizure Video Analyzer – Research Prototype &nbsp;|&nbsp;
  Not a medical device &nbsp;|&nbsp; {ts}
</footer>
</body>
</html>"""
