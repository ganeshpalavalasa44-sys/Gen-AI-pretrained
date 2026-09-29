"""
Utility functions for label loading, visualization, metrics, and video frame annotation.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.config import LABELS_PATH, ACTIVITY_CATEGORIES


def load_labels(labels_path: Optional[Path] = None) -> List[str]:
    """Load Kinetics-600 action class labels from file."""
    path = labels_path or LABELS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def get_activity_category(action_name: str) -> str:
    """Map a fine-grained Kinetics action to a macro category."""
    action_lower = action_name.lower()
    for category, actions in ACTIVITY_CATEGORIES.items():
        for act in actions:
            if act in action_lower:
                return category
    return "General Action"


def format_predictions(
    probabilities: np.ndarray,
    labels: List[str],
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Format raw output probabilities into structured prediction items.
    """
    probs = np.squeeze(probabilities)
    
    # If logits instead of probabilities, apply softmax
    if np.min(probs) < 0.0 or not np.isclose(np.sum(probs), 1.0, atol=1e-2):
        exp_p = np.exp(probs - np.max(probs))
        probs = exp_p / np.sum(exp_p)

    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(probs[idx])
        if score < threshold:
            continue
        label_text = labels[idx] if idx < len(labels) else f"Action #{idx}"
        results.append({
            "rank": rank,
            "label": label_text,
            "confidence": score,
            "percentage": f"{score * 100:.2f}%",
            "category": get_activity_category(label_text)
        })
    return results


def draw_hud_overlay(
    frame: np.ndarray,
    top_prediction: Optional[Dict] = None,
    fps: Optional[float] = None,
    frame_index: Optional[int] = None,
    total_frames: Optional[int] = None
) -> np.ndarray:
    """
    Draw a clean, modern HUD overlay with activity label and confidence on an OpenCV frame.
    """
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]

    # Overlay banner at top
    overlay = out_frame.copy()
    banner_height = 80
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 33), -1)
    
    # Add alpha blend for glassy transparent banner
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0, out_frame)

    # Accent line under banner
    cv2.line(out_frame, (0, banner_height), (w, banner_height), (0, 200, 255), 2)

    # Draw Model / System Badge
    badge_text = "MoViNet AI | HAR"
    cv2.putText(
        out_frame, badge_text, (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA
    )

    # Draw FPS & Frame Info
    info_parts = []
    if fps is not None:
        info_parts.append(f"{fps:.1f} FPS")
    if frame_index is not None and total_frames:
        info_parts.append(f"Frame {frame_index}/{total_frames}")
    if info_parts:
        info_str = " | ".join(info_parts)
        cv2.putText(
            out_frame, info_str, (w - 220, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 190, 205), 1, cv2.LINE_AA
        )

    # Draw Activity & Confidence
    if top_prediction:
        label = top_prediction.get("label", "Analyzing...").title()
        conf = top_prediction.get("confidence", 0.0)
        category = top_prediction.get("category", "")
        pct = f"{conf * 100:.1f}%"

        # Label title
        cv2.putText(
            out_frame, f"Activity: {label}", (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Confidence pill and bar on right
        cv2.putText(
            out_frame, f"Confidence: {pct} ({category})", (w - 380, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA
        )

        # Progress bar under label
        bar_w = int(w * 0.35)
        bar_x = 16
        bar_y = 66
        cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (50, 55, 65), -1)
        fill_w = int(bar_w * min(max(conf, 0.0), 1.0))
        cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 6), (0, 255, 180), -1)
    else:
        cv2.putText(
            out_frame, "Buffering frames for MoViNet analysis...", (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 170, 185), 1, cv2.LINE_AA
        )

    return out_frame


def plot_top_predictions_bar(predictions: List[Dict], title: str = "Top Human Activity Predictions") -> plt.Figure:
    """Generate a clean horizontal bar chart of top predictions for Streamlit/reporting."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    if not predictions:
        ax.text(0.5, 0.5, "No predictions above threshold", ha='center', va='center', color='white')
        return fig

    labels = [p["label"].title() for p in reversed(predictions)]
    scores = [p["confidence"] * 100 for p in reversed(predictions)]

    colors = ["#00d4aa", "#38bdf8", "#818cf8", "#c084fc", "#f472b6"]
    bar_colors = (colors * 3)[:len(labels)]

    bars = ax.barh(labels, scores, color=bar_colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, max(max(scores) * 1.25, 20))
    ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=11, fontweight="medium")
    ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#334155")
    ax.spines['left'].set_color("#334155")
    ax.grid(axis='x', linestyle='--', alpha=0.25, color="#64748b")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%", va='center', ha='left', color='#f8fafc',
            fontweight='bold', fontsize=10
        )

    plt.tight_layout()
    return fig
