"""
Streamlit Web Application for Human Activity Recognition using MoViNet.
Features video upload, live webcam input, synthetic motion generator, HUD annotation, and analytics.
"""

import os
import tempfile
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    DEFAULT_NUM_FRAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    ACTIVITY_CATEGORIES,
    LABELS_PATH,
    SAMPLE_VIDEOS_DIR
)
from src.model import MoViNetClassifier
from src.video_processor import VideoProcessor
from src.utils import (
    load_labels,
    format_predictions,
    plot_top_predictions_bar,
    draw_hud_overlay,
    get_activity_category
)

# Set page configuration
st.set_page_config(
    page_title="MoViNet Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4aa 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #00d4aa;
        margin-bottom: 10px;
    }
    .action-badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        color: #00d4aa;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_classifier(model_key: str):
    """Cache loaded MoViNet model instance in memory across app reruns."""
    return MoViNetClassifier(model_key=model_key)


@st.cache_data
def get_all_labels():
    """Load Kinetics-600 label list."""
    return load_labels(LABELS_PATH)


def ensure_sample_videos():
    """Ensure sample videos exist for quick user demonstration."""
    SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    samples = {
        "jumping_jacks.mp4": "jumping_jacks",
        "squats.mp4": "squat",
        "walking.mp4": "walking"
    }
    created_paths = {}
    for filename, action in samples.items():
        sample_path = SAMPLE_VIDEOS_DIR / filename
        if not sample_path.exists():
            VideoProcessor.generate_synthetic_action_video(
                action_name=action,
                output_path=sample_path,
                duration_sec=3.0,
                fps=15
            )
        created_paths[action.replace("_", " ").title()] = str(sample_path)
    return created_paths


def main():
    # Header Banner
    st.markdown('<div class="main-title">🏃 MoViNet Human Activity Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time video frame analysis and human activity classification '
        'powered by Google Research MoViNet (Kinetics-600)</div>',
        unsafe_allow_html=True
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        model_choice = st.selectbox(
            "MoViNet Variant",
            options=list(MODEL_REGISTRY.keys()),
            format_func=lambda k: f"{MODEL_REGISTRY[k]['name']} ({MODEL_REGISTRY[k]['params']} params)",
            index=0
        )
        model_info = MODEL_REGISTRY[model_choice]

        st.caption(f"**Resolution**: {model_info['resolution']}x{model_info['resolution']} | **Architecture**: 3D Causal Convolutions")

        st.divider()
        st.subheader("🎯 Inference Settings")

        num_frames = st.slider("Frames per Clip", min_value=8, max_value=32, value=16, step=4)
        top_k = st.slider("Top Predictions", min_value=3, max_value=10, value=5)
        confidence_thresh = st.slider("Confidence Threshold (%)", min_value=0, max_value=80, value=5) / 100.0

        st.divider()
        st.subheader("🏷️ Category Filter")
        category_filter = st.selectbox(
            "Filter Kinetics Domain",
            options=["All 600 Classes"] + list(ACTIVITY_CATEGORIES.keys())
        )

        st.divider()
        st.markdown("""
        **About MoViNet:**
        Mobile Video Networks are computationally efficient video classification models using streamable causal 3D convolutions with memory state.
        """)

    # Load Model
    with st.spinner(f"Loading {model_info['name']}..."):
        try:
            classifier = get_cached_classifier(model_choice)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    # Create Sample Videos if needed
    sample_video_dict = ensure_sample_videos()

    # Main Tabs
    tab_analyze, tab_timeline, tab_categories, tab_guide = st.tabs([
        "📹 Video Activity Recognition",
        "📈 Temporal Dynamics",
        "📚 Predefined Activities (Kinetics-600)",
        "🚀 Model Card & Fine-Tuning"
    ])

    with tab_analyze:
        st.subheader("Select Video Source")

        source_type = st.radio(
            "Input Mode",
            ["📁 Upload Video File", "🎬 Preloaded Test Samples", "🧪 Synthetic Action Generator", "📷 Live Camera Recording"],
            horizontal=True
        )

        active_video_path = None

        if source_type == "📁 Upload Video File":
            uploaded_file = st.file_uploader(
                "Upload video file (MP4, AVI, MOV, WEBM)",
                type=["mp4", "avi", "mov", "webm", "mkv"]
            )
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                active_video_path = tfile.name
                tfile.close()

        elif source_type == "🎬 Preloaded Test Samples":
            selected_sample = st.selectbox("Choose Sample Clip", list(sample_video_dict.keys()))
            active_video_path = sample_video_dict[selected_sample]

        elif source_type == "🧪 Synthetic Action Generator":
            col_gen1, col_gen2 = st.columns([2, 1])
            with col_gen1:
                gen_action = st.selectbox(
                    "Motion Type",
                    ["jumping_jacks", "squat", "walking"],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            with col_gen2:
                if st.button("Generate Motion Clip", type="primary", use_container_width=True):
                    gen_path = VideoProcessor.generate_synthetic_action_video(
                        action_name=gen_action,
                        duration_sec=3.0,
                        fps=15
                    )
                    st.session_state["synth_video"] = gen_path

            if "synth_video" in st.session_state:
                active_video_path = st.session_state["synth_video"]

        elif source_type == "📷 Live Camera Recording":
            cam_file = st.camera_input("Take a photo / test pose")
            if cam_file is not None:
                st.info("Converting snapshot into video clip sequence for MoViNet...")
                bytes_data = cam_file.getvalue()
                nparr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                active_video_path = tfile.name
                tfile.close()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(active_video_path, fourcc, 10.0, (img.shape[1], img.shape[0]))
                for _ in range(16):
                    writer.write(img)
                writer.release()

        # If a video is available for classification
        if active_video_path and os.path.exists(active_video_path):
            st.divider()

            col_vid, col_results = st.columns([1.1, 1.2])

            with col_vid:
                st.markdown("### 🎬 Video Preview")
                st.video(active_video_path)

                try:
                    meta = VideoProcessor.read_video_metadata(active_video_path)
                    st.caption(
                        f"📊 Resolution: {meta['width']}x{meta['height']} | "
                        f"FPS: {meta['fps']:.1f} | Duration: {meta['duration_sec']:.2f}s "
                        f"({meta['frame_count']} frames)"
                    )
                except Exception:
                    pass

            with col_results:
                st.markdown("### 🎯 Classification Results")

                with st.spinner("Extracting frames & running MoViNet inference..."):
                    # Uniform frame extraction
                    frames = VideoProcessor.extract_uniform_frames(
                        active_video_path,
                        num_frames=num_frames,
                        target_size=(classifier.resolution, classifier.resolution)
                    )

                    t_start = time.time()
                    predictions = classifier.predict_clip(frames, top_k=top_k, threshold=confidence_thresh)
                    infer_ms = (time.time() - t_start) * 1000

                if predictions:
                    top_1 = predictions[0]

                    # Filter by category if selected
                    if category_filter != "All 600 Classes":
                        allowed_actions = ACTIVITY_CATEGORIES.get(category_filter, [])
                        filtered_preds = [p for p in predictions if any(a in p["label"].lower() for a in allowed_actions)]
                        if filtered_preds:
                            top_1 = filtered_preds[0]

                    # Metric Display
                    st.markdown(f"""
                    <div class="metric-card">
                        <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Recognized Activity</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">{top_1['label'].title()}</div>
                        <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
                            <span class="action-badge">Confidence: {top_1['percentage']}</span>
                            <span style="color: #cbd5e1; font-size: 0.9rem;">Category: <strong>{top_1['category']}</strong></span>
                            <span style="color: #64748b; font-size: 0.85rem;">Latency: {infer_ms:.1f} ms</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Top-K Bar Plot
                    fig = plot_top_predictions_bar(predictions, title=f"Top-{len(predictions)} Activity Probabilities")
                    st.pyplot(fig)
                    plt.close(fig)

                else:
                    st.warning("No action detected above the selected confidence threshold.")

            # Sampled Frames Filmstrip
            st.divider()
            st.markdown("### 🎞️ Sampled Video Frames")
            st.caption(f"Showing {len(frames)} uniformly sampled frames passed to MoViNet ({classifier.resolution}x{classifier.resolution})")

            cols = st.columns(min(8, len(frames)))
            for idx, frame in enumerate(frames[:8]):
                with cols[idx]:
                    st.image(frame, caption=f"Frame {idx+1}", use_container_width=True)

            # Video HUD Annotation section
            st.divider()
            st.markdown("### 🖥️ Full Video HUD Annotation")
            st.caption("Process every frame of the video with rolling MoViNet temporal classification and render the futuristic CV HUD.")

            if st.button("Generate HUD-Annotated Video", type="secondary"):
                with st.spinner("Processing video stream & rendering HUD overlay..."):
                    out_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    out_path = out_temp.name
                    out_temp.close()

                    result = VideoProcessor.annotate_video_stream(
                        input_path=active_video_path,
                        output_path=out_path,
                        classifier=classifier,
                        window_size=num_frames,
                        inference_interval=4
                    )

                    st.success(f"Annotation complete! Processed {result['processed_frames']} frames.")
                    col_annot_v, col_annot_d = st.columns([1.5, 1])
                    with col_annot_v:
                        st.video(out_path)
                    with col_annot_d:
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Annotated Video",
                                data=f.read(),
                                file_name="annotated_movinet_output.mp4",
                                mime="video/mp4"
                            )

    with tab_timeline:
        st.subheader("📈 Temporal Action Progression")
        st.markdown(
            "MoViNet sliding-window temporal analysis tracks action transitions and activity dynamics over the continuous duration of the clip."
        )

        if active_video_path and os.path.exists(active_video_path):
            if st.button("Compute Temporal Timeline", type="primary"):
                with st.spinner("Executing sliding window analysis across video frames..."):
                    # Load all frames
                    all_frames = VideoProcessor.extract_uniform_frames(
                        active_video_path,
                        num_frames=32,
                        target_size=(classifier.resolution, classifier.resolution)
                    )
                    timeline_data = classifier.predict_sliding_window(
                        all_frames,
                        window_size=12,
                        stride=4,
                        top_k=3
                    )

                    if timeline_data:
                        timeline_records = []
                        for item in timeline_data:
                            pri = item.get("primary")
                            timeline_records.append({
                                "Window": f"F{item['start_frame']}-F{item['end_frame']}",
                                "Predicted Activity": pri["label"].title() if pri else "Unknown",
                                "Confidence (%)": round(pri["confidence"] * 100, 2) if pri else 0.0,
                                "Category": pri["category"] if pri else "None"
                            })

                        df_timeline = pd.DataFrame(timeline_records)
                        st.dataframe(df_timeline, use_container_width=True)

                        # Line Chart of confidence over time
                        st.line_chart(df_timeline.set_index("Window")["Confidence (%)"])
        else:
            st.info("Please select or upload a video in the first tab to analyze temporal dynamics.")

    with tab_categories:
        st.subheader("📚 Predefined Kinetics-600 Activities")
        st.markdown(
            "MoViNet is pre-trained on the **Kinetics-600** dataset, classifying 600 rich human activities across sports, fitness, domestic tasks, dance, and music."
        )

        labels = get_all_labels()
        search_query = st.text_input("🔍 Search Predefined Activities", placeholder="e.g. yoga, dancing, basketball, cooking...")

        if search_query:
            matched_labels = [l for l in labels if search_query.lower() in l.lower()]
            st.write(f"Found **{len(matched_labels)}** matching activities:")
            st.write(", ".join([f"`{m}`" for m in matched_labels]))
        else:
            col_cat1, col_cat2 = st.columns(2)
            categories = list(ACTIVITY_CATEGORIES.items())
            mid = len(categories) // 2

            with col_cat1:
                for cat, acts in categories[:mid]:
                    with st.expander(f"📁 {cat} ({len(acts)} featured classes)"):
                        st.write(", ".join([f"`{a}`" for a in acts]))

            with col_cat2:
                for cat, acts in categories[mid:]:
                    with st.expander(f"📁 {cat} ({len(acts)} featured classes)"):
                        st.write(", ".join([f"`{a}`" for a in acts]))

            with st.expander(f"📋 View All {len(labels)} Predefined Activity Classes"):
                st.write(", ".join([f"`{l}`" for l in labels]))

    with tab_guide:
        st.subheader("🚀 MoViNet Model Architecture & Transfer Learning")
        st.markdown("""
        ### About MoViNet (Mobile Video Networks)
        MoViNet (Kondratyuk et al., Google Research 2021) is a family of computation-efficient video recognition architectures designed for mobile and edge devices.
        
        #### Key Innovations:
        1. **3D Causal Convolutions:** Convolutions operate causally along the temporal dimension without looking ahead into future frames, enabling online streaming.
        2. **Stream Buffers:** Replaces 3D feature representations with lightweight 2D frame-by-frame processing and internal memory states.
        3. **Neural Architecture Search (NAS):** Optimized simultaneously for accuracy, FLOPs, and latency on mobile platforms.
        
        ---
        
        ### Fine-Tuning MoViNet on Custom Activities
        To train MoViNet on your own dataset of videos, use the included transfer learning script:
        
        ```bash
        # 1. Structure your dataset
        dataset/
          ├── train/
          │     ├── pushups/
          │     └── jumping_jacks/
          └── val/
                ├── pushups/
                └── jumping_jacks/
        
        # 2. Run transfer learning
        python src/train_transfer.py --train_dir dataset/train --val_dir dataset/val --epochs 10 --batch_size 4
        ```
        """)


if __name__ == "__main__":
    main()
