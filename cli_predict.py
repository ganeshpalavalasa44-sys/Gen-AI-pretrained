"""
Command-Line Interface (CLI) for MoViNet Human Activity Recognition.
Supports video file analysis, synthetic action generation, live webcam inference, and video annotation.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from src.config import MODEL_REGISTRY, DEFAULT_MODEL_KEY, LABELS_PATH
from src.model import MoViNetClassifier
from src.video_processor import VideoProcessor
from src.utils import draw_hud_overlay


def print_banner():
    print("=" * 65)
    print("   MoViNet Human Activity Recognition System (HAR)")
    print("   Google Research MoViNet Mobile Video Networks")
    print("=" * 65)


def print_predictions_table(predictions):
    print("\n[+] Top Human Activity Predictions:")
    print("-" * 65)
    print(f" {'Rank':<6} | {'Activity Name':<32} | {'Confidence':<12} | {'Category'}")
    print("-" * 65)
    for p in predictions:
        pct = p["percentage"]
        label = p["label"].title()
        cat = p["category"]
        print(f" #{p['rank']:<5} | {label:<32} | {pct:<12} | {cat}")
    print("-" * 65)


def run_video_classification(args):
    print(f"[*] Initializing MoViNet ({args.model})...")
    classifier = MoViNetClassifier(model_key=args.model)

    video_path = args.video
    if args.synthetic:
        print(f"[*] Generating synthetic motion clip for: {args.synthetic}...")
        video_path = VideoProcessor.generate_synthetic_action_video(
            action_name=args.synthetic,
            duration_sec=3.0,
            fps=20
        )
        print(f"[+] Synthetic video generated at: {video_path}")

    if not video_path or not Path(video_path).exists():
        print(f"[!] Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    # Read Metadata
    meta = VideoProcessor.read_video_metadata(video_path)
    print(f"[*] Video Specs: {meta['width']}x{meta['height']} | {meta['fps']:.1f} FPS | {meta['duration_sec']:.2f}s ({meta['frame_count']} frames)")

    if args.output:
        print(f"[*] Annotating video stream with MoViNet HUD -> {args.output}...")
        result = VideoProcessor.annotate_video_stream(
            input_path=video_path,
            output_path=args.output,
            classifier=classifier,
            window_size=args.frames
        )
        print(f"[+] Video annotation complete! Saved to {result['output_path']}")

    # Overall Clip Prediction
    print(f"[*] Extracting {args.frames} uniform frames across clip...")
    frames = VideoProcessor.extract_uniform_frames(
        video_path,
        num_frames=args.frames,
        target_size=(classifier.resolution, classifier.resolution)
    )

    t0 = time.time()
    predictions = classifier.predict_clip(frames, top_k=args.top_k)
    inference_time = (time.time() - t0) * 1000

    print_predictions_table(predictions)
    print(f"[i] MoViNet Inference Latency: {inference_time:.2f} ms")


def run_webcam_stream(args):
    print(f"[*] Starting Real-Time Webcam Stream (Device #{args.webcam})...")
    print("[*] Press 'q' in the video window to quit.")
    classifier = MoViNetClassifier(model_key=args.model)

    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"[!] Failed to open webcam #{args.webcam}.")
        sys.exit(1)

    window_size = args.frames
    frame_buffer = []
    top_pred = None
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (classifier.resolution, classifier.resolution))
            frame_buffer.append(resized)

            if len(frame_buffer) > window_size:
                frame_buffer.pop(0)

            if len(frame_buffer) == window_size and frame_idx % 4 == 0:
                clip_tensor = np.array(frame_buffer, dtype=np.float32) / 255.0
                preds = classifier.predict_clip(clip_tensor, top_k=1)
                if preds:
                    top_pred = preds[0]

            elapsed = time.time() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0

            annotated = draw_hud_overlay(
                frame=frame,
                top_prediction=top_pred,
                fps=fps,
                frame_index=frame_idx
            )

            cv2.imshow("MoViNet Human Activity Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Classify predefined human activities using MoViNet.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device ID (e.g. 0).")
    parser.add_argument("--synthetic", type=str, choices=["jumping_jacks", "squat", "walking"], default=None,
                        help="Generate and classify a synthetic action animation.")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), default=DEFAULT_MODEL_KEY,
                        help="MoViNet model architecture variant.")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames per video clip.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display.")
    parser.add_argument("--output", type=str, default=None, help="Save annotated video with HUD.")

    args = parser.parse_args()

    if args.webcam is not None:
        run_webcam_stream(args)
    elif args.video or args.synthetic:
        run_video_classification(args)
    else:
        # Default behavior: run on synthetic action
        print("[*] No input specified. Running demonstration on synthetic 'jumping_jacks'...")
        args.synthetic = "jumping_jacks"
        run_video_classification(args)


if __name__ == "__main__":
    main()
