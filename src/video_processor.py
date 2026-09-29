"""
Video Processing, Frame Extraction, Temporal Sampling, HUD Annotation, and Synthetic Clip Generation.
"""

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils import draw_hud_overlay


class VideoProcessor:
    """Utilities for decoding, sampling, transforming, and annotating video streams."""

    @staticmethod
    def read_video_metadata(video_path: Union[str, Path]) -> Dict:
        """Read metadata (fps, frame count, duration, resolution) from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration
        }

    @staticmethod
    def extract_uniform_frames(
        video_path: Union[str, Path],
        num_frames: int = 16,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Extract `num_frames` uniformly spaced frames across the video duration.
        Returns a float32 or uint8 array of shape [num_frames, H, W, 3] in RGB.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback for streams or incomplete headers
            raw_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
            total_frames = len(raw_frames)
            cap.release()

            if total_frames == 0:
                raise ValueError("Video contains 0 readable frames.")

            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            selected = [raw_frames[i] for i in indices]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            selected = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    selected.append(frame)
                else:
                    if selected:
                        selected.append(selected[-1].copy())
                    else:
                        selected.append(np.zeros((target_size[0] or 172, target_size[1] or 172, 3), dtype=np.uint8))
            cap.release()

        # Process frames (BGR -> RGB, resize)
        processed = []
        for f in selected:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if target_size:
                f_rgb = cv2.resize(f_rgb, target_size, interpolation=cv2.INTER_AREA)
            processed.append(f_rgb)

        return np.array(processed, dtype=np.uint8)

    @staticmethod
    def annotate_video_stream(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        classifier,
        window_size: int = 16,
        inference_interval: int = 4
    ) -> Dict:
        """
        Process a video file, classify human actions in a rolling window, overlay
        HUD telemetry on each frame, and save the resulting annotated MP4 video.
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use standard MP4 codec for cross-platform compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_buffer = []
        top_pred = None
        frame_idx = 0
        all_detections = []

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            # Prepare frame for model buffer (RGB, resized)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_for_model = cv2.resize(
                frame_rgb, (classifier.resolution, classifier.resolution),
                interpolation=cv2.INTER_LINEAR
            )
            frame_buffer.append(resized_for_model)

            # Keep buffer size to window_size
            if len(frame_buffer) > window_size:
                frame_buffer.pop(0)

            # Run inference periodically
            if len(frame_buffer) == window_size and (frame_idx % inference_interval == 0 or top_pred is None):
                clip_tensor = np.array(frame_buffer, dtype=np.float32) / 255.0
                predictions = classifier.predict_clip(clip_tensor, top_k=3)
                if predictions:
                    top_pred = predictions[0]
                    all_detections.append({
                        "frame": frame_idx,
                        "time_sec": round(frame_idx / fps, 2),
                        "top_activity": top_pred["label"],
                        "confidence": top_pred["confidence"],
                        "category": top_pred["category"]
                    })

            # Calculate current processing FPS
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else fps

            # Draw HUD
            annotated_frame = draw_hud_overlay(
                frame=frame,
                top_prediction=top_pred,
                fps=current_fps,
                frame_index=frame_idx,
                total_frames=total_frames
            )

            writer.write(annotated_frame)

        cap.release()
        writer.release()

        return {
            "output_path": str(output_path),
            "processed_frames": frame_idx,
            "total_detections": len(all_detections),
            "detections": all_detections
        }

    @staticmethod
    def generate_synthetic_action_video(
        action_name: str = "jumping_jacks",
        output_path: Optional[Union[str, Path]] = None,
        duration_sec: float = 3.0,
        fps: int = 20,
        width: int = 640,
        height: int = 480
    ) -> str:
        """
        Generate a synthetic video demonstrating human motion (e.g. jumping jacks, squat, walking).
        Allows testing without downloading external footage.
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = temp_file.name
            temp_file.close()

        total_frames = int(duration_sec * fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

        center_x = width // 2
        center_y = int(height * 0.48)

        for i in range(total_frames):
            # Clean dark gradient canvas
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            # Add subtle grid floor
            cv2.line(canvas, (0, int(height * 0.82)), (width, int(height * 0.82)), (50, 50, 60), 2)
            for gx in range(0, width, 40):
                cv2.line(canvas, (gx, int(height * 0.82)), (gx, height), (35, 35, 45), 1)

            t = i / fps
            phase = (math.sin(t * math.pi * 2.0) + 1.0) / 2.0  # 0 to 1

            if action_name == "jumping_jacks":
                # Arm angle: from 20 deg (down) to 150 deg (overhead)
                arm_angle = 20 + phase * 130
                # Leg spread: from 10 to 65 px
                leg_spread = 15 + int(phase * 60)
                body_y = center_y - int(phase * 15)  # slight jump bounce

                head_pos = (center_x, body_y - 80)
                torso_bottom = (center_x, body_y)

                # Draw Head
                cv2.circle(canvas, head_pos, 22, (0, 230, 255), -1)
                # Draw Torso
                cv2.line(canvas, (center_x, head_pos[1] + 22), torso_bottom, (0, 230, 255), 6)

                # Draw Arms
                rad = math.radians(arm_angle)
                arm_len = 55
                left_hand = (int(center_x - arm_len * math.sin(rad)), int((head_pos[1] + 30) - arm_len * math.cos(rad)))
                right_hand = (int(center_x + arm_len * math.sin(rad)), int((head_pos[1] + 30) - arm_len * math.cos(rad)))
                cv2.line(canvas, (center_x, head_pos[1] + 30), left_hand, (0, 230, 255), 5)
                cv2.line(canvas, (center_x, head_pos[1] + 30), right_hand, (0, 230, 255), 5)

                # Draw Legs
                cv2.line(canvas, torso_bottom, (center_x - leg_spread, int(height * 0.82)), (0, 230, 255), 5)
                cv2.line(canvas, torso_bottom, (center_x + leg_spread, int(height * 0.82)), (0, 230, 255), 5)

            elif action_name == "squat":
                squat_depth = int(phase * 50)
                body_y = center_y + squat_depth
                head_pos = (center_x, body_y - 80)
                torso_bottom = (center_x, body_y)

                cv2.circle(canvas, head_pos, 22, (50, 200, 255), -1)
                cv2.line(canvas, (center_x, head_pos[1] + 22), torso_bottom, (50, 200, 255), 6)

                # Arms forward
                cv2.line(canvas, (center_x, head_pos[1] + 30), (center_x + 50, head_pos[1] + 30), (50, 200, 255), 5)

                # Knees bend
                knee_l = (center_x - 35, torso_bottom[1] + 35)
                knee_r = (center_x + 35, torso_bottom[1] + 35)
                cv2.line(canvas, torso_bottom, knee_l, (50, 200, 255), 5)
                cv2.line(canvas, torso_bottom, knee_r, (50, 200, 255), 5)
                cv2.line(canvas, knee_l, (center_x - 35, int(height * 0.82)), (50, 200, 255), 5)
                cv2.line(canvas, knee_r, (center_x + 35, int(height * 0.82)), (50, 200, 255), 5)

            else:  # walking
                walk_x = int((center_x - 120) + (i / total_frames) * 240)
                leg_swing = math.sin(t * math.pi * 3.0) * 35
                arm_swing = -leg_swing * 0.8

                head_pos = (walk_x, center_y - 80)
                torso_bottom = (walk_x, center_y)

                cv2.circle(canvas, head_pos, 22, (100, 255, 120), -1)
                cv2.line(canvas, (walk_x, head_pos[1] + 22), torso_bottom, (100, 255, 120), 6)

                # Arms
                cv2.line(canvas, (walk_x, head_pos[1] + 30), (int(walk_x + arm_swing), head_pos[1] + 75), (100, 255, 120), 5)
                cv2.line(canvas, (walk_x, head_pos[1] + 30), (int(walk_x - arm_swing), head_pos[1] + 75), (100, 255, 120), 5)

                # Legs
                cv2.line(canvas, torso_bottom, (int(walk_x + leg_swing), int(height * 0.82)), (100, 255, 120), 5)
                cv2.line(canvas, torso_bottom, (int(walk_x - leg_swing), int(height * 0.82)), (100, 255, 120), 5)

            # Label on synthetic video
            cv2.putText(
                canvas, f"Synthetic Activity: {action_name.replace('_', ' ').title()}",
                (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1, cv2.LINE_AA
            )

            writer.write(canvas)

        writer.release()
        return str(output_path)
