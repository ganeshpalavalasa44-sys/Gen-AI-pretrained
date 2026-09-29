"""
Transfer Learning and Fine-Tuning Pipeline for MoViNet on Custom Human Activity Datasets.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

from src.config import MODEL_REGISTRY, DEFAULT_MODEL_KEY
from src.video_processor import VideoProcessor


def build_transfer_model(
    model_key: str = DEFAULT_MODEL_KEY,
    num_classes: int = 10,
    num_frames: int = 16,
    dropout_rate: float = 0.25,
    trainable_backbone: bool = False
):
    """
    Build a custom Keras model with a frozen or fine-tunable MoViNet backbone.
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    model_info = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    resolution = model_info["resolution"]
    handle = model_info["handle"]

    # Input tensor: [batch_size, num_frames, height, width, 3]
    inputs = tf.keras.Input(
        shape=(num_frames, resolution, resolution, 3),
        dtype=tf.float32,
        name="video_frames"
    )

    # MoViNet Backbone Layer
    backbone = hub.KerasLayer(handle, trainable=trainable_backbone, name="movinet_backbone")
    features = backbone(inputs)

    # Custom classification head
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(features)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="action_probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model_key}_custom_classifier")
    return model


class VideoDatasetGenerator:
    """
    Keras Sequence generator to load batches of video clips from a dataset directory.
    Expected folder structure:
        dataset/
            train/
                activity_1/
                    clip1.mp4
                activity_2/
                    clip2.mp4
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (172, 172),
        batch_size: int = 4,
        shuffle: bool = True
    ):
        self.dataset_dir = Path(dataset_dir)
        self.num_frames = num_frames
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Discover classes
        self.classes = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Discover video files
        self.samples = []
        for cls_name in self.classes:
            cls_folder = self.dataset_dir / cls_name
            for video_path in cls_folder.glob("*.*"):
                if video_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                    self.samples.append((video_path, self.class_to_idx[cls_name]))

        if not self.samples:
            raise ValueError(f"No video files found in {dataset_dir} matching structure: <dataset_dir>/<class>/*.mp4")

        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_videos = []
        batch_labels = []

        for b_idx in batch_indices:
            v_path, label = self.samples[b_idx]
            try:
                frames = VideoProcessor.extract_uniform_frames(
                    v_path, num_frames=self.num_frames, target_size=self.target_size
                )
                batch_videos.append(frames.astype(np.float32) / 255.0)
                batch_labels.append(label)
            except Exception as e:
                # Handle corrupted video file
                continue

        if not batch_videos:
            return (
                np.zeros((1, self.num_frames, *self.target_size, 3), dtype=np.float32),
                np.zeros((1,), dtype=np.int32)
            )

        return np.array(batch_videos), np.array(batch_labels)


def train_custom_model(
    train_dir: str,
    val_dir: Optional[str] = None,
    output_dir: str = "./checkpoints",
    model_key: str = DEFAULT_MODEL_KEY,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3
):
    """Run full transfer learning training loop."""
    import tensorflow as tf

    train_gen = VideoDatasetGenerator(
        dataset_dir=train_dir,
        batch_size=batch_size
    )

    val_gen = None
    if val_dir and os.path.exists(val_dir):
        val_gen = VideoDatasetGenerator(
            dataset_dir=val_dir,
            batch_size=batch_size,
            shuffle=False
        )

    num_classes = len(train_gen.classes)
    print(f"Discovered {num_classes} action classes: {train_gen.classes}")

    model = build_transfer_model(
        model_key=model_key,
        num_classes=num_classes,
        num_frames=train_gen.num_frames
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]

    print(f"Starting MoViNet training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "custom_movinet")
    model.save(save_path)
    print(f"Model saved successfully to {save_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MoViNet on custom activity dataset.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory.")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation data directory.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output model directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    train_custom_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
