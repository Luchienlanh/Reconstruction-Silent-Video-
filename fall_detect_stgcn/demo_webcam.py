#!/usr/bin/env python
"""Live webcam demo for fall detection using any trained checkpoint.

Supports all model types: lstm, speech_tcn, snn, pose_r2plus1d, spiking_stgcn.
Uses MediaPipe Pose for real-time skeleton extraction.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import torch

from fall_detect_stgcn.labels import label_is_fall
from fall_detect_stgcn.mediapipe_backend import draw_pose, pose_estimator
from fall_detect_stgcn.pose_features import INPUT_DIM, normalize_pose_sequence, resample_sequence
from fall_detect_stgcn.models.registry import build_model


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(
        model_type=str(config.get("model_type", "lstm")),
        input_dim=int(config.get("input_dim", INPUT_DIM)),
        num_classes=int(config["num_classes"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        bidirectional=bool(config.get("bidirectional", True)),
        transformer_layers=int(config.get("transformer_layers", 1)),
        feature_set=str(config.get("feature_set", "full")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def predict(
    model: torch.nn.Module,
    config: dict[str, object],
    buffer: collections.deque[np.ndarray],
    mask_buffer: collections.deque[float],
    device: torch.device,
) -> tuple[int, float, float, float, np.ndarray]:
    keypoints = np.stack(list(buffer), axis=0)
    mask = np.asarray(list(mask_buffer), dtype=np.float32)
    features = normalize_pose_sequence(keypoints, mask)
    features = resample_sequence(features, int(config["seq_len"]))
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    label = int(probs.argmax())
    task = str(config["task"])
    if task == "multiclass":
        fall_score = float(probs[:5].sum())
    else:
        fall_score = float(probs[1])
    pose_ratio = float(mask.mean())
    return label, float(probs[label]), fall_score, pose_ratio, probs


def draw_status(
    frame: np.ndarray,
    model_type: str,
    label_name: str,
    confidence: float,
    fall_score: float,
    pose_ratio: float,
    is_fall: bool,
    ready: bool,
) -> None:
    h, w = frame.shape[:2]
    panel_w = min(560, w - 20)

    status = "FALL ALERT" if is_fall and ready else "normal"
    color = (0, 0, 255) if is_fall and ready else (0, 180, 0)

    # Background panel
    cv2.rectangle(frame, (12, 12), (panel_w, 108), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (panel_w, 108), (80, 80, 80), 1)

    # Status line
    cv2.putText(frame, status, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # Model type badge
    cv2.putText(frame, f"[{model_type}]", (220, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Label + confidence
    cv2.putText(
        frame,
        f"{label_name}  conf={confidence:.2f}",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    # Fall score + pose ratio
    cv2.putText(
        frame,
        f"fall={fall_score:.2f}  pose={pose_ratio:.2f}",
        (24, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a live webcam fall-detection demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="runs/upfall_spiking_stgcn/best_model.pt")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--predict-every", type=int, default=3)
    parser.add_argument("--alert-threshold", type=float, default=0.75)
    parser.add_argument("--alert-frames", type=int, default=7)
    parser.add_argument("--min-pose-ratio", type=float, default=0.75)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--pose-model", default="pretrained_models/pose_landmarker_lite.task")
    parser.add_argument("--pose-model-variant", default="lite", choices=["lite", "full", "heavy"])
    parser.add_argument("--download-pose-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = get_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)
    all_label_names = list(config["label_names"])
    task = str(config["task"])
    seq_len = int(config["seq_len"])
    model_type = str(config.get("model_type", "unknown"))

    print(f"Model: {model_type} | Task: {task} | Classes: {len(all_label_names)} | Device: {device}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {args.camera_index}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    keypoint_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=seq_len)
    mask_buffer: collections.deque[float] = collections.deque(maxlen=seq_len)
    fall_votes: collections.deque[int] = collections.deque(maxlen=args.alert_frames)

    label_name = "warming_up"
    confidence = 0.0
    fall_score = 0.0
    pose_ratio = 0.0
    fall_now = False
    frame_index = 0

    window_title = f"Fall Detection Demo [{model_type}]"

    with pose_estimator(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        pose_model=args.pose_model,
        pose_model_variant=args.pose_model_variant,
        download_model=args.download_pose_model,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints, found = pose.detect(rgb, timestamp_ms=frame_index * 33)
            keypoint_buffer.append(keypoints)
            mask_buffer.append(float(found))

            draw_pose(frame, keypoints, found)

            ready = len(keypoint_buffer) >= seq_len
            if ready and frame_index % args.predict_every == 0:
                label, confidence, fall_score, pose_ratio, _ = predict(
                    model, config, keypoint_buffer, mask_buffer, device
                )
                label_name = str(all_label_names[label])
                fall_now = (
                    label_is_fall(label, task)
                    and fall_score >= args.alert_threshold
                    and pose_ratio >= args.min_pose_ratio
                )
                fall_votes.append(int(fall_now))

            alert = ready and sum(fall_votes) >= max(1, args.alert_frames // 2 + 1)
            draw_status(frame, model_type, label_name, confidence, fall_score, pose_ratio, alert, ready)

            cv2.imshow(window_title, frame)
            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
