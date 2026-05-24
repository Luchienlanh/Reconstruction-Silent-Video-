from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from srcV2.data.build_cache import LIP_LANDMARK_IDXS, box_from_landmarks, scan_raw_pairs
from srcV2.utils.common import safe_name, seed_everything


NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


@dataclass
class FrameMetrics:
    detected: bool
    frontal: bool = False
    crop_ok: bool = False
    nose_offset: float = 999.0
    eye_roll: float = 999.0
    mouth_width_px: float = 0.0
    mouth_open: float = 0.0
    crop_inside_ratio: float = 0.0


class FaceLandmarkDetector:
    def __init__(self, args):
        import mediapipe as mp

        self.mp = mp
        self.backend = "solutions"
        self.detector = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=args.min_face_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
            return

        self.backend = "tasks"
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = Path(args.face_landmarker_model)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe solutions API is unavailable and task model was not found: {model_path}"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=args.min_face_confidence,
            min_face_presence_confidence=args.min_face_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            result = self.detector.process(rgb)
            if not result.multi_face_landmarks:
                return None
            lms = result.multi_face_landmarks[0].landmark
            return np.asarray([[lm.x, lm.y] for lm in lms], dtype=np.float32)

        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        return np.asarray([[lm.x, lm.y] for lm in lms], dtype=np.float32)


def crop_inside_ratio(box: tuple[float, float, float], width: int, height: int) -> float:
    x1, y1, side = box
    x2 = x1 + side
    y2 = y1 + side
    ix1 = max(0.0, x1)
    iy1 = max(0.0, y1)
    ix2 = min(float(width), x2)
    iy2 = min(float(height), y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / max(side * side, 1e-6))


def frame_metrics(lm: np.ndarray | None, width: int, height: int, args) -> tuple[FrameMetrics, np.ndarray | None, tuple[float, float, float] | None]:
    if lm is None or lm.shape[0] <= RIGHT_CHEEK:
        return FrameMetrics(detected=False), None, None

    lip = lm[LIP_LANDMARK_IDXS]
    cheek_left = lm[LEFT_CHEEK]
    cheek_right = lm[RIGHT_CHEEK]
    face_width = abs(float(cheek_right[0] - cheek_left[0])) * width
    if face_width < 1.0:
        return FrameMetrics(detected=False), lip, None

    cheek_mid_x = (float(cheek_left[0]) + float(cheek_right[0])) * 0.5
    nose_offset = abs(float(lm[NOSE_TIP, 0]) - cheek_mid_x) * width / face_width
    eye_roll = abs(float(lm[LEFT_EYE_OUTER, 1]) - float(lm[RIGHT_EYE_OUTER, 1])) * height / face_width

    mouth_width_px = np.linalg.norm((lm[RIGHT_MOUTH] - lm[LEFT_MOUTH]) * np.asarray([width, height], dtype=np.float32))
    mouth_open_px = np.linalg.norm((lm[LOWER_LIP] - lm[UPPER_LIP]) * np.asarray([width, height], dtype=np.float32))
    mouth_open = float(mouth_open_px / max(mouth_width_px, 1e-6))

    box = box_from_landmarks(lip, width, height, args.crop_margin)
    inside = crop_inside_ratio(box, width, height)
    frontal = nose_offset <= args.max_nose_offset and eye_roll <= args.max_eye_roll
    crop_ok = inside >= args.min_crop_inside and mouth_width_px >= args.min_mouth_width_px
    return (
        FrameMetrics(
            detected=True,
            frontal=bool(frontal),
            crop_ok=bool(crop_ok),
            nose_offset=float(nose_offset),
            eye_roll=float(eye_roll),
            mouth_width_px=float(mouth_width_px),
            mouth_open=float(mouth_open),
            crop_inside_ratio=float(inside),
        ),
        lip,
        box,
    )


def sampled_frame_indices(video_path: Path, max_frames: int) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return []
    n = min(max_frames, total)
    if n <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, total - 1, n)))


def lip_motion(lips: list[np.ndarray | None], metrics: list[FrameMetrics], width: int, height: int) -> float:
    motions = []
    prev = None
    prev_width = None
    scale = np.asarray([width, height], dtype=np.float32)
    for lip, m in zip(lips, metrics):
        if lip is None or not m.detected:
            continue
        lip_px = lip * scale
        if prev is not None and prev_width is not None:
            motions.append(float(np.abs(lip_px - prev).mean() / max(prev_width, 1e-6)))
        prev = lip_px
        prev_width = m.mouth_width_px
    if not motions:
        return 0.0
    return float(np.mean(motions))


def draw_preview_cell(frame: np.ndarray, lip: np.ndarray | None, box: tuple[float, float, float] | None, metric: FrameMetrics) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if box is not None:
        x1, y1, side = box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x1 + side), int(y1 + side)), (0, 255, 255), 2)
    if lip is not None:
        pts = (lip * np.asarray([w, h], dtype=np.float32)).astype(np.int32)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
    color = (0, 220, 0) if metric.frontal and metric.crop_ok else (0, 0, 255)
    text = f"det={int(metric.detected)} front={int(metric.frontal)} crop={metric.crop_inside_ratio:.2f} open={metric.mouth_open:.2f}"
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return cv2.resize(out, (240, 135), interpolation=cv2.INTER_AREA)


def save_preview(path: Path, cells: list[np.ndarray]) -> None:
    if not cells:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    blank = np.zeros_like(cells[0])
    padded = cells + [blank] * (rows * cols - len(cells))
    grid_rows = [np.concatenate(padded[r * cols : (r + 1) * cols], axis=1) for r in range(rows)]
    grid = np.concatenate(grid_rows, axis=0)
    cv2.imwrite(str(path), grid)


def analyze_video(video_path: Path, detector: FaceLandmarkDetector, args) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"accepted": False, "reasons": ["cannot_open_video"], "video": str(video_path)}

    indices = sampled_frame_indices(video_path, args.max_sampled_frames)
    if not indices:
        cap.release()
        return {"accepted": False, "reasons": ["no_frames"], "video": str(video_path)}

    index_set = set(indices)
    metrics: list[FrameMetrics] = []
    lips: list[np.ndarray | None] = []
    boxes: list[tuple[float, float, float] | None] = []
    preview_cells: list[np.ndarray] = []
    frame_idx = -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        if frame_idx not in index_set:
            continue
        timestamp_ms = int(round(frame_idx * 1000.0 / fps))
        lm = detector.detect(frame, timestamp_ms)
        metric, lip, box = frame_metrics(lm, frame.shape[1], frame.shape[0], args)
        metrics.append(metric)
        lips.append(lip)
        boxes.append(box)
        if args.save_previews != "none" and len(preview_cells) < args.preview_frames:
            preview_cells.append(draw_preview_cell(frame, lip, box, metric))
        if len(metrics) >= len(indices):
            break
    cap.release()

    sampled = max(1, len(metrics))
    detected = [m for m in metrics if m.detected]
    detection_rate = len(detected) / sampled
    frontal_rate = sum(1 for m in detected if m.frontal) / max(1, len(detected))
    crop_ok_rate = sum(1 for m in detected if m.crop_ok) / max(1, len(detected))
    open_values = np.asarray([m.mouth_open for m in detected], dtype=np.float32)
    mouth_open_std = float(open_values.std()) if open_values.size else 0.0
    mouth_open_range = float(open_values.max() - open_values.min()) if open_values.size else 0.0
    motion = lip_motion(lips, metrics, max(1, width), max(1, height))
    median_mouth_width = float(np.median([m.mouth_width_px for m in detected])) if detected else 0.0
    median_nose_offset = float(np.median([m.nose_offset for m in detected])) if detected else 999.0
    median_eye_roll = float(np.median([m.eye_roll for m in detected])) if detected else 999.0

    reasons = []
    if detection_rate < args.min_detection_rate:
        reasons.append("low_face_detection")
    if frontal_rate < args.min_frontal_rate:
        reasons.append("not_frontal_enough")
    if crop_ok_rate < args.min_crop_ok_rate:
        reasons.append("mouth_crop_not_safe")
    if median_mouth_width < args.min_mouth_width_px:
        reasons.append("mouth_too_small")
    if mouth_open_std < args.min_mouth_open_std and mouth_open_range < args.min_mouth_open_range:
        reasons.append("low_mouth_open_variation")
    if motion < args.min_lip_motion:
        reasons.append("low_lip_motion")

    accepted = not reasons
    return {
        "accepted": accepted,
        "reasons": reasons,
        "video": str(video_path),
        "sampled_frames": sampled,
        "width": width,
        "height": height,
        "detection_rate": detection_rate,
        "frontal_rate": frontal_rate,
        "crop_ok_rate": crop_ok_rate,
        "mouth_open_std": mouth_open_std,
        "mouth_open_range": mouth_open_range,
        "lip_motion": motion,
        "median_mouth_width_px": median_mouth_width,
        "median_nose_offset": median_nose_offset,
        "median_eye_roll": median_eye_roll,
        "_preview_cells": preview_cells,
    }


def copy_pair(video_path: Path, audio_path: Path, output_dir: Path) -> Path:
    dst = output_dir / safe_name(video_path)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, dst / "video.mp4")
    shutil.copy2(audio_path, dst / "audio.wav")
    for extra_name in ("transcript.txt", "text.txt", "metadata.json"):
        extra = video_path.with_name(extra_name)
        if extra.exists():
            shutil.copy2(extra, dst / extra.name)
    return dst


def configure_preset(args) -> None:
    if args.preset == "loose":
        args.min_detection_rate = min(args.min_detection_rate, 0.65)
        args.min_frontal_rate = min(args.min_frontal_rate, 0.55)
        args.min_crop_ok_rate = min(args.min_crop_ok_rate, 0.70)
        args.min_mouth_open_std = min(args.min_mouth_open_std, 0.006)
        args.min_lip_motion = min(args.min_lip_motion, 0.004)
        args.min_mouth_width_px = min(args.min_mouth_width_px, 12.0)
    elif args.preset == "strict":
        args.min_detection_rate = max(args.min_detection_rate, 0.90)
        args.min_frontal_rate = max(args.min_frontal_rate, 0.80)
        args.min_crop_ok_rate = max(args.min_crop_ok_rate, 0.90)
        args.min_mouth_open_std = max(args.min_mouth_open_std, 0.010)
        args.min_lip_motion = max(args.min_lip_motion, 0.007)
        args.min_mouth_width_px = max(args.min_mouth_width_px, 18.0)


def run(args) -> None:
    seed_everything(args.seed)
    configure_preset(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    kept_dir = output_dir / "kept"
    pairs = scan_raw_pairs(args.raw_dir)
    if args.limit is not None:
        pairs = pairs[: max(1, min(int(args.limit), len(pairs)))]
    if not pairs:
        raise RuntimeError(f"No video.mp4/audio.wav pairs found under {args.raw_dir}")

    detector = FaceLandmarkDetector(args)

    results = []
    try:
        for video_path, audio_path in tqdm(pairs, desc="filter-videos"):
            item = analyze_video(video_path, detector, args)
            item["audio"] = str(audio_path)
            cells = item.pop("_preview_cells", [])
            if args.save_previews == "all" or (args.save_previews == "kept" and item["accepted"]) or (args.save_previews == "rejected" and not item["accepted"]):
                save_preview(preview_dir / f"{safe_name(video_path)}.jpg", cells)
            if item["accepted"] and args.copy_kept:
                item["copied_to"] = str(copy_pair(video_path, audio_path, kept_dir))
            results.append(item)
    finally:
        detector.close()

    kept = [r for r in results if r["accepted"]]
    summary = {
        "raw_dir": str(args.raw_dir),
        "output_dir": str(output_dir),
        "total": len(results),
        "kept": len(kept),
        "rejected": len(results) - len(kept),
        "keep_rate": len(kept) / max(1, len(results)),
        "thresholds": {
            "preset": args.preset,
            "min_detection_rate": args.min_detection_rate,
            "min_frontal_rate": args.min_frontal_rate,
            "min_crop_ok_rate": args.min_crop_ok_rate,
            "min_mouth_open_std": args.min_mouth_open_std,
            "min_mouth_open_range": args.min_mouth_open_range,
            "min_lip_motion": args.min_lip_motion,
            "max_nose_offset": args.max_nose_offset,
            "max_eye_roll": args.max_eye_roll,
            "min_crop_inside": args.min_crop_inside,
            "min_mouth_width_px": args.min_mouth_width_px,
        },
        "items": results,
    }
    manifest_path = output_dir / args.manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    kept_list = output_dir / "kept_videos.txt"
    with open(kept_list, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(item["video"] + "\n")

    print(f"[done] kept={len(kept)}/{len(results)} keep_rate={summary['keep_rate']:.3f}")
    print(f"[manifest] {manifest_path}")
    if args.copy_kept:
        print(f"[kept_dir] {kept_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Filter raw clips to frontal speaking presenter videos with safe mouth crops.")
    parser.add_argument("--raw-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="Dataset_Output_FrontalSpeaking_Filter")
    parser.add_argument("--manifest-name", default="frontal_speaking_manifest.json")
    parser.add_argument("--preset", choices=["loose", "normal", "strict"], default="normal")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-sampled-frames", type=int, default=48)
    parser.add_argument("--crop-margin", type=float, default=1.8)
    parser.add_argument("--min-face-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    parser.add_argument("--min-detection-rate", type=float, default=0.80)
    parser.add_argument("--min-frontal-rate", type=float, default=0.70)
    parser.add_argument("--min-crop-ok-rate", type=float, default=0.85)
    parser.add_argument("--max-nose-offset", type=float, default=0.11)
    parser.add_argument("--max-eye-roll", type=float, default=0.08)
    parser.add_argument("--min-crop-inside", type=float, default=0.90)
    parser.add_argument("--min-mouth-width-px", type=float, default=14.0)
    parser.add_argument("--min-mouth-open-std", type=float, default=0.008)
    parser.add_argument("--min-mouth-open-range", type=float, default=0.025)
    parser.add_argument("--min-lip-motion", type=float, default=0.005)
    parser.add_argument("--copy-kept", action="store_true", help="Copy accepted video.mp4/audio.wav pairs into output-dir/kept.")
    parser.add_argument("--save-previews", choices=["none", "kept", "rejected", "all"], default="kept")
    parser.add_argument("--preview-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
