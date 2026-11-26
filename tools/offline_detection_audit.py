#!/usr/bin/env python3
"""Offline detector audit for recorded video.

This utility only reads a video file and reports aggregate detection/performance
metrics. It does not capture the desktop, interact with a game, track targets,
or send mouse/keyboard input.

Example:
    python tools/offline_detection_audit.py \
        --video path/to/recording.mp4 \
        --model models/sunxds_0.8.0.onnx \
        --device cpu \
        --imgsz 320

Outputs:
    runs/offline_audit/<video-name>_summary.json
    runs/offline_audit/<video-name>_frames.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def percentile(values: list[float], value: float) -> float:
    """Return a percentile, or zero when no samples are available."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def class_name(names: Any, class_id: int) -> str:
    """Resolve Ultralytics class names without assuming dict/list storage."""
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure YOLO detector performance on a recorded video only."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video file.")
    parser.add_argument("--model", required=True, type=Path, help="YOLO model path.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu or 0. Defaults to cpu.",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    parser.add_argument(
        "--conf", type=float, default=0.20, help="Confidence threshold (0 to 1)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to analyze; 0 means the whole video.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/offline_audit"),
        help="Directory for JSON summary and per-frame CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.is_file():
        print(f"ERROR: video file does not exist: {args.video}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"ERROR: model file does not exist: {args.model}", file=sys.stderr)
        return 2
    if args.imgsz <= 0:
        print("ERROR: --imgsz must be greater than zero.", file=sys.stderr)
        return 2
    if args.max_frames < 0:
        print("ERROR: --max-frames cannot be negative.", file=sys.stderr)
        return 2
    if not 0.0 <= args.conf <= 1.0:
        print("ERROR: --conf must be between 0 and 1.", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video: {args.video}", file=sys.stderr)
        return 2

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(f"Loading model: {args.model}")
    try:
        model = YOLO(str(args.model), task="detect")
    except Exception as exc:
        cap.release()
        print(f"ERROR: could not load model: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    csv_path = args.out_dir / f"{stem}_frames.csv"
    json_path = args.out_dir / f"{stem}_summary.json"

    inference_ms: list[float] = []
    total_frame_ms: list[float] = []
    decode_ms_samples: list[float] = []
    per_class: dict[int, dict[str, float]] = defaultdict(
        lambda: {"detections": 0, "confidence_sum": 0.0}
    )

    frame_count = 0
    frames_with_detections = 0
    total_detections = 0
    valid_boxes = 0
    invalid_boxes = 0
    out_of_frame_boxes = 0
    run_started = time.perf_counter()

    fieldnames = [
        "frame_index",
        "video_time_seconds",
        "decode_ms",
        "inference_ms",
        "total_frame_ms",
        "detection_count",
        "valid_box_count",
        "invalid_box_count",
        "out_of_frame_box_count",
        "mean_confidence",
        "class_counts",
    ]

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while args.max_frames == 0 or frame_count < args.max_frames:
                frame_started = time.perf_counter()
                decode_started = time.perf_counter()
                ok, frame = cap.read()
                decode_duration = (time.perf_counter() - decode_started) * 1000.0

                if not ok or frame is None:
                    break

                frame_index = frame_count
                frame_count += 1
                decode_ms_samples.append(decode_duration)

                infer_started = time.perf_counter()
                try:
                    results = model.predict(
                        source=frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        device=args.device,
                        verbose=False,
                        stream=False,
                        save=False,
                        show=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Inference failed at video frame {frame_index}: {exc}"
                    ) from exc
                infer_duration = (time.perf_counter() - infer_started) * 1000.0
                inference_ms.append(infer_duration)

                result = results[0] if results else None
                boxes = getattr(result, "boxes", None) if result is not None else None
                xyxy = (
                    boxes.xyxy.detach().cpu().numpy()
                    if boxes is not None and len(boxes) > 0
                    else np.empty((0, 4), dtype=np.float32)
                )
                confs = (
                    boxes.conf.detach().cpu().numpy()
                    if boxes is not None and len(boxes) > 0
                    else np.empty((0,), dtype=np.float32)
                )
                classes = (
                    boxes.cls.detach().cpu().numpy().astype(int)
                    if boxes is not None and len(boxes) > 0
                    else np.empty((0,), dtype=np.int64)
                )

                frame_detection_count = int(len(xyxy))
                frame_valid = 0
                frame_invalid = 0
                frame_outside = 0
                class_counts: dict[str, int] = defaultdict(int)

                if frame_detection_count:
                    frames_with_detections += 1

                for coords, confidence, class_id in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = (float(value) for value in coords)
                    finite = all(math.isfinite(value) for value in (x1, y1, x2, y2))
                    geometry_ok = finite and x2 > x1 and y2 > y1

                    if geometry_ok:
                        frame_valid += 1
                        valid_boxes += 1
                        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                            frame_outside += 1
                            out_of_frame_boxes += 1
                    else:
                        frame_invalid += 1
                        invalid_boxes += 1

                    name = class_name(model.names, int(class_id))
                    class_counts[name] += 1
                    per_class[int(class_id)]["detections"] += 1
                    per_class[int(class_id)]["confidence_sum"] += float(confidence)

                total_detections += frame_detection_count
                mean_confidence = (
                    float(np.mean(confs)) if len(confs) else 0.0
                )
                total_duration = (time.perf_counter() - frame_started) * 1000.0
                total_frame_ms.append(total_duration)

                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "video_time_seconds": (
                            round(frame_index / video_fps, 6) if video_fps > 0 else ""
                        ),
                        "decode_ms": round(decode_duration, 4),
                        "inference_ms": round(infer_duration, 4),
                        "total_frame_ms": round(total_duration, 4),
                        "detection_count": frame_detection_count,
                        "valid_box_count": frame_valid,
                        "invalid_box_count": frame_invalid,
                        "out_of_frame_box_count": frame_outside,
                        "mean_confidence": round(mean_confidence, 6),
                        "class_counts": json.dumps(dict(class_counts), sort_keys=True),
                    }
                )

                if frame_count % 100 == 0:
                    print(f"Analyzed {frame_count} frames...")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        cap.release()

    wall_seconds = time.perf_counter() - run_started
    wall_fps = frame_count / wall_seconds if wall_seconds > 0 else 0.0
    per_class_summary = {}
    for class_id, values in sorted(per_class.items()):
        count = int(values["detections"])
        per_class_summary[class_name(model.names, class_id)] = {
            "class_id": class_id,
            "detections": count,
            "mean_confidence": (
                round(values["confidence_sum"] / count, 6) if count else 0.0
            ),
        }

    summary = {
        "mode": "offline_recorded_video_only",
        "video": str(args.video),
        "model": str(args.model),
        "device": args.device,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "video_metadata": {
            "width": width,
            "height": height,
            "reported_fps": video_fps,
            "reported_frame_count": reported_frame_count,
        },
        "results": {
            "analyzed_frames": frame_count,
            "frames_with_detections": frames_with_detections,
            "total_detections": total_detections,
            "valid_boxes": valid_boxes,
            "invalid_geometry_boxes": invalid_boxes,
            "out_of_frame_boxes": out_of_frame_boxes,
            "wall_time_seconds": round(wall_seconds, 4),
            "overall_processing_fps": round(wall_fps, 3),
            "decode_ms_mean": round(statistics.fmean(decode_ms_samples), 4)
            if decode_ms_samples
            else 0.0,
            "inference_ms_mean": round(statistics.fmean(inference_ms), 4)
            if inference_ms
            else 0.0,
            "inference_ms_median": round(statistics.median(inference_ms), 4)
            if inference_ms
            else 0.0,
            "inference_ms_p95": round(percentile(inference_ms, 95), 4),
            "total_frame_ms_mean": round(statistics.fmean(total_frame_ms), 4)
            if total_frame_ms
            else 0.0,
        },
        "per_class": per_class_summary,
        "outputs": {"per_frame_csv": str(csv_path)},
        "limitations": [
            "This report measures offline inference on the supplied recording only.",
            "It does not measure live capture latency or game input latency.",
            "Detection counts alone do not establish precision or recall; labeled ground truth is required.",
        ],
    }

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nOffline audit complete.")
    print(f"Frames analyzed: {frame_count}")
    print(f"Overall processing FPS: {wall_fps:.2f}")
    print(f"Mean inference latency: {summary['results']['inference_ms_mean']:.2f} ms")
    print(f"P95 inference latency: {summary['results']['inference_ms_p95']:.2f} ms")
    print(f"JSON report: {json_path}")
    print(f"Frame CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
