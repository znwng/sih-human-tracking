#!/usr/bin/env python3
"""Demo-ready, privacy-first CCTV safety intelligence MVP.

The application intentionally uses anonymous tracker IDs only.  It provides
operator selection, lightweight ReID-based recovery, virtual zones, motion
state estimates, explainable alerts, and saved evidence frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.utils.reid import ReID


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONE = {
    "restricted": [[0.68, 0.08], [0.97, 0.08], [0.97, 0.92], [0.68, 0.92]]
}
COLORS = {"restricted": (45, 70, 230)}


@dataclass
class TrackState:
    center: np.ndarray
    frame: int
    activity: str = "STANDING"
    in_zones: set[str] = field(default_factory=set)


class SafetyMVP:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(args.model))
        self.reid = ReID(str(args.reid_model), imgsz=448, device=self.device)
        self.video = cv2.VideoCapture(str(args.input))
        if not self.video.isOpened():
            raise RuntimeError(f"Could not open input video: {args.input}")

        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.zones = self.load_zones(args.zones)
        self.track_states: dict[int, TrackState] = {}
        self.reference_id: int | None = None
        self.reference_gallery: deque[np.ndarray] = deque(maxlen=args.gallery_size)
        self.reference_center: np.ndarray | None = None
        self.lost_frames = 0
        self.click: tuple[int, int] | None = None
        self.alert_cooldowns: dict[tuple[int, str], float] = {}
        self.alerts: deque[dict[str, Any]] = deque(maxlen=8)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
            (self.width + args.panel_width, self.height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output}")
        self.alert_file = args.alerts.open("w", newline="", encoding="utf-8")
        self.alert_writer = csv.DictWriter(
            self.alert_file,
            fieldnames=["timestamp_s", "frame", "camera", "track_id", "event", "detail", "evidence"],
        )
        self.alert_writer.writeheader()

    def load_zones(self, path: Path | None) -> dict[str, np.ndarray]:
        source = DEFAULT_ZONE
        if path is not None:
            with path.open(encoding="utf-8") as file:
                source = json.load(file)
        zones: dict[str, np.ndarray] = {}
        for name, points in source.items():
            points_array = np.asarray(points, dtype=np.float32)
            if points_array.shape != (4, 2):
                raise ValueError(f"Zone '{name}' must have exactly four [x, y] points")
            if np.max(points_array) <= 1.0:
                points_array *= np.array([self.width, self.height])
            zones[name] = points_array.astype(np.int32)
        return zones

    @staticmethod
    def color_for_id(track_id: int) -> tuple[int, int, int]:
        return ((track_id * 97) % 256, (track_id * 231) % 256, (track_id * 123) % 256)

    @staticmethod
    def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return float(np.dot(first, second) / denominator) if denominator else -1.0

    def embedding(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        bbox = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]], dtype=np.float32)
        embeddings = self.reid(frame, bbox)
        if embeddings is None or len(embeddings) == 0:
            return None
        value = embeddings[0]
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        return np.asarray(value, dtype=np.float32).flatten()

    def in_zone(self, center: np.ndarray, zone: np.ndarray) -> bool:
        return cv2.pointPolygonTest(zone, (float(center[0]), float(center[1])), False) >= 0

    def motion_state(self, track_id: int, center: np.ndarray, frame_number: int) -> tuple[str, set[str]]:
        previous = self.track_states.get(track_id)
        zones = {name for name, polygon in self.zones.items() if self.in_zone(center, polygon)}
        activity = "STANDING"
        if previous:
            elapsed = max((frame_number - previous.frame) / self.fps, 1 / self.fps)
            normalized_speed = np.linalg.norm(center - previous.center) / np.hypot(self.width, self.height) / elapsed
            activity = "STANDING" if normalized_speed < 0.012 else "WALKING" if normalized_speed < self.args.running_speed else "RUNNING"
        self.track_states[track_id] = TrackState(center=center, frame=frame_number, activity=activity, in_zones=zones)
        return activity, zones

    def emit_alert(self, frame: np.ndarray, frame_number: int, track_id: int, event: str, detail: str) -> None:
        now = frame_number / self.fps
        key = (track_id, event)
        if now - self.alert_cooldowns.get(key, -999) < self.args.alert_cooldown:
            return
        self.alert_cooldowns[key] = now
        stamp = f"{now:07.2f}".replace(".", "_")
        evidence_path = self.args.evidence_dir / f"{event}_id{track_id}_{stamp}.jpg"
        cv2.imwrite(str(evidence_path), frame)
        record = {
            "timestamp_s": f"{now:.2f}", "frame": frame_number, "camera": self.args.camera_name,
            "track_id": track_id, "event": event, "detail": detail, "evidence": str(evidence_path),
        }
        self.alert_writer.writerow(record)
        self.alert_file.flush()
        self.alerts.appendleft(record)
        print(f"ALERT | {event} | anonymous ID {track_id} | {detail}")

    def select_reference(self, frame: np.ndarray, boxes: np.ndarray, ids: list[int]) -> None:
        if self.click is None:
            return
        x, y = self.click
        self.click = None
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                value = self.embedding(frame, box)
                if value is not None:
                    self.reference_id = track_id
                    self.reference_gallery.clear()
                    self.reference_gallery.append(value)
                    self.reference_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    self.lost_frames = 0
                    print(f"Selected anonymous reference ID {track_id}")
                return

    def recover_reference(self, frame: np.ndarray, boxes: np.ndarray, ids: list[int]) -> bool:
        if self.reference_id is None:
            return False
        if self.reference_id in ids:
            index = ids.index(self.reference_id)
            value = self.embedding(frame, boxes[index])
            if value is not None:
                self.reference_gallery.append(value)
            self.reference_center = np.array([(boxes[index][0] + boxes[index][2]) / 2, (boxes[index][1] + boxes[index][3]) / 2])
            self.lost_frames = 0
            return True
        self.lost_frames += 1
        if not self.reference_gallery or self.lost_frames > int(self.args.reacquire_seconds * self.fps):
            return False
        best: tuple[float, int, np.ndarray] | None = None
        for index, box in enumerate(boxes):
            value = self.embedding(frame, box)
            if value is None:
                continue
            appearance = max(self.cosine_similarity(value, item) for item in self.reference_gallery)
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            distance = np.linalg.norm(center - self.reference_center) / np.hypot(self.width, self.height) if self.reference_center is not None else 0
            score = 0.8 * appearance + 0.2 * max(0.0, 1.0 - distance / 0.4)
            if best is None or score > best[0]:
                best = (score, index, value)
        if best and best[0] >= self.args.reid_threshold:
            _, index, value = best
            old_id, self.reference_id = self.reference_id, ids[index]
            self.reference_gallery.append(value)
            self.reference_center = np.array([(boxes[index][0] + boxes[index][2]) / 2, (boxes[index][1] + boxes[index][3]) / 2])
            self.lost_frames = 0
            self.emit_alert(frame, self.frame_number, self.reference_id, "REIDENTIFIED", f"anonymous ID changed from {old_id} to {self.reference_id}")
            return True
        return False

    def draw_panel(self, frame: np.ndarray, frame_number: int, reference_active: bool) -> np.ndarray:
        panel = np.full((self.height, self.args.panel_width, 3), (24, 25, 31), dtype=np.uint8)
        cv2.putText(panel, "STRIDER VISION", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (80, 220, 255), 2)
        cv2.putText(panel, "Privacy-first safety intelligence", (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        lines = [
            f"CAMERA: {self.args.camera_name}",
            f"TIME: {frame_number / self.fps:06.1f}s",
            f"TRACKS: {len(self.track_states)}",
            f"REFERENCE: {'ACTIVE' if reference_active else 'SEARCHING' if self.reference_id else 'NONE'}",
            f"ANON ID: {self.reference_id if self.reference_id is not None else '-'}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (20, 112 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
        cv2.putText(panel, "RECENT EXPLAINABLE ALERTS", (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 255), 1)
        y = 305
        for alert in self.alerts:
            cv2.putText(panel, f"{alert['event']}  ID {alert['track_id']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 160, 255), 1)
            cv2.putText(panel, alert["detail"][:42], (20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (220, 220, 220), 1)
            y += 52
        cv2.putText(panel, "No face recognition. Anonymous IDs only.", (20, self.height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (155, 155, 155), 1)
        return np.hstack((frame, panel))

    def on_click(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and x < self.width:
            self.click = (x, y)

    def run(self) -> None:
        if not self.args.headless:
            cv2.namedWindow(self.args.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.args.window_name, self.on_click)
        self.frame_number = 0
        print(f"Processing {self.args.input} on {self.device}; output: {self.args.output}")
        while True:
            success, frame = self.video.read()
            if not success:
                break
            self.frame_number += 1
            if self.args.max_frames and self.frame_number > self.args.max_frames:
                break
            results = self.model.track(frame, classes=[0], conf=self.args.confidence, persist=True, tracker=str(self.args.tracker), verbose=False)
            result = results[0]
            boxes = np.empty((0, 4), dtype=np.float32)
            ids: list[int] = []
            confidences = np.empty(0, dtype=np.float32)
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().numpy()
            self.select_reference(frame, boxes, ids)
            reference_active = self.recover_reference(frame, boxes, ids)
            if self.reference_id is not None and not reference_active and self.lost_frames == int(self.args.lost_alert_seconds * self.fps):
                self.emit_alert(frame, self.frame_number, self.reference_id, "REFERENCE_LOST", f"not visible for {self.args.lost_alert_seconds:.1f} seconds")
            for box, track_id, confidence in zip(boxes, ids, confidences):
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                old_zones = self.track_states.get(track_id, TrackState(center, self.frame_number)).in_zones
                activity, zones = self.motion_state(track_id, center, self.frame_number)
                for zone in zones - old_zones:
                    self.emit_alert(frame, self.frame_number, track_id, "RESTRICTED_ZONE_ENTRY", f"entered {zone} zone")
                if activity == "RUNNING" and "restricted" in zones:
                    self.emit_alert(frame, self.frame_number, track_id, "RUNNING_IN_RESTRICTED_ZONE", "high-speed movement in restricted zone")
                x1, y1, x2, y2 = map(int, box)
                is_reference = track_id == self.reference_id
                color = (0, 255, 255) if is_reference else self.color_for_id(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{'SELECTED' if is_reference else 'ANON'}-{track_id} | {activity} | {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            for name, polygon in self.zones.items():
                cv2.polylines(frame, [polygon], True, COLORS.get(name, (255, 120, 40)), 2)
                cv2.putText(frame, name.upper(), tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.get(name, (255, 120, 40)), 2)
            dashboard = self.draw_panel(frame, self.frame_number, reference_active)
            self.writer.write(dashboard)
            if not self.args.headless:
                cv2.imshow(self.args.window_name, dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        self.video.release()
        self.writer.release()
        self.alert_file.close()
        cv2.destroyAllWindows()
        print(f"Done. Video: {self.args.output}; alert log: {self.args.alerts}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="STRIDER VISION CCTV safety intelligence MVP")
    value.add_argument("--input", type=Path, default=ROOT / "datasets/CustomVideos/strangers4.mp4")
    value.add_argument("--output", type=Path, default=ROOT / "output/mvp_demo.mp4")
    value.add_argument("--alerts", type=Path, default=ROOT / "output/alerts.csv")
    value.add_argument("--evidence-dir", type=Path, default=ROOT / "output/evidence")
    value.add_argument("--model", type=Path, default=ROOT / "yolo26n.pt")
    value.add_argument("--reid-model", type=Path, default=ROOT / "yolo26n-reid.onnx")
    value.add_argument("--tracker", type=Path, default=ROOT / "custom_botsort.yaml")
    value.add_argument("--zones", type=Path, help="JSON map of named zones to four normalized or pixel points")
    value.add_argument("--camera-name", default="CAM-A")
    value.add_argument("--confidence", type=float, default=0.5)
    value.add_argument("--running-speed", type=float, default=0.08, help="normalized diagonal lengths per second")
    value.add_argument("--reid-threshold", type=float, default=0.70)
    value.add_argument("--reacquire-seconds", type=float, default=6.0)
    value.add_argument("--lost-alert-seconds", type=float, default=3.0)
    value.add_argument("--alert-cooldown", type=float, default=4.0)
    value.add_argument("--gallery-size", type=int, default=20)
    value.add_argument("--panel-width", type=int, default=390)
    value.add_argument("--max-frames", type=int, help="process only this many frames (useful for smoke tests)")
    value.add_argument("--window-name", default="STRIDER VISION Safety MVP")
    value.add_argument("--headless", action="store_true", help="write outputs without opening an OpenCV window")
    return value


if __name__ == "__main__":
    SafetyMVP(parser().parse_args()).run()
