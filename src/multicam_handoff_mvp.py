#!/usr/bin/env python3
"""STRIDER VISION two-camera anonymous ReID handoff demonstration."""

from __future__ import annotations

import argparse
import csv
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
ZONE = np.array([[0.68, 0.08], [0.97, 0.08], [0.97, 0.92], [0.68, 0.92]], dtype=np.float32)


@dataclass
class Camera:
    name: str
    path: Path
    model_path: Path
    tracker: Path
    video: cv2.VideoCapture = field(init=False)
    model: YOLO = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    fps: float = field(init=False)
    states: dict[int, tuple[np.ndarray, int, set[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.video = cv2.VideoCapture(str(self.path))
        if not self.video.isOpened():
            raise RuntimeError(f"Could not open {self.name}: {self.path}")
        self.model = YOLO(str(self.model_path))  # Separate tracker state per camera.
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) or 25.0

    def read(self, confidence: float) -> tuple[np.ndarray | None, np.ndarray, list[int], np.ndarray]:
        ok, frame = self.video.read()
        if not ok:
            return None, np.empty((0, 4), np.float32), [], np.empty(0, np.float32)
        result = self.model.track(frame, classes=[0], conf=confidence, persist=True, tracker=str(self.tracker), verbose=False)[0]
        if result.boxes is None or result.boxes.id is None:
            return frame, np.empty((0, 4), np.float32), [], np.empty(0, np.float32)
        return frame, result.boxes.xyxy.cpu().numpy(), result.boxes.id.int().cpu().tolist(), result.boxes.conf.cpu().numpy()

    def close(self) -> None:
        self.video.release()


class MultiCameraMVP:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.cameras = [
            Camera(args.camera_a_name, args.camera_a, args.model, args.tracker),
            Camera(args.camera_b_name, args.camera_b, args.model, args.tracker),
        ]
        for _ in range(args.camera_b_offset):
            self.cameras[1].video.read()
        self.reid = ReID(str(args.reid_model), imgsz=448, device=self.device)
        self.reference_camera: int | None = None
        self.reference_id: int | None = None
        self.gallery: deque[np.ndarray] = deque(maxlen=args.gallery_size)
        self.reference_center: np.ndarray | None = None
        self.click: tuple[int, int] | None = None
        self.missing = 0
        self.frame_number = 0
        self.alerts: deque[dict[str, str]] = deque(maxlen=7)
        self.cooldown: dict[str, float] = {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.alert_csv = args.alerts.open("w", newline="", encoding="utf-8")
        self.alert_writer = csv.DictWriter(self.alert_csv, fieldnames=["time_s", "event", "camera", "anonymous_id", "detail", "evidence"])
        self.alert_writer.writeheader()
        height = self.cameras[0].height
        self.writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), self.cameras[0].fps, (self.cameras[0].width * 2 + args.panel_width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create {args.output}")

    def embedding(self, frame: np.ndarray, box: np.ndarray, camera: Camera) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(camera.width, x2), min(camera.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        values = self.reid(frame, np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]], dtype=np.float32))
        if values is None or len(values) == 0:
            return None
        value = values[0]
        return np.asarray(value.cpu().numpy() if hasattr(value, "cpu") else value, dtype=np.float32).flatten()

    @staticmethod
    def similarity(first: np.ndarray, second: np.ndarray) -> float:
        norm = np.linalg.norm(first) * np.linalg.norm(second)
        return float(np.dot(first, second) / norm) if norm else -1.0

    def event(self, frame: np.ndarray, event: str, camera: Camera, track_id: int, detail: str) -> None:
        now = self.frame_number / self.cameras[0].fps
        key = f"{event}:{camera.name}:{track_id}"
        if now - self.cooldown.get(key, -999) < self.args.alert_cooldown:
            return
        self.cooldown[key] = now
        evidence = self.args.evidence_dir / f"{event.lower()}_{camera.name}_{track_id}_{now:.1f}.jpg"
        cv2.imwrite(str(evidence), frame)
        item = {"time_s": f"{now:.1f}", "event": event, "camera": camera.name, "anonymous_id": str(track_id), "detail": detail, "evidence": str(evidence)}
        self.alert_writer.writerow(item)
        self.alert_csv.flush()
        self.alerts.appendleft(item)
        print(f"{event}: {camera.name} anonymous ID {track_id} — {detail}")

    def zone_points(self, camera: Camera) -> np.ndarray:
        return (ZONE * np.array([camera.width, camera.height])).astype(np.int32)

    def select(self, data: list[tuple[np.ndarray | None, np.ndarray, list[int], np.ndarray]]) -> None:
        if self.click is None:
            return
        click_x, click_y = self.click
        self.click = None
        camera_index = 0 if click_x < self.cameras[0].width else 1
        local_x = click_x if camera_index == 0 else click_x - self.cameras[0].width
        frame, boxes, ids, _ = data[camera_index]
        if frame is None:
            return
        for box, track_id in zip(boxes, ids):
            if box[0] <= local_x <= box[2] and box[1] <= click_y <= box[3]:
                value = self.embedding(frame, box, self.cameras[camera_index])
                if value is not None:
                    self.reference_camera, self.reference_id = camera_index, track_id
                    self.gallery.clear(); self.gallery.append(value)
                    self.reference_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    self.missing = 0
                    print(f"Selected anonymous subject {track_id} on {self.cameras[camera_index].name}")
                return

    def update_reference(self, data: list[tuple[np.ndarray | None, np.ndarray, list[int], np.ndarray]]) -> bool:
        if self.reference_camera is None or self.reference_id is None:
            return False
        frame, boxes, ids, _ = data[self.reference_camera]
        if frame is not None and self.reference_id in ids:
            index = ids.index(self.reference_id)
            value = self.embedding(frame, boxes[index], self.cameras[self.reference_camera])
            if value is not None: self.gallery.append(value)
            self.reference_center = np.array([(boxes[index][0] + boxes[index][2]) / 2, (boxes[index][1] + boxes[index][3]) / 2])
            self.missing = 0
            return True
        self.missing += 1
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for camera_index, (candidate_frame, candidate_boxes, candidate_ids, _) in enumerate(data):
            if candidate_frame is None: continue
            for index, box in enumerate(candidate_boxes):
                value = self.embedding(candidate_frame, box, self.cameras[camera_index])
                if value is not None and self.gallery:
                    candidates.append((max(self.similarity(value, known) for known in self.gallery), camera_index, index, value))
        if candidates:
            score, camera_index, index, value = max(candidates)
            if score >= self.args.reid_threshold:
                previous = self.cameras[self.reference_camera].name
                self.reference_camera, self.reference_id = camera_index, data[camera_index][2][index]
                self.gallery.append(value); self.missing = 0
                self.event(data[camera_index][0], "CROSS_CAMERA_HANDOFF", self.cameras[camera_index], self.reference_id, f"reidentified from {previous}; similarity {score:.2f}")
                return True
        return False

    def decorate(self, camera_index: int, frame: np.ndarray, boxes: np.ndarray, ids: list[int], confs: np.ndarray) -> np.ndarray:
        camera = self.cameras[camera_index]
        polygon = self.zone_points(camera)
        cv2.polylines(frame, [polygon], True, (45, 70, 230), 2)
        cv2.putText(frame, "RESTRICTED ZONE", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, .55, (45, 70, 230), 2)
        for box, track_id, confidence in zip(boxes, ids, confs):
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            inside = cv2.pointPolygonTest(polygon, (float(center[0]), float(center[1])), False) >= 0
            old = camera.states.get(track_id)
            activity = "STANDING"
            if old:
                speed = np.linalg.norm(center - old[0]) / np.hypot(camera.width, camera.height) * camera.fps
                activity = "WALKING" if speed < self.args.running_speed else "RUNNING"
            old_zones = old[2] if old else set()
            zones = {"restricted"} if inside else set()
            camera.states[track_id] = (center, self.frame_number, zones)
            if inside and "restricted" not in old_zones: self.event(frame, "RESTRICTED_ZONE_ENTRY", camera, track_id, "entered protected area")
            if inside and activity == "RUNNING": self.event(frame, "RUNNING_IN_RESTRICTED_ZONE", camera, track_id, "high-speed movement detected")
            selected = camera_index == self.reference_camera and track_id == self.reference_id
            color = (0, 255, 255) if selected else ((track_id * 97) % 256, (track_id * 231) % 256, (track_id * 123) % 256)
            x1, y1, x2, y2 = map(int, box); cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{'SELECTED' if selected else 'ANON'}-{track_id} | {activity} | {confidence:.2f}"
            cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .47, color, 2)
        cv2.rectangle(frame, (0, 0), (220, 40), (22, 25, 31), -1)
        cv2.putText(frame, camera.name, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (80, 220, 255), 2)
        return frame

    def panel(self, height: int, active: bool) -> np.ndarray:
        panel = np.full((height, self.args.panel_width, 3), (22, 25, 31), np.uint8)
        cv2.putText(panel, "STRIDER VISION", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, .76, (80, 220, 255), 2)
        cv2.putText(panel, "Anonymous cross-camera safety", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, .40, (215, 215, 215), 1)
        status = "ACTIVE" if active else "SEARCHING" if self.reference_id is not None else "CLICK A SUBJECT"
        cv2.putText(panel, f"REFERENCE: {status}", (18, 108), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 255), 1)
        cv2.putText(panel, f"CAMERA: {self.cameras[self.reference_camera].name if self.reference_camera is not None else '-'}", (18, 135), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1)
        cv2.putText(panel, f"ANON ID: {self.reference_id if self.reference_id is not None else '-'}", (18, 162), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1)
        cv2.putText(panel, "EXPLAINABLE EVENTS", (18, 215), cv2.FONT_HERSHEY_SIMPLEX, .52, (80, 220, 255), 1)
        y = 245
        for alert in self.alerts:
            cv2.putText(panel, f"{alert['event']}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, .43, (70, 160, 255), 1)
            cv2.putText(panel, f"{alert['camera']} | ID {alert['anonymous_id']}", (18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (220, 220, 220), 1)
            y += 46
        cv2.putText(panel, "No face recognition", (18, height - 30), cv2.FONT_HERSHEY_SIMPLEX, .40, (155, 155, 155), 1)
        return panel

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and y < self.cameras[0].height and x < self.cameras[0].width * 2: self.click = (x, y)

    def run(self) -> None:
        if not self.args.headless:
            cv2.namedWindow("STRIDER VISION Multi-Camera", cv2.WINDOW_NORMAL); cv2.setMouseCallback("STRIDER VISION Multi-Camera", self.mouse)
        while True:
            self.frame_number += 1
            data = [camera.read(self.args.confidence) for camera in self.cameras]
            if all(item[0] is None for item in data): break
            for index, item in enumerate(data):
                if item[0] is None: data[index] = (np.zeros((self.cameras[0].height, self.cameras[0].width, 3), np.uint8), np.empty((0, 4), np.float32), [], np.empty(0, np.float32))
            self.select(data); active = self.update_reference(data)
            left = self.decorate(0, data[0][0], data[0][1], data[0][2], data[0][3])
            right = self.decorate(1, data[1][0], data[1][1], data[1][2], data[1][3])
            if right.shape != left.shape: right = cv2.resize(right, (left.shape[1], left.shape[0]))
            dashboard = np.hstack((left, right, self.panel(left.shape[0], active)))
            self.writer.write(dashboard)
            if not self.args.headless:
                cv2.imshow("STRIDER VISION Multi-Camera", dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            if self.args.max_frames and self.frame_number >= self.args.max_frames: break
        for camera in self.cameras: camera.close()
        self.writer.release(); self.alert_csv.close(); cv2.destroyAllWindows()
        print(f"Done. Output: {self.args.output}")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STRIDER VISION two-camera anonymous handoff MVP")
    parser.add_argument("--camera-a", type=Path, default=ROOT / "datasets/CustomVideos/strangers1.mp4")
    parser.add_argument("--camera-b", type=Path, default=ROOT / "datasets/CustomVideos/strangers4.mp4")
    parser.add_argument("--camera-a-name", default="CAM-A"); parser.add_argument("--camera-b-name", default="CAM-B")
    parser.add_argument("--camera-b-offset", type=int, default=90, help="start CAM-B this many frames later for a simulated handoff")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo26n.pt"); parser.add_argument("--reid-model", type=Path, default=ROOT / "yolo26n-reid.onnx"); parser.add_argument("--tracker", type=Path, default=ROOT / "custom_botsort.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "output/multicam_handoff.mp4"); parser.add_argument("--alerts", type=Path, default=ROOT / "output/multicam_alerts.csv"); parser.add_argument("--evidence-dir", type=Path, default=ROOT / "output/multicam_evidence")
    parser.add_argument("--confidence", type=float, default=.5); parser.add_argument("--reid-threshold", type=float, default=.70); parser.add_argument("--running-speed", type=float, default=.08); parser.add_argument("--gallery-size", type=int, default=20); parser.add_argument("--alert-cooldown", type=float, default=4.0); parser.add_argument("--panel-width", type=int, default=340)
    parser.add_argument("--headless", action="store_true"); parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    MultiCameraMVP(arguments()).run()
