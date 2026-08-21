# STRIDER VISION

STRIDER VISION is a privacy-first CCTV safety intelligence prototype. It detects
and tracks people using temporary anonymous IDs, preserves a selected person's
continuity through short occlusions, and produces explainable safety alerts with
saved evidence.

The project does **not** use facial recognition or store a person-identity
database.

## Capabilities

- Real-time person detection and multi-object tracking with YOLO and BoT-SORT
- Anonymous person-of-interest selection by clicking a track
- Appearance-based ReID recovery after a short occlusion or tracker ID switch
- Two-camera anonymous handoff demonstration
- Virtual restricted zones and motion-state estimates
- Explainable alerts for zone entry, running in a protected area, lost reference,
  and cross-camera handoff
- Rendered dashboard, CSV alert log, and timestamped evidence frames

## Repository layout

```text
.
├── src/
│   ├── safety_mvp.py             # Single-camera dashboard MVP
│   ├── multicam_handoff_mvp.py   # Two-camera ReID handoff MVP
│   └── behavior_analysis/        # Experimental video-action classifier
├── run_mvp.sh                    # Single-camera launcher
├── run_multicam_demo.sh          # Two-camera launcher
├── zones.demo.json               # Normalized restricted-zone configuration
├── custom_botsort.yaml           # BoT-SORT + ReID tracker configuration
├── Dockerfile                    # Development environment
└── requirements.txt              # Python dependencies
```

## Setup

### Docker development environment

```sh
./dev.sh build
./dev.sh run
```

The shared development-container configuration is CPU-compatible, so teammates
without an NVIDIA GPU can run the project normally. PyTorch automatically falls
back to CPU when CUDA is unavailable.

To enable GPU acceleration on a machine with NVIDIA Container Toolkit, add
`"--gpus=all"` as an additional value in that machine's local `runArgs` array
before reopening the container. Keep this local-only change out of shared Git
commits so it does not block CPU-only teammates.

### Local assets

Models, datasets, recordings, checkpoints, evidence, and rendered videos are
intentionally ignored by Git. Before running the demos, provide these local
files:

```text
yolo26n.pt
yolo26n-reid.onnx
datasets/CustomVideos/<your-video>.mp4
```

Install dependencies outside Docker when needed:

```sh
pip install -r requirements.txt
```

## Single-camera demo

```sh
./run_mvp.sh \
  --input datasets/CustomVideos/strangers1.mp4 \
  --zones zones.demo.json
```

Click a person to select an anonymous person of interest. Press `q` to stop.
The dashboard video, alert CSV, and evidence frames are written under `output/`.

For a reliable non-interactive backup video:

```sh
./run_mvp.sh --headless --input datasets/CustomVideos/strangers1.mp4
```

## Two-camera handoff demo

```sh
./run_multicam_demo.sh \
  --camera-a datasets/CustomVideos/strangers1.mp4 \
  --camera-b datasets/CustomVideos/strangers4.mp4
```

Select a person in either panel. When their source track is lost, the system
searches both feeds using the anonymous appearance gallery and records a
`CROSS_CAMERA_HANDOFF` event when the similarity score exceeds the configured
threshold. Use a single feed with `--camera-b-offset 90` to demonstrate a
controlled, simulated handoff.

## Privacy and scope

STRIDER VISION is a prototype for operator-assisted safety monitoring. It uses
temporary tracker IDs and local appearance embeddings only; it does not perform
facial recognition, identify individuals, or make automated enforcement
decisions. The activity labels are trajectory-based motion estimates, not a
validated action-recognition model.

## GitHub hygiene

Do not commit CCTV recordings, personal data, model weights, generated videos,
evidence frames, checkpoints, or local runtime state. These paths are excluded
in `.gitignore` by design.
