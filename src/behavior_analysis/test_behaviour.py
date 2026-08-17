import cv2
import torch
import torch.nn.functional as F
import numpy as np

from model import create_model


VIDEO = "/workspace/datasets/CCTV-Action-Recognition-Dataset-Kaggle/Videos/Videos/run/YOUTUBE_YouTubeCCTV022_run_1.mp4"
CHECKPOINT = "/workspace/checkpoints/behavior_best.pth"

CLASSES = ["stand", "walk", "run"]
NUM_FRAMES = 16


def load_frames(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = torch.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
    ).long()

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        success, frame = cap.read()

        if success:
            frames.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

    cap.release()

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    return np.stack(frames)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model, weights = create_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # -----------------------------
    # Load 16 frames
    # -----------------------------

    frames = load_frames(VIDEO)

    tensor = torch.from_numpy(frames)

    # [T, H, W, C] -> [T, C, H, W]
    tensor = tensor.permute(0, 3, 1, 2)

    # Same preprocessing used during training
    tensor = weights.transforms()(tensor)

    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)

    # -----------------------------
    # Prediction
    # -----------------------------

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )[0]

    prediction = probabilities.argmax().item()

    behavior = CLASSES[prediction]

    confidence = (
        probabilities[prediction].item() * 100
    )

    print(
        f"Prediction: {behavior} "
        f"({confidence:.2f}%)"
    )

    # -----------------------------
    # Play video
    # -----------------------------

    cap = cv2.VideoCapture(VIDEO)

    while True:

        success, frame = cap.read()

        if not success:
            break

        text = (
            f"{behavior.upper()} "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        cv2.imshow(
            "Behavior Detection",
            frame,
        )

        # Press Q to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()