import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

input_path = "datasets/CustomVideos/PeopleWalkingOnStreet-1.mp4"
output_path = "people_tracking.mp4"

video = cv2.VideoCapture(input_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height),
)

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        classes=[0],
        conf=0.5,
        persist=True,
        tracker="custom_botsort.yaml",
        verbose=False,
    )

    annotated = results[0].plot()

    writer.write(annotated)

video.release()
writer.release()

print(f"Saved output to: {output_path}")
