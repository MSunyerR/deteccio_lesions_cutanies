"""
Codi per fer prediccions amb Ultralytics
"""

from ultralytics import YOLO

model = YOLO("./runs/detect/train27/weights/best.pt")


results = model.predict(
    source=".\\dataset\\images\\test\\",
    save=True,
    project="yolo11s1024init",
    name="results",
    show_labels=False,
    show_conf=False,
    imgsz=1024
)


