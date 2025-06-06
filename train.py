"""
Codi per fer entrenar i fer validació amb ultralytics
"""


from ultralytics import YOLO
from ultralytics import RTDETR



validation=False

if __name__ == "__main__":

    if validation:
        model = YOLO("./runs/detect/train27/weights/best.pt")

        results = model.val(data="dataset/data.yaml", imgsz=1024)

        print(results)
    else:

        model = YOLO("yolo11x.yaml")    #yaml per entrenar desde 0 o .pt per carregar pesos

        results = model.train(data="dataset/data.yaml", epochs=10000, imgsz=1024,  augment=True, single_cls=True, batch=8, pretrained=False, lrf=5e-5, lr0=5e-5,
                              optimizer="SGD", weight_decay=0.0001, warmup_epochs=15.0, warmup_momentum=0.8, warmup_bias_lr=0.1,conf=0.01, iou=0.1 )



