# detection/detection_yolo.py

from ultralytics import YOLO
import config


class PigdetectorYOLO:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        """
        Ejecuta la detección y el seguimiento interno de YOLO (StrongSORT).
        Devuelve los resultados brutos.
        """
        results = self.model.track(
            source=frame,
            conf=0.35,
            iou=0.7,
            imgsz=640,          # más calidad de entrada
            agnostic_nms=False,
            augment=True,
            persist=True,
            max_det=6,
            tracker=config.TRACKER_TYPE,
            verbose=False
        )
        return results