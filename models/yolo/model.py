# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from engine.model import Model
from models import yolo
from nn.tasks import DetectionModel

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """YOLO モデルを初期化"""
        path = Path(model)
        # デフォルトの YOLO の初期化を継続
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """タスクの種類ごとにモデル、バリデーター、予測器、トレーナーをマッピングする。"""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }

