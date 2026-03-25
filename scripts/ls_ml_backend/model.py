"""
Label Studio ML Backend - YOLO ボール検出モデル

Label Studio からの画像に対して YOLO 推論を実行し、
プレラベルを返す。

環境変数:
    YOLO_MODEL_PATH  : YOLOモデルパス (default: runs/detect/train/weights/best.pt)
    BALL_DATASET_ROOT: ball_dataset のルートパス (default: data/ball_dataset)
    CONF_THRESHOLD   : 検出信頼度の閾値 (default: 0.25)
"""

import os
from urllib.parse import parse_qs, urlparse

from label_studio_ml.model import LabelStudioMLBase

# クラスID → Label Studio ラベル名マッピング
LABEL_MAP = {0: "ball_alive", 1: "ball_faint", 2: "ball_status"}

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "runs/detect/train/weights/best.pt")
DATASET_ROOT = os.environ.get("BALL_DATASET_ROOT", "data/ball_dataset")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))


class YOLOBallDetector(LabelStudioMLBase):
    """YOLO を使ったボール検出 ML Backend。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ultralytics import YOLO

        model_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLOモデルが見つかりません: {model_path}\n"
                "先に仮モデルを学習してください:\n"
                "  venv\\Scripts\\python.exe -m ultralytics train "
                "model=yolov8n.pt data=data/ball_dataset/data.yaml epochs=20 batch=16"
            )
        self.model = YOLO(model_path)
        print(f"[ML Backend] モデルロード完了: {model_path}")

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            image_url = task["data"].get("image", "")
            image_path = self._url_to_local_path(image_url)

            if not image_path or not os.path.exists(image_path):
                print(f"[ML Backend] 画像が見つかりません: {image_path}")
                predictions.append({"result": [], "score": 0.0})
                continue

            results = self.model(image_path, conf=CONF_THRESHOLD, verbose=False)[0]
            img_h, img_w = results.orig_shape[:2]

            annotations = []
            scores = []

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = LABEL_MAP.get(cls_id, "ball_alive")

                annotations.append({
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": x1 / img_w * 100,
                        "y": y1 / img_h * 100,
                        "width": (x2 - x1) / img_w * 100,
                        "height": (y2 - y1) / img_h * 100,
                        "rectanglelabels": [label],
                    },
                })
                scores.append(conf)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            predictions.append({"result": annotations, "score": avg_score})
            print(f"[ML Backend] {os.path.basename(image_path)}: {len(annotations)} 件検出 (avg_conf={avg_score:.2f})")

        return predictions

    def _url_to_local_path(self, url: str) -> str:
        """
        Label Studio の画像URL をローカルファイルパスに変換する。

        Label Studio ローカルストレージ形式:
            /data/local-files/?d=Users%5Crotat%5C...%5Cimage.png
            → d パラメータはドライブレター省略の絶対パス (例: Users\rotat\...)
        """
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)

        if "d" in qs:
            rel_path = qs["d"][0]
            # ドライブレターを取得して補完 (例: C:)
            drive = os.path.splitdrive(os.getcwd())[0]
            return drive + os.sep + rel_path

        # fallback: URL をそのままパスとして使う
        return url
