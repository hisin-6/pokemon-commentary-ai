"""
Label Studio ML Backend - YOLO ボール検出

label_studio_ml.api の init_app を使った正規の起動方法。
use_reloader=False で Windows での即時終了バグを回避。
"""
import os
import sys

# model.py と同じディレクトリを import パスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from label_studio_ml.api import init_app
from model import YOLOBallDetector

app = init_app(model_class=YOLOBallDetector, model_dir=os.environ.get("MODEL_DIR", "tmp/ls_ml_jobs"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9090))
    print(f"[ML Backend] Starting on port {port}")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
