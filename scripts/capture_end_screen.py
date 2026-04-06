"""
終了画面キャプチャスクリプト
対戦終了画面（勝ち・負け・降参）のトレーニング画像を収集する。

使い方:
    venv\Scripts\python.exe scripts/capture_end_screen.py

操作:
    s キー  : 今のフレームを保存
    q キー  : 終了
    Ctrl+C  : 終了

保存先: data/end_screen_dataset/raw/
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

CAMERA_INDEX = 3
SAVE_DIR = Path("data/end_screen_dataset/raw")
WINDOW_NAME = "End Screen Capture (s=save, q=quit)"


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] カメラ {CAMERA_INDEX} を開けません。OBS仮想カメラが起動しているか確認してください。")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    saved_count = 0
    print(f"[INFO] キャプチャ開始。保存先: {SAVE_DIR}")
    print("[INFO] 対戦終了画面が出たら s キーで保存。q キーで終了。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] フレーム取得失敗。リトライ...")
            time.sleep(0.1)
            continue

        # プレビュー表示（1280x720 にリサイズ）
        preview = cv2.resize(frame, (1280, 720))
        cv2.putText(
            preview,
            f"saved: {saved_count}  |  s=save  q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
            path = SAVE_DIR / f"{ts}.png"
            cv2.imwrite(str(path), frame)
            saved_count += 1
            print(f"[SAVE] {path}  (合計 {saved_count} 枚)")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] {saved_count} 枚保存しました → {SAVE_DIR}")


if __name__ == "__main__":
    main()
