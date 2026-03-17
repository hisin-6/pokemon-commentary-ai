"""
学習データ収集ツール（Windows で実行）
--------------------------------------------------
ポケモン対戦画面をキャプチャして YOLO 学習用画像を蓄積する。

特徴:
  - 差分検出でほぼ同じフレームをスキップ（重複排除）
  - ROI の切り出し画像も同時保存（アノテーション確認用）
  - 状態異常・ボール数別の収集状況をリアルタイム表示

使用方法:
  # フルフレーム収集（推奨）
  venv\\Scripts\\python.exe tools/collect_training_data.py

  # モニター指定
  venv\\Scripts\\python.exe tools/collect_training_data.py --monitor 2

  # 収集間隔を短くする
  venv\\Scripts\\python.exe tools/collect_training_data.py --interval 1.5

収集のコツ:
  - 状態異常（やけど・まひ等）が発生したターンを多く含めること
  - 相手・自分両方の状態異常を収集する
  - ボール数が変わった直後（ポケモンが倒れた瞬間）も収集する
  - 最低 50〜100 枚 / クラスを目安にする
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.capture.screen_capture import DiffDetector
from src.capture.yolo_detector import ROIS, YoloDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── 設定 ────────────────────────────────────────────────────────────────────

SAVE_DIR      = Path("data/yolo_dataset/images/train")
ROI_SAVE_DIR  = Path("data/yolo_dataset/images/train/_roi_preview")  # ROI切り出し確認用
DIFF_SKIP_THRESHOLD = 5.0   # 差分スコアがこれ以下なら重複としてスキップ


# ─── メイン収集ループ ─────────────────────────────────────────────────────────

def _collect_loop(get_frame, interval: float, session_id: str) -> None:
    """フレーム取得関数を受け取って収集ループを実行する共通処理。"""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ROI_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    detector = DiffDetector()
    saved = 0
    skipped = 0

    while True:
        start = time.perf_counter()

        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        _, score = detector.detect(frame)

        if score <= DIFF_SKIP_THRESHOLD and saved > 0:
            skipped += 1
            sys.stdout.write(f"\r  保存済み: {saved} 枚 | スキップ: {skipped} 枚 | 差分スコア: {score:.1f}  ")
            sys.stdout.flush()
        else:
            filename = f"{session_id}_{saved:04d}.png"
            save_path = SAVE_DIR / filename
            cv2.imwrite(str(save_path), frame)
            _save_roi_preview(frame, SAVE_DIR / filename, ROI_SAVE_DIR / filename)
            saved += 1
            sys.stdout.write(f"\r  保存済み: {saved} 枚 | スキップ: {skipped} 枚 | 差分スコア: {score:.1f}  ")
            sys.stdout.flush()

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, interval - elapsed))


def collect(interval: float, monitor_index: int) -> None:
    try:
        import mss
    except ImportError:
        log.error("mss がインストールされていません: pip install mss")
        sys.exit(1)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("  ポケモン対戦 学習データ収集ツール（mss）")
    print("=" * 60)
    print(f"  保存先  : {SAVE_DIR}")
    print(f"  間隔    : {interval} 秒")
    print(f"  セッション: {session_id}")
    print("  Ctrl+C で終了")
    print("=" * 60)

    with mss.mss() as sct:
        if monitor_index >= len(sct.monitors):
            log.error(f"モニター {monitor_index} は存在しません（0〜{len(sct.monitors)-1}）")
            sys.exit(1)

        monitor = sct.monitors[monitor_index]
        log.info(f"キャプチャ開始: モニター{monitor_index} ({monitor['width']}x{monitor['height']})")

        def get_frame():
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        _collect_loop(get_frame, interval, session_id)


def collect_from_camera(interval: float, camera_index: int) -> None:
    """OBS仮想カメラ等からキャプチャして学習データを収集する。"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error(f"カメラ {camera_index} を開けませんでした")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # ウォームアップ
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("  ポケモン対戦 学習データ収集ツール（OBS仮想カメラ）")
    print("=" * 60)
    print(f"  カメラ  : {camera_index} ({w}x{h})")
    print(f"  保存先  : {SAVE_DIR}")
    print(f"  間隔    : {interval} 秒")
    print(f"  セッション: {session_id}")
    print("  Ctrl+C で終了")
    print("=" * 60)

    def get_frame():
        ret, frame = cap.read()
        return frame if ret else None

    try:
        _collect_loop(get_frame, interval, session_id)
    finally:
        cap.release()


def _save_roi_preview(
    frame: np.ndarray,
    original_path: Path,
    preview_path: Path,
) -> None:
    """ROI領域を可視化した確認用画像を保存する。"""
    vis = frame.copy()
    h, w = vis.shape[:2]

    colors = {
        "opponent_status": (0, 255, 255),
        "player_status":   (0, 255, 255),
        "opponent_balls":  (255, 165, 0),
        "player_balls":    (255, 165, 0),
    }

    for roi_name, roi_ratio in ROIS.items():
        x1 = int(w * roi_ratio[0])
        y1 = int(h * roi_ratio[1])
        x2 = int(w * roi_ratio[2])
        y2 = int(h * roi_ratio[3])
        color = colors.get(roi_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, roi_name, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), vis)


# ─── エントリポイント ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ポケモン対戦 YOLO学習データ収集ツール（Windows実行）"
    )
    parser.add_argument("--interval", type=float, default=2.0,
                        help="キャプチャ間隔（秒、デフォルト: 2.0）")
    parser.add_argument("--monitor", type=int, default=1,
                        help="キャプチャ対象モニター番号（デフォルト: 1）")
    parser.add_argument("--camera", type=int, default=None,
                        help="OBS仮想カメラのデバイス番号（指定時はmssの代わりにカメラを使用）")
    args = parser.parse_args()

    try:
        if args.camera is not None:
            collect_from_camera(interval=args.interval, camera_index=args.camera)
        else:
            collect(interval=args.interval, monitor_index=args.monitor)
    except KeyboardInterrupt:
        print("\n\n収集終了！")
        print(f"画像は {SAVE_DIR} に保存されました。")
        print()
        print("【次のステップ】")
        print("  1. LabelImg でアノテーション:")
        print(f"     labelImg {SAVE_DIR} {Path('data/yolo_dataset/classes.txt')}")
        print("  2. 生成された .txt ラベルを data/yolo_dataset/labels/train/ に移動")
        print("  3. 学習を実行:")
        print("     yolo detect train data=data/yolo_dataset/data.yaml model=yolov8n.pt epochs=50")


if __name__ == "__main__":
    main()
