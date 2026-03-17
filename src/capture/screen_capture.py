"""
Sprint 1: 画面キャプチャ + EasyOCR テキスト認識
--------------------------------------------------
2つのモードで動作する:
  - ファイルモード: --image <path> でスクリーンショットを読み込んでOCR（WSL2でもOK）
  - ライブモード:  --live でmssを使ってリアルタイムキャプチャ（Windowsで実行）
"""

from __future__ import annotations

import argparse
import time
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import easyocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── OCRリーダー（初回起動時にモデルDL、約500MB） ───────────────────────────

def init_reader(gpu: bool = True) -> easyocr.Reader:
    log.info("EasyOCRリーダーを初期化中（初回はモデルDLで1〜2分かかります）...")
    reader = easyocr.Reader(["ja", "en"], gpu=gpu)
    log.info("EasyOCRリーダー初期化完了")
    return reader


# ─── OCR処理 ────────────────────────────────────────────────────────────────

def run_ocr(reader: easyocr.Reader, image: np.ndarray) -> list[dict]:
    """
    画像に対してOCRを実行し、認識結果のリストを返す。

    Returns:
        [{"text": str, "confidence": float, "bbox": list}, ...]
    """
    results = reader.readtext(image)
    parsed = []
    for (bbox, text, confidence) in results:
        parsed.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bbox": bbox,
        })
    return parsed


def print_ocr_results(results: list[dict]) -> None:
    if not results:
        log.info("テキストが検出されませんでした")
        return

    log.info(f"検出テキスト数: {len(results)}")
    print("\n" + "=" * 50)
    print(f"{'テキスト':<30} {'信頼度':>8}")
    print("-" * 50)
    for r in results:
        marker = "✓" if r["confidence"] >= 0.5 else "?"
        print(f"{marker} {r['text']:<28} {r['confidence']:>8.1%}")
    print("=" * 50 + "\n")


# ─── 差分検出 ───────────────────────────────────────────────────────────────

class DiffDetector:
    """
    OpenCV差分検出でターン切替などのイベントを検知する。
    """

    DIFF_THRESHOLD = 30    # フレーム間のピクセル平均差分の閾値
    MIN_CHANGE_AREA = 1000  # 変化領域の最小面積（px²）

    def __init__(self):
        self._prev_frame: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Returns:
            (イベント発生フラグ, 差分スコア)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_frame = gray
            return False, 0.0

        diff = cv2.absdiff(self._prev_frame, gray)
        score = float(diff.mean())

        # 変化領域の面積チェック
        _, thresh = cv2.threshold(diff, self.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        changed_area = int(np.sum(thresh > 0))

        self._prev_frame = gray
        event_detected = score > self.DIFF_THRESHOLD and changed_area > self.MIN_CHANGE_AREA
        return event_detected, score


# ─── ファイルモード ──────────────────────────────────────────────────────────

def run_file_mode(
    image_path: str,
    reader: easyocr.Reader,
    yolo_model_path: str | None = None,
) -> None:
    """
    指定した画像ファイルにOCR（+ オプションでYOLO）を実行する（WSL2でのテスト用）。
    """
    path = Path(image_path)
    if not path.exists():
        log.error(f"ファイルが見つかりません: {image_path}")
        sys.exit(1)

    log.info(f"画像読み込み: {path}")
    image = cv2.imread(str(path))
    if image is None:
        log.error("画像の読み込みに失敗しました")
        sys.exit(1)

    log.info(f"画像サイズ: {image.shape[1]}x{image.shape[0]}")

    start = time.perf_counter()
    results = run_ocr(reader, image)
    elapsed = time.perf_counter() - start

    print_ocr_results(results)
    log.info(f"OCR処理時間: {elapsed:.2f}秒")

    if yolo_model_path is not None or yolo_model_path == "":
        _run_yolo_on_image(image, model_path=yolo_model_path or None, save_prefix=path.stem)


def _run_yolo_on_image(
    image: np.ndarray,
    model_path: str | None,
    save_prefix: str = "frame",
) -> None:
    """画像に対して YoloDetector を実行してログ出力＋可視化保存する。"""
    import sys
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.capture.yolo_detector import YoloDetector

    detector = YoloDetector(model_path=model_path)
    state = detector.detect(image)

    log.info(f"[YOLO] {state.summary()} (mode={state.mode})")
    if state.detections:
        for d in state.detections:
            log.info(f"  [{d.roi_name}] {d.label} {d.confidence:.1%}")

    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    save_path = debug_dir / f"yolo_{save_prefix}.png"
    vis = detector.draw_detections(image, state)
    cv2.imwrite(str(save_path), vis)
    log.info(f"YOLO可視化画像を保存: {save_path}")


# ─── ライブキャプチャモード ─────────────────────────────────────────────────

def run_live_mode(
    reader: easyocr.Reader,
    interval: float = 1.0,
    monitor_index: int = 1,
    yolo_model_path: str | None = None,
) -> None:
    """
    mssでリアルタイム画面キャプチャしOCRを実行する。
    Windows Pythonで実行する必要がある。
    Ctrl+C で終了。
    """
    try:
        import mss
    except ImportError:
        log.error("mssがインストールされていません: pip install mss")
        sys.exit(1)

    diff_detector = DiffDetector()
    turn = 0

    yolo_detector = None
    if yolo_model_path is not None:
        import sys
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.capture.yolo_detector import YoloDetector
        yolo_detector = YoloDetector(model_path=yolo_model_path if yolo_model_path else None)
        log.info(f"YOLOv8 有効: {yolo_model_path or 'yolov8n.pt（デフォルト）'}")

    log.info(f"ライブキャプチャ開始（間隔: {interval}秒、Ctrl+Cで終了）")

    with mss.mss() as sct:
        # 利用可能なモニター一覧を表示（0=全体, 1=プライマリ, 2=セカンダリ...）
        log.info(f"利用可能なモニター数: {len(sct.monitors) - 1}")
        for i, m in enumerate(sct.monitors):
            label = "全体" if i == 0 else f"モニター{i}"
            log.info(f"  [{i}] {label}: {m['width']}x{m['height']} (left={m['left']}, top={m['top']})")

        if monitor_index >= len(sct.monitors):
            log.error(f"モニター {monitor_index} は存在しません（利用可能: 0〜{len(sct.monitors)-1}）")
            sys.exit(1)

        monitor = sct.monitors[monitor_index]
        log.info(f"キャプチャ対象: モニター{monitor_index} ({monitor['width']}x{monitor['height']})")

        try:
            while True:
                start = time.perf_counter()

                # スクリーンショット取得
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 差分検出
                event_detected, score = diff_detector.detect(frame)
                log.debug(f"差分スコア: {score:.2f}")

                # YOLO検出（毎フレーム）
                if yolo_detector is not None:
                    state = yolo_detector.detect(frame)
                    if state.detections:
                        log.info(f"[YOLO] {state.summary()}")
                        for d in state.detections:
                            log.info(f"  [{d.roi_name}] {d.label} {d.confidence:.1%}")

                if event_detected:
                    turn += 1
                    log.info(f"[ターン {turn}] イベント検知！ 差分スコア: {score:.2f} → OCR実行")

                    results = run_ocr(reader, frame)
                    print_ocr_results(results)

                    # スクリーンショットを保存（デバッグ用）
                    debug_dir = Path("debug")
                    debug_dir.mkdir(exist_ok=True)
                    save_path = debug_dir / f"turn_{turn:03d}.png"
                    cv2.imwrite(str(save_path), frame)
                    log.info(f"スクリーンショット保存: {save_path}")

                    # YOLO可視化画像を保存（デバッグ用）
                    if yolo_detector is not None:
                        vis = yolo_detector.draw_detections(frame, state)
                        yolo_save_path = debug_dir / f"yolo_turn_{turn:03d}.png"
                        cv2.imwrite(str(yolo_save_path), vis)
                        log.info(f"YOLO可視化画像を保存: {yolo_save_path}")

                elapsed = time.perf_counter() - start
                sleep_time = max(0.0, interval - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            log.info(f"終了します（検知ターン数: {turn}）")


# ─── カメラキャプチャモード（OBS仮想カメラ用） ──────────────────────────────

def run_camera_mode(
    reader: easyocr.Reader,
    camera_index: int = 0,
    interval: float = 1.0,
    yolo_model_path: str | None = None,
) -> None:
    """
    cv2.VideoCapture でカメラ（OBS仮想カメラ等）からキャプチャしOCR+YOLOを実行する。
    OBS仮想カメラは固定解像度で出力されるためROIのズレが発生しない。
    Ctrl+C で終了。
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error(f"カメラ {camera_index} を開けませんでした（OBS仮想カメラが起動中か確認してください）")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # カメラ起動直後は黒フレームが続くためウォームアップ
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"カメラ {camera_index} オープン完了: {w}x{h}")

    yolo_detector = None
    if yolo_model_path is not None:
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.capture.yolo_detector import YoloDetector
        yolo_detector = YoloDetector(model_path=yolo_model_path if yolo_model_path else None)
        log.info(f"YOLOv8 有効: {yolo_model_path or 'yolov8n.pt（デフォルト）'}")

    diff_detector = DiffDetector()
    turn = 0

    log.info(f"カメラキャプチャ開始（間隔: {interval}秒、Ctrl+Cで終了）")

    try:
        while True:
            start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                log.warning("フレームの取得に失敗しました。再試行します...")
                time.sleep(0.5)
                continue

            # 差分検出
            event_detected, score = diff_detector.detect(frame)
            log.debug(f"差分スコア: {score:.2f}")

            # YOLO検出（毎フレーム）
            if yolo_detector is not None:
                state = yolo_detector.detect(frame)
                if state.detections:
                    log.info(f"[YOLO] {state.summary()}")
                    for d in state.detections:
                        log.info(f"  [{d.roi_name}] {d.label} {d.confidence:.1%}")

            if event_detected:
                turn += 1
                log.info(f"[ターン {turn}] イベント検知！ 差分スコア: {score:.2f} → OCR実行")

                results = run_ocr(reader, frame)
                print_ocr_results(results)

                debug_dir = Path("debug")
                debug_dir.mkdir(exist_ok=True)
                save_path = debug_dir / f"turn_{turn:03d}.png"
                cv2.imwrite(str(save_path), frame)
                log.info(f"スクリーンショット保存: {save_path}")

                if yolo_detector is not None:
                    vis = yolo_detector.draw_detections(frame, state)
                    yolo_save_path = debug_dir / f"yolo_turn_{turn:03d}.png"
                    cv2.imwrite(str(yolo_save_path), vis)
                    log.info(f"YOLO可視化画像を保存: {yolo_save_path}")

            elapsed = time.perf_counter() - start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        log.info(f"終了します（検知ターン数: {turn}）")
    finally:
        cap.release()


# ─── エントリポイント ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ポケモン対戦画面キャプチャ + OCR（Sprint 1）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="テスト用画像ファイルのパス（WSL2可）")
    group.add_argument("--live", action="store_true", help="ライブキャプチャモード（Windows推奨）")
    group.add_argument("--camera", metavar="DEVICE_INDEX", type=int,
                        help="OBS仮想カメラ等のカメラデバイスからキャプチャ（0始まりのデバイス番号）")

    parser.add_argument("--interval", type=float, default=1.0, help="キャプチャ間隔（秒、デフォルト: 1.0）")
    parser.add_argument("--monitor", type=int, default=1, help="キャプチャ対象モニター番号（1=プライマリ, 2=セカンダリ、デフォルト: 1）")
    parser.add_argument("--cpu", action="store_true", help="CPUモードで実行（GPU無効）")
    parser.add_argument("--yolo", metavar="MODEL_PATH", nargs="?", const="",
                        help="YOLOv8 検出を有効化（省略で yolov8n.pt、パス指定でカスタムモデル使用）")

    args = parser.parse_args()

    use_gpu = not args.cpu
    reader = init_reader(gpu=use_gpu)

    if args.image:
        run_file_mode(args.image, reader, yolo_model_path=args.yolo)
    elif args.live:
        run_live_mode(reader, interval=args.interval, monitor_index=args.monitor, yolo_model_path=args.yolo)
    else:
        run_camera_mode(reader, camera_index=args.camera, interval=args.interval, yolo_model_path=args.yolo)


if __name__ == "__main__":
    main()
