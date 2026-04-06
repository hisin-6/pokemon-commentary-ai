"""
診断用OCRロギングパイプライン
============================
対戦画面からOCRテキスト・YOLO検出結果をひたすら記録する。
実況・Bedrock・音声合成は一切行わない。

目的:
  - どのフレームでどのテキストが取れているかを生データで記録
  - ユーザーが「ここが試合開始」「ここが1ターン目」と注釈を書き込む
  - ターン検出・ポケモン名検出の改善に活用

出力:
  logs/ocr_diag_YYYYMMDD_HHMMSS.txt   人間が読める形式（注釈書き込み用）
  logs/ocr_diag_YYYYMMDD_HHMMSS.jsonl 構造化データ（プログラム解析用）

使用例:
  venv\\Scripts\\python.exe scripts/ocr_logger.py
  venv\\Scripts\\python.exe scripts/ocr_logger.py --camera 3 --ball-model runs/detect/train7/weights/best.pt
  venv\\Scripts\\python.exe scripts/ocr_logger.py --camera 3 --ball-model runs/detect/train7/weights/best.pt --end-model runs/detect/train_end_screen2/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.capture.screen_capture import DiffDetector, init_reader, run_ocr

# ─── ロギング設定 ────────────────────────────────────────────────────────────

def _setup_logging(txt_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(txt_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)


log = logging.getLogger(__name__)

# 画面上半分（y < この値）= 相手エリア（1080p基準）
_OPPONENT_Y_MAX = 500
# コマンドメニューエリア（y > この値）= 技選択UI
_COMMAND_Y_MIN = 700


# ─── OCR生データの整形 ──────────────────────────────────────────────────────

def _area_label(center_y: float) -> str:
    """Y座標からエリアラベルを返す。"""
    if center_y < _OPPONENT_Y_MAX:
        return "相手"
    elif center_y > _COMMAND_Y_MIN:
        return "コマンド"
    else:
        return "自分"


_DISPLAY_CONF_MIN = 0.3  # TXT表示用しきい値（JSONL は全件保存）


def _format_ocr_raw(ocr_results: list[dict]) -> tuple[str, list[dict]]:
    """
    OCR結果を整形する。
    - JSONL: 全件保存（structured に全データ）
    - TXT表示: conf >= _DISPLAY_CONF_MIN のみ表示、除外件数を末尾に付記

    Returns:
        (表示用文字列, 構造化リスト)
    """
    if not ocr_results:
        return "(テキストなし)", []

    structured = []
    display_parts = []
    low_conf_count = 0

    for r in ocr_results:
        text = r["text"].strip()
        conf = r["confidence"]
        bbox = r.get("bbox", [])
        if bbox:
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            area = _area_label(center_y)
        else:
            center_x, center_y = 960.0, 540.0
            area = "不明"

        structured.append({
            "text": text,
            "conf": round(conf, 3),
            "area": area,
            "cx": round(center_x, 0),
            "cy": round(center_y, 0),
        })

        if conf >= _DISPLAY_CONF_MIN:
            display_parts.append(f"{text}({conf:.2f},{area})")
        else:
            low_conf_count += 1

    if not display_parts:
        display_str = "(テキストなし)"
    else:
        display_str = " / ".join(display_parts)
        if low_conf_count > 0:
            display_str += f"  (+{low_conf_count}件 low_conf)"

    return display_str, structured


# ─── YOLOボール検出サマリー ──────────────────────────────────────────────────

def _format_ball_state(state) -> tuple[str, dict]:
    """BattleState からボール情報を整形する。"""
    pb = state.player_balls
    ob = state.opponent_balls

    # 終了画面スコア
    end_scores = []
    for d in state.detections:
        if d.label == "battle_end":
            end_scores.append(f"{d.confidence:.2f}")

    ball_str = (
        f"自alive={pb.alive} faint={pb.faint} / "
        f"相alive={ob.alive} faint={ob.faint}"
    )
    if end_scores:
        ball_str += f" / 終了画面={','.join(end_scores)}"

    ball_dict = {
        "player_alive": pb.alive,
        "player_faint": pb.faint,
        "opponent_alive": ob.alive,
        "opponent_faint": ob.faint,
        "end_screen_scores": end_scores,
    }
    return ball_str, ball_dict


# ─── メインループ ────────────────────────────────────────────────────────────

def run(
    camera_index: int,
    ball_model_path: str | None,
    end_model_path: str | None,
    conf: float,
    debug_balls: bool = False,
) -> None:
    # 出力ファイル準備
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"ocr_diag_{ts}.txt"
    jsonl_path = log_dir / f"ocr_diag_{ts}.jsonl"

    _setup_logging(txt_path)

    log.info("=" * 70)
    log.info(f"OCR診断ロガー 開始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"カメラ: {camera_index} / テキストログ: {txt_path} / JSONL: {jsonl_path}")
    log.info("注釈の追加方法: ログファイルに # BATTLE_START / # TURN_1 / # BATTLE_END 等を書き込む")
    log.info("=" * 70)

    # カメラ初期化
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error(f"カメラ {camera_index} を開けませんでした（OBS仮想カメラが起動中か確認してください）")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(10):
        cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"カメラ {camera_index} オープン完了: {w}x{h}")

    # EasyOCR初期化
    reader = init_reader(gpu=True)

    # YOLO初期化（オプション）
    yolo = None
    if ball_model_path or end_model_path:
        from src.capture.yolo_detector import YoloDetector
        yolo = YoloDetector(
            ball_model_path=ball_model_path,
            end_model_path=end_model_path,
            conf=conf,
        )

    diff_detector = DiffDetector()
    start_time = time.perf_counter()
    frame_no = 0

    log.info("キャプチャ開始（Ctrl+C で終了）")
    log.info("-" * 70)

    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("フレーム取得失敗、再試行...")
                time.sleep(0.1)
                continue

            frame_no += 1
            elapsed = time.perf_counter() - start_time
            elapsed_str = f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}"

            # 差分スコア
            _, diff_score = diff_detector.detect(frame)

            # OCR（フィルタなし・全テキスト）
            ocr_results = run_ocr(reader, frame)
            ocr_str, ocr_structured = _format_ocr_raw(ocr_results)

            # YOLO（ボール検出は conf=0.15 の detect_balls() を使う）
            ball_str = ""
            ball_dict: dict = {}
            if yolo is not None:
                dbg_dir = "debug/ball_rois" if (debug_balls and frame_no <= 30) else None
                if dbg_dir:
                    Path(dbg_dir).mkdir(parents=True, exist_ok=True)
                state = yolo.detect_balls(frame, debug_dir=dbg_dir)
                # 終了画面検出結果を state.detections にマージ
                if yolo._end_model is not None:
                    end_results = yolo._end_model(
                        frame, conf=0.6, device=yolo._device, verbose=False
                    )
                    from src.capture.yolo_detector import Detection
                    for r in end_results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            state.detections.append(Detection(
                                class_id=0,
                                label="battle_end",
                                confidence=float(box.conf[0]),
                                bbox=tuple(int(v) for v in box.xyxy[0]),
                                roi_name="full",
                            ))
                ball_str, ball_dict = _format_ball_state(state)

            # コンソール＋テキストログ出力
            diff_marker = " !!!" if diff_score > 30 else ("  ! " if diff_score > 10 else "    ")
            log.info(f"[{elapsed_str}] [F{frame_no:04d}] diff={diff_score:5.1f}{diff_marker}")
            log.info(f"  OCR({len(ocr_results)}件): {ocr_str}")
            if ball_str:
                log.info(f"  YOLO: {ball_str}")

            # JSONL出力
            record = {
                "frame": frame_no,
                "elapsed": round(elapsed, 2),
                "elapsed_str": elapsed_str,
                "diff": round(diff_score, 2),
                "ocr": ocr_structured,
                "yolo": ball_dict,
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()

    except KeyboardInterrupt:
        log.info("-" * 70)
        log.info(f"終了（フレーム数: {frame_no} / 経過時間: {elapsed_str}）")
    finally:
        cap.release()
        jsonl_file.close()
        log.info(f"テキストログ: {txt_path}")
        log.info(f"JSONL: {jsonl_path}")


# ─── エントリポイント ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="診断用OCRロギングパイプライン（実況なし・生データ記録）"
    )
    parser.add_argument("--camera", type=int, default=3,
                        help="OBS仮想カメラのデバイス番号（デフォルト: 3）")
    parser.add_argument("--ball-model", metavar="PATH", default=None,
                        help="ボール検出モデルのパス（省略可）")
    parser.add_argument("--end-model", metavar="PATH", default=None,
                        help="終了画面検出モデルのパス（省略可）")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO信頼度閾値（デフォルト: 0.3）")
    parser.add_argument("--debug-balls", action="store_true",
                        help="最初の30フレームのボールROI切り出し画像を debug/ball_rois/ に保存")
    args = parser.parse_args()

    run(
        camera_index=args.camera,
        ball_model_path=args.ball_model,
        end_model_path=args.end_model,
        conf=args.conf,
        debug_balls=args.debug_balls,
    )


if __name__ == "__main__":
    main()
