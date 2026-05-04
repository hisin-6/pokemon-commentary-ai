"""
COMMAND テキストの bbox 座標を動画から収集するスクリプト。

「COMMAND」または類似文字列（OCR誤読含む）が検出されたフレームの
座標・同フレームの全テキストを CSV に出力する。

使い方:
    venv\Scripts\python.exe scripts/detect_command_bbox.py --input D:\path\to\battle.mp4
    venv\Scripts\python.exe scripts/detect_command_bbox.py --input D:\path\to\battle.mp4 --skip 15 --output logs/command_bbox.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.capture.screen_capture import init_reader, run_ocr

# COMMAND に似た文字列にマッチする正規表現（OCR誤読対応）
# C/c, O/0, M, A/a, N, D の順で、多少の誤読（1文字程度の置換）を許容
_COMMAND_RE = re.compile(
    r'[Cc][O0oQ][Mm][Mm][Aa@][Nn][Dd]',
    re.IGNORECASE,
)


def _is_command_like(text: str) -> bool:
    """COMMAND またはそれに近い OCR 誤読テキストかどうか判定する。"""
    t = text.strip().upper().replace('0', 'O').replace('@', 'A')
    return bool(_COMMAND_RE.search(text)) or t == "COMMAND"


def _bbox_center(bbox) -> tuple[float, float]:
    """EasyOCR bbox（4点リスト）から中心座標を返す。"""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bbox_topleft(bbox) -> tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description="COMMAND bbox 座標収集")
    parser.add_argument("--input", required=True, help="動画ファイルパス")
    parser.add_argument("--skip", type=int, default=15, help="フレームスキップ数（デフォルト15）")
    parser.add_argument("--output", default="", help="CSV 出力先（省略時は logs/command_bbox_<timestamp>.csv）")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else Path("logs") / f"command_bbox_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_path.parent.mkdir(exist_ok=True)

    print("EasyOCR 初期化中...")
    reader = init_reader(gpu=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"動画を開けません: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画: {args.input}  fps={fps:.1f}  総フレーム={total}  skip={args.skip}")
    print(f"出力: {out_path}")

    frame_idx = 0
    found = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "time_sec",
            "text", "conf",
            "cx", "cy", "x1", "y1",
            "other_texts",
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.skip == 0:
                results = run_ocr(reader, frame, preprocess_hp=False)
                all_texts = [r["text"] for r in results]

                for r in results:
                    if _is_command_like(r["text"]):
                        bbox = r["bbox"]
                        cx, cy = _bbox_center(bbox)
                        x1, y1 = _bbox_topleft(bbox)
                        others = " | ".join(t for t in all_texts if t != r["text"])
                        writer.writerow([
                            frame_idx,
                            round(frame_idx / fps, 2),
                            r["text"],
                            round(r["confidence"], 3),
                            round(cx, 1), round(cy, 1),
                            round(x1, 1), round(y1, 1),
                            others,
                        ])
                        found += 1

                if frame_idx % (args.skip * 50) == 0:
                    print(f"  フレーム {frame_idx}/{total} 処理中... (COMMAND検出: {found}件)")

            frame_idx += 1

    cap.release()
    print(f"\n完了: COMMAND系テキスト {found} 件 → {out_path}")


if __name__ == "__main__":
    main()
