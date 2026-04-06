"""
HPバー座標グリッド確認スクリプト

既存のデバッグ画像に座標グリッド（50px間隔）を重ねて保存する。
出力画像でHPバーの正確なピクセル座標を読み取れる。

使い方:
  venv\Scripts\python.exe scripts/hpbar_grid.py debug\ocr_turn_004.png
"""

import sys
from pathlib import Path
import cv2
import numpy as np


def add_grid(src_path: str, grid_step: int = 50) -> None:
    img = cv2.imread(src_path)
    if img is None:
        print(f"[ERROR] 読み込めません: {src_path}", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]
    out = img.copy()

    # 縦線
    for x in range(0, w, grid_step):
        color = (0, 255, 0) if x % 100 == 0 else (0, 180, 0)
        thick = 2 if x % 100 == 0 else 1
        cv2.line(out, (x, 0), (x, h), color, thick)
        if x % 100 == 0:
            cv2.putText(out, str(x), (x + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 横線
    for y in range(0, h, grid_step):
        color = (0, 255, 0) if y % 100 == 0 else (0, 180, 0)
        thick = 2 if y % 100 == 0 else 1
        cv2.line(out, (0, y), (w, y), color, thick)
        if y % 100 == 0:
            cv2.putText(out, str(y), (2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # HPバーが存在しそうな範囲をハイライト（現在の推定値）
    HIGHLIGHT = [
        # (x1, y1, x2, y2, label, color_bgr)
        (  10, 840, 240, 1000, "player_s0 ROI(current)", (255, 100,   0)),
        ( 260, 840, 490, 1000, "player_s1 ROI(current)", (255, 150,   0)),
        (1330,  80, 1570,  240, "opp_s0 ROI(current)",   (  0, 100, 255)),
        (1590,  80, 1830,  240, "opp_s1 ROI(current)",   (  0, 150, 255)),
    ]
    for x1, y1, x2, y2, label, col in HIGHLIGHT:
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    src = Path(src_path)
    dst = src.parent / f"{src.stem}_grid.png"
    cv2.imwrite(str(dst), out)
    print(f"保存: {dst}  (画像サイズ: {w}x{h})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python hpbar_grid.py <画像パス>")
        sys.exit(1)
    add_grid(sys.argv[1])
