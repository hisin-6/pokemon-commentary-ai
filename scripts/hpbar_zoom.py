"""
HPバー領域ズーム確認スクリプト

HPバーが存在しそうな領域を切り出して拡大保存する。
実際の座標確認用。

使い方:
  venv\Scripts\python.exe scripts/hpbar_zoom.py debug\ocr_turn_004.png
"""

import sys
from pathlib import Path
import cv2
import numpy as np


# 確認したいROI候補（現在の推定値・1920x1080基準）
ZOOM_ROIS = [
    # (x1, y1, x2, y2, label)
    # 自分側: HPバーが y=948-968 あたりと仮定して水平スキャン
    (  10, 940, 590, 975, "player_hpbar_strip"),
    # 相手側: HPバーが y=74-97 あたりと仮定
    (1290,  72, 1870,  100, "opponent_hpbar_strip"),
]


def zoom_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                scale: float = 4.0, grid: int = 10) -> np.ndarray:
    """領域を切り出してグリッド付きで拡大する。"""
    crop = img[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]
    out = cv2.resize(crop, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_NEAREST)
    oh, ow = out.shape[:2]
    step = int(grid * scale)

    # 縦線（ステップはoriginal px単位）
    for xi in range(0, w, grid):
        px = int(xi * scale)
        col = (0, 255, 0) if xi % 50 == 0 else (0, 120, 0)
        th = 2 if xi % 50 == 0 else 1
        cv2.line(out, (px, 0), (px, oh), col, th)
        if xi % 50 == 0:
            label = str(x1 + xi)
            cv2.putText(out, label, (px + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 横線
    for yi in range(0, h, grid):
        py = int(yi * scale)
        col = (0, 255, 0) if yi % 50 == 0 else (0, 120, 0)
        th = 2 if yi % 50 == 0 else 1
        cv2.line(out, (0, py), (ow, py), col, th)
        if yi % 50 == 0:
            label = str(y1 + yi)
            cv2.putText(out, label, (2, py + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return out


def main():
    if len(sys.argv) < 2:
        print("使い方: python hpbar_zoom.py <画像パス>")
        sys.exit(1)

    src_path = sys.argv[1]
    img = cv2.imread(src_path)
    if img is None:
        print(f"[ERROR] 読み込めません: {src_path}", file=sys.stderr)
        sys.exit(1)

    src = Path(src_path)
    debug_dir = src.parent

    for x1, y1, x2, y2, label in ZOOM_ROIS:
        zoomed = zoom_region(img, x1, y1, x2, y2, scale=6.0, grid=5)
        out_path = debug_dir / f"hpbar_zoom_{label}.png"
        cv2.imwrite(str(out_path), zoomed)
        print(f"保存: {out_path}")

    print("\n完了。hpbar_zoom_*.png でHPバーの正確な座標を読み取ってください。")
    print("各ピクセルが4倍に拡大されています（グリッド=10px間隔）。")


if __name__ == "__main__":
    main()
