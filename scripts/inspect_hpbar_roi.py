"""
HPバーROIのピクセル確認スクリプト

各スロットのROIをズームアップし、HP色検出マスクを重ねて表示する。
座標がHPバーに正確に当たっているか目視確認できる。

使い方:
    python scripts/inspect_hpbar_roi.py <画像パス> [--out debug/hpbar_inspect.png]
    python scripts/inspect_hpbar_roi.py <動画パス> --frame 150 [--out debug/hpbar_inspect.png]

出力:
    各スロット（player_0/1, opp_0/1）を ZOOM 倍率で拡大した行を並べた画像。
    上段: 元のROIクロップ（赤枠）
    中段: HP色検出マスク（白=HP色あり）
    下段: マスク適用後のピクセル（HP色のみ残す）
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── hpbar_analyzer.py と同じ定数 ──────────────────────────────────────────
_HP_COLOR_RANGES = [
    (np.array([18, 100, 120]), np.array([42, 255, 255])),   # 橙/黄
    (np.array([42, 100, 120]), np.array([85, 255, 255])),   # 緑
    (np.array([ 0, 120, 120]), np.array([10, 255, 255])),   # 赤
    (np.array([170, 120, 120]), np.array([180, 255, 255])), # 赤（折り返し）
]

SLOTS = {
    "player_0": dict(x1=178,  x2=405,  y1=1005, y2=1010, color=(0, 220, 0)),
    "player_1": dict(x1=574,  x2=801,  y1=1005, y2=1010, color=(0, 180, 0)),
    "opp_0":    dict(x1=1222, x2=1450, y1=110,  y2=130,  color=(0, 120, 255)),
    "opp_1":    dict(x1=1618, x2=1846, y1=110,  y2=130,  color=(0, 80,  220)),
}

ZOOM = 6          # 拡大倍率
GAP  = 8          # パネル間の隙間(px)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _mask_hp(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _HP_COLOR_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def _first_seg_width(col_colored: np.ndarray, gap_tol: int = 8) -> int:
    """hpbar_analyzer._first_segment_width と同実装"""
    in_seg, start, gap = False, 0, 0
    for i, v in enumerate(col_colored):
        if v:
            if not in_seg:
                start = i
                in_seg = True
            gap = 0
        else:
            if in_seg:
                gap += 1
                if gap > gap_tol:
                    return i - gap - start
    if in_seg:
        return len(col_colored) - gap - start
    return 0


def _make_panel(label: str, roi: np.ndarray, color: tuple) -> np.ndarray:
    """1スロット分の3行パネルを作成して返す。"""
    h, w = roi.shape[:2]
    zh, zw = h * ZOOM, w * ZOOM

    # 行1: 元画像（ズーム）+ 列スキャン結果
    row1 = cv2.resize(roi, (zw, zh), interpolation=cv2.INTER_NEAREST)

    # HP色マスク計算
    mask = _mask_hp(roi)
    col_colored = mask.any(axis=0)
    seg_w = _first_seg_width(col_colored)
    full_w = w  # 満タン幅 = ROI幅
    pct = seg_w / full_w * 100 if full_w > 0 else 0

    # 行2: マスク（グレースケール→BGR）
    row2 = cv2.resize(mask, (zw, zh), interpolation=cv2.INTER_NEAREST)
    row2 = cv2.cvtColor(row2, cv2.COLOR_GRAY2BGR)

    # 行3: マスク適用（HP色ピクセルのみ残す）
    masked_roi = roi.copy()
    masked_roi[mask == 0] = 0
    row3 = cv2.resize(masked_roi, (zw, zh), interpolation=cv2.INTER_NEAREST)

    # 各行に枠と列スキャン指示線を描く
    for row in (row1, row2, row3):
        cv2.rectangle(row, (0, 0), (zw - 1, zh - 1), color, 2)
        # セグメント終端を縦線で示す
        if seg_w > 0:
            lx = seg_w * ZOOM
            cv2.line(row, (lx, 0), (lx, zh - 1), (0, 0, 255), 2)

    # ラベルとHP%をrow1に描く
    txt = f"{label}  seg={seg_w}/{full_w}px  HP≈{pct:.1f}%"
    cv2.putText(row1, txt, (4, zh - 6), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(row1, txt, (4, zh - 6), FONT, 0.55, (0, 0, 0),       1, cv2.LINE_AA)

    # 3行を縦に結合
    gap_strip = np.zeros((GAP, zw, 3), dtype=np.uint8)
    panel = np.vstack([row1, gap_strip, row2, gap_strip, row3])
    return panel


def inspect(frame: np.ndarray, out_path: str) -> None:
    panels = []
    for name, s in SLOTS.items():
        roi = frame[s["y1"]:s["y2"], s["x1"]:s["x2"]]
        if roi.size == 0:
            continue
        p = _make_panel(name, roi, s["color"])
        panels.append(p)

    # 4パネルを横に並べる（高さを揃える）
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        dh = max_h - p.shape[0]
        if dh > 0:
            p = np.vstack([p, np.zeros((dh, p.shape[1], 3), dtype=np.uint8)])
        padded.append(p)

    sep = np.zeros((max_h, GAP * 2, 3), dtype=np.uint8)
    combined = padded[0]
    for p in padded[1:]:
        combined = np.hstack([combined, sep, p])

    # 凡例を上部に追加
    legend_h = 55
    legend = np.zeros((legend_h, combined.shape[1], 3), dtype=np.uint8)
    lines = [
        "Row1: ROI crop (zoom x{})  |  Row2: HP color mask (white=hit)  |  Row3: masked pixels".format(ZOOM),
        "Red vertical line = first-segment end (= current HP width)  |  coords from hpbar_analyzer.py",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(legend, txt, (6, 18 + i * 20), FONT, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
    result = np.vstack([legend, combined])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, result)
    print(f"[INFO] 保存: {out_path}")

    # 各スロットのHP推定値をコンソールにも出力
    print("\n=== HP推定サマリー ===")
    for name, s in SLOTS.items():
        roi = frame[s["y1"]:s["y2"], s["x1"]:s["x2"]]
        if roi.size == 0:
            print(f"  {name}: ROIが空")
            continue
        mask = _mask_hp(roi)
        col_colored = mask.any(axis=0)
        seg_w = _first_seg_width(col_colored)
        full_w = s["x2"] - s["x1"]
        pct = seg_w / full_w * 100 if full_w > 0 else 0
        print(f"  {name}: seg={seg_w}/{full_w}px  HP≈{pct:.1f}%"
              f"  ROI=x[{s['x1']}-{s['x2']}] y[{s['y1']}-{s['y2']}]")


def load_frame(path: str, frame_no: int) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"[ERROR] 読み込めません: {path}")
        return img
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        sys.exit(f"[ERROR] 開けません: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[ERROR] フレーム {frame_no} を取得できません")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="HPバーROIのピクセル確認ツール")
    ap.add_argument("input", help="画像または動画パス")
    ap.add_argument("--frame", type=int, default=0, help="動画のフレーム番号")
    ap.add_argument("--out", default="debug/hpbar_inspect.png", help="出力先")
    args = ap.parse_args()

    frame = load_frame(args.input, args.frame)
    print(f"[INFO] 読み込み完了: {frame.shape[1]}x{frame.shape[0]}  frame={args.frame}")
    inspect(frame, args.out)


if __name__ == "__main__":
    main()
