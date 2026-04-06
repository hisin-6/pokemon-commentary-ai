"""
HPバー座標精密測定スクリプト

画像の特定y行をスキャンして、HPバー色（緑/黄/橙/赤）の
x座標範囲を自動検出し、スロットごとの正確なROIを出力する。

使い方:
  venv\Scripts\python.exe scripts/hpbar_measure.py debug\ocr_turn_004.png
"""

import sys
from pathlib import Path
import cv2
import numpy as np


# ── HPバー色（HSV）──────────────────────────────────────────────────
HP_COLOR_RANGES_HSV = [
    (np.array([ 18, 100, 120]), np.array([ 42, 255, 255])),  # 橙/黄
    (np.array([ 42, 100, 120]), np.array([ 85, 255, 255])),  # 緑
    (np.array([  0, 120, 120]), np.array([ 10, 255, 255])),  # 赤
    (np.array([170, 120, 120]), np.array([180, 255, 255])),  # 赤（折り返し）
]

# ── スキャンするy行の候補（1920x1080基準）──────────────────────────
SCAN_Y_RANGES = {
    "player":   (940, 975),
    "opponent": ( 72, 100),
}

# ── 自分/相手側のx範囲（大まかな区域）──────────────────────────────
X_REGIONS = {
    "player":   (  0,  620),   # 画面左側
    "opponent": (1280, 1920),   # 画面右側
}


def get_hp_mask_row(frame: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    """指定y範囲・x範囲でHPバー色マスクを取得する（列方向に集約）。"""
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HP_COLOR_RANGES_HSV:
        mask |= cv2.inRange(hsv, lo, hi)
    # 列方向に集約: 1列でも着色ピクセルがあれば True
    return mask.any(axis=0)  # shape: (width,)


def find_hp_bar_segments(col_mask: np.ndarray, x_offset: int,
                         min_width: int = 30, gap_tolerance: int = 8
                         ) -> list[tuple[int, int]]:
    """
    連続する着色列のセグメントを検出する。
    min_width  以上の幅のセグメントのみ返す（ノイズ除去）。
    gap_tolerance px 以内のギャップはセグメントを繋げる。
    戻り値: [(x_start, x_end), ...]（絶対座標）
    """
    segments = []
    in_seg = False
    start = 0
    gap = 0

    for i, v in enumerate(col_mask):
        if v:
            if not in_seg:
                start = i
                in_seg = True
            gap = 0
        else:
            if in_seg:
                gap += 1
                if gap > gap_tolerance:
                    end = i - gap
                    if end - start >= min_width:
                        segments.append((x_offset + start, x_offset + end))
                    in_seg = False
                    gap = 0

    if in_seg:
        end = len(col_mask) - gap
        if end - start >= min_width:
            segments.append((x_offset + start, x_offset + end))

    return segments


def measure(image_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 読み込めません: {image_path}", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"画像サイズ: {w}x{h}\n")

    results: dict[str, list[tuple[int, int]]] = {}

    for side, (y1, y2) in SCAN_Y_RANGES.items():
        x1, x2 = X_REGIONS[side]
        col_mask = get_hp_mask_row(img, y1, y2, x1, x2)
        segs = find_hp_bar_segments(col_mask, x_offset=x1)
        results[side] = segs

        print(f"=== {side.upper()} (y={y1}-{y2}, x={x1}-{x2}) ===")
        for i, (sx, ex) in enumerate(segs):
            width = ex - sx
            print(f"  セグメント{i}: x={sx}-{ex}  幅={width}px")

        # 可視化: 検出セグメントを元画像に描画
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for sx, ex in segs:
            cv2.rectangle(vis, (sx, y1), (ex, y2), (0, 0, 255), 3)
            cv2.putText(vis, f"x={sx}-{ex}", (sx, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        out_path = Path(image_path).parent / f"hpbar_measure_{side}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"  → 保存: {out_path}\n")

    # まとめ
    print("=" * 60)
    print("【HpBarAnalyzer 用 ROI 候補】")
    print("自分側セグメント:", results.get("player", []))
    print("相手側セグメント:", results.get("opponent", []))
    print()

    player_segs = results.get("player", [])
    opponent_segs = results.get("opponent", [])
    y_p1, y_p2 = SCAN_Y_RANGES["player"]
    y_o1, y_o2 = SCAN_Y_RANGES["opponent"]

    if len(player_segs) >= 2:
        s0, s1 = player_segs[0], player_segs[1]
        print(f"player_slot0 ROI: ({s0[0]}, {y_p1}, {s0[1]}, {y_p2})")
        print(f"player_slot1 ROI: ({s1[0]}, {y_p1}, {s1[1]}, {y_p2})")
    if len(opponent_segs) >= 2:
        s0, s1 = opponent_segs[0], opponent_segs[1]
        print(f"opponent_slot0 ROI: ({s0[0]}, {y_o1}, {s0[1]}, {y_o2})")
        print(f"opponent_slot1 ROI: ({s1[0]}, {y_o1}, {s1[1]}, {y_o2})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python hpbar_measure.py <画像パス>")
        sys.exit(1)
    measure(sys.argv[1])
