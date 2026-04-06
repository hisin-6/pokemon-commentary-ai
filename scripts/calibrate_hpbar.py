"""
HPバー座標キャリブレーションスクリプト

動画の特定フレームからHPバーのROI（関心領域）を検出して保存する。
実行後 debug/hpbar_calib_*.png を確認し、HPバーの座標を記録する。

使い方:
  venv\Scripts\python.exe scripts/calibrate_hpbar.py --input "D:\ゲーム録画\2026-03-28 23-55-54.mp4"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ==============================================================================
# HPバーの候補ROI（1920x1080 基準）
# 実際のゲームに合わせてここを調整する
# ==============================================================================

# (x1, y1, x2, y2, ラベル)
CANDIDATE_ROIS = [
    # ── 自分側（画面左下） ──────────────────────────────────────
    (  10, 840, 240, 1000, "player_slot0"),
    ( 260, 840, 490, 1000, "player_slot1"),

    # ── 相手側（画面右上） ──────────────────────────────────────
    (1330,  80, 1570,  240, "opponent_slot0"),
    (1590,  80, 1830,  240, "opponent_slot1"),
]

# HPバー色（HSV）
HP_COLOR_RANGES = [
    # 緑（高HP）
    (np.array([40,  80, 80]), np.array([90, 255, 255])),
    # 黄（中HP）
    (np.array([15,  80, 80]), np.array([40, 255, 255])),
    # 赤（低HP）
    (np.array([ 0, 100, 80]), np.array([10, 255, 255])),
    (np.array([170, 100, 80]), np.array([180, 255, 255])),
]

ROI_COLORS = {
    "player_slot0":   (  0, 200,   0),  # 緑
    "player_slot1":   (  0, 150,   0),  # 濃緑
    "opponent_slot0": (200,   0,   0),  # 赤
    "opponent_slot1": (150,   0,   0),  # 濃赤
}


def detect_hpbar_in_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """ROI内のHPバー候補ピクセルを検出してマスクと割合を返す。"""
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HP_COLOR_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)

    # 水平方向に走査: 左から右へ連続する着色列の最右端を検出
    col_has_color = mask.any(axis=0)   # shape: (width,)
    if col_has_color.any():
        rightmost = int(np.where(col_has_color)[0][-1])
        total_w = x2 - x1
        pct = rightmost / total_w * 100
    else:
        rightmost = 0
        pct = 0.0

    return mask, rightmost, pct


def annotate_frame(frame: np.ndarray, rois, results) -> np.ndarray:
    """ROIと検出結果をフレームに描画する。"""
    out = frame.copy()

    for (x1, y1, x2, y2, label), (mask, rightmost, pct) in zip(rois, results):
        color = ROI_COLORS[label]

        # ROI枠
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # HPバー検出位置（縦線）
        if pct > 0:
            bar_x = x1 + rightmost
            cv2.line(out, (bar_x, y1), (bar_x, y2), (0, 255, 255), 2)
            cv2.putText(out, f"{pct:.1f}%", (x1 + 4, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # HPバー着色マスクをROI内に半透明でオーバーレイ
        roi_out = out[y1:y2, x1:x2]
        overlay = roi_out.copy()
        overlay[mask > 0] = (0, 255, 200)
        cv2.addWeighted(overlay, 0.4, roi_out, 0.6, 0, roi_out)

    return out


def process_video(video_path: str, target_seconds: list[float]) -> None:
    """指定秒数のフレームを抽出してキャリブレーション画像を保存する。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    for sec in target_seconds:
        frame_no = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] フレーム {frame_no} ({sec}s) を読み取れませんでした")
            continue

        results = [
            detect_hpbar_in_roi(frame, x1, y1, x2, y2)
            for x1, y1, x2, y2, _ in CANDIDATE_ROIS
        ]

        annotated = annotate_frame(frame, CANDIDATE_ROIS, results)
        out_path = debug_dir / f"hpbar_calib_{sec:.0f}s.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"[{sec:.0f}s] 保存: {out_path}")
        for (x1, y1, x2, y2, label), (_, rightmost, pct) in zip(CANDIDATE_ROIS, results):
            print(f"  {label:20s}: bar_right={x1+rightmost:4d}px  HP≈{pct:5.1f}%  ROI=({x1},{y1})-({x2},{y2})")

    cap.release()
    print("\n完了。debug/hpbar_calib_*.png を確認してROI座標を調整してください。")


def main():
    parser = argparse.ArgumentParser(description="HPバーROIキャリブレーション")
    parser.add_argument("--input", required=True, help="動画ファイルのパス")
    parser.add_argument(
        "--seconds", nargs="+", type=float,
        default=[5, 30, 60, 90, 120],
        help="抽出する秒数リスト（デフォルト: 5 30 60 90 120）"
    )
    args = parser.parse_args()
    process_video(args.input, args.seconds)


if __name__ == "__main__":
    main()
