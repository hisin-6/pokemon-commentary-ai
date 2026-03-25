"""
ボール検出デバッグスクリプト
使い方:
  venv/Scripts/python.exe test_yolo.py debug/yolo_turn_start.png
  venv/Scripts/python.exe test_yolo.py debug/yolo_turn_start.png --conf 0.05
  venv/Scripts/python.exe test_yolo.py debug/  # フォルダ内の最新5枚を処理
"""

import argparse
import sys
from pathlib import Path

import cv2

BALL_ROIS = {
    "opponent_balls": (0.83, 0.01, 1.00, 0.20),
    "player_balls":   (0.00, 0.72, 0.10, 0.97),
}
BALL_CLASS_NAMES = {0: "ball_alive", 1: "ball_faint", 2: "ball_status"}


def crop_roi(frame, ratio):
    h, w = frame.shape[:2]
    x1 = int(w * ratio[0])
    y1 = int(h * ratio[1])
    x2 = int(w * ratio[2])
    y2 = int(h * ratio[3])
    return frame[y1:y2, x1:x2], (x1, y1)


def run(image_path: Path, model_path: str, conf: float):
    from ultralytics import YOLO
    model = YOLO(model_path)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] 画像を読み込めません: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"\n{'='*60}")
    print(f"画像: {image_path.name}  ({w}x{h})")
    print(f"{'='*60}")

    total_passed = 0
    for roi_name, roi_ratio in BALL_ROIS.items():
        crop, (x_off, y_off) = crop_roi(frame, roi_ratio)
        if crop.size == 0:
            print(f"  [{roi_name}] ROIが空")
            continue

        ch, cw = crop.shape[:2]

        # ROIクロップ画像を保存
        roi_out = image_path.parent / f"_roi_{image_path.stem}_{roi_name}.png"
        cv2.imwrite(str(roi_out), crop)

        # conf=0.01 で生スコアを全件確認
        results = model(crop, conf=0.01, verbose=False)
        all_boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                label = BALL_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                all_boxes.append((label, score, x1, y1, x2, y2))

        passed = [b for b in all_boxes if b[1] >= conf]
        total_passed += len(passed)

        print(f"\n  [{roi_name}] ROI={cw}x{ch}  ROI保存→ {roi_out.name}")
        if not all_boxes:
            print(f"    ⚠ conf=0.01 でも検出ゼロ（モデルがROI画像に反応していない）")
        else:
            print(f"    conf=0.01: {len(all_boxes)}件  conf>={conf}: {len(passed)}件")
            for label, score, x1, y1, x2, y2 in sorted(all_boxes, key=lambda b: -b[1])[:10]:
                marker = "✅" if score >= conf else "  "
                print(f"    {marker} {label:12s} conf={score:.3f}  box=({x1},{y1},{x2},{y2})")

    print(f"\n合計検出数 (conf>={conf}): {total_passed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="画像ファイルまたはフォルダ")
    parser.add_argument("--model", default="runs/detect/train7/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        images = sorted(p.glob("yolo_turn*.png"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        print(f"フォルダモード: 最新{len(images)}枚を処理")
    else:
        images = [p]

    for img in images:
        run(img, args.model, args.conf)


if __name__ == "__main__":
    main()
