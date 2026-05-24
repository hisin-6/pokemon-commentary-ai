#!/usr/bin/env python3
"""
動画から対戦メッセージテキストクロップを収集し、EasyOCRファインチューニング用アノテーションを生成する。

Usage:
    # 新規収集
    python scripts/collect_ocr_crops.py --input "C:/path/to/battle.mp4" [--interval 5] [--out data/ocr_crops]
    # 既存データに追記
    python scripts/collect_ocr_crops.py --input "C:/path/to/battle2.mp4" --append [--out data/ocr_crops]
"""
import argparse
import json
import os
import sys

import cv2
import easyocr

# パイプラインの BattleMessageParser と同じメッセージROI
MSG_X1, MSG_Y1, MSG_X2, MSG_Y2 = 0, 740, 900, 930


def _video_id(video_path: str) -> str:
    """ファイル名からvideo_idを生成（スペース→アンダースコア）。"""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return stem.replace(" ", "_").replace(":", "")


def extract_frames(video_path: str, out_dir: str, interval_sec: float, prefix: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けませんでした: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * interval_sec))
    print(f"  FPS={fps:.1f}, 総フレーム={total}, 抽出ステップ={step}f ({interval_sec}秒ごと)")

    saved = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            path = os.path.join(out_dir, f"{prefix}_frame_{frame_idx:06d}.png")
            cv2.imwrite(path, frame)
            saved.append(path)
        frame_idx += 1
    cap.release()
    print(f"  {len(saved)} 枚保存 → {out_dir}")
    return saved


def collect_crops(frame_paths: list, crops_dir: str, reader: easyocr.Reader) -> list:
    os.makedirs(crops_dir, exist_ok=True)
    annotations = []
    total = len(frame_paths)

    for i, img_path in enumerate(frame_paths):
        if i % 50 == 0:
            print(f"  {i}/{total} フレーム処理中 ...")
        img = cv2.imread(img_path)
        if img is None:
            continue
        roi = img[MSG_Y1:MSG_Y2, MSG_X1:MSG_X2]
        results = reader.readtext(roi)

        for j, (bbox, text, conf) in enumerate(results):
            text = text.strip()
            if len(text) < 2:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx1 = max(0, int(min(xs)))
            cy1 = max(0, int(min(ys)))
            cx2 = min(roi.shape[1], int(max(xs)))
            cy2 = min(roi.shape[0], int(max(ys)))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = roi[cy1:cy2, cx1:cx2]
            stem = os.path.basename(img_path)[:-4]
            name = f"{stem}_{j:03d}.png"
            cv2.imwrite(os.path.join(crops_dir, name), crop)
            annotations.append({"file": name, "ocr_text": text, "conf": round(conf, 3)})

    return annotations


def main():
    parser = argparse.ArgumentParser(description="EasyOCRファインチューニング用データ収集")
    parser.add_argument("--input", required=True, help="入力動画ファイルパス")
    parser.add_argument("--interval", type=float, default=5.0, help="フレーム抽出間隔（秒）デフォルト5")
    parser.add_argument("--out", default="data/ocr_crops", help="出力ディレクトリ（デフォルト: data/ocr_crops）")
    parser.add_argument("--append", action="store_true", help="既存のannotations.jsonに追記する")
    args = parser.parse_args()

    video_path = args.input.replace("\\", "/")
    prefix = _video_id(video_path)

    frames_dir = os.path.join(args.out, "frames")
    crops_dir  = os.path.join(args.out, "text_crops")
    ann_path   = os.path.join(args.out, "annotations.json")

    print("=== Step 1: フレーム抽出 ===")
    frame_paths = extract_frames(video_path, frames_dir, args.interval, prefix)

    print("\n=== Step 2: EasyOCR 初期化 ===")
    reader = easyocr.Reader(["ja"], gpu=True)

    print("\n=== Step 3: テキストクロップ生成 ===")
    new_annotations = collect_crops(frame_paths, crops_dir, reader)

    os.makedirs(args.out, exist_ok=True)

    # appendモード: 既存データをロードしてマージ
    if args.append and os.path.exists(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            existing = json.load(f)
        merged = existing + new_annotations
        print(f"  既存 {len(existing)} 件 + 新規 {len(new_annotations)} 件 = {len(merged)} 件")
    else:
        merged = new_annotations

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完了 ===")
    print(f"  合計 {len(merged)} 件 → {ann_path}")
    print(f"  次のステップ: annotations.json の ocr_text を手修正してください")


if __name__ == "__main__":
    main()
