"""
終了画面検出データセット準備スクリプト

Label Studio JSON エクスポート → YOLO 形式変換 + train/val 分割

使い方:
    # Label Studio からエクスポートした JSON を指定
    venv\\Scripts\\python.exe scripts/prepare_end_screen_dataset.py annotations.json

    # val 比率を変更（デフォルト: 0.2）
    venv\\Scripts\\python.exe scripts/prepare_end_screen_dataset.py annotations.json --val-ratio 0.15

出力先:
    data/end_screen_dataset/
    ├── images/train/   # 学習用画像
    ├── images/val/     # 検証用画像
    ├── labels/train/   # 学習用ラベル（YOLO形式）
    └── labels/val/     # 検証用ラベル（YOLO形式）
"""

import argparse
import json
import os
import random
import re
import shutil
from urllib.parse import parse_qs, urlparse, unquote

# Label Studio ラベル名 → YOLO クラスID
LABEL_MAP = {"battle_end": 0}

# 画像の元フォルダ
RAW_DIR = "data/end_screen_dataset/raw"
DATASET_DIR = "data/end_screen_dataset"


def _extract_stem(image_url: str) -> str:
    """Label Studio の画像URL からファイル名（拡張子なし）を取得する。"""
    parsed = urlparse(image_url)
    qs = parse_qs(parsed.query)
    if "d" in qs:
        d_path = unquote(qs["d"][0])
        filename = d_path.replace("\\", "/").split("/")[-1]
    else:
        filename = os.path.basename(image_url.split("?")[0])
    return os.path.splitext(filename)[0]


def convert_and_split(annotations_file: str, val_ratio: float, seed: int) -> None:
    with open(annotations_file, encoding="utf-8") as f:
        tasks = json.load(f)

    # アノテーション済みタスクだけ処理
    annotated = []
    skipped = 0

    for task in tasks:
        image_url = task["data"].get("image", "")
        stem = _extract_stem(image_url)

        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        results = annotations[0].get("result", [])
        lines = []

        for r in results:
            if r.get("type") != "rectanglelabels":
                continue

            v = r["value"]
            labels = v.get("rectanglelabels", [])
            if not labels:
                continue

            label = labels[0]
            cls_id = LABEL_MAP.get(label)
            if cls_id is None:
                print(f"  警告: 未知のラベル '{label}' をスキップ")
                continue

            # Label Studio: 左上基準・パーセント → YOLO: 中心基準・正規化
            x_pct = v["x"] / 100
            y_pct = v["y"] / 100
            w_pct = v["width"] / 100
            h_pct = v["height"] / 100

            x_center = x_pct + w_pct / 2
            y_center = y_pct + h_pct / 2

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_pct:.6f} {h_pct:.6f}")

        annotated.append((stem, lines))

    print(f"アノテーション済み: {len(annotated)} 件 / スキップ: {skipped} 件")

    # train/val 分割
    random.seed(seed)
    random.shuffle(annotated)
    val_count = max(1, int(len(annotated) * val_ratio))
    val_items = annotated[:val_count]
    train_items = annotated[val_count:]

    print(f"train: {len(train_items)} 件 / val: {val_count} 件")

    # ディレクトリ作成
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # ファイルをコピー＆ラベル書き出し
    missing_images = []

    for split, items in [("train", train_items), ("val", val_items)]:
        for stem, lines in items:
            # 画像コピー
            # Label Studio が UUID プレフィックスを付ける場合に対応
            # 例: "01a3f13b-20260329_222540_806818" → "20260329_222540_806818"
            raw_stem = re.sub(r'^[0-9a-f]{8}-', '', stem)
            src_img = os.path.join(RAW_DIR, f"{raw_stem}.png")
            dst_img = os.path.join(DATASET_DIR, "images", split, f"{stem}.png")
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                missing_images.append(src_img)

            # ラベル書き出し
            dst_lbl = os.path.join(DATASET_DIR, "labels", split, f"{stem}.txt")
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    if missing_images:
        print(f"\n警告: 画像が見つからなかったファイル ({len(missing_images)} 件):")
        for p in missing_images:
            print(f"  {p}")

    print(f"\n完了！データセット出力先: {DATASET_DIR}")
    print("\n学習コマンド:")
    print(
        "venv\\Scripts\\yolo.exe train "
        "model=yolov8n.pt "
        f"data={DATASET_DIR}/data.yaml "
        "imgsz=640 epochs=50 batch=8 patience=15 "
        "name=train_end_screen"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio JSON → YOLO 形式変換 + train/val 分割"
    )
    parser.add_argument("annotations_file", help="Label Studio からエクスポートした JSON ファイル")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="検証データの割合 (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="シャッフル用ランダムシード (default: 42)",
    )
    args = parser.parse_args()
    convert_and_split(args.annotations_file, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
