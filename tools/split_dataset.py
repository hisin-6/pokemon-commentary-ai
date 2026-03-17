"""
train/val 分割スクリプト
アノテーション済み画像（labelsにtxtがあるもの）を 8:2 で分割する

使い方:
  ドライラン（何が移動されるか確認）:
    venv\Scripts\python.exe tools/split_dataset.py

  実際に移動:
    venv\Scripts\python.exe tools/split_dataset.py --execute
"""

import argparse
import random
import shutil
from pathlib import Path

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"

TRAIN_IMAGES = DATASET_ROOT / "images" / "train"
TRAIN_LABELS = DATASET_ROOT / "labels" / "train"
VAL_IMAGES   = DATASET_ROOT / "images" / "val"
VAL_LABELS   = DATASET_ROOT / "labels" / "val"

VAL_RATIO = 0.2
SEED = 42


def main(execute: bool):
    # アノテーション済みのラベルファイルを取得
    label_files = sorted(TRAIN_LABELS.glob("*.txt"))
    total = len(label_files)

    if total == 0:
        print("ラベルファイルが見つかりません。パスを確認してください。")
        print(f"  探した場所: {TRAIN_LABELS}")
        return

    # ランダムに val 分を選ぶ
    random.seed(SEED)
    val_count = max(1, int(total * VAL_RATIO))
    val_labels = random.sample(label_files, val_count)
    train_count = total - val_count

    print(f"アノテーション済み合計: {total} 枚")
    print(f"  → train: {train_count} 枚 / val: {val_count} 枚 ({VAL_RATIO*100:.0f}%)")
    print()

    # 移動対象のペアを収集
    pairs = []
    missing_images = []

    for label_path in sorted(val_labels):
        stem = label_path.stem
        # 対応する画像を探す（png / jpg / jpeg に対応）
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = TRAIN_IMAGES / (stem + ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            missing_images.append(stem)
            continue

        pairs.append((image_path, label_path))

    if missing_images:
        print(f"⚠ 対応する画像が見つからないラベル ({len(missing_images)} 件):")
        for name in missing_images[:10]:
            print(f"  {name}.txt")
        if len(missing_images) > 10:
            print(f"  ... 他 {len(missing_images) - 10} 件")
        print()

    # 移動先の確認・実行
    if not execute:
        print("【ドライラン】以下のファイルが val/ に移動されます（--execute で実行）:")
        for img, lbl in pairs[:10]:
            print(f"  {img.name}")
        if len(pairs) > 10:
            print(f"  ... 他 {len(pairs) - 10} 件")
        print()
        print(f"合計 {len(pairs)} ペアを移動予定")
    else:
        VAL_IMAGES.mkdir(parents=True, exist_ok=True)
        VAL_LABELS.mkdir(parents=True, exist_ok=True)

        moved = 0
        for img_src, lbl_src in pairs:
            shutil.move(str(img_src), VAL_IMAGES / img_src.name)
            shutil.move(str(lbl_src), VAL_LABELS / lbl_src.name)
            moved += 1

        print(f"✅ 移動完了: {moved} ペア")
        print(f"  train/images: {len(list(TRAIN_IMAGES.glob('*.png'))) + len(list(TRAIN_IMAGES.glob('*.jpg')))} 枚")
        print(f"  val/images:   {len(list(VAL_IMAGES.glob('*.png'))) + len(list(VAL_IMAGES.glob('*.jpg')))} 枚")
        print(f"  train/labels: {len(list(TRAIN_LABELS.glob('*.txt')))} 枚")
        print(f"  val/labels:   {len(list(VAL_LABELS.glob('*.txt')))} 枚")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train/val データセット分割")
    parser.add_argument("--execute", action="store_true", help="実際にファイルを移動する（指定なしはドライラン）")
    args = parser.parse_args()

    main(execute=args.execute)
