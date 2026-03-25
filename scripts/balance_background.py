"""
背景画像（空ラベルファイル）を削減してクラスバランスを改善するスクリプト。

使い方:
  python scripts/balance_background.py --dry-run   # 削除対象を確認するだけ
  python scripts/balance_background.py              # 実際に削除

ラベルファイルのみ削除。対応する画像は残す（YOLOが自動スキップ）。
削除対象ファイルリストは deleted_bg_labels.txt に保存（復元用）。
"""

import argparse
import os
import random
from pathlib import Path


LABEL_DIR = Path("data/ball_dataset/labels/train")
LOG_FILE = Path("data/ball_dataset/deleted_bg_labels.txt")
TARGET_BG_COUNT = 720  # オブジェクト画像数と合わせる
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="削除せずに対象を表示するだけ")
    parser.add_argument("--target", type=int, default=TARGET_BG_COUNT, help=f"残す背景画像数（デフォルト: {TARGET_BG_COUNT}）")
    args = parser.parse_args()

    # 空ラベルファイルを収集
    all_labels = list(LABEL_DIR.glob("*.txt"))
    empty_labels = [f for f in all_labels if f.stat().st_size == 0]
    has_objects = [f for f in all_labels if f.stat().st_size > 0]

    print(f"=== データセット現状 ===")
    print(f"  ラベルファイル総数: {len(all_labels)}")
    print(f"  オブジェクトあり: {len(has_objects)}")
    print(f"  背景のみ（空）: {len(empty_labels)}")

    target_bg = min(args.target, len(empty_labels))
    n_delete = len(empty_labels) - target_bg

    print(f"\n=== 整理計画 ===")
    print(f"  残す背景ファイル数: {target_bg}")
    print(f"  削除する背景ファイル数: {n_delete}")
    print(f"  整理後の比率: {len(has_objects)}(obj) : {target_bg}(bg) = 1:{target_bg/len(has_objects):.2f}")

    if n_delete <= 0:
        print("\n削除対象なし。終了。")
        return

    # 削除対象をランダム選択
    random.seed(SEED)
    to_delete = random.sample(empty_labels, n_delete)
    to_delete.sort()

    if args.dry_run:
        print(f"\n[DRY RUN] 削除対象 {n_delete} 件（最初の10件）:")
        for f in to_delete[:10]:
            print(f"  {f.name}")
        if n_delete > 10:
            print(f"  ... あと {n_delete - 10} 件")
        print("\n実際に削除するには --dry-run なしで実行してください。")
        return

    # 削除対象を記録してから削除
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        for path in to_delete:
            f.write(str(path.resolve()) + "\n")
    print(f"\n削除対象リストを保存: {LOG_FILE}")

    # 削除実行
    deleted = 0
    for path in to_delete:
        path.unlink()
        deleted += 1

    print(f"削除完了: {deleted} 件")
    print(f"\n=== 整理後のデータセット ===")
    remaining = list(LABEL_DIR.glob("*.txt"))
    remaining_empty = [f for f in remaining if f.stat().st_size == 0]
    print(f"  ラベルファイル総数: {len(remaining)}")
    print(f"  オブジェクトあり: {len(has_objects)}")
    print(f"  背景のみ（空）: {len(remaining_empty)}")

    print(f"\n次のコマンドで学習を開始してください:")
    print(f"  venv\\Scripts\\yolo.exe train model=yolov8s.pt data=data/ball_dataset/data.yaml imgsz=640 epochs=150 batch=8 cls=2.0 fl_gamma=1.5 cos_lr=true patience=30 name=train7")


if __name__ == "__main__":
    main()
