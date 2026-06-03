#!/usr/bin/env python3
"""
annotations.json から LMDB用 train/val txt を生成する。
削除フラグ(deleted=True)のものは除外。
train:val = 9:1 でランダム分割。

使い方（どちらのPythonでもOK）:
  venv/bin/python scripts/prepare_lmdb_data.py
  venv\Scripts\python.exe scripts\prepare_lmdb_data.py
"""
import json
import random
from pathlib import Path

ANNO_FILE  = Path(__file__).parent.parent / "data/ocr_crops/annotations.json"
CROPS_DIR  = Path(__file__).parent.parent / "data/ocr_crops/text_crops"
OUT_DIR    = Path(__file__).parent.parent / "data/ocr_crops"
TRAIN_TXT  = OUT_DIR / "annotations_train.txt"
VAL_TXT    = OUT_DIR / "annotations_val.txt"

SEED       = 42
VAL_RATIO  = 0.1

def main():
    with open(ANNO_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # 削除フラグなし・ラベルが空でないものだけ残す
    valid = [d for d in data if not d.get("deleted") and d.get("ocr_text", "").strip()]
    print(f"有効件数: {len(valid)}  (削除: {sum(1 for d in data if d.get('deleted'))}件除外)")

    # 存在する画像だけに絞る
    missing = [d for d in valid if not (CROPS_DIR / d["file"]).exists()]
    if missing:
        print(f"  ※画像が見つからない {len(missing)}件をスキップ:")
        for d in missing[:5]:
            print(f"    {d['file']}")
    valid = [d for d in valid if (CROPS_DIR / d["file"]).exists()]
    print(f"画像確認済み: {len(valid)}件")

    # シャッフル → train/val 分割
    random.seed(SEED)
    random.shuffle(valid)
    val_n   = max(1, int(len(valid) * VAL_RATIO))
    val     = valid[:val_n]
    train   = valid[val_n:]

    def write_txt(items, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in items:
                f.write(f"{d['file']}\t{d['ocr_text']}\n")

    write_txt(train, TRAIN_TXT)
    write_txt(val,   VAL_TXT)

    print(f"\n生成完了:")
    print(f"  train: {len(train)}件 → {TRAIN_TXT}")
    print(f"  val  : {len(val)}件 → {VAL_TXT}")
    print()
    print("次のステップ（Windowsのコマンドプロンプトで実行）:")
    print("  venv\\Scripts\\python.exe scripts\\create_lmdb.py")

if __name__ == "__main__":
    main()
