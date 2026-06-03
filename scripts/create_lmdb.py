#!/usr/bin/env python3
"""
LMDB データセット作成スクリプト（Windows の venv で実行すること）。
事前に prepare_lmdb_data.py で train/val txt を生成しておくこと。

使い方:
  venv\Scripts\python.exe scripts\create_lmdb.py
"""
import sys
import os

# deep-text-recognition-benchmark の create_lmdb_dataset をそのまま使う
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "tools", "deep-text-recognition-benchmark"))

from create_lmdb_dataset import createDataset

INPUT_PATH = os.path.join(project_root, "data", "ocr_crops", "text_crops")
TRAIN_TXT  = os.path.join(project_root, "data", "ocr_crops", "annotations_train.txt")
VAL_TXT    = os.path.join(project_root, "data", "ocr_crops", "annotations_val.txt")
TRAIN_LMDB = os.path.join(project_root, "data", "ocr_lmdb", "train")
VAL_LMDB   = os.path.join(project_root, "data", "ocr_lmdb", "val")

if __name__ == "__main__":
    print("=== LMDB 作成開始 ===")

    print(f"\n[1/2] train LMDB を作成中...")
    print(f"  入力: {INPUT_PATH}")
    print(f"  GT  : {TRAIN_TXT}")
    print(f"  出力: {TRAIN_LMDB}")
    createDataset(inputPath=INPUT_PATH, gtFile=TRAIN_TXT, outputPath=TRAIN_LMDB)
    print("  → 完了")

    print(f"\n[2/2] val LMDB を作成中...")
    print(f"  入力: {INPUT_PATH}")
    print(f"  GT  : {VAL_TXT}")
    print(f"  出力: {VAL_LMDB}")
    createDataset(inputPath=INPUT_PATH, gtFile=VAL_TXT, outputPath=VAL_LMDB)
    print("  → 完了")

    print("\n=== LMDB 作成完了 ===")
    print(f"  {TRAIN_LMDB}")
    print(f"  {VAL_LMDB}")
    print("\n次のステップ:")
    print("  venv\\Scripts\\python.exe scripts\\finetune_ocr.py")
