#!/usr/bin/env python3
r"""
ファインチューニング済みOCRモデルのセットアップスクリプト。
saved_models/pokemon_finetune/best_accuracy.pth を EasyOCR に登録する。

実行:
  venv\Scripts\python.exe scripts\setup_finetune_model.py
"""
import os
import sys
import shutil
import yaml

# finetune_ocr.py から CHARACTERS をインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from finetune_ocr import CHARACTERS


def main():
    easyocr_base = os.path.join(os.path.expanduser('~'), '.EasyOCR')
    model_dir    = os.path.join(easyocr_base, 'model')
    network_dir  = os.path.join(easyocr_base, 'user_network')
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)

    # モデルファイルのコピー
    project_root = os.path.dirname(script_dir)
    src = os.path.join(project_root, 'saved_models', 'pokemon_finetune', 'best_accuracy.pth')
    dst = os.path.join(model_dir, 'pokemon_g2.pth')
    if not os.path.exists(src):
        print(f"[ERROR] モデルが見つかりません: {src}")
        sys.exit(1)
    shutil.copy2(src, dst)
    print(f"モデルコピー完了: {dst}")

    # YAML設定ファイルの生成
    config = {
        'network_params': {
            'input_channel': 1,
            'output_channel': 256,
            'hidden_size': 256,
        },
        'imgH': 32,
        'lang_list': ['ja'],
        'character_list': CHARACTERS,
    }
    yaml_path = os.path.join(network_dir, 'pokemon_g2.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, width=float('inf'))
    print(f"YAML生成完了: {yaml_path}")

    # ネットワーク定義モジュール（EasyOCR が importlib でロードする）
    py_path = os.path.join(network_dir, 'pokemon_g2.py')
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write("from easyocr.model.vgg_model import Model\n")
    print(f"ネットワーク定義生成完了: {py_path}")

    print("\n=== セットアップ完了 ===")
    print("  次回 pipeline.py 起動時に pokemon_g2 モデルが自動で使われます")


if __name__ == '__main__':
    main()
