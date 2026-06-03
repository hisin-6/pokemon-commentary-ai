# EasyOCR ファインチューニング サイクルマニュアル

ポケモン対戦画面のOCR精度を継続的に改善するためのサイクル手順。
新しい動画データを積み上げるたびにこのサイクルを回す。

---

## ツール概要

| スクリプト | 実行環境 | 役割 |
|-----------|---------|------|
| `scripts/collect_ocr_crops.py` | PowerShell | 動画からテキストクロップを収集 |
| `scripts/annotation_reviewer.py` | WSL2 or PowerShell | ブラウザUIで目視確認 |
| `scripts/prepare_lmdb_data.py` | PowerShell | train/val txtファイルを生成 |
| `scripts/create_lmdb.py` | PowerShell | LMDBデータセットを作成 |
| `scripts/finetune_ocr.py` | PowerShell | ファインチューニング実行 |
| `scripts/setup_finetune_model.py` | PowerShell | EasyOCRにモデルを登録 |
| `scripts/test_ocr_areas.py` | PowerShell | OCR精度を動画で検証 |

---

## ステップ1: クロップ収集

未使用動画から10本ずつ選んで実行する。

```powershell
# 1本目（既存に追記）
venv\Scripts\python.exe scripts\collect_ocr_crops.py --input "D:\ゲーム録画\YYYY-MM-DD HH-MM-SS.mp4" --append

# 2本目以降も同様に --append をつける
venv\Scripts\python.exe scripts\collect_ocr_crops.py --input "D:\ゲーム録画\YYYY-MM-DD HH-MM-SS.mp4" --append
```

> **注意**: `--append` を忘れると既存の `annotations.json` が上書きされる

- 出力先: `data/ocr_crops/annotations.json`（追記）
- 画像保存先: `data/ocr_crops/text_crops/`

---

## ステップ2: 目視確認

```powershell
venv\Scripts\python.exe scripts\annotation_reviewer.py
```

ブラウザで `http://localhost:5001` を開く。

**操作方法**:
| キー | 動作 |
|-----|------|
| `→` | 確認済みにして次へ |
| `Enter` | ラベルを編集して保存・次へ |
| `D` | 削除フラグ（LMDB作成時スキップ） |
| `←` | 前に戻る |

**効率的な確認順序**:
1. 「未確認のみ」チェックを入れる（確認済みをスキップ）
2. 「低信頼（conf<0.5）」フィルターで重点確認
3. 残りは「全件」+「未確認のみ」で流す

---

## ステップ3: LMDB作成

```powershell
# train/val txtを生成（削除フラグ除外・9:1分割）
venv\Scripts\python.exe scripts\prepare_lmdb_data.py

# LMDBデータセットを作成
venv\Scripts\python.exe scripts\create_lmdb.py
```

- 出力: `data/ocr_lmdb/train/` / `data/ocr_lmdb/val/`

---

## ステップ4: ファインチューニング

### 初回
```powershell
venv\Scripts\python.exe scripts\finetune_ocr.py --num_iter 5000
```

### 2回目以降（前回の pokemon_g2 をベースにする）
```powershell
# 前回登録済みモデルを pretrained にコピー（ステップ6完了後に必ず実行）
copy C:\Users\rotat\.EasyOCR\model\pokemon_g2.pth pretrained\pokemon_g2.pth

# pokemon_g2 ベースで学習
venv\Scripts\python.exe scripts\finetune_ocr.py --saved_model pretrained\pokemon_g2.pth --num_iter 7000
```

> **`--num_iter` の目安**: データ1000件あたり約3000〜5000iter。データ3000件超なら5000〜7000iter推奨（10000はBest iter=3000付近で収束し過学習気味になる実績あり）。

**確認ポイント**: ログに `loaded 44/44 layers (shape-matched, FT mode)` が出ればOK。
43/44以下なら文字数不一致（`CHARACTERS` を確認）。

- 保存先: `saved_models/pokemon_finetune/best_accuracy.pth`

---

## ステップ5: モデル登録

```powershell
# バージョン付きでバックアップしてから登録（ベストモデルの上書き消失を防ぐ）
# N はサイクル番号（v1, v2, v3 ...）
copy saved_models\pokemon_finetune\best_accuracy.pth pretrained\pokemon_g2_vN.pth

# EasyOCRに登録
venv\Scripts\python.exe scripts\setup_finetune_model.py
```

- `~/.EasyOCR/model/pokemon_g2.pth` にコピー
- 次回 pipeline / collect_ocr_crops 起動時から自動で使われる
- **登録後に `pretrained\pokemon_g2.pth` も更新しておく**（次サイクルのベースモデルになる）

```powershell
# 登録完了後に実行
copy C:\Users\rotat\.EasyOCR\model\pokemon_g2.pth pretrained\pokemon_g2.pth
```

---

## ステップ6: 精度検証

```powershell
venv\Scripts\python.exe scripts\test_ocr_areas.py "D:\ゲーム録画\2026-04-12 16-14-39.mp4"
```

- ログ: `logs/ocr_areas_YYYYMMDD_HHMMSS.txt`
- 書き起こし（`records/2026-04-12 16-14-39書き起こし.md`）と比較して改善確認
- 前回ログと diff して改善・悪化ポイントを確認する

---

## 使用済み動画の管理

**書き起こし済み（TRANSCRIPT_FILES）6本**: `fix_annotations.py` の照合対象

| 動画 | クロップ数 | 削除済 | 有効 | 書き起こし |
|------|----------|--------|------|-----------|
| 2026-04-12 16-06-11 | 154 | 4 | 150 | － |
| 2026-04-12 16-14-39 | 116 | 1 | 115 | ✅ |
| 2026-04-12 16-29-27 | 125 | 6 | 119 | － |
| 2026-04-12 16-43-26 | 153 | 2 | 151 | － |
| 2026-04-12 16-51-41 | 125 | 4 | 121 | － |
| 2026-04-12 16-58-03 | 112 | 0 | 112 | － |
| 2026-04-12 17-04-52 | 133 | 1 | 132 | － |
| 2026-04-12 17-12-38 | 78 | 1 | 77 | － |
| 2026-04-12 17-18-57 | 114 | 4 | 110 | － |
| 2026-04-12 17-24-26 | 79 | 5 | 74 | － |
| 2026-04-12 17-30-42 | 169 | 3 | 166 | － |
| 2026-04-12 17-40-16 | 123 | 16 | 107 | － |
| 2026-04-12 17-47-57 | 152 | 3 | 149 | － |
| 2026-04-12 17-57-51 | 206 | 1 | 205 | － |
| 2026-04-12 18-12-45 | 142 | 1 | 141 | ✅ |
| 2026-04-12 18-37-29 | 136 | 0 | 136 | － |
| 2026-04-12 18-45-51 | 94 | 3 | 91 | － |
| 2026-04-12 18-53-04 | 162 | 2 | 160 | － |
| 2026-04-12 19-01-37 | 124 | 4 | 120 | － |
| 2026-04-12 19-09-29 | 123 | 3 | 120 | － |
| 2026-04-12 19-16-58 | 159 | 6 | 153 | － |
| 2026-04-12 19-28-37 | 140 | 4 | 136 | － |
| 2026-04-13 06-22-25 | 41 | 5 | 36 | － |
| 2026-04-13 06-25-46 | 159 | 1 | 158 | ✅ |
| 2026-04-13 06-34-11 | 186 | 7 | 179 | － |
| 2026-04-13 07-00-19 | 101 | 1 | 100 | ✅ |
| 2026-04-13 07-07-31 | 125 | 0 | 125 | － |
| 2026-04-13 07-18-18 | 101 | 2 | 99 | － |
| 2026-04-13 07-23-41 | 110 | 2 | 108 | － |
| 2026-04-13 07-38-30 | 126 | 3 | 123 | － |
| 2026-04-13 20-47-00 | 99 | 1 | 98 | － |
| 2026-04-13 21-22-29 | 177 | 6 | 171 | － |
| 2026-04-13 21-38-13 | 138 | 5 | 133 | － |
| 2026-04-14 08-15-22 | 372 | 8 | 364 | ✅ |
| 2026-04-14 20-14-17 | 123 | 7 | 116 | ✅ |
| **合計** | **4777** | **122** | **4655** | |

> このテーブルは以下コマンドで再集計できる（WSL2で実行）:
> ```powershell
> # WSL2で実行
> python3 -c "
> import json
> from collections import defaultdict
> with open('data/ocr_crops/annotations.json') as f:
>     data = json.load(f)
> counts = defaultdict(int); deleted = defaultdict(int)
> for d in data:
>     parts = d['file'].split('_'); vid = parts[0]+' '+parts[1]
>     counts[vid] += 1
>     if d.get('deleted'): deleted[vid] += 1
> for vid in sorted(counts):
>     print(vid, counts[vid], deleted[vid], counts[vid]-deleted[vid])
> "
> ```

**未使用**: `D:\ゲーム録画\` 内の残り動画（10本ずつ追加推奨）

---

## OCR精度の推移

| 日付 | データ数 | Best Accuracy | ベースモデル | 主な改善・備考 |
|------|---------|---------------|------------|---------|
| 2026-05-07 | 1200件 | 80.7% | japanese_g2 | 失敗（文字数不一致） |
| 2026-05-25 | 994件(train) | **96.2%** (v1) | japanese_g2 | 全レイヤーロード成功・フシギバナ/ウェザーボール改善 |
| 2026-05-26 | 2829件(train) | **96.764%** (v2) | pokemon_g2 v1 | つ→っ改善・ポケモン名安定化・Best iter=3000 ※バックアップ未保存で消失 |
| 2026-05-26 | 3873件(train) | 93.303% (v3) | pokemon_g2 v1 | val退行（っ→つ混入ラベルの影響疑い・val setも拡大で非比較） |
| 2026-05-26 | 3873件(train) | 93.303% (v4) | pokemon_g2 v1 | v3と同結果・val set拡大の影響が支配的と判断・v1に戻す |
| 2026-05-28 | 4229件(train) | **98.222%** (v5) | pokemon_g2 v1 | 全4777件目視確認後・ばつくん完全修正・インファイト75%改善・ソーラーピーム未改善 |

> **現行モデル**: v5（98.222%）で稼働中。`pretrained\pokemon_g2_v5.pth` にバックアップ済み。
