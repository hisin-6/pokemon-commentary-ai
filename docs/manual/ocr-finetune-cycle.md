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

未使用動画から10〜20本ずつ選んで実行する。

```powershell
# 1本目（既存に追記）
venv\Scripts\python.exe scripts\collect_ocr_crops.py --input "D:\ゲーム録画\YYYY-MM-DD HH-MM-SS.mp4" --append

# 2本目以降も同様に --append をつける
venv\Scripts\python.exe scripts\collect_ocr_crops.py --input "D:\ゲーム録画\YYYY-MM-DD HH-MM-SS.mp4" --append
```

> **注意**: `--append` を忘れると既存の `annotations.json` が上書きされる

- 出力先: `data/ocr_crops/annotations.json`（追記）
- 画像保存先: `data/ocr_crops/text_crops/`

### 収集対象ROI（2026-06-07〜 7ROI対応）

| ROI名 | 画面範囲 | 収集内容 | スケール |
|-------|---------|---------|---------|
| `msg` | y=740〜930 左半分 | 技名・システムメッセージ・ポケモン名 | 等倍 |
| `ability_player` | y=450〜570 左側 | 自分側特性・道具名 | 等倍 |
| `ability_opp` | y=450〜570 右側 | 相手側特性・道具名 | 等倍 |
| `name_plr0` | y=930〜970 左 | 自分スロット0のポケモン名 | **2倍** |
| `name_plr1` | y=930〜970 中 | 自分スロット1のポケモン名 | **2倍** |
| `name_opp0` | y=50〜90 右側 | 相手スロット0のポケモン名 | **2倍** |
| `name_opp1` | y=50〜90 右端 | 相手スロット1のポケモン名 | **2倍** |

**アノテーション形式の変化**:
- ファイル名: `{動画stem}_{roi名}_{連番}.png`（v10以前は `{stem}_{連番}.png`）
- エントリに `"roi"` フィールドが追加された（どのROI由来か識別用）
- v10以前のエントリには `roi` フィールドがない（`"msg"` 扱いで問題なし）

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

**収集済み動画**: 全31,792件（学習有効29,839件・削除フラグ1,953件・全件reviewed完了・2026-07-05時点。2026-06-07時点の99本・12,730件から大幅増加）  
**書き起こし済み（検証用）6本**: `records/` 以下の書き起こし対応動画

| 期間 | 本数 | 備考 |
|------|------|------|
| 2026-04系 | 42本 | v5〜v8学習ベース |
| 2026-06-02〜 | 5本 | v8時点で追加 |
| 2026-06-03〜 | 6本 | v9学習で追加 |
| 2026-06-04〜06 | 22本 | v10学習で追加 |
| 2026-06-06〜07 | 22本 | 録画済み・収集済み |
| **合計** | **99本** | |

> 動画ごとの件数を再集計するコマンド（WSL2で実行）:
> ```bash
> python3 -c "
> import json
> from collections import defaultdict
> with open('data/ocr_crops/annotations.json') as f:
>     data = json.load(f)
> counts = defaultdict(int); deleted = defaultdict(int)
> for d in data:
>     roi = d.get('roi', 'msg')
>     # roi付きファイル名: {stem}_{roi}_{seq}.png → stemは最初の部分
>     parts = d['file'].split('_')
>     # roi名がファイル名中にある場合とない場合を考慮
>     vid = parts[0]+' '+parts[1] if len(parts) >= 2 else d['file']
>     counts[vid] += 1
>     if d.get('deleted'): deleted[vid] += 1
> for vid in sorted(counts):
>     print(vid, counts[vid], deleted[vid], counts[vid]-deleted[vid])
> "
> ```

**未使用**: `D:\ゲーム録画\` 内の残り動画（10〜20本ずつ追加推奨）

---

## OCR精度の推移

| 日付 | データ数(train) | Best Accuracy | ベースモデル | 主な改善・備考 |
|------|----------------|---------------|------------|---------|
| 2026-05-07 | 1200件 | 80.7% | japanese_g2 | 失敗（文字数不一致） |
| 2026-05-25 | 994件 | **96.2%** (v1) | japanese_g2 | 全レイヤーロード成功・フシギバナ/ウェザーボール改善 |
| 2026-05-26 | 2829件 | **96.764%** (v2) | pokemon_g2 v1 | つ→っ改善・ポケモン名安定化・Best iter=3000 ※バックアップ未保存で消失 |
| 2026-05-26 | 3873件 | 93.303% (v3) | pokemon_g2 v1 | val退行（っ→つ混入ラベルの影響疑い） |
| 2026-05-26 | 3873件 | 93.303% (v4) | pokemon_g2 v1 | v3と同結果・v1に戻す |
| 2026-05-28 | 4229件 | **98.222%** (v5) | pokemon_g2 v1 | 全4777件確認・ばつくん完全修正・インファイト75%改善 |
| 2026-05-31 | 4583件 | **99.628%** (v6) | pokemon_g2 v5 | インファイト100%達成・ペリッパー改善 |
| 2026-05-31 | 5120件 | **99.678%** (v7) | pokemon_g2 v6 | 精度わずか向上 |
| 2026-06-02 | 5775件 | **98.822%** (v8) | pokemon_g2 v7 | Best iter=1で早期収束・ペリッパー完全修正 |
| 2026-06-05 | 8490件 | **98.800%** (v9) | pokemon_g2 v8 | ペリッパー・ソーラービーム完全修正・しろいハーブ退行 |
| 2026-06-07 | 11489件 | **99.246%** (v10) | pokemon_g2 v9 | +0.446%・ability/name ROI初収録 |
| 2026-07-05 | 29839件 | **99.391%** (v11) | pokemon_g2 v10 | 全31,792件レビュー後（学習有効29,839件）。こだわりスカーフ・フシギバナ・ねっぷう完全定着 |

> **現行モデル**: v11（99.391%）で稼働中。`pretrained\pokemon_g2_v11.pth` にバックアップ済み。（2026-07-09見直し。旧記載のv10は1世代前）

### v11時点の残存誤読（2026-07-05 書き起こし6本全数検証確定）

| 誤読 | 正解 | 優先度 | 備考 |
|------|------|-------|------|
| バッグンだ | バツグンだ | 最高 | 68%誤読（50回中34回）・v10比で悪化傾向 |
| ダブルウイング | ダブルウィング | 高 | 100%誤読（4/4）・小さいィ→大きいイ |
| ムーンフオース | ムーンフォース | 高 | 100%誤読（8/8）・小さいォ→大きいオ |
| しろいハープ | しろいハーブ | 中 | 92%正読まで改善（46/50）・残数収集で解消見込み |
| ペリッパーー（長音重複） | ペリッパー | 低 | 94.6%正読まで改善（66/70）・ほぼ解消 |

**v11で解消済み**: こだわりスカーフ（694回）・フシギバナ（92回）・ねっぷう（6回）は完全正読
