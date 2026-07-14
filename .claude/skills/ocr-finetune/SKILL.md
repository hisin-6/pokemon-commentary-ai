# OCR ファインチューニング スキル

EasyOCR `pokemon_g2` モデルの改善サイクルに関する知識。
詳細手順: `docs/manual/ocr-finetune-cycle.md`

## 現行モデル
- `~/.EasyOCR/model/pokemon_g2.pth`（本番稼働中）
- ベースは `japanese_g2`（2216文字・`＊` 含む）
- 精度: **v10（99.246%）**（2026-06-07登録）
- バージョン管理: `pretrained\pokemon_g2_vN.pth` に都度バックアップ必須

## OCRが読み取るテキストの種類・収集ROI

クロップ収集は以下7つのROIから行う（2026-06-07〜）。

| ROI名 | 画面上の位置 | 収集内容 | スケール |
|-------|------------|---------|---------|
| `msg` | y=740〜930（メッセージボックス） | 技名・システムメッセージ・ポケモン名 | 等倍 |
| `ability_player` | y=450〜570 左側 | 自分側の特性・道具表示 | 等倍 |
| `ability_opp` | y=450〜570 右側 | 相手側の特性・道具表示 | 等倍 |
| `name_plr0` | y=930〜970 左 | 自分スロット0のポケモン名 | **2倍** |
| `name_plr1` | y=930〜970 中 | 自分スロット1のポケモン名 | **2倍** |
| `name_opp0` | y=50〜90 右側 | 相手スロット0のポケモン名 | **2倍** |
| `name_opp1` | y=50〜90 右端 | 相手スロット1のポケモン名 | **2倍** |

アノテーションの各エントリには `roi` フィールドが付く（どのエリアのクロップか識別用）。  
ファイル名は `{動画stem}_{roi名}_{連番}.png`。

誤読を議論するときは「何が（ポケモン名 or 技名 or …）・どのROIで誤読されているか」を区別すること。

## 残存誤読と改善見込み（v10時点）

| 誤読 | 正解 | 種類 | 頻度 | 状況 |
|------|------|------|------|------|
| バッグンだ | バツグンだ | システム | **全動画・多数** | 最優先・msg ROI |
| しろいハープ | しろいハーブ | アイテム名 | 全動画 | ability ROIで頻出・次優先 |
| ねっぶう | ねっぷう | 技名 | 一部 | データ不足 |
| ダブルウイング | ダブルウィング | 技名 | 少数 | 小文字ィ→大文字イ |
| ムーンフオース | ムーンフォース | 技名 | 少数 | 小文字ォ→大文字オ |
| ペリッパーー | ペリッパー | ポケモン名 | 少数 | name ROIで長音二重 |

### 改善に必要なデータ量の目安（実績ベース）
- **17件以上** → 高確率で改善（インファイト17件で100%正読達成）
- **3件以下** → 改善困難

## サイクル概要

```
動画選択 → クロップ収集（7ROI） → 目視確認 → LMDB作成 → 学習 → 登録 → 検証
```

## コマンド（PowerShellで実行）

```powershell
# 1. クロップ収集（--appendで追記・7ROIを自動収集）
venv\Scripts\python.exe scripts\collect_ocr_crops.py --input "D:\ゲーム録画\XXXX.mp4" --append

# 2. 目視確認（http://localhost:5001）
venv\Scripts\python.exe scripts\annotation_reviewer.py

# 3. LMDB作成
venv\Scripts\python.exe scripts\prepare_lmdb_data.py
venv\Scripts\python.exe scripts\create_lmdb.py

# 4. ファインチューニング（pokemon_g2ベース）
copy C:\Users\rotat\.EasyOCR\model\pokemon_g2.pth pretrained\pokemon_g2.pth
venv\Scripts\python.exe scripts\finetune_ocr.py --saved_model pretrained\pokemon_g2.pth --num_iter 7000

# 5. モデル登録（バックアップ必須→登録→pretrained更新）
copy saved_models\pokemon_finetune\best_accuracy.pth pretrained\pokemon_g2_vN.pth
venv\Scripts\python.exe scripts\setup_finetune_model.py
copy C:\Users\rotat\.EasyOCR\model\pokemon_g2.pth pretrained\pokemon_g2.pth

# 6. 精度検証（書き起こしのある動画で比較）
venv\Scripts\python.exe scripts\test_ocr_areas.py "D:\ゲーム録画\2026-04-12 16-14-39.mp4"
```

## 注意事項
- 学習ログに `loaded 44/44 layers` が出ることを確認（43以下は文字数不一致）
- `--num_iter`: データ3000件超なら5000〜7000推奨（Best iter=3000付近で早期収束する実績あり）
- `collect_ocr_crops.py` の `--append` を忘れると既存データが消える
- name系ROI（name_plr0/1・name_opp0/1）は高さ40pxと小さいため2倍スケールアップ済み
- v10以前のクロップ（msg ROIのみ）には `roi` フィールドがない。レビューツールでは問題なく扱える

## 処理済み動画
- 全99本・12,730件（全件reviewed完了）（2026-06-07時点）
- 詳細は `docs/manual/ocr-finetune-cycle.md` を参照
