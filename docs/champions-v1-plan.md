# ポケモンチャンピオンズ対応 第一弾計画

**ブランチ**: `dev/champions-v1`  
**作成日**: 2026-04-08  
**更新日**: 2026-04-18（タスク状態のみ2026-07-09に見直し）  

---

## 確定済み UI 変更点（実機確認）

| # | 変更内容 | 影響箇所 |
|---|----------|---------|
| 1 | 相手の HP 表示が **数字（X/Y）→ パーセント（XX%）** に変更 | `pipeline.py` OCR パース、`hpbar_analyzer.py` |
| 2 | 自分の HP は **数字のまま変わらず** | 影響なし |
| 3 | 状態異常アイコンが**新デザイン**に変更 | train4 相当のモデル再学習が必要 |
| 4 | 残りポケモンを示すボールアイコンが**新デザイン・新位置・縦→横並び**に変更 | `yolo_detector.py` ROI 座標、train7 相当のモデル再学習が必要 |
| 5 | **使用可能ポケモンが大幅に削減**（将来のアップデートで順次追加予定） | `data/pokedb.sqlite`、`src/pokedb/classifier.py` |

---

## タスク一覧

### 最優先（起動確認・ROI 系）

| # | タスク | ファイル | 状態 |
|---|--------|----------|------|
| 1 | ROI 座標の確認・再キャリブレーション | `yolo_detector.py` ROIS、`pipeline.py` 定数 | ✅ 2026-04-18 |
| 2 | HP バーピクセル解析の確認 | `hpbar_analyzer.py` | ✅ 2026-04-13 |
| 3 | OBS カメラ接続・フレーム取得確認 | `src/pipeline.py` | ✅ 完了（デバイス番号3・解像度1920x1080で稼働中） |

### 高優先（OCR・テキスト系）

| # | タスク | ファイル | 状態 |
|---|--------|----------|------|
| 4 | **相手 HP% の OCR パース追加** | `src/pipeline.py` | ✅ 2026-04-13 |
| 5 | メッセージボックスの OCR 確認 | `src/pipeline.py` | ✅ 2026-04-18（動画01:09確認） |
| 6 | switch-in 検出 regex の確認 | `src/pipeline.py` | ✅ 完了（`_OPPONENT_SWITCH_IN_RE`/`_DUAL_OPPONENT_SWITCH_IN_RE`等でその後も継続改修） |
| 7 | 技名・ポケモン名 OCR 精度確認 | `src/pipeline.py` | 🔧 2026-04-18（HP `/`→`1` 誤読あり） |
| 8 | **特性・道具発動メッセージ OCR 追加** | `src/pipeline.py` `_scan_ability_msg` | ✅ 2026-04-18 |

### 高優先（PokeDB 系）

チャンピオンズでは使用可能ポケモンが絞られているため、fuzzy マッチの精度向上のために DB をゲームモード対応にする。

| # | タスク | ファイル | 状態 |
|---|--------|----------|------|
| 8 | **`champions_pokemon` テーブル追加**（pokemon_id の許可リスト管理） | `data/pokedb.sqlite`、`scripts/build_pokedb.py` | 未着手 |
| 9 | **`PokeClassifier` に `game_mode` フィルター追加**（Champions モードでは許可リストのみ検索対象にする） | `src/pokedb/classifier.py` | 未着手 |
| 10 | **チャンピオンズ使用可能ポケモン一覧の収集・登録**（実機 or 公式情報から収集） | `scripts/build_pokedb.py` or 手動 SQL | 未着手 |
| 11 | 新ポケモン・新技がある場合は PokeAPI キャッシュ更新 | `scripts/build_pokedb.py` | 未着手 |

### 高優先（YOLO 検出系）

UI デザイン変更によりモデルの再学習が必要。
**動作確認と並行して画像収集を開始する。**

| # | タスク | モデル | 状態 |
|---|--------|--------|------|
| 12 | ボール検出の動作確認 + ROI 位置調整 | `train7/best.pt` | 未着手 |
| 13 | 状態異常アイコン検出の確認 | `train4/best.pt` | 未着手 |
| 14 | 終了画面検出の確認 | `train_end_screen2/best.pt` | 未着手 |

#### 画像収集（動作確認と並行）

| 対象 | 収集ツール | 保存先 |
|------|-----------|--------|
| ボール（alive/faint/status） | `scripts/ocr_logger.py` または手動キャプチャ | `data/ball_dataset/` |
| 状態異常アイコン（新デザイン） | 手動キャプチャ | 既存データセット or 新規 |
| 終了画面 | `scripts/capture_end_screen.py`（s キーで保存） | `data/end_screen_dataset/` |

> **収集方針**: リリース直後から対戦をプレイ or 配信を録画し、各クラス最低 30〜50 枚を目標に収集。
> Label Studio でアノテーション後に再学習。

### 中優先（YOLO モデル再学習）

動作確認で精度が不十分と判断した場合に実施。

| # | タスク | 状態 |
|---|--------|------|
| 15 | Label Studio でアノテーション（ボール・状態異常・終了画面） | 未着手 |
| 16 | ボール検出 train8 学習実行 | 未着手 |
| 17 | 状態異常検出 再学習（新デザイン対応） | 未着手 |
| 18 | 終了画面検出 再学習（必要に応じて） | 未着手 |

### 低優先（引き継ぎ課題）

| # | タスク | 状態 |
|---|--------|------|
| 19 | T4 ターン分割対策の検証 | 実装済み・検証待ち |
| 20 | バドレックスのブリザードランス未検出 | 継続調査 |

---

## PokeDB 対応方針（詳細）

### 現状
- `data/pokedb.sqlite` に全 1025 匹（第 1 世代〜第 9 世代）を収録
- チャンピオンズは使用可能ポケモンが大幅削減（将来バージョンアップで順次追加予定）
- 現状のまま全匹を fuzzy マッチ対象にすると、チャンピオンズに登場しないポケモンへの誤マッチが増える

### 方針
1. **既存 DB は削除・変更しない**（SV との共用、将来の再追加に備える）
2. **`champions_pokemon` テーブルを追加**し、チャンピオンズで使用可能な `pokemon_id` を管理
3. **`PokeClassifier(game_mode="champions")` フラグ**を追加し、Champions モードでは `champions_pokemon` に含まれるポケモンのみを fuzzy マッチ対象にする
4. チャンピオンズにポケモンが追加されたら `champions_pokemon` テーブルに行を追加するだけで対応可能

### DB スキーマ（追加予定）

```sql
CREATE TABLE IF NOT EXISTS champions_pokemon (
    pokemon_id INTEGER PRIMARY KEY,
    added_version TEXT,          -- 追加されたゲームバージョン（例: "1.0.0"）
    note TEXT                    -- 備考
);
```

### PokeClassifier 変更方針

```python
# 現状（SV 対応）
clf = PokeClassifier()

# チャンピオンズ対応
clf = PokeClassifier(game_mode="champions")
# → fuzzy マッチを champions_pokemon に絞り込む
# → SV 対応モードではこれまで通り全 1025 匹から検索
```

---

## 2026-04-18 実装記録

### visualize_coords.py 整理・拡充
- `MSG_X_MIN = 120` 追加（右端と左右対称のマージン）
- ボールアイコン YOLO ROI を追加:
  - `BALL_ROI_OPP = (0.86, 0.15, 0.93, 0.19)` / `BALL_ROI_PLR = (0.04, 0.80, 0.11, 0.84)`
- 状態異常エリアを **per-pokemon** 4 ROI に変更（旧: 広域 2 ROI）:
  - `opp_status_0/1`（右上エリア）・`player_status_0/1`（左下エリア）
- 特性・道具発動メッセージエリアを追加:
  - `ability_msg_player`（x=0-555, y=450-570）/ `ability_msg_opp`（x=1365-1920, y=450-570）
- SV 時代の不要定数を削除: `PLAYER_Y_THRESHOLD`, `COMMAND_Y_MIN`, 旧広域 STATUS_ROI, `draw_hline`

### yolo_detector.py ROI 更新
- `ROIS` を per-pokemon 4 スロットに変更（`_0`/`_1` サフィックスでスロット判定）
- `_OPP_SLOT_SPLIT_X` / `_PLR_SLOT_SPLIT_X` 定数を削除（ROI 名サフィックスで代替）
- `BattleState`: `opponent_status_0/1`, `player_status_0/1` フィールドに変更
- `_extract_status_slot(detections, roi_name)` メソッド追加

### pipeline.py 更新
- `BattleMessageParser.MSG_X_MIN = 120` 追加・フィルター条件を左端チェックに更新
- `_STATUS_ICON_AREAS` / `_ABILITY_MSG_AREAS` 定数を `PipelineRunner` に追加
- `_scan_ability_msg()`: per-frame で ability/item メッセージエリアをスキャンして `dict[str, str]` を返す新メソッド
- `_sync_status_from_ocr_bbox()`: スケール正規化 + per-slot エリアマッチングに更新
- `_build_game_state()`: `ability_msg_player`/`ability_msg_opp` キーを追加

### scripts/test_ocr_areas.py 新規作成
- 動画に対して OCR のみを実行し、定義済みエリアごとにテキストを表示する軽量確認ツール
- 時刻ベース処理: `--step`, `--start`, `--end` オプション（秒指定）
- 15 エリア定義（MSG ROI・ability・status・name・HP）
- HP エリアの `/` → `1` 誤読補正: `_HP_SLASH_RE = re.compile(r'^(\d{1,3})1(\d{2,3})$')`
- コマンドメニュー文字列フィルター（`--no-filter` で無効化可）
- ログ自動保存: `logs/ocr_areas_YYYYMMDD_HHMMSS.txt`
- 手書き注釈テンプレートをログに出力（画面種別・検出テキスト記入欄）

### OCR 動作確認（動画 00:00〜01:09）
| エリア | 結果 |
|--------|------|
| MSG ROI | おおむね読み取れる（誤字あり） |
| ability_player / ability_opp | かなり読み取れる |
| ポケモン名・HP | フォントの違いで読み取り率低め |
| HP `/` → `1` 誤読 | `_HP_SLASH_RE` で補正済み（例: `01205` → `0/205`） |
| コマンドメニュー文字 | `_UI_FILTER` で除去 |

---

## 相手 HP% OCR 対応方針（詳細）

### 現状
`parse_ocr_results()` で `r'\b(\d{1,3})/(\d{1,3})\b'` パターンにより `X/Y` 形式を検出。
相手 HP には数値が画面に表示されていたため、このパターンで捕捉できていた。

### 変更点
チャンピオンズでは相手 HP が `XX%` 形式で表示される。

### 対応
- `r'\b(\d{1,3})%'` パターンを追加し、y 座標が `_PLAYER_Y_THRESHOLD` 未満（相手エリア）のものを `hp_opponent_pct_by_slot` として格納
- 既存の `X/Y` パターンは自分側専用として残す
- `hpbar_analyzer.py` のピクセル解析は HP バーの色が残存する場合は引き続き利用（要確認）

---

## 作業手順（リリース後）

1. OBS でチャンピオンズの対戦画面をキャプチャ
2. `ocr_logger.py` を起動して生 OCR データを記録
   ```
   venv\Scripts\python.exe scripts/ocr_logger.py --ball-model runs/detect/train7/weights/best.pt --end-model runs/detect/train_end_screen2/weights/best.pt
   ```
3. **スクショを撮影して ROI 座標・HP バー位置を確認** → ずれていれば修正
4. YOLO 検出をライブで流してボール・状態異常・終了画面の精度確認
5. OCR・regex の誤読・未検出をログで確認
6. 問題があれば各タスクに着手

---

## 完了基準

- [ ] チャンピオンズの対戦 1 試合を通してログが正常に出力される
- [ ] 相手 HP が `XX%` 形式で正しく記録される
- [ ] ボール検出・状態異常検出・終了画面検出が動作する
- [ ] switch-in・技名・ポケモン名が正しく記録される（チャンピオンズ収録ポケモンのみ）
- [ ] VOICEVOX 実況音声が出力される
- [ ] PokeDB の Champions フィルターが機能する

---

## YOLO 再学習コマンド（参考）

```bash
# ボール検出 train8
venv\Scripts\yolo.exe train model=runs/detect/train7/weights/best.pt data=data/ball_dataset/data.yaml imgsz=640 epochs=50 batch=8 cls=2.0 cos_lr=true patience=20 name=train8

# 状態異常 再学習（ベースモデルは適宜変更）
venv\Scripts\yolo.exe train model=yolov8n.pt data=data/status_dataset/data.yaml imgsz=640 epochs=50 batch=8 name=train_status_champ
```
