# アノテーション作業マニュアル

## 概要

YOLOv8 学習用のバウンディングボックスを付ける作業。
ツール: **LabelImg**

---

## 起動方法

```powershell
venv\Scripts\labelImg.exe data\yolo_dataset\images\train data\yolo_dataset\classes.txt
```

### 初回のみ設定

1. **保存先の指定**: `Change Save Dir` → `data\yolo_dataset\labels\train` を選択
2. **フォーマット変更**: 左パネルの `PascalVOC` ボタン → `YOLO` に変更
3. **Auto Save Mode ON**: `View` → `Auto Save Mode` にチェック（推奨）

---

## 操作キー一覧

| キー | 操作 |
|------|------|
| `W` | バウンディングボックスを描く |
| `D` | 次の画像へ |
| `A` | 前の画像へ |
| `Ctrl+S` | 保存 |
| `Del` | 選択中のボックスを削除 |
| `Ctrl++` | 拡大 |

> ⚠️ **注意**: `Ctrl+W` は「検証済みマーク」が付く。誤押しした場合は `D` → `A` で戻ると復帰する。

---

## クラス定義（全10クラス）

| ID | クラス名 | 内容 | 備考 |
|----|---------|------|------|
| 0 | poison | どく | ✅ |
| 1 | bad_poison | どくどく | ⚠️ 画面上にアイコン表示なし・アノテ不要 |
| 2 | burn | やけど | ✅ |
| 3 | paralysis | まひ | ✅ |
| 4 | freeze | こおり | ✅ |
| 5 | sleep | ねむり | ✅ |
| 6 | confusion | こんらん | ⚠️ 画面上にアイコン表示なし・アノテ不要 |
| 7 | ball_alive | 生存ポケモンのボール（通常色） | ✅ |
| 8 | ball_faint | 瀕死ポケモンのボール | ✅ |
| 9 | ball_status | 状態異常ポケモンのボール（黄色っぽい） | ✅ |

---

## アノテーション方針

### 状態異常アイコン

| 画像の内容 | 対応 |
|-----------|------|
| 状態異常アイコンが写っている | `W` でボックスを描いて保存 |
| バトル画面だが状態異常なし | 保存せずに `D` で次へ（ネガティブ例として保持） |
| 技アニメーション中・HPバー非表示 | 保存せずに `D` で次へ |

- **収集は相手ポケモン側のみでOK**（アイコンのデザインは相手・自分側で同じため自分側も検出できる）
- 1枚に複数のアイコンがある場合はアイコンの数だけボックスを描く

### ボールアイコン

- **1つ1つ個別にバウンディングボックスをつける**（まとめてNG）
  - 個別に検出することで生存ポケモン数のカウントが可能になる
- ボールは小さいので `Ctrl++` で拡大してから作業する
- 4〜6個縦並びの全てにラベルを貼る

#### ボールの状態の見分け方

| 見た目 | ラベル |
|--------|--------|
| 通常色（白・青系） | `ball_alive` |
| 暗い・×マーク | `ball_faint` |
| 黄色っぽい | `ball_status` |

---

## ボックスの描き方

1. `W` キーを押す
2. アイコンの左上から右下にドラッグして囲む（ぴったり or 少し大きめでOK）
3. クラス名の選択ダイアログが出る → 該当クラスを選ぶ
4. `Ctrl+S` で保存（Auto Save Mode ONなら不要）

---

## ラベル数の確認方法

アノテーション済み画像数とクラスごとのラベル数を確認するコマンド：

```bash
# アノテーション済み画像数（labelsフォルダ内のtxtファイル数）
ls data/yolo_dataset/labels/train/ | wc -l

# クラスごとのラベル数
cat data/yolo_dataset/labels/train/*.txt | awk '{print $1}' | sort | uniq -c | sort -rn
```

出力例：
```
464 7   → ball_alive
146 4   → freeze
104 9   → ball_status
 83 8   → ball_faint
 43 3   → paralysis
 12 2   → burn
  6 0   → poison
```

クラスIDは `data/yolo_dataset/classes.txt` の行番号（0始まり）と対応している。

---

## 目標枚数（クラスごと）

| クラス | 目標 | 理由 |
|--------|------|------|
| sleep | 50枚以上 | 初期学習で少なかった |
| freeze | 50枚以上 | ゼロから |
| ball_alive | 50枚以上 | 全バトル画像に写っているので稼ぎやすい |
| ball_faint | 30枚以上 | ポケモンが倒れた場面限定 |
| ball_status | 30枚以上 | 状態異常シーン限定 |

---

## 学習前の準備（アノテーション完了後）

### train/val 分割

```powershell
venv\Scripts\python.exe tools\split_dataset.py
```

自動で 8:2 に分割される（SEED=42 固定なので再現性あり）。

### 追加学習（既存モデルからファインチューニング）

初回学習済みの `best.pt` を引き継いで追加学習する:

```powershell
venv\Scripts\yolo detect train data=data/yolo_dataset/data.yaml model=runs/detect/train/weights/best.pt epochs=50 imgsz=640 batch=8
```

> ⚠️ `yolov8n.pt` ではなく `best.pt` を指定すること！

### 動作確認

```powershell
venv\Scripts\python.exe src/capture/screen_capture.py --yolo --model runs/detect/train/weights/best.pt --live
```

---

## ボール検出専用データセット（Sprint 7 追加作業）

### 背景

フルフレーム学習ではボールアイコン（平均27×25px）が小さすぎてYOLOv8nが検出できなかった（Recall=0）。
ROIクロップ画像に対して学習することでスケール問題を解消し、yolov8sで再学習する。

### データセット生成（スクリプト自動実行済み）

```bash
python scripts/crop_ball_rois.py
```

- 出力先: `data/ball_dataset/`
- 既存アノテーション（クラスID 7/8/9）をROI座標系に自動変換済み
- train: 842枚（自動ラベル165枚・ネガティブ677枚）/ val: 210枚（自動ラベル49枚・ネガティブ161枚）

### アノテーションツール（ボール専用データセット）

> **Sprint 7 以降は Label Studio + ML Backend を使用**（LabelImg から移行済み）
> セットアップ手順: `scripts/ls_ml_backend/README.md` を参照

```powershell
# Label Studio 起動
$env:LOCAL_FILES_SERVING_ENABLED = "true"
venv\Scripts\label-studio.exe start --port 8080

# ML Backend 起動（別ウィンドウ）
$env:YOLO_MODEL_PATH = "runs/detect/pretrain_ball/weights/best.pt"
$env:BALL_DATASET_ROOT = "data/ball_dataset"
venv\Scripts\python.exe scripts/ls_ml_backend/_wsgi.py
```

### クラス定義（3クラスのみ）

| ID | クラス名 | 内容 |
|----|---------|------|
| 0 | ball_alive | 生存ポケモンのボール（白・通常色） |
| 1 | ball_faint | 瀕死ポケモンのボール（暗い・×マーク） |
| 2 | ball_status | 状態異常ポケモンのボール（黄色っぽい） |

### 作業方針

- `_opponent.png` → 画面右上のボール列クロップ
- `_player.png` → 画面左下のボール列クロップ
- YOLO の自動プレラベルを確認・修正して Submit
- **ボールなし画像**: 枠を描かずそのまま **Submit**（Skip は使わない）

### 再学習コマンド（アノテーション完了後）

```powershell
# Step 1: Label Studio から JSON エクスポート → プロジェクトルートに保存
# Step 2: YOLO 形式に変換
venv\Scripts\python.exe scripts/export_to_yolo.py annotations.json

# Step 3: 本番モデル再学習
venv\Scripts\yolo.exe train model=yolov8s.pt data=data/ball_dataset/data.yaml imgsz=640 epochs=100 batch=8
```

> ⚠️ モデルは `yolov8n.pt` → **`yolov8s.pt`** に変更（小物体検出精度向上のため）
> ⚠️ `python -m ultralytics` は動かない。`yolo.exe` を使うこと。

### yolo_detector.py への組み込み（学習完了後）

学習完了後に `runs/detect/trainXX/weights/best.pt` を確認し、
`pipeline.py` の `--model` 引数を新しいモデルパスに変更する。
ただし新モデルはボール専用（クラスID 0/1/2）なので `CUSTOM_CLASS_NAMES` の更新も必要。

---

## 終了画面検出データセット（Sprint 7 追加作業・2026-03-30）

### 概要

「勝負に勝った！」「負けた！」「降参が選ばれました」の左下テキストオーバーレイを検出する。
タイムアウト機構の代替として YOLO で確実に終了を検知するために導入。

### 画像収集

```powershell
# OBS仮想カメラ（番号3）起動後に実行。sキーで保存。
venv\Scripts\python.exe scripts\capture_end_screen.py
```

- 保存先: `data/end_screen_dataset/raw/`
- 勝ち・負け・降参 各10〜15枚が目安（通信エラーは3枚未満なら除外）

### アノテーション（Label Studio）

ML Backend は不要。手動アノテーションで十分。

1. Label Studio 起動（`$env:LOCAL_FILES_SERVING_ENABLED = "true"` を設定してから）
2. プロジェクト作成 → `data\end_screen_dataset\raw\` の画像をインポート
3. Labeling Interface に以下の XML を設定:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="battle_end" background="#FF0000"/>
  </RectangleLabels>
</View>
```

4. **終了テキストが表示されている画像**: 左下テキスト部分を `battle_end` でBBox
5. **終了テキストなし（バトル中フレーム等）**: BBoxなしで **Submit**（ネガティブ例として重要）
6. **Skip は使わない**（未作業扱いになりエクスポートに含まれない）

### データセット準備 & 学習

```powershell
# Step 1: Label Studio から JSON エクスポート → プロジェクトルートに保存
# Step 2: YOLO形式変換 + train/val分割
venv\Scripts\python.exe scripts\prepare_end_screen_dataset.py annotations.json

# Step 3: 学習
venv\Scripts\yolo.exe train model=yolov8n.pt data=data/end_screen_dataset/data.yaml imgsz=640 epochs=50 batch=8 patience=15 name=train_end_screen
```

### クラス定義

| ID | クラス名 | 内容 |
|----|---------|------|
| 0 | battle_end | バトル終了テキストオーバーレイ（左下） |

### pipeline への組み込み

```powershell
# --end-model 引数で指定
venv\Scripts\python.exe src/pipeline.py --end-model runs/detect/train_end_screen2/weights/best.pt ...
```

### 既知の問題（2026-03-30 時点）

- **バトル中に誤発火**: 技アニメーション等を終了画面と誤認識する
  - 暫定対策: バトル開始25秒後から検査・3フレーム連続確認
  - **根本解決策**: バトル中フレームを negative サンプルに追加して再学習が必要

---

## 既知の問題と対処

### LabelImg が Python 3.12 でエラーになる

以下のファイルを修正済み（再発したら確認する）:

- `venv/Lib/site-packages/labelImg/labelImg.py` L965: `int()` キャスト追加
- `venv/Lib/site-packages/libs/canvas.py` L526, L530-531: `int()` キャスト追加
