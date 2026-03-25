# Label Studio ML Backend - セットアップ手順

## 全体フロー

```
① 仮モデル学習（既存ラベル済み画像で短時間学習）
        ↓
② Label Studio 起動
        ↓
③ ML Backend 起動
        ↓
④ Label Studio でプロジェクト設定・画像インポート
        ↓
⑤ 自動プレラベル → ブラウザで確認・修正 → Submit
        ↓
⑥ JSON エクスポート → YOLO 形式変換
        ↓
⑦ 本番モデル再学習（yolov8s・100エポック）
        ↓
⑧ 次回以降は最新モデルでプレラベル精度アップ → ①に戻る
```

---

## Step 1: 仮モデル学習（PowerShell）

> **注意**: `python -m ultralytics` は動かない。`yolo.exe` を使うこと。

```powershell
venv\Scripts\yolo.exe train model=yolov8n.pt data=data/ball_dataset/data.yaml epochs=20 batch=16 imgsz=640 name=pretrain_ball
```

完了後: `runs/detect/pretrain_ball/weights/best.pt` が生成される

確認コマンド:
```powershell
Test-Path runs\detect\pretrain_ball\weights\best.pt
# True が返ればOK
```

---

## Step 2: Label Studio 起動（PowerShell）

> **注意**: `LOCAL_FILES_SERVING_ENABLED=true` を必ず設定してから起動すること。
> ポート 8080 が使用中の場合は自動で 8081 に切り替わる。どちらでも動作は同じ。

```powershell
$env:LOCAL_FILES_SERVING_ENABLED = "true"
venv\Scripts\label-studio.exe start --port 8080
# ブラウザで http://localhost:8080（または 8081）にアクセス
# アカウント作成 → ログイン
```

> **アカウント登録について**: Label Studio は完全にローカルで動作するため、
> 登録したメールアドレスやパスワードは外部に送信されません。
> 普段使いのメアドでも、適当なダミーのメアドでも、どちらでも問題ありません。

---

## Step 3: ML Backend 起動（PowerShell・別ウィンドウ）

> **注意**:
> - `label-studio-ml.exe start` は動かない
> - `python -m label_studio_ml` も動かない
> - `python scripts/ls_ml_backend/_wsgi.py` で直接起動すること
> - Step 1 の仮モデルが存在しないと起動に失敗する

```powershell
$env:YOLO_MODEL_PATH = "runs/detect/pretrain_ball/weights/best.pt"
$env:BALL_DATASET_ROOT = "data/ball_dataset"
$env:CONF_THRESHOLD = "0.25"

venv\Scripts\python.exe scripts/ls_ml_backend/_wsgi.py
```

`Running on http://<ローカルIP>:9090/ (Press CTRL+C to quit)` が表示され、
プロンプトが戻ってこなければ起動成功。

ヘルスチェック確認:
```powershell
Invoke-WebRequest -Uri http://localhost:9090/health -UseBasicParsing -TimeoutSec 5
# StatusCode: 200 が返ればOK
```

---

## Step 4: Label Studio プロジェクト設定

### 4-1. プロジェクト作成
「Create Project」→ 名前: `BallDetection`

### 4-2. ラベル設定（Labeling Setup）
「Object Detection with Bounding Boxes」を選択後、以下のXMLに書き換え:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="ball_alive"  background="#00FF00"/>
    <Label value="ball_faint"  background="#888888"/>
    <Label value="ball_status" background="#FFD700"/>
  </RectangleLabels>
</View>
```

### 4-3. ローカルストレージ設定
Settings → Cloud Storage → Add Source Storage:
- Type: `Local files`
- Absolute local path: `C:\Users\rotat\AITuberProject\data\ball_dataset\images\train`
- **Import Method**: `Files - Automatically creates a task for each storage object`
- **File Name Filter**: `.*\.(jpg|jpeg|png)$`

「Add Storage」→「Sync Storage」ボタンを押して画像をインポート

> **注意**: Import Method のデフォルトは `Tasks`（JSON専用）、
> File Name Filter のデフォルトも JSON 系のみ。
> **どちらも必ず変更すること**。変更しないと画像が読み込まれずエラーになる。

### 4-4. ML Backend 接続
Settings → Model → Add Model:
- Name: `YOLOBallDetector`（任意）
- URL: `http://localhost:9090`
- Authentication: `No Authentication`

「Validate and Save」でモデル接続完了

---

## Step 5: プレラベル生成・確認・修正

### プレラベルの一括生成
タスク一覧で全タスクを選択 → 上部「Actions」→「**Retrieve predictions**」

### アノテーション作業
タスク一覧の「**Label All Tasks**」ボタンから開始すると連続作業モードになる。

操作方法:
- ラベルを1回クリックして選択 → そのままドラッグで連続描画（毎回クリック不要）
- ショートカット: `1`=ball_alive / `2`=ball_faint / `3`=ball_status
- 枠の削除: クリックして選択 → `Del`
- Submit → 自動で次の画像へ

> **ボールがない画像**: 枠を描かずそのまま **Submit**（Skip ではなく）。
> 「ボールなし」も学習データとして重要。

---

## Step 6: エクスポート → YOLO 変換

```
Label Studio → Export → JSON → プロジェクトルートに保存
```

```powershell
venv\Scripts\python.exe scripts/export_to_yolo.py annotations.json
# → data/ball_dataset/labels/train/ に YOLO 形式 txt が出力される
```

---

## Step 7: 本番モデル再学習

> **注意**: `python -m ultralytics` は動かない。`yolo.exe` を使うこと。

```powershell
venv\Scripts\yolo.exe train model=yolov8s.pt data=data/ball_dataset/data.yaml epochs=100 batch=8 imgsz=640
```

RTX 3080 で 30〜60 分程度。
完了後のモデル: `runs/detect/train/weights/best.pt`

---

## 次のゲームが来たとき（再利用手順）

1. 新しいゲームの画像を `data/ball_dataset/images/train/` に追加
2. Label Studio の「Sync Storage」で画像を取り込む
3. 最新の `best.pt` を `YOLO_MODEL_PATH` に設定して ML Backend 起動
4. Actions → Retrieve predictions で自動プレラベル生成
5. 確認・修正 → Export → `export_to_yolo.py` → 再学習

モデルが育つほどプレラベルの精度が上がり、確認作業がどんどん楽になる。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `No module named ultralytics.__main__` | `-m ultralytics` は使えない | `yolo.exe train ...` を使う |
| `label-studio-ml.exe` でモジュールエラー | .exe が別 Python を使う | `python.exe scripts/ls_ml_backend/_wsgi.py` で起動 |
| ML Backend がすぐ終了する | `use_reloader=True`（デフォルト）の問題 | `_wsgi.py` に `use_reloader=False` が設定済み |
| Local storage で画像が読み込まれない | Import Method / Filter のデフォルトが JSON 用 | Import Method を `Files`、Filter を `.*\.(jpg\|jpeg\|png)$` に変更 |
| ML Backend に接続できない | モデルファイルが存在しない | Step 1 の仮モデル学習を先に完了させる |
| ポート 8080 が使用中 | 別プロセスが占有 | 自動で 8081 に切り替わるのでそのまま使用 |
| プレラベルが表示されない | `/predict` が呼ばれていない | `init_app` 方式の `_wsgi.py` を使う（シンプル Flask 版では動かない） |
