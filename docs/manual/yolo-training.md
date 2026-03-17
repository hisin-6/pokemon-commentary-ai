# YOLOv8 学習フロー マニュアル

## 概要

ポケモン対戦画面の状態異常アイコン・ボールアイコンを検出する YOLOv8 モデルの
データ収集〜学習〜動作確認までの手順書。

---

## クラス定義

| ID | クラス名 | 内容 |
|----|---------|------|
| 0 | poison | どく |
| 1 | bad_poison | どくどく |
| 2 | burn | やけど |
| 3 | paralysis | まひ |
| 4 | freeze | こおり |
| 5 | sleep | ねむり |
| 6 | confusion | こんらん |
| 7 | ball_alive | 生存ポケモンのボール（通常色） |
| 8 | ball_faint | 瀕死ポケモンのボール |
| 9 | ball_status | 状態異常ポケモンのボール（黄色っぽい） |

---

## Step 1: 画像収集

### 実行コマンド（Windows PowerShell）

```powershell
# デフォルト（2秒間隔・モニター1）
venv\Scripts\python.exe tools\collect_training_data.py

# 間隔を短くする
venv\Scripts\python.exe tools\collect_training_data.py --interval 1.0

# モニターを指定する
venv\Scripts\python.exe tools\collect_training_data.py --monitor 2
```

終了: `Ctrl+C`

### 仕組み

- OpenCV 差分検出でほぼ同じフレームを自動スキップ（重複排除）
- ファイル名は `{セッション日時}_{連番}.png` 形式 → **既存ファイルは上書きされない**
- 保存先: `data/yolo_dataset/images/train/`
- ROI 可視化プレビュー画像も `_roi_preview/` に同時保存

### 収集のコツ

| 欲しいクラス | ゲームでの意識 |
|------------|-------------|
| sleep / freeze | 相手ポケモンが状態異常になるシーンを長く経過させる |
| ball_alive | バトル中なら常に映るので枚数が稼ぎやすい |
| ball_faint | ポケモンが倒れた直後の場面を意識する |
| ball_status | 状態異常ポケモンがいるターンを意識する |

- **収集は相手ポケモン側のみでOK**（アイコンデザインが同じため自分側も検出できる）
- 目標: 各クラス 30〜50 枚以上

---

## Step 2: アノテーション

詳細は [annotation-guide.md](annotation-guide.md) を参照。

```powershell
venv\Scripts\labelImg.exe data\yolo_dataset\images\train data\yolo_dataset\classes.txt
```

ラベルの保存先: `data/yolo_dataset/labels/train/`

---

## Step 3: train/val 分割

```powershell
venv\Scripts\python.exe tools\split_dataset.py
```

- 比率: train 8 : val 2（SEED=42 固定）
- 画像（.png）とラベル（.txt）がセットで分割される

---

## Step 4: 学習

### 初回学習

```powershell
venv\Scripts\yolo detect train data=data/yolo_dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8
```

### 追加学習（ファインチューニング）

クラス追加・データ追加後は既存モデルを引き継いで学習する:

```powershell
venv\Scripts\yolo detect train data=data/yolo_dataset/data.yaml model=runs/detect/train/weights/best.pt epochs=50 imgsz=640 batch=8
```

> ⚠️ `venv\Scripts\python.exe -m yolo` や `-m ultralytics` は動かない。`venv\Scripts\yolo` を直接使うこと。

### 学習結果の確認

| 指標 | 説明 | 目安 |
|------|------|------|
| mAP50 | 主要な精度指標 | 0.9 以上が目標 |
| Precision | 誤検出の少なさ | 高いほど良い |
| Recall | 見逃しの少なさ | 高いほど良い |

学習済みモデルの保存先: `runs/detect/train/weights/best.pt`

---

## Step 5: 動作確認

```powershell
venv\Scripts\python.exe src/capture/screen_capture.py --yolo --model runs/detect/train/weights/best.pt --live
```

---

## ディレクトリ構成

```
data/yolo_dataset/
├── data.yaml               # クラス定義・データセットパス
├── classes.txt             # LabelImg 用クラスリスト
├── images/
│   ├── train/              # 学習用画像（収集スクリプトの保存先）
│   │   └── _roi_preview/   # ROI 可視化プレビュー（参考用）
│   └── val/                # 検証用画像（split_dataset.py で自動分割）
└── labels/
    ├── train/              # 学習用ラベル（LabelImg の保存先）
    └── val/                # 検証用ラベル（split_dataset.py で自動分割）

runs/detect/train/
└── weights/
    ├── best.pt             # 最良モデル（追加学習・推論に使用）
    └── last.pt             # 最終エポックのモデル
```

---

## ROI 定義（yolo_detector.py）

```python
ROIS = {
    "opponent_status": (0.57, 0.00, 1.00, 0.28),  # 右上: 相手HP・状態
    "player_status":   (0.00, 0.69, 0.43, 1.00),  # 左下: 自分HP・状態
    "opponent_balls":  (0.83, 0.01, 1.00, 0.20),  # 右上端: 相手ボール
    "player_balls":    (0.00, 0.72, 0.10, 0.97),  # 左下端: 自分ボール
}
```

16:9 固定 UI（スカーレット・バイオレット / 剣盾）を前提とした座標。

---

## 将来の拡張メモ（未着手）

### 相手HPバーのOpenCV検出（後回し）

相手のHPは数値表示がなく**バーのみ**のため、YOLOではなくOpenCVで対応する予定。

- HPバーのROI（固定座標）を切り取り、**緑ピクセルの横幅**を測定してHP%を算出
- HPバーは 緑→黄→赤 と色変化するので色でも大まかな残量が分かる
- 自分のHP数値はEasyOCRで読み取り可能（既存パイプラインで対応）

**優先度: 低（全体パイプライン統合・実況動作を優先）**

---

## ボール系クラスの検出精度問題と対策

### 問題
`imgsz=640` ではボールアイコンが小さすぎて解像度不足になり、Recall=0（全く検出できない）になる。

状態異常系（poison/burn/paralysis/sleep/freeze）は問題なし。

### 対策：`imgsz=1280` で学習する
```powershell
venv\Scripts\yolo detect train data=data/yolo_dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=1280 batch=4
```
- `batch=4` に下げること（VRAMが増えるため）
- 学習時間は約3時間 → `imgsz=1280` だとさらに長くなる可能性あり

### 新作ポケモンへの対応
新作ゲームが出た場合、UIデザイン・ボールアイコンのデザインが変わる可能性があるため**再学習が必要**。

手順：
1. 新作の対戦画面で画像収集（`tools/collect_training_data.py`）
2. LabelImgでアノテーション（各クラス100〜150枚目安）
3. `tools/split_dataset.py` で train/val 分割
4. `imgsz=1280 batch=4` で学習
5. 動作確認

ROI座標（`yolo_detector.py`）もUIレイアウトが変わった場合は要更新。

---

## 学習履歴

| 日付 | 内容 | 結果 |
|------|------|------|
| 2026-03-09 | 初回学習（poison / burn / paralysis / sleep 4クラス・268枚） | mAP50: 0.991 / mAP50-95: 0.669 |
| 2026-03-12 | 追加学習（freeze / sleep追加 / ball_alive / ball_faint / ball_status 追加） | mAP50: 0.988 / mAP50-95: 0.659（val未評価クラスあり・動作確認要） |
| 2026-03-17 | OBSカメラデータで全クラス再学習（527枚・imgsz=640）→ train3 | mAP50: 0.753 / 状態異常系は0.99前後だがボール系Recall=0 |
| 2026-03-17 | imgsz=1280に変更して再学習（527枚・batch=4）→ train4 | mAP50: 0.758 / 状態異常系は良好・ボール系はball_statusのみRecall改善（0.565）、ball_alive/ball_faintはRecall=0のまま |

### train4 クラス別詳細（現時点の最良モデル）

| クラス | Precision | Recall | mAP50 | 評価 |
|--------|-----------|--------|-------|------|
| poison | 1.000 | 0.930 | 0.995 | ✅ 優秀 |
| burn | 0.946 | 1.000 | 0.991 | ✅ 優秀 |
| paralysis | 0.994 | 1.000 | 0.995 | ✅ 優秀 |
| sleep | 1.000 | 0.931 | 0.990 | ✅ 優秀 |
| freeze | 0.578 | 0.879 | 0.774 | ✅ 実用レベル |
| ball_status | 0.395 | 0.565 | 0.493 | ⚠️ 要改善 |
| ball_alive | 1.000 | 0.000 | 0.390 | ❌ 検出不可 |
| ball_faint | 1.000 | 0.000 | 0.439 | ❌ 検出不可 |

### ボール系 Recall=0 の原因と今後の対策

`imgsz=1280` にしても ball_alive / ball_faint の Recall=0 が改善しなかったため、解像度不足以外の原因が疑われる。

**原因候補：**
1. **ドメインギャップ**: 学習データがOBSカメラ映像なのに対し、val側の画像との差異
2. **アノテーション品質**: バウンディングボックスが小さすぎ・ずれている可能性
3. **データ量不足**: ボールは1枚の画像に複数あるが、パターンのバリエーションが足りない

**今後試すこと（優先度: 低）：**
- アノテーションをやり直す（ボックスをやや大きめに描く）
- val側にも新クラスを均等に含めて再分割する
- ボール系だけ別モデルとして学習する

**現在の方針: 状態異常系（poison/burn/paralysis/sleep/freeze）だけで実況パイプライン統合を優先する。ボール残数カウントは後回し。**
