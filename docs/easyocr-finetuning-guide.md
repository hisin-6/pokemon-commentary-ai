# EasyOCR ファインチューニング手順書

本プロジェクトのポケモン技名テキスト認識精度を上げるため、EasyOCR の認識モデル（CRNN）をゲーム画面データでファインチューニングする手順をまとめる。

---

## 前提

| 項目 | 内容 |
|------|------|
| 対象モデル | EasyOCR 日本語認識モデル（`japanese_g2.pth`）|
| 学習フレームワーク | [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) |
| GPU | RTX 3080（VRAM 10GB）|
| 目標データ数 | 500〜1000 件（技名・ポケモン名テキストクロップ）|

---

## Phase 1: データ収集

### 1-1. 動画からフレームを切り出す

```bash
# 10秒ごとに1フレーム抽出（battle_start〜battle_end 区間の動画が望ましい）
python scripts/extract_frames.py --input "C:\path\to\battle.mp4" --interval 10 --out data/ocr_crops/frames/
```

スクリプトが未作成の場合は以下で代用：
```python
import cv2, os
cap = cv2.VideoCapture("battle.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
i = 0
while cap.read()[0]:
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(fps * 10) == 0:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"frames/frame_{i:05d}.png", frame)
            i += 1
```

### 1-2. メッセージボックスROIをクロップする

`scripts/test_ocr_areas.py` の ROI 定義（`MSG_ROI`）を参照して、メッセージボックス領域を切り出す。

```python
# ROI例（pipeline.py の _MSG_BOX_ROI に合わせること）
MSG_ROI = (0, 900, 1920, 1080)   # x1,y1,x2,y2
```

### 1-3. CRAFT でテキスト領域を自動検出する

EasyOCR の CRAFT 検出器を使って各フレームからテキストクロップを生成する。

```python
import easyocr, cv2, os, json

reader = easyocr.Reader(['ja'], gpu=True)
out_dir = "data/ocr_crops/text_crops"
os.makedirs(out_dir, exist_ok=True)
annotations = []

for img_path in sorted(glob("data/ocr_crops/frames/*.png")):
    img = cv2.imread(img_path)
    results = reader.readtext(img)
    for i, (bbox, text, conf) in enumerate(results):
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        crop = img[y1:y2, x1:x2]
        name = f"{os.path.basename(img_path)[:-4]}_{i:03d}.png"
        cv2.imwrite(os.path.join(out_dir, name), crop)
        annotations.append({"file": name, "ocr_text": text, "conf": conf})

with open("data/ocr_crops/annotations.json", "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)
```

### 1-4. ラベルを手修正する

`annotations.json` を開いて `ocr_text` を正しいテキストに修正する。EasyOCR が誤読したケースを中心に修正する（`プテラ`→`フーラ` 等）。

- 低品質クロップ（blur / 部分欠け）は削除
- 目標: 500〜1000 件の正解ラベル付きクロップ

---

## Phase 2: 学習環境構築

### 2-1. リポジトリをクローン

```bash
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
cd deep-text-recognition-benchmark
pip install lmdb fire Pillow torch torchvision
```

### 2-2. LMDB データセットを作成

```bash
python create_lmdb_dataset.py \
  --inputPath data/ocr_crops/text_crops/ \
  --gtFile data/ocr_crops/annotations_gt.txt \
  --outputPath data/ocr_lmdb/train/
```

`annotations_gt.txt` のフォーマット（タブ区切り）：
```
frame_00001_000.png	プテラの
frame_00001_001.png	いわなだれ
```

検証用（val）データも同様に作成する（全体の10〜20%）。

### 2-3. ベースモデルをダウンロード

EasyOCR のモデルキャッシュから `japanese_g2.pth` をコピーする。

```bash
# Windowsの場合: C:\Users\<user>\.EasyOCR\model\japanese_g2.pth
# WSL2の場合:
cp ~/.EasyOCR/model/japanese_g2.pth pretrained/japanese_g2.pth
```

---

## Phase 3: ファインチューニング

### 3-1. 学習コマンド

```bash
python train.py \
  --train_data data/ocr_lmdb/train/ \
  --valid_data data/ocr_lmdb/val/ \
  --saved_model pretrained/japanese_g2.pth \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --batch_size 32 \
  --num_iter 5000 \
  --valInterval 500 \
  --FT \
  --character "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン゛゜ーっんァィゥェォャュョ" \
  --exp_name pokemon_finetune
```

- `--FT`: ファインチューニングモード（既存重みを初期値として使用）
- RTX 3080 で 5000 iter ≒ 1〜3時間程度
- `saved_models/pokemon_finetune/best_accuracy.pth` に最良モデルが保存される

### 3-2. 精度確認

```bash
python test.py \
  --eval_data data/ocr_lmdb/val/ \
  --saved_model saved_models/pokemon_finetune/best_accuracy.pth \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn
```

Accuracy が元モデルより改善していることを確認する。

---

## Phase 4: パイプラインへの統合

### 4-1. EasyOCR カスタムモデルのロード

`src/capture/screen_capture.py` の `init_reader()` を修正して、ファインチューニング済みモデルを使用する。

```python
import easyocr

def init_reader() -> easyocr.Reader:
    return easyocr.Reader(
        ['ja'],
        gpu=True,
        recog_network='japanese_g2',          # ベースネットワーク名
        model_storage_directory='models/',     # カスタムモデル格納先
        user_network_directory='models/',
    )
```

カスタムモデルを `models/` に配置：
```
models/
├── japanese_g2.pth        # ファインチューニング済みモデル（best_accuracy.pth をリネーム）
└── japanese_g2.yaml       # ネットワーク定義（元モデルの .yaml をコピー）
```

### 4-2. 効果測定

1. `venv\Scripts\python.exe src/pipeline.py --input <検証動画>` でログを取る
2. `records/書き起こし.md` と比較して技名・ポケモン名の誤読件数を比較する
3. 改善が見られない場合はデータ数を増やして再学習する

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `character` 未設定で文字化け | `--character` に使用する全文字を含める（英数字も含む場合は追加） |
| VRAM OOM | `--batch_size` を 16 に下げる |
| 精度が上がらない | データ数が少ない可能性。500件未満なら収集を増やす |
| 元モデルより悪化 | 過学習の可能性。`--num_iter` を 3000 に下げて再試行 |
| `japanese_g2.yaml` が見つからない | `~/.EasyOCR/user_network/` または EasyOCR インストール先の `easyocr/model/` を確認 |

---

## 参考リンク

- [EasyOCR fine-tuning guide (公式 GitHub)](https://github.com/JaidedAI/EasyOCR/blob/master/trainer/README.md)
- [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- [create_lmdb_dataset.py](https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/create_lmdb_dataset.py)
