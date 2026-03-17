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

## 既知の問題と対処

### LabelImg が Python 3.12 でエラーになる

以下のファイルを修正済み（再発したら確認する）:

- `venv/Lib/site-packages/labelImg/labelImg.py` L965: `int()` キャスト追加
- `venv/Lib/site-packages/libs/canvas.py` L526, L530-531: `int()` キャスト追加
