# screen_capture.py 使用マニュアル

**対象ファイル**: `src/capture/screen_capture.py`
**2026-07-09更新**: 旧CLI（`--image`/`--live`/`--camera`）はSprint 1時代の到達不能コードだったため削除済み。現在はライブラリ関数のみを提供するモジュール。

---

## 概要

このファイルはCLIエントリポイントではなく、`src/pipeline.py`や`scripts/`配下の各ツールが import して使うコアユーティリティ集。

| 関数/クラス | 役割 |
|------------|------|
| `init_reader(gpu: bool)` | EasyOCRリーダーを初期化（`pokemon_g2`ファインチューニングモデルがあれば優先使用） |
| `run_ocr(reader, image, preprocess_hp=False, preprocess_dense=False)` | 画像に対してOCRを実行し `[{"text", "confidence", "bbox"}, ...]` を返す |
| `print_ocr_results(results)` | OCR結果をコンソールに整形表示（デバッグ用） |
| `DiffDetector` | OpenCV差分検出でフレーム間の変化イベントを検知するクラス |

---

## 実際の起動手段

対戦画面のリアルタイム認識を試したい場合は、このファイルを直接実行するのではなく `src/pipeline.py` を使う。

```powershell
# カメラ（ライブ・OBS仮想カメラ経由）
venv\Scripts\python.exe src/pipeline.py --end-model runs/detect/train_end_screen2/weights/best.pt --ec2-url http://<EC2-IP>:5000 --conf 0.3

# 動画ファイル（検証用）
venv\Scripts\python.exe src/pipeline.py --input "D:\ゲーム録画\battle.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --ec2-url http://<EC2-IP>:5000 --conf 0.3
```

（`--model`＝状態異常YOLO・`--ball-model`＝ボール数YOLOは2026-07-15より未指定がデフォルト。状態異常はテキストOCRで代替済み、ボール数は現在未使用。再度使いたい場合のみ`--model runs/detect/train4/weights/best.pt`等を追加する）

OCRの生データだけを動画から確認したい場合は `scripts/ocr_logger.py`（フィルタなし・全フレームOCR診断ロガー）や `scripts/test_ocr_areas.py`（定義済みROIごとの確認）を使う。どちらも `init_reader`/`run_ocr` をこのファイルから import している。

---

## よくあるトラブル

### OCRが日本語を誤認識する
- 信頼度が低い結果は精度が不十分
- ポケモン名の後補正は `PokeClassifier`（`src/pokedb/classifier.py`）が担当
- ファインチューニング状況は `docs/manual/ocr-finetune-cycle.md` を参照

### 差分検出でイベントが検知されない / されすぎる
`DiffDetector` クラスの定数を調整する（`screen_capture.py` の `DIFF_THRESHOLD`/`MIN_CHANGE_AREA`）。

```python
DIFF_THRESHOLD = 30    # 大きくすると検知しにくくなる（誤検知が減る）
MIN_CHANGE_AREA = 1000  # 大きくすると小さな変化を無視する
```
