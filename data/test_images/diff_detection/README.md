# 差分検出テスト用画像

Sprint 1 の OpenCV 差分検出チューニングに使う画像置き場。
「変化あり」「変化なし」のペアを用意することで閾値を調整できる。

## 推奨スペック

| 項目 | 推奨 |
|------|------|
| 形式 | **PNG** |
| 解像度 | 実際のキャプチャと同じサイズ（統一すること） |
| 命名規則 | `scene_<シーン名>_<連番>.png` |

## 集めてほしいペア画像

連続した場面を2枚セットで用意する。

### パターン1: 変化「あり」ペア（イベントとして検知してほしい）

| ファイル名 | 説明 |
|-----------|------|
| `scene_turn_before.png` | 技選択画面（ターン前） |
| `scene_turn_after.png` | 技を選んだ直後（アニメ開始前） |
| `scene_switch_before.png` | 交代前 |
| `scene_switch_after.png` | 交代後（別ポケモンが出た状態） |
| `scene_faint_before.png` | 気絶直前（HP が少ない状態） |
| `scene_faint_after.png` | 気絶後（倒れたアニメ終わり） |

### パターン2: 変化「なし」ペア（イベントとして検知してほしくない）

| ファイル名 | 説明 |
|-----------|------|
| `scene_idle_1.png` | 何も起きていないターン待機中（1枚目） |
| `scene_idle_2.png` | 同じ状態の1秒後（2枚目） |
| `scene_thinking_1.png` | 相手の行動選択待ち（1枚目） |
| `scene_thinking_2.png` | 同じ状態の1秒後（2枚目） |

## 使い方（手動で差分スコアを確認するスクリプト）

```python
import cv2
import numpy as np

img1 = cv2.imread("data/test_images/diff_detection/scene_turn_before.png")
img2 = cv2.imread("data/test_images/diff_detection/scene_turn_after.png")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray1, gray2)
print(f"差分スコア: {diff.mean():.2f}")  # 30以上でイベント検知される設定
```

変化「あり」ペアは30以上、変化「なし」ペアは30未満になるのが理想。
ずれる場合は `screen_capture.py` の `DIFF_THRESHOLD` を調整する。
