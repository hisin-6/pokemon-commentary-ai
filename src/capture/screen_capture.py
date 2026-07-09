"""
画面キャプチャ + EasyOCR テキスト認識のコアユーティリティ。

`init_reader` / `run_ocr` / `DiffDetector` は `src/pipeline.py` や `scripts/` 配下の
各ツールから利用される。CLIエントリポイントとしては現在使われておらず、実運用の
起動手段は `src/pipeline.py --camera` / `--input` を参照。
"""

from __future__ import annotations

import logging

import os
import cv2
import numpy as np
import easyocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── OCRリーダー（初回起動時にモデルDL、約500MB） ───────────────────────────

def init_reader(gpu: bool = True) -> easyocr.Reader:
    log.info("EasyOCRリーダーを初期化中（初回はモデルDLで1〜2分かかります）...")
    user_network_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR", "user_network")
    pokemon_model = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model", "pokemon_g2.pth")
    if os.path.exists(pokemon_model):
        log.info("pokemon_g2 ファインチューニングモデルを使用します")
        reader = easyocr.Reader(
            ["ja"],
            gpu=gpu,
            recog_network="pokemon_g2",
            user_network_directory=user_network_dir,
        )
    else:
        log.warning("pokemon_g2 が見つからないため japanese_g2 にフォールバックします")
        reader = easyocr.Reader(["ja", "en"], gpu=gpu)
    log.info("EasyOCRリーダー初期化完了")
    return reader




# ─── OCR処理 ────────────────────────────────────────────────────────────────

# ─── HPテキストROI前処理（チャンピオンズ用） ────────────────────────────────
# hp_opp: HPバー（黄緑グラデーション）がROI上部に混入してOCR誤読する。
#   → フルフレームOCRから除外し、個別前処理OCRで差し替える。
# hp_plr: フルフレームOCRで正しく読めるため、マスクせず通常フローに流す。
_HP_OPP_ROIS_1920 = [
    (1330, 120,  1450, 170),    # hp_opp0
    (1720, 120,  1840, 170),    # hp_opp1
]
_HP_THRESH = 160       # この輝度以上のピクセル（白テキスト）を保持
_DENSE_SCALE = 2       # dense scan 前処理: スケールアップ倍率
_DENSE_THRESH = 160    # dense scan 前処理: 白テキスト抽出閾値（メッセージボックスの白文字）


def _scale_roi(x1, y1, x2, y2, w, h):
    sx, sy = w / 1920, h / 1080
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def _ocr_hp_opp_rois(reader: easyocr.Reader, frame: np.ndarray):
    """
    hp_opp ROIを個別前処理してOCRし、フレーム座標系のbboxつき結果リストを返す。
    HPバー混入によるノイズをthreshold除去してから認識する。
    """
    h, w = frame.shape[:2]
    results = []
    for (x1, y1, x2, y2) in _HP_OPP_ROIS_1920:
        rx1, ry1, rx2, ry2 = _scale_roi(x1, y1, x2, y2, w, h)
        roi = frame[ry1:ry2, rx1:rx2].copy()
        if roi.size == 0:
            continue

        # 白テキスト抽出: 閾値超えのピクセルのみ白、他は黒
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, _HP_THRESH, 255, cv2.THRESH_BINARY)
        ocr_img = np.zeros_like(roi)
        ocr_img[binary == 255] = [255, 255, 255]

        for (bbox, text, conf) in reader.readtext(ocr_img):
            # bbox座標をフレーム全体の座標系に変換
            adjusted = [[pt[0] + rx1, pt[1] + ry1] for pt in bbox]
            results.append({
                "text": text,
                "confidence": round(conf, 3),
                "bbox": adjusted,
            })
    return results


def run_ocr(reader: easyocr.Reader, image: np.ndarray,
            preprocess_hp: bool = False,
            preprocess_dense: bool = False):
    """
    画像に対してOCRを実行し、認識結果のリストを返す。

    Args:
        preprocess_hp: Trueのとき hp_opp ROIをフルフレームOCRから除外し、
                       個別前処理OCRで差し替える（チャンピオンズ対応）。
                       hp_plr はフルフレームOCRで正しく読めるためマスクしない。
        preprocess_dense: Trueのとき dense scan 用前処理を適用する。
                          2×スケールアップ + 白テキスト抽出でOCR精度を向上。
                          bboxはオリジナルスケールに変換して返す。
    Returns:
        [{"text": str, "confidence": float, "bbox": list}, ...]
    """
    if preprocess_dense:
        # 2× スケールアップ（CUBIC補間で文字エッジを保持）
        # 白テキスト抽出は副作用（誤検出増加）があるためスケールアップのみ適用
        scaled = cv2.resize(
            image,
            (image.shape[1] * _DENSE_SCALE, image.shape[0] * _DENSE_SCALE),
            interpolation=cv2.INTER_CUBIC,
        )
        raw = reader.readtext(scaled)
        results = []
        for (bbox, text, conf) in raw:
            # bbox座標をオリジナルスケールに戻す
            orig_bbox = [[pt[0] / _DENSE_SCALE, pt[1] / _DENSE_SCALE] for pt in bbox]
            results.append({"text": text, "confidence": round(conf, 3), "bbox": orig_bbox})
        return results

    if preprocess_hp:
        h, w = image.shape[:2]
        # フルフレームOCRでhp_opp ROI部分のみ黒塗り（HPバーノイズ抑制）
        masked = image.copy()
        for (x1, y1, x2, y2) in _HP_OPP_ROIS_1920:
            rx1, ry1, rx2, ry2 = _scale_roi(x1, y1, x2, y2, w, h)
            masked[ry1:ry2, rx1:rx2] = 0
        main_results = reader.readtext(masked)
        parsed = [{"text": t, "confidence": round(c, 3), "bbox": b}
                  for (b, t, c) in main_results]
        # hp_opp ROIを個別前処理OCRで補完
        parsed.extend(_ocr_hp_opp_rois(reader, image))
        return parsed

    results = reader.readtext(image)
    return [{"text": t, "confidence": round(c, 3), "bbox": b}
            for (b, t, c) in results]


def print_ocr_results(results: list[dict]) -> None:
    if not results:
        log.info("テキストが検出されませんでした")
        return

    log.info(f"検出テキスト数: {len(results)}")
    print("\n" + "=" * 50)
    print(f"{'テキスト':<30} {'信頼度':>8}")
    print("-" * 50)
    for r in results:
        marker = "✓" if r["confidence"] >= 0.5 else "?"
        print(f"{marker} {r['text']:<28} {r['confidence']:>8.1%}")
    print("=" * 50 + "\n")


# ─── 差分検出 ───────────────────────────────────────────────────────────────

class DiffDetector:
    """
    OpenCV差分検出でターン切替などのイベントを検知する。
    """

    DIFF_THRESHOLD = 30    # フレーム間のピクセル平均差分の閾値
    MIN_CHANGE_AREA = 1000  # 変化領域の最小面積（px²）

    def __init__(self):
        self._prev_frame: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Returns:
            (イベント発生フラグ, 差分スコア)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_frame = gray
            return False, 0.0

        diff = cv2.absdiff(self._prev_frame, gray)
        score = float(diff.mean())

        # 変化領域の面積チェック
        _, thresh = cv2.threshold(diff, self.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        changed_area = int(np.sum(thresh > 0))

        self._prev_frame = gray
        event_detected = score > self.DIFF_THRESHOLD and changed_area > self.MIN_CHANGE_AREA
        return event_detected, score
