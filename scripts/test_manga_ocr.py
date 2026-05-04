"""
EasyOCR vs MangaOCR 比較テスト
================================
固定ROI（HP・ポケモン名エリア）で両モデルの認識結果を並べて表示する。
MangaOCRは認識専用（検出なし）なので固定ROIとの相性確認が目的。

使い方:
    venv\Scripts\python.exe scripts/test_manga_ocr.py <動画ファイル> <秒>

例:
    venv\Scripts\python.exe scripts/test_manga_ocr.py "D:\録画\battle.mp4" 91
"""

import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.capture.screen_capture import init_reader

# ── テスト対象ROI定義（1920x1080基準） ────────────────────────────────────────
ROIS = {
    "hp_plr0":   (240,  1000, 410,  1050),
    "hp_plr1":   (640,  1000, 810,  1050),
    "hp_opp0":   (1330, 120,  1450, 170),
    "hp_opp1":   (1720, 120,  1840, 170),
    "name_plr0": (155,  930,  335,  970),
    "name_plr1": (555,  930,  735,  970),
    "name_opp0": (1200, 50,   1380, 90),
    "name_opp1": (1600, 50,   1780, 90),
    "msg":       (120,  740,  1800, 930),
}

SCALE_FACTORS = [1, 2, 3]  # ROIを何倍に拡大してOCRするか


def crop_roi(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    h, w = frame.shape[:2]
    sx, sy = w / 1920, h / 1080
    return frame[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)].copy()


def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def run_easyocr(reader, roi: np.ndarray) -> str:
    results = reader.readtext(roi)
    return " ".join(t for _, t, c in results if c >= 0.3) or "（なし）"


def run_manga_ocr(mocr, roi: np.ndarray) -> str:
    pil = np_to_pil(roi)
    text = mocr(pil).strip()
    return text or "（なし）"


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_manga_ocr.py <video> <sec>")
        sys.exit(1)

    video, sec = sys.argv[1], float(sys.argv[2])

    # フレーム取得
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("フレーム取得失敗")
        sys.exit(1)

    print(f"フレームサイズ: {frame.shape[1]}x{frame.shape[0]}  ({int(sec//60):02d}:{sec%60:05.2f})")

    # モデル初期化
    print("\nEasyOCR 初期化中...")
    easy_reader = init_reader()

    print("MangaOCR 初期化中...")
    from manga_ocr import MangaOcr
    mocr = MangaOcr()

    # ── 各ROIで比較 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'ROI':<12} {'スケール':>6}  {'EasyOCR':<25} {'MangaOCR':<25}")
    print("=" * 70)

    out_dir = Path("debug/manga_ocr")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (x1, y1, x2, y2) in ROIS.items():
        roi = crop_roi(frame, x1, y1, x2, y2)
        if roi.size == 0:
            continue

        for scale in SCALE_FACTORS:
            if scale == 1:
                roi_s = roi
            else:
                roi_s = cv2.resize(roi, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

            easy_text = run_easyocr(easy_reader, roi_s)
            manga_text = run_manga_ocr(mocr, roi_s)

            marker = "★" if easy_text != manga_text else " "
            print(f"{marker}{name:<11} {scale:>5}x  {easy_text:<25} {manga_text:<25}")

        # 元画像を保存
        save_path = out_dir / f"{name}_{int(sec)}s.png"
        cv2.imwrite(str(save_path), roi)

    print("=" * 70)
    print(f"\nROI画像保存先: {out_dir}/")


if __name__ == "__main__":
    main()
