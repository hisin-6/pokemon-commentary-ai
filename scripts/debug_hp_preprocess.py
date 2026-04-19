"""
HPテキストROIの前処理効果を可視化するデバッグスクリプト。
動画の指定フレームからhp_plr0/hp_plr1エリアを切り出し、
いくつかの閾値でテキスト抽出した画像を debug/hp_preprocess/ に保存する。
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# 1920x1080基準のHP数値テキストROI (x1, y1, x2, y2)
HP_TEXT_ROIS = {
    "hp_plr0": (240, 1000, 410, 1050),
    "hp_plr1": (640,  1000, 810, 1050),
    "hp_opp0": (1330, 120,  1450, 170),
    "hp_opp1": (1720, 120,  1840, 170),
}

def load_frame(video_path: str, sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"フレーム取得失敗: {sec}s")
    return frame

def extract_roi(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    h, w = frame.shape[:2]
    sx1 = int(x1 * w / 1920)
    sy1 = int(y1 * h / 1080)
    sx2 = int(x2 * w / 1920)
    sy2 = int(y2 * h / 1080)
    return frame[sy1:sy2, sx1:sx2].copy()

def preprocess_white_text(roi: np.ndarray, thresh: int) -> np.ndarray:
    """明るいピクセル（白テキスト）だけ残して黒背景に変換"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    result = np.zeros_like(roi)
    result[binary == 255] = [255, 255, 255]
    return result

def preprocess_dark_bg_removal(roi: np.ndarray) -> np.ndarray:
    """背景（暗い＋彩度の高い色）を除去してテキストを残す"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 彩度が高い（色ある）→ 背景のHP棒色など → 除去
    sat_mask = hsv[:, :, 1] > 80  # 彩度>80は背景
    # 明度が低い → 除去
    val_mask = hsv[:, :, 2] < 80
    bg_mask = sat_mask | val_mask
    result = roi.copy()
    result[bg_mask] = [0, 0, 0]
    return result

def make_panel(original: np.ndarray, processed_list: list, labels: list) -> np.ndarray:
    """オリジナルと前処理画像を横に並べた比較パネルを作成"""
    target_h = 80
    imgs = [original] + processed_list
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = max(1, int(w * scale))
        resized.append(cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_NEAREST))

    max_w = max(r.shape[1] for r in resized)
    padded = []
    for img in resized:
        h, w = img.shape[:2]
        pad = np.zeros((h, max_w, 3), dtype=np.uint8)
        pad[:, :w] = img
        padded.append(pad)

    panel = np.hstack(padded)

    # ラベル追加
    all_labels = ["original"] + labels
    x = 0
    for label, img in zip(all_labels, padded):
        cv2.putText(panel, label, (x + 2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        x += img.shape[1]
    return panel

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_hp_preprocess.py <video> <sec>")
        print("  例: python debug_hp_preprocess.py \"D:\\動画.mp4\" 152")
        sys.exit(1)

    video = sys.argv[1]
    sec = float(sys.argv[2])

    print(f"フレーム取得: {sec}s")
    frame = load_frame(video, sec)

    out_dir = Path("debug/hp_preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for name, (x1, y1, x2, y2) in HP_TEXT_ROIS.items():
        roi = extract_roi(frame, x1, y1, x2, y2)

        processed = [
            preprocess_white_text(roi, 140),
            preprocess_white_text(roi, 160),
            preprocess_white_text(roi, 180),
            preprocess_dark_bg_removal(roi),
        ]
        labels = ["thresh140", "thresh160", "thresh180", "sat_removal"]

        panel = make_panel(roi, processed, labels)
        label_bar = np.zeros((20, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, f"[{name}]  ({x1},{y1})-({x2},{y2})",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)
        panels.append(np.vstack([label_bar, panel]))

    max_w = max(p.shape[1] for p in panels)
    padded_panels = []
    for p in panels:
        h, w = p.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        padded_panels.append(p)
    result = np.vstack(padded_panels)
    out_path = out_dir / f"hp_preprocess_{int(sec)}s.png"
    cv2.imwrite(str(out_path), result)
    print(f"保存: {out_path}")

if __name__ == "__main__":
    main()
