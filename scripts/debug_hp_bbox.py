"""
フレームのOCR全結果とbbox座標を出力して、
7967135などの問題テキストがどこから来ているか調べる。
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.capture.screen_capture import init_reader

HP_TEXT_ROIS = {
    "hp_plr0": (240,  1000, 410,  1050),
    "hp_plr1": (640,  1000, 810,  1050),
    "hp_opp0": (1330, 120,  1450, 170),
    "hp_opp1": (1720, 120,  1840, 170),
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_hp_bbox.py <video> <sec>")
        sys.exit(1)

    video, sec = sys.argv[1], float(sys.argv[2])
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("フレーム取得失敗"); sys.exit(1)

    h, w = frame.shape[:2]
    print(f"フレームサイズ: {w}x{h}")
    sx, sy = w / 1920, h / 1080

    print("\n--- フルフレームOCR 全結果 ---")
    reader = init_reader()
    results = reader.readtext(frame)

    # 可視化用フレーム
    vis = frame.copy()
    out_dir = Path("debug/hp_bbox")
    out_dir.mkdir(parents=True, exist_ok=True)

    for (bbox, text, conf) in results:
        pts = [[int(p[0]), int(p[1])] for p in bbox]
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4

        # どのHP ROIに入るか判定
        roi_hit = None
        for name, (x1, y1, x2, y2) in HP_TEXT_ROIS.items():
            rx1, ry1 = int(x1*sx), int(y1*sy)
            rx2, ry2 = int(x2*sx), int(y2*sy)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                roi_hit = name
                break

        marker = f" ★{roi_hit}" if roi_hit else ""
        print(f"  [{int(cx):4d},{int(cy):4d}] conf={conf:.2f}  '{text}'{marker}")

        # 枠を描画
        color = (0, 0, 255) if roi_hit else (200, 200, 200)
        thickness = 2 if roi_hit else 1
        cv2.polylines(vis, [np.array(pts)], True, color, thickness)
        cv2.putText(vis, text[:10], (pts[0][0], pts[0][1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # HP ROI枠を描画
    for name, (x1, y1, x2, y2) in HP_TEXT_ROIS.items():
        rx1, ry1 = int(x1*sx), int(y1*sy)
        rx2, ry2 = int(x2*sx), int(y2*sy)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(vis, name, (rx1, ry1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    out = out_dir / f"bbox_{int(sec)}s.png"
    cv2.imwrite(str(out), vis)
    print(f"\n可視化画像: {out}")

    # HP ROIの前処理画像も保存
    print("\n--- HP ROI 前処理画像 ---")
    for name, (x1, y1, x2, y2) in HP_TEXT_ROIS.items():
        rx1, ry1 = int(x1*sx), int(y1*sy)
        rx2, ry2 = int(x2*sx), int(y2*sy)
        roi = frame[ry1:ry2, rx1:rx2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        # 上部トリム量を試す（HPバーがROI上端に混入する対策）
        trim_ratios = [0.0, 0.2, 0.4]
        trim_results = []
        for tr in trim_ratios:
            trim_px = int(roi.shape[0] * tr)
            roi_t = roi[trim_px:, :]
            gray_t = cv2.cvtColor(roi_t, cv2.COLOR_BGR2GRAY)
            _, bin_t = cv2.threshold(gray_t, 160, 255, cv2.THRESH_BINARY)
            img_t = np.full_like(roi_t, 255)
            img_t[bin_t == 255] = [0, 0, 0]
            res = reader.readtext(img_t)
            texts = [f"'{t}'({c:.2f})" for _, t, c in res]
            label = f"trim{int(tr*100)}%"
            print(f"  {name} [{label}]: {texts if texts else '(なし)'}")
            trim_results.append((label, img_t))

        # 白bg黒文字（ベースライン、トリムなし）
        img_base = np.full_like(roi, 255)
        img_base[binary == 255] = [0, 0, 0]

        def deskew(img, shear):
            """イタリック補正: shear>0 で右傾きを補正"""
            h2, w2 = img.shape[:2]
            extra = int(abs(shear) * h2)
            new_w = w2 + extra
            M = np.float32([[1, shear, max(0, -shear * h2)],
                             [0, 1,    0]])
            return cv2.warpAffine(img, M, (new_w, h2),
                                  borderValue=(255, 255, 255))

        # シアー量ごとに試す（正 = 右傾き補正、負 = 左傾き補正）
        shear_values = [0.2, 0.3, 0.4, 0.5]
        variants = [("base(0.0)", img_base)]
        for s in shear_values:
            ds = deskew(img_base, s)
            variants.append((f"shear{s}", ds))

        for label, img in variants:
            res = reader.readtext(img)
            texts = [f"'{t}'({c:.2f})" for _, t, c in res]
            print(f"  {name} [{label}]: {texts if texts else '(なし)'}")

        # 横並び保存（オリジナル + トリムバリアント + シアーバリアント）
        target_h = roi.shape[0] * 4
        panels = [cv2.resize(roi, (int(roi.shape[1] * target_h / roi.shape[0]), target_h),
                             interpolation=cv2.INTER_NEAREST)]
        for _, img in trim_results:
            h2, w2 = img.shape[:2]
            panels.append(cv2.resize(img, (int(w2 * target_h / h2), target_h),
                                     interpolation=cv2.INTER_NEAREST))
        for _, img in variants:
            h2, w2 = img.shape[:2]
            panels.append(cv2.resize(img, (int(w2 * target_h / h2), target_h),
                                     interpolation=cv2.INTER_NEAREST))
        roi_out = out_dir / f"{name}_{int(sec)}s.png"
        cv2.imwrite(str(roi_out), np.hstack(panels))
        print(f"    → {roi_out}")

if __name__ == "__main__":
    main()
