import sys
import cv2
from pathlib import Path

save = "--save" in sys.argv

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{i}] {w}x{h}")
        if save:
            # 数フレーム読み捨ててから保存（カメラ起動直後は黒画面になることがある）
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if ret:
                path = f"debug/camera_{i}.png"
                Path("debug").mkdir(exist_ok=True)
                cv2.imwrite(path, frame)
                print(f"  → 保存: {path}")
        cap.release()
    else:
        print(f"[{i}] なし")
