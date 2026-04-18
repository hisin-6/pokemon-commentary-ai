"""
対戦画面の座標定義を可視化するスクリプト

使い方:
    python scripts/visualize_coords.py <画像ファイルパス> [--out 出力パス]
    python scripts/visualize_coords.py <動画ファイルパス> --frame 150 [--out 出力パス]

出力: 各ROI・閾値ラインを重ねた画像（デフォルト: debug/coords_visual.png）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MSG_X_MIN = 120              # BattleMessageParser: メッセージボックス左端（右マージンと対称）
MSG_X_MAX = 1800             # BattleMessageParser: メッセージボックス右端（右マージン 120px）
MSG_Y_MIN = 750             # BattleMessageParser: メッセージボックス上端
MSG_Y_MAX = 930             # BattleMessageParser: メッセージボックス下端

# ── ボールアイコンエリア（yolo_detector.py ROIS と同値）
# YOLO ROI（比率: x_min, y_min, x_max, y_max）
BALL_ROI_OPP = (0.86, 0.15, 0.93, 0.19)     # opponent_balls: 右上端
BALL_ROI_PLR = (0.04, 0.80, 0.11, 0.84)     # player_balls:   左下端


# ── 状態異常アイコンエリア（per-pokemon / 実測で要調整） ──────────────────
# 名前エリアの左上あたりに表示される想定
STATUS_ICON_CHAMP = {
    "opp_status_0 (Champ)":    dict(x1=1135, x2=1215, y1=20,  y2=80,  color=(255, 100, 100)),
    "opp_status_1 (Champ)":    dict(x1=1535, x2=1615, y1=20,  y2=80,  color=(220,  80,  80)),
    "player_status_0 (Champ)": dict(x1=105,  x2=170,  y1=900, y2=960, color=(100, 100, 255)),
    "player_status_1 (Champ)": dict(x1=505,  x2=570,  y1=900, y2=960, color=( 80,  80, 220)),
}

# ── 特性・道具発動メッセージエリア（チャンピオンズ / 実測で要調整） ──────────
# 自分: 左端〜player_name_1左端(x=555)、画面縦中央あたり
# 相手: その左右対称（x=1365〜右端）
ABILITY_MSG_CHAMP = {
    "ability_msg_player (Champ)": dict(x1=0,    x2=555,  y1=450, y2=570, color=(255, 220, 80)),
    "ability_msg_opp (Champ)":    dict(x1=1365, x2=1920, y1=450, y2=570, color=(255, 180, 60)),
}

# ── ポケモン名エリア ──────────────────────────────────────────────────────
# チャンピオンズ（1920x1080 基準 / 実測で要調整）
POKEMON_NAME_CHAMP = {
    "opp_name_0 (Champ)":    dict(x1=1200, x2=1380, y1=50,   y2=90,  color=(180, 255, 80)),
    "opp_name_1 (Champ)":    dict(x1=1600, x2=1780, y1=50,   y2=90,  color=(140, 220, 60)),
    "player_name_0 (Champ)": dict(x1=155,  x2=335,  y1=930,  y2=970,  color=(80,  255, 180)),
    "player_name_1 (Champ)": dict(x1=555,  x2=735,  y1=930,  y2=970,  color=(60,  220, 140)),
}

# ── hpbar_analyzer.py 由来の座標定数 ────────────────────────────────────
# 自分HPバー（SV・チャンピオンズ共通）
HP_SLOTS_PLAYER = {
    "player_0": dict(x1=240, x2=410, y1=1000, y2=1050, color=(0, 220, 0)),
    "player_1": dict(x1=640, x2=810, y1=1000, y2=1050, color=(0, 180, 0)),
}
# 相手HPバー チャンピオンズ（実測 2026-04-13）
HP_SLOTS_CHAMP = {
    "opp_0 (Champ)": dict(x1=1330, x2=1450, y1=120, y2=170, color=(0, 120, 255)),
    "opp_1 (Champ)": dict(x1=1720, x2=1840, y1=120, y2=170, color=(0, 80, 220)),
}


def draw_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, label: str, alpha: float = 0.25) -> None:
    """半透明塗りつぶし + 枠 + ラベルを描画する"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # ラベル背景
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    lx, ly = x1 + 4, max(y1 + th + 4, th + 4)
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (lx, ly), font, scale, color, thick, cv2.LINE_AA)


def annotate(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    img = frame.copy()

    # ── 1. メッセージボックスROI ──────────────────────────────────────
    draw_region(img,
                MSG_X_MIN, MSG_Y_MIN, MSG_X_MAX, MSG_Y_MAX,
                (255, 100, 255),
                f"MSG ROI  x={MSG_X_MIN}-{MSG_X_MAX} y={MSG_Y_MIN}-{MSG_Y_MAX}")

    # ── 3. ボールアイコン YOLO ROI ───────────────────────────────────
    for ratio, color, label in [
        (BALL_ROI_OPP, (255, 200,  0),  "opponent_balls ROI (YOLO)"),
        (BALL_ROI_PLR, (200, 255,  0),  "player_balls ROI (YOLO)"),
    ]:
        rx1, ry1, rx2, ry2 = int(w * ratio[0]), int(h * ratio[1]), int(w * ratio[2]), int(h * ratio[3])
        draw_region(img, rx1, ry1, rx2, ry2, color, label, alpha=0.15)

    # ── 4. 自分HPバースロット ─────────────────────────────────────────
    for label, s in HP_SLOTS_PLAYER.items():
        draw_region(img, s["x1"], s["y1"], s["x2"], s["y2"],
                    s["color"], label)

    # ── 5. チャンピオンズ 相手HPバースロット ──────────────────────────
    for label, s in HP_SLOTS_CHAMP.items():
        draw_region(img, s["x1"], s["y1"], s["x2"], s["y2"],
                    s["color"], label)

    # ── 6. ポケモン名エリア（チャンピオンズ） ────────────────────────
    for label, s in POKEMON_NAME_CHAMP.items():
        draw_region(img, s["x1"], s["y1"], s["x2"], s["y2"],
                    s["color"], label)

    # ── 7. 状態異常アイコンエリア（per-pokemon / チャンピオンズ） ──────
    for label, s in STATUS_ICON_CHAMP.items():
        draw_region(img, s["x1"], s["y1"], s["x2"], s["y2"],
                    s["color"], label)

    # ── 8. 特性・道具発動メッセージエリア（チャンピオンズ） ──────────
    for label, s in ABILITY_MSG_CHAMP.items():
        draw_region(img, s["x1"], s["y1"], s["x2"], s["y2"],
                    s["color"], label)

    # ── 凡例 ──────────────────────────────────────────────────────────
    legends = [
        ((255, 100, 255), "MSG ROI (BattleMessageParser)"),
        ((255, 200,  0),  "opponent/player_balls ROI (YOLO)"),
        ((0, 220, 0),     "HP player_0 / player_1"),
        ((0, 120, 255),   "HP opp_0 / opp_1 (Champ)"),
        ((180, 255, 80),  "Pokemon name (Champ)"),
        ((255, 100, 100), "status icon opp_0/1 (Champ) ※要調整"),
        ((100, 100, 255), "status icon plr_0/1 (Champ) ※要調整"),
        ((255, 220,  80), "ability/item msg player (Champ) ※要調整"),
        ((255, 180,  60), "ability/item msg opp (Champ) ※要調整"),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (color, text) in enumerate(legends):
        ly = 20 + i * 22
        cv2.rectangle(img, (w - 380, ly - 14), (w - 366, ly), color, -1)
        cv2.putText(img, text, (w - 360, ly - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def load_frame(path: str, frame_no: int) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"[ERROR] 画像を読み込めません: {path}")
        return img

    # 動画
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        sys.exit(f"[ERROR] ファイルを開けません: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[ERROR] フレーム {frame_no} を取得できません")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="対戦画面の座標定義を可視化する")
    ap.add_argument("input", help="画像ファイルまたは動画ファイルのパス")
    ap.add_argument("--frame", type=int, default=0,
                    help="動画の場合に使用するフレーム番号（デフォルト: 0）")
    ap.add_argument("--out", default="debug/coords_visual.png",
                    help="出力先パス（デフォルト: debug/coords_visual.png）")
    args = ap.parse_args()

    frame = load_frame(args.input, args.frame)
    print(f"[INFO] 読み込み完了: {frame.shape[1]}x{frame.shape[0]} フレーム={args.frame}")

    annotated = annotate(frame)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"[INFO] 保存: {out_path}")


if __name__ == "__main__":
    main()
