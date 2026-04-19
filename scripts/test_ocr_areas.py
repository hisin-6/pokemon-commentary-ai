"""
各エリアのOCRテキスト検出確認スクリプト
=========================================
動画ファイルに対してOCRを実行し、定義済みエリアごとに検出テキストを表示する。
YOLO・EC2・Bedrock 不要。OCRのみ。

使い方:
    venv\\Scripts\\python.exe scripts/test_ocr_areas.py <動画ファイル> [オプション]

オプション:
    --step  SEC    SEC 秒ごとに1回OCRを実行（デフォルト: 1.0）
    --start SEC    開始時刻（秒、デフォルト: 0）
    --end   SEC    終了時刻（秒）。省略で最後まで
    --save-frames  エリアを可視化した画像を debug/ocr_areas/ に保存

出力例:
    [00:10.00]
      >>> MSG ROI         : ゴリランダーのねこだまし
          ability_player  : （なし）
          ability_opp     : （なし）
      >>> status_plr0     : まひ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import re

import cv2
import numpy as np

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.capture.screen_capture import init_reader, run_ocr

# HP テキスト正規化
# スラッシュが1や7に誤読されるケースを補正: "1551155"→"155/155", "1557155"→"155/155"
_HP_SLASH_RE = re.compile(r'^(\d{1,3})[17](\d{2,3})$')
_HP_PCT_RE   = re.compile(r'^(\d{1,3})%$')


def _normalize_hp_text(t: str) -> str:
    """HPエリアのOCR誤読を補正する。"""
    # O→0、{→8（hp_opp1でよく発生）
    t = t.replace('O', '0').replace('o', '0').replace('{', '8')
    m = _HP_SLASH_RE.match(t)
    if m:
        cur, mx = int(m.group(1)), int(m.group(2))
        if cur <= mx:  # 分子 > 分母は誤補正として弾く
            return f"{cur}/{mx}"
    return t

# ─── エリア定義（visualize_coords.py / pipeline.py と同値） ──────────────────

AREAS = {
    # メッセージ
    "MSG ROI":        dict(x1=120,  x2=1800, y1=740,  y2=930,  color=(255, 100, 255)),
    # 特性・道具発動
    "ability_player": dict(x1=0,    x2=555,  y1=450,  y2=570,  color=(255, 220, 80)),
    "ability_opp":    dict(x1=1365, x2=1920, y1=450,  y2=570,  color=(255, 180, 60)),
    # 状態異常アイコン
    "status_opp0":    dict(x1=1135, x2=1215, y1=20,   y2=80,   color=(255, 100, 100)),
    "status_opp1":    dict(x1=1535, x2=1615, y1=20,   y2=80,   color=(220,  80,  80)),
    "status_plr0":    dict(x1=105,  x2=170,  y1=900,  y2=960,  color=(100, 100, 255)),
    "status_plr1":    dict(x1=505,  x2=570,  y1=900,  y2=960,  color=( 80,  80, 220)),
    # ポケモン名（チャンピオンズ）
    "name_opp0":      dict(x1=1200, x2=1380, y1=50,   y2=90,   color=(180, 255, 80)),
    "name_opp1":      dict(x1=1600, x2=1780, y1=50,   y2=90,   color=(140, 220, 60)),
    "name_plr0":      dict(x1=155,  x2=335,  y1=930,  y2=970,  color=(80,  255, 180)),
    "name_plr1":      dict(x1=555,  x2=735,  y1=930,  y2=970,  color=(60,  220, 140)),
    # HPバー（チャンピオンズ）
    "hp_opp0":        dict(x1=1330, x2=1450, y1=120,  y2=170,  color=(0,  120, 255)),
    "hp_opp1":        dict(x1=1720, x2=1840, y1=120,  y2=170,  color=(0,   80, 220)),
    "hp_plr0":        dict(x1=240,  x2=410,  y1=1000, y2=1050, color=(0,  220, 0)),
    "hp_plr1":        dict(x1=640,  x2=810,  y1=1000, y2=1050, color=(0,  180, 0)),
}

# コマンドメニュー・UIノイズとして除外するワード（pipeline.py の _UI_WORDS 等と同値）
_UI_FILTER: frozenset[str] = frozenset({
    "たたかう", "ポケモン", "にげる", "相手を見る", "様子を見る",
    "どうぐ", "テラスタル", "ダイマックス", "キョダイマックス",
    "もどる", "つかう", "すてる", "わたす", "とじる",
})


def classify_token(cx: float, cy: float, frame_w: int, frame_h: int) -> str | None:
    """OCRトークンの中心座標をエリアに振り分ける。複数ヒットは最小面積のエリアを優先。"""
    scale_x = 1920 / frame_w
    scale_y = 1080 / frame_h
    ax = cx * scale_x
    ay = cy * scale_y
    matches = []
    for name, a in AREAS.items():
        if a["x1"] <= ax <= a["x2"] and a["y1"] <= ay <= a["y2"]:
            area_size = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            matches.append((area_size, name))
    if not matches:
        return None
    return min(matches)[1]  # 面積が最小のエリアを優先


def draw_areas(frame: np.ndarray) -> np.ndarray:
    """エリア枠を描画したフレームを返す。"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    sx, sy = w / 1920, h / 1080
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, a in AREAS.items():
        x1 = int(a["x1"] * sx); y1 = int(a["y1"] * sy)
        x2 = int(a["x2"] * sx); y2 = int(a["y2"] * sy)
        color = a["color"]
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(name, font, 0.5, 1)
        lx, ly = x1 + 3, max(y1 + th + 3, th + 3)
        cv2.rectangle(vis, (lx - 1, ly - th - 1), (lx + tw + 1, ly + 1), (0, 0, 0), -1)
        cv2.putText(vis, name, (lx, ly), font, 0.5, color, 1, cv2.LINE_AA)
    return vis


def draw_tokens(vis: np.ndarray, tokens: list[dict], frame_w: int, frame_h: int) -> np.ndarray:
    """OCRトークンをエリア色で描画する。"""
    sx, sy = vis.shape[1] / frame_w, vis.shape[0] / frame_h
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in tokens:
        bbox = t.get("bbox")
        if not bbox:
            continue
        cx = sum(p[0] for p in bbox) / 4
        cy = sum(p[1] for p in bbox) / 4
        area_name = classify_token(cx, cy, frame_w, frame_h)
        color = AREAS[area_name]["color"] if area_name else (180, 180, 180)
        pts = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in bbox], np.int32)
        cv2.polylines(vis, [pts], True, color, 1)
        cv2.putText(vis, t["text"], (pts[0][0], pts[0][1] - 3), font, 0.4, color, 1, cv2.LINE_AA)
    return vis


def _emit(line: str, log_file) -> None:
    """コンソールとログファイルに同時出力する。"""
    print(line)
    if log_file:
        log_file.write(line + "\n")


def process_video(
    video_path: str,
    step_sec: float,
    start_sec: float,
    end_sec: float | None,
    save_frames: bool,
    filter_ui: bool,
    log_path: Path | None,
) -> None:
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"ログ保存先: {log_path}")

    try:
        _emit(f"動画: {video_path}", log_file)
        _emit(f"OCR間隔: {step_sec}秒ごと  UIフィルター: {'ON' if filter_ui else 'OFF'}", log_file)
        _emit("", log_file)

        reader = init_reader()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _emit(f"[ERROR] 動画を開けません: {video_path}", log_file)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_sec = total_frames / fps
        end_sec = min(end_sec, total_sec) if end_sec else total_sec

        out_dir = Path("debug/ocr_areas")
        if save_frames:
            out_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        cur_sec = start_sec

        while cur_sec < end_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, cur_sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw = frame.shape[:2]
            ocr_results = run_ocr(reader, frame)

            # エリアごとにトークンを振り分け
            buckets: dict[str, list[str]] = {name: [] for name in AREAS}
            for r in ocr_results:
                if r.get("confidence", 0) < 0.3:
                    continue
                text = r["text"].strip()
                if filter_ui and text in _UI_FILTER:
                    continue
                bbox = r.get("bbox")
                if not bbox:
                    continue
                cx = sum(p[0] for p in bbox) / 4
                cy = sum(p[1] for p in bbox) / 4
                area_name = classify_token(cx, cy, fw, fh)
                if area_name:
                    if area_name.startswith("hp_"):
                        text = _normalize_hp_text(text)
                        # 有効なHP形式（XX/YY または XX%）のみ通す
                        if not re.match(r'^\d{1,3}/\d{1,3}$', text) and not re.match(r'^\d{1,3}%$', text):
                            continue
                    buckets[area_name].append(text)

            # 何かテキストがあるときだけ出力
            has_any = any(buckets[name] for name in AREAS)
            if has_any:
                mm = int(cur_sec) // 60
                ss = cur_sec - mm * 60
                time_label = f"[{mm:02d}:{ss:05.2f}]"

                # ── OCR結果（コンソール + ログ）──────────────────────────
                _emit(time_label, log_file)
                for name in AREAS:
                    tokens = buckets[name]
                    disp = " ".join(tokens) if tokens else "（なし）"
                    marker = ">>>" if tokens else "   "
                    _emit(f"  {marker} {name:<16}: {disp}", log_file)
                _emit("", log_file)

                # ── 手書き注釈欄（ログファイルのみ）─────────────────────
                if log_file:
                    log_file.write("  ┌─ 注釈 " + "─" * 50 + "\n")
                    log_file.write("  │ [画面種別] 対戦アニメーション / 選出 / コマンド / 他: \n")
                    log_file.write("  │\n")
                    for name in AREAS:
                        tokens = buckets[name]
                        if tokens:
                            disp = " ".join(tokens)
                            log_file.write(f"  │ {name:<16}: {disp}\n")
                    log_file.write("  └" + "─" * 58 + "\n")
                    log_file.write("\n")

            if save_frames:
                vis = draw_areas(frame.copy())
                vis = draw_tokens(vis, ocr_results, fw, fh)
                mm = int(cur_sec) // 60
                ss = cur_sec - mm * 60
                out_path = out_dir / f"{mm:02d}m{ss:05.2f}s.jpg"
                cv2.imwrite(str(out_path), vis)

            processed += 1
            cur_sec += step_sec

        cap.release()
        _emit(f"完了: {processed} 回OCRを実行しました", log_file)
        if save_frames:
            _emit(f"画像保存先: {out_dir}/", log_file)

    finally:
        if log_file:
            log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="各エリアのOCRテキスト検出確認")
    parser.add_argument("video", help="動画ファイルパス")
    parser.add_argument("--step",       type=float, default=1.0,  help="OCR実行間隔（秒、デフォルト: 1.0）")
    parser.add_argument("--start",      type=float, default=0.0,  help="開始時刻（秒）")
    parser.add_argument("--end",        type=float, default=None, help="終了時刻（秒）")
    parser.add_argument("--out",        type=str,   default=None, help="ログ保存先ファイルパス（省略時は logs/ocr_areas_YYYYMMDD_HHMMSS.txt に自動保存）")
    parser.add_argument("--save-frames", action="store_true",     help="エリア可視化画像を debug/ocr_areas/ に保存")
    parser.add_argument("--no-filter",   action="store_true",     help="UIワードフィルターを無効化（デフォルト: 有効）")
    args = parser.parse_args()

    import datetime
    if args.out:
        log_path = Path(args.out)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"logs/ocr_areas_{ts}.txt")

    process_video(
        video_path=args.video,
        step_sec=args.step,
        start_sec=args.start,
        end_sec=args.end,
        save_frames=args.save_frames,
        filter_ui=not args.no_filter,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
