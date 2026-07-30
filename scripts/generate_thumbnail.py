"""実況動画のサムネイル自動生成（改善ロードマップ⑥）。

パス1の素材（manifest.jsonl・states.jsonl）から「盛り上がった瞬間」
（KO/battle_end・HPの急激な減少）を機械的に選び、その時刻のフレームを
元動画から抜き出してテキストを焼き込んだサムネイル画像を出力する。

候補の優先度: battle_end（試合結果） > faint（KO） > HP急変（states.jsonlの
連続スナップショット間の大幅なHP減少）。イベントが無いレンダー（fillers.jsonl
のみ等）では選べないためエラーになる。

使い方:
    python scripts/generate_thumbnail.py renders/16-14-39
    python scripts/generate_thumbnail.py renders/16-14-39 --time 230.5 --label "きめ台詞！"

ffmpeg が PATH に必要。Pillow (PIL) が必要（venv に導入済み）。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 同ディレクトリのパス2実装からロード関数・折り返しロジックを再利用
sys.path.insert(0, str(Path(__file__).parent))
from render_commentary_video import (  # noqa: E402
    _wrap_jp,
    load_manifest,
    load_states,
    resolve_video_path,
)

logger = logging.getLogger(__name__)

# イベント種別ごとの「盛り上がり」スコア（KOがHP急変より優先されるよう上限を分離）
_EVENT_SCORE = {"battle_end": 100.0, "faint": 80.0}
# HP急変とみなす最低減少幅（パーセントポイント）
_DEFAULT_HP_SWING_THRESHOLD = 30.0
# HP急変候補の最大スコア（KOイベントの最低スコアより低く抑える）
_HP_SWING_SCORE_CAP = 79.0

# サムネイル焼き込みフォント（biim風レイアウトと同じmeiryo太字）
_FONT_PATH = "/mnt/c/Windows/Fonts/meiryob.ttc"
# 折り返し前の粗い上限（この文字数を超える分は最終的に切り捨てて「…」を付ける）
_LABEL_MAX_CHARS = 60
# 下部帯に焼き込む最大行数
_LABEL_MAX_LINES = 2


def _collect_event_candidates(manifest: list) -> list:
    """manifest.jsonl のKO/battle_endイベントを候補として返す。"""
    candidates = []
    for e in manifest:
        score = _EVENT_SCORE.get(e.get("event_type"))
        if score is None:
            continue
        candidates.append({
            "time": float(e["event_time"]),
            "score": score,
            "reason": e["event_type"],
            "label": e.get("commentary", ""),
        })
    return candidates


def _collect_hp_swing_candidates(states: list, threshold: float = _DEFAULT_HP_SWING_THRESHOLD) -> list:
    """states.jsonl の連続スナップショット間で大きなHP減少を検出する。

    同じ陣営・同じ名前のポケモンが前回スナップショットより threshold pt 以上
    減った時刻を候補として返す。スロット入れ替え（交代）によるHP変化の
    誤検出を避けるため、名前一致のみで前回値と比較する（厳密な同一個体
    追跡ではないが「機械的な候補選び」としては十分な精度）。
    """
    candidates = []
    prev_hp: dict = {}  # (side, name) -> hp_pct
    for state in states:
        t = float(state["time"])
        for side_key in ("player", "opponent"):
            for mon in state.get(side_key, []):
                name = mon.get("name")
                pct = mon.get("hp_pct")
                if not name or pct is None:
                    continue
                key = (side_key, name)
                prev = prev_hp.get(key)
                if prev is not None:
                    drop = prev - pct
                    if drop >= threshold:
                        side_label = "自分" if side_key == "player" else "相手"
                        candidates.append({
                            "time": t,
                            "score": min(_HP_SWING_SCORE_CAP, drop),
                            "reason": "hp_swing",
                            "label": f"{side_label}の{name} 残りHP{pct}%！",
                        })
                prev_hp[key] = pct
    return candidates


def select_thumbnail_moment(manifest: list, states: list,
                           hp_swing_threshold: float = _DEFAULT_HP_SWING_THRESHOLD) -> dict:
    """サムネイルに使う「盛り上がった瞬間」を1件選ぶ。

    Returns:
        {"time": 秒, "score": スコア, "reason": "battle_end"/"faint"/"hp_swing", "label": テキスト}
    Raises:
        ValueError: 候補が1件も見つからない場合
    """
    candidates = (_collect_event_candidates(manifest)
                  + _collect_hp_swing_candidates(states, hp_swing_threshold))
    if not candidates:
        raise ValueError("サムネイル候補（KO/battle_end/HP急変）が見つかりませんでした")
    return max(candidates, key=lambda c: c["score"])


def _truncate_label(text: str, max_chars: int = _LABEL_MAX_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


def build_extract_frame_command(ffmpeg: str, video: Path, time_sec: float, out_png: Path) -> list:
    """指定時刻のフレームを1枚抜き出すffmpegコマンドを組み立てる（入力側-ssの高速シーク）。"""
    return [ffmpeg, "-y", "-ss", str(max(0.0, time_sec)), "-i", str(video),
            "-frames:v", "1", "-q:v", "2", str(out_png)]


def extract_frame(video: Path, time_sec: float, out_png: Path) -> None:
    """元動画から指定時刻のフレームを1枚抜き出す。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません（PATHを確認してください）")
    subprocess.run(build_extract_frame_command(ffmpeg, video, time_sec, out_png),
                   check=True, capture_output=True)


def _wrap_to_lines(draw, text: str, font, max_width: float, max_lines: int = _LABEL_MAX_LINES) -> list:
    """テキストを実測フォント幅から算出した折り返し文字数で複数行にする。

    ``_wrap_jp``（render_commentary_video.py・ASS字幕の折り返しと同じロジック）を
    再利用し、max_lines を超える分は末尾行を「…」付きで切り詰める。
    """
    sample_width = draw.textlength("あ", font=font) or 1.0
    chars_per_line = max(4, int(max_width / sample_width))
    wrapped_lines = _wrap_jp(text, width=chars_per_line).split("\\N")
    if len(wrapped_lines) <= max_lines:
        return wrapped_lines
    lines = wrapped_lines[:max_lines]
    lines[-1] = _truncate_label(lines[-1], max_chars=max(1, chars_per_line - 1))
    return lines


def compose_thumbnail(frame_png: Path, out_png: Path, label: str) -> None:
    """抜き出したフレームに下部帯＋テキスト（実測幅で折り返し）を焼き込む（PIL）。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(frame_png).convert("RGB")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = int(h * 0.26)
    draw.rectangle([0, h - bar_h, w, h], fill=(10, 12, 20, 195))

    font_path = _FONT_PATH if Path(_FONT_PATH).exists() else None
    pad_x = int(w * 0.04)
    max_width = w - 2 * pad_x
    # 2行分を帯の高さに収めるフォントサイズ
    font_size = max(1, int(bar_h / _LABEL_MAX_LINES * 0.62))
    font = ImageFont.truetype(font_path, size=font_size) if font_path else ImageFont.load_default()

    text = _truncate_label(label, max_chars=_LABEL_MAX_CHARS)
    lines = _wrap_to_lines(draw, text, font, max_width) if text else []

    line_height = int(font_size * 1.25)
    total_text_h = line_height * len(lines)
    start_y = h - bar_h + max(0, (bar_h - total_text_h) // 2)
    for i, line in enumerate(lines):
        draw.text((pad_x, start_y + i * line_height), line, font=font,
                  fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))

    composed = Image.alpha_composite(img.convert("RGBA"), overlay)
    composed.convert("RGB").save(out_png)


def generate_thumbnail(render_dir: Path, video: Path = None, out: Path = None,
                       time_override: float = None, label_override: str = None,
                       hp_swing_threshold: float = _DEFAULT_HP_SWING_THRESHOLD) -> dict:
    """サムネイル生成の一連の流れ。戻り値は選ばれた瞬間の情報。"""
    manifest = load_manifest(render_dir)
    states = load_states(render_dir)

    if time_override is not None:
        moment = {"time": time_override, "score": 0.0, "reason": "manual",
                  "label": label_override or ""}
    else:
        moment = select_thumbnail_moment(manifest, states, hp_swing_threshold)
        if label_override is not None:
            moment["label"] = label_override

    if video is None:
        with (render_dir / "render_info.json").open(encoding="utf-8") as fp:
            info = json.load(fp)
        video = resolve_video_path(info["video"])
    if not video.exists():
        raise FileNotFoundError(f"元動画が見つかりません: {video}")

    out = out or (render_dir / "thumbnail.png")

    with tempfile.TemporaryDirectory() as tmp:
        frame_png = Path(tmp) / "frame.png"
        extract_frame(video, moment["time"], frame_png)
        compose_thumbnail(frame_png, out, moment["label"])

    return {"out": str(out), **moment}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="実況動画のサムネイル自動生成")
    parser.add_argument("render_dir", type=Path, help="renders/<動画名> ディレクトリ")
    parser.add_argument("--video", help="元動画パス（省略時はrender_info.jsonから）")
    parser.add_argument("--out", type=Path, help="出力先PNG（既定: <render_dir>/thumbnail.png）")
    parser.add_argument("--time", type=float, help="自動選択せずこの秒数のフレームを使う")
    parser.add_argument("--label", help="焼き込みテキストを上書きする")
    parser.add_argument("--hp-swing-threshold", type=float, default=_DEFAULT_HP_SWING_THRESHOLD,
                        help=f"HP急変とみなす最低減少幅・pt（既定{_DEFAULT_HP_SWING_THRESHOLD}）")
    args = parser.parse_args(argv)

    video = resolve_video_path(args.video) if args.video else None
    result = generate_thumbnail(
        args.render_dir, video=video, out=args.out,
        time_override=args.time, label_override=args.label,
        hp_swing_threshold=args.hp_swing_threshold)

    logger.info("サムネイル生成完了: %s", result["out"])
    logger.info("  選択根拠: %s（t=%.1fs, score=%.1f）", result["reason"], result["time"], result["score"])
    logger.info("  焼き込みテキスト: %s", result["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
