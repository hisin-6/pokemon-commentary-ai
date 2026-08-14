"""実況動画のサムネイル自動生成（改善ロードマップ⑥）。

パス1の素材（manifest.jsonl・states.jsonl）から「盛り上がった瞬間」
（KO・HPの急激な減少）を機械的に選び、その時刻のフレームを元動画から
抜き出してテキストを焼き込んだサムネイル画像を出力する。
あわせて、AIVTuberであることを示すバッジ・アバターの顔・構築（自分/相手の
ポケモンアイコン）を焼き込める（2026-08-04・他AIVTuberサムネ研究を踏まえて刷新）。

候補の優先度: faint（KO） > HP急変（states.jsonlの連続スナップショット間の
大幅なHP減少）。battle_end（勝敗）は既定では候補にしない
（結果そのものを見せてしまいネタバレになるため。旧仕様に戻したい場合は
--allow-result-spoiler を指定）。イベントが無いレンダー（fillers.jsonlのみ等）
では選べないためエラーになる。

使い方:
    python scripts/generate_thumbnail.py renders/16-14-39
    python scripts/generate_thumbnail.py renders/16-14-39 --time 230.5 --label "きめ台詞！"
    python scripts/generate_thumbnail.py renders/16-14-39 --team opponent \\
        --avatar-video "renders/16-14-39/2026-08-04 14-26-47.mp4"

ffmpeg が PATH に必要。Pillow (PIL) が必要（venv に導入済み）。
ポケモンアイコンは初回のみPokeAPIの公式スプライトをネットワーク取得してキャッシュする
（data/pokeapi_cache/icons/）。ネットワーク不可・図鑑DB未整備の場合はアイコン無しで続行する。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# 同ディレクトリのパス2実装からロード関数・折り返しロジック・アバタークロマキー定数を再利用
sys.path.insert(0, str(Path(__file__).parent))
from render_commentary_video import (  # noqa: E402
    _AVATAR_CHROMA,
    _AVATAR_SIMILARITY,
    _wrap_jp,
    load_manifest,
    load_states,
    resolve_video_path,
)

logger = logging.getLogger(__name__)

# イベント種別ごとの「盛り上がり」スコア（KOがHP急変より優先されるよう上限を分離）。
# battle_end は _collect_event_candidates 側で allow_result_spoiler=False の時に除外する。
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

# ── AIVTuber要素（改善ロードマップ⑥続き・2026-08-04）───────────────────────
_BADGE_TEXT = "AI実況"
_BADGE_COLOR = (255, 64, 129, 235)
_CHARACTER_NAME = "花圓くれぴ"
# 2026-08-14: persona="neutral"用（3Dモデル一時差し替え検証用）のキャラ名表記。
# pipeline.py/server.pyのpersona分岐と同じパターン
_CHARACTER_NAME_NEUTRAL = "VOICEVOX：四国めたん"

# ── 構築（チーム編成）アイコン ─────────────────────────────────────────────
# pokedb.sqlite（scripts/build_pokedb.py がPokeAPIから構築した図鑑DB）で
# 和名→PokeAPI図鑑番号を引き、公式スプライトをローカルキャッシュして使う。
_POKEDB_PATH = Path("data/pokedb.sqlite")
_ICON_CACHE_DIR = Path("data/pokeapi_cache/icons")
_SPRITE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{id}.png"
_MAX_ROSTER_ICONS = 6

# ── アバターの顔（v2c録画からクロマキー抜きして使う）────────────────────────
# 65秒(ニュートラル)/71秒台(Fun発火)の実機フレーム比較で顔まわりがちょうど収まると
# 確認済みのクロップ値（2026-08-04・HairSample_Female.vrm実測）。モデルが変わったら
# 要再計測（`--avatar-crop`で上書き可）。
_AVATAR_FACE_CROP_DEFAULT = "400:400:800:250"


def _collect_event_candidates(manifest: list, allow_result_spoiler: bool = False) -> list:
    """manifest.jsonl のKO（・battle_end）イベントを候補として返す。

    allow_result_spoiler=False（既定）では battle_end を候補から除外する
    （試合結果を見せてしまうネタバレ対策・2026-08-04）。
    """
    scores = dict(_EVENT_SCORE)
    if not allow_result_spoiler:
        scores.pop("battle_end", None)
    candidates = []
    for e in manifest:
        score = scores.get(e.get("event_type"))
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
                           hp_swing_threshold: float = _DEFAULT_HP_SWING_THRESHOLD,
                           allow_result_spoiler: bool = False) -> dict:
    """サムネイルに使う「盛り上がった瞬間」を1件選ぶ。

    Returns:
        {"time": 秒, "score": スコア, "reason": "faint"/"hp_swing"/"battle_end", "label": テキスト}
    Raises:
        ValueError: 候補が1件も見つからない場合
    """
    candidates = (_collect_event_candidates(manifest, allow_result_spoiler)
                  + _collect_hp_swing_candidates(states, hp_swing_threshold))
    if not candidates:
        kinds = "KO/battle_end/HP急変" if allow_result_spoiler else "KO/HP急変"
        raise ValueError(f"サムネイル候補（{kinds}）が見つかりませんでした")
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


def build_avatar_face_command(ffmpeg: str, avatar_video: Path, time_sec: float, crop: str,
                              out_png: Path, chroma: str = _AVATAR_CHROMA,
                              similarity: float = _AVATAR_SIMILARITY) -> list:
    """アバター録画（v2c・グリーンバック）から指定時刻の顔まわりをクロップ＋
    クロマキー抜きしたRGBA PNGを書き出すffmpegコマンドを組み立てる。
    render_commentary_video.pyのアバター合成と同じフィルタ（chromakey+despill）を再利用。
    """
    vf = (f"crop={crop},chromakey={chroma}:{similarity}:0.08,"
          f"despill=type=green:mix=0.5:expand=0,format=rgba")
    return [ffmpeg, "-y", "-ss", str(max(0.0, time_sec)), "-i", str(avatar_video),
            "-frames:v", "1", "-vf", vf, str(out_png)]


def extract_avatar_face(avatar_video: Path, time_sec: float, crop: str, out_png: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません（PATHを確認してください）")
    subprocess.run(build_avatar_face_command(ffmpeg, avatar_video, time_sec, crop, out_png),
                   check=True, capture_output=True)


def _collect_roster(states: list, side: str) -> list[str]:
    """states.jsonl から指定side（'player'/'opponent'）に登場したポケモン名を
    初出順・重複なしで返す。

    ⚠️ 対戦画面に実際に映ったポケモンのみが対象（選出画面の6体プレビュー読み取りは
    未実装）。ダブルバトルでは選出された数体（フルの6体ではない）になるのが通常。
    """
    names: list[str] = []
    seen: set[str] = set()
    for state in states:
        for mon in state.get(side, []):
            name = mon.get("name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _resolve_pokemon_id(name_ja: str, pokedb_path: Path = _POKEDB_PATH) -> int | None:
    """pokedb.sqlite（PokeAPI由来の図鑑DB）で和名からPokeAPIの図鑑番号を引く。"""
    if not pokedb_path.exists():
        return None
    conn = sqlite3.connect(pokedb_path)
    try:
        row = conn.execute("SELECT id FROM pokemon WHERE name_ja = ?", (name_ja,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def fetch_pokemon_icon(name_ja: str, cache_dir: Path = _ICON_CACHE_DIR,
                       pokedb_path: Path = _POKEDB_PATH) -> Path | None:
    """ポケモン和名からPokeAPI公式スプライトPNGのローカルキャッシュ済みパスを返す
    （未取得なら初回だけダウンロード）。図鑑DBに無い名前・ネットワーク不可などで
    取得できない場合はNone（呼び出し側はアイコン無しで続行すること）。
    """
    pid = _resolve_pokemon_id(name_ja, pokedb_path)
    if pid is None:
        logger.warning("図鑑DBに無い名前のためアイコンをスキップ: %s", name_ja)
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{pid}.png"
    if out_path.exists():
        return out_path
    try:
        urllib.request.urlretrieve(_SPRITE_URL.format(id=pid), out_path)
    except Exception as exc:
        logger.warning("ポケモンアイコン取得失敗: %s (id=%s): %s", name_ja, pid, exc)
        return None
    return out_path


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


def compose_thumbnail(frame_png: Path, out_png: Path, label: str,
                      avatar_face_png: Path | None = None,
                      roster_icon_pngs: list[Path] | None = None,
                      character_name: str = _CHARACTER_NAME,
                      big_logo_text: str = "AI自動実況") -> None:
    """抜き出したフレームに下部帯＋テキスト（実測幅で折り返し）を焼き込む（PIL）。

    avatar_face_png / roster_icon_pngs を渡すと、右上にアバターの顔・キャプション帯の
    直上に構築アイコン列も焼き込む（AIVTuberサムネ刷新・2026-08-04）。どちらも省略時は
    従来通りゲーム画面＋テキスト帯＋AI実況バッジのみのシンプル版になる。
    character_name: バッジ下に焼き込むキャラ名（既定=花圓くれぴ・persona="neutral"時は
    呼び出し側から_CHARACTER_NAME_NEUTRALを渡すこと）。
    big_logo_text: label=""（盛り上がりシーンの字幕なし）時に代わりに表示する大きな
    ロゴテキスト。幅に収まらなければ_wrap_to_linesで自動2行折り返しする。
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(frame_png).convert("RGB")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_path = _FONT_PATH if Path(_FONT_PATH).exists() else None

    bar_h = int(h * 0.26)
    draw.rectangle([0, h - bar_h, w, h], fill=(10, 12, 20, 195))

    pad_x = int(w * 0.04)
    max_width = w - 2 * pad_x
    # 2行分を帯の高さに収めるフォントサイズ
    font_size = max(1, int(bar_h / _LABEL_MAX_LINES * 0.62))
    font = ImageFont.truetype(font_path, size=font_size) if font_path else ImageFont.load_default()

    text = _truncate_label(label, max_chars=_LABEL_MAX_CHARS)
    lines = _wrap_to_lines(draw, text, font, max_width) if text else []

    if lines:
        line_height = int(font_size * 1.25)
        total_text_h = line_height * len(lines)
        start_y = h - bar_h + max(0, (bar_h - total_text_h) // 2)
        for i, line in enumerate(lines):
            draw.text((pad_x, start_y + i * line_height), line, font=font,
                      fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    else:
        # 2026-08-14: label未指定（盛り上がりシーンの字幕なし）の場合、
        # 実況キャプションの代わりに大きなロゴテキストを表示する
        # （--no-roster-iconsと組み合わせてテキストのみのシンプルなサムネにする用途）
        big_size = int(bar_h * 0.55)
        big_font = ImageFont.truetype(font_path, size=big_size) if font_path else font
        if "\n" in big_logo_text:
            # 明示的な改行が指定されている場合はそれを優先する（自動折り返しだと
            # "AI"のような英字トークンの途中で割れて不自然になることがあるため。
            # 2026-08-14実機で発見: "ポケモンダブルバトルA"/"I自動実況"のように
            # 割れた）
            big_lines = big_logo_text.split("\n")
        else:
            big_lines = _wrap_to_lines(draw, big_logo_text, big_font, max_width, max_lines=2)
        if len(big_lines) > 1:
            # 2行になる場合は帯の高さに収まるようフォントだけ縮小する
            # （行の内容=big_linesは固定。縮小後に再wrapすると1行に収まり直して
            # 折り返し結果が消えてしまうバグを2026-08-14に踏んだため、再wrapしない）
            big_size = int(bar_h * 0.36)
            big_font = ImageFont.truetype(font_path, size=big_size) if font_path else font
            max_line_w = max(draw.textbbox((0, 0), ln, font=big_font)[2] for ln in big_lines)
            while max_line_w > max_width and big_size > 10:
                big_size = int(big_size * 0.9)
                big_font = ImageFont.truetype(font_path, size=big_size) if font_path else font
                max_line_w = max(draw.textbbox((0, 0), ln, font=big_font)[2] for ln in big_lines)

        big_line_height = int(big_size * 1.15)
        total_big_h = big_line_height * len(big_lines)
        start_big_y = h - bar_h + max(0, (bar_h - total_big_h) // 2)
        for i, ln in enumerate(big_lines):
            bbox = draw.textbbox((0, 0), ln, font=big_font)
            draw.text((pad_x, start_big_y + i * big_line_height - bbox[1]), ln,
                      font=big_font, fill=(102, 204, 255, 255),
                      stroke_width=4, stroke_fill=(0, 0, 0, 255))

    # ── AI実況バッジ＋キャラ名（左上）──────────────────────────────────────
    badge_font = ImageFont.truetype(font_path, size=max(1, int(h * 0.035))) if font_path else font
    badge_pad = int(h * 0.015)
    badge_text_w = draw.textlength(_BADGE_TEXT, font=badge_font)
    badge_w = int(badge_text_w + badge_pad * 4)
    badge_h = int(h * 0.06)
    badge_x, badge_y = int(w * 0.03), int(h * 0.03)
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=badge_h // 2, fill=_BADGE_COLOR)
    draw.text((badge_x + badge_w / 2, badge_y + badge_h / 2), _BADGE_TEXT,
              font=badge_font, fill=(255, 255, 255, 255), anchor="mm",
              stroke_width=2, stroke_fill=(0, 0, 0, 200))

    name_font = ImageFont.truetype(font_path, size=max(1, int(h * 0.032))) if font_path else font
    draw.text((badge_x, badge_y + badge_h + int(h * 0.012)), character_name,
              font=name_font, fill=(255, 255, 255, 255),
              stroke_width=3, stroke_fill=(0, 0, 0, 255))

    composed = Image.alpha_composite(img.convert("RGBA"), overlay)

    # ── アバターの顔（右上・クロマキー抜き済みRGBAをそのまま貼る）──────────────
    if avatar_face_png is not None and Path(avatar_face_png).exists():
        face = Image.open(avatar_face_png).convert("RGBA")
        face_w = int(h * 0.34)
        face_h = int(face_w * face.height / face.width)
        face = face.resize((face_w, face_h))
        fx = w - face_w - int(w * 0.025)
        fy = int(h * 0.03)
        composed.alpha_composite(face, (fx, fy))

    # ── 構築アイコン（キャプション帯のすぐ上に横並び）───────────────────────
    if roster_icon_pngs:
        icon_size = int(h * 0.09)
        strip_h = int(icon_size * 1.3)
        strip_y = h - bar_h - strip_h

        strip_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(strip_overlay).rectangle(
            [0, strip_y, w, strip_y + strip_h], fill=(10, 12, 20, 160))
        composed = Image.alpha_composite(composed, strip_overlay)

        gap = int(icon_size * 0.25)
        total_w = len(roster_icon_pngs) * icon_size + (len(roster_icon_pngs) - 1) * gap
        icon_x = max(pad_x, (w - total_w) // 2)
        icon_y = strip_y + (strip_h - icon_size) // 2
        for icon_path in roster_icon_pngs:
            try:
                icon = Image.open(icon_path).convert("RGBA").resize((icon_size, icon_size))
            except Exception:
                continue
            composed.alpha_composite(icon, (icon_x, icon_y))
            icon_x += icon_size + gap

    composed.convert("RGB").save(out_png)


def generate_thumbnail(render_dir: Path, video: Path = None, out: Path = None,
                       time_override: float = None, label_override: str = None,
                       hp_swing_threshold: float = _DEFAULT_HP_SWING_THRESHOLD,
                       allow_result_spoiler: bool = False,
                       team: str | None = "player",
                       avatar_video: Path = None,
                       avatar_time: float = None,
                       avatar_offset: float = 0.0,
                       avatar_crop: str = _AVATAR_FACE_CROP_DEFAULT,
                       pokedb_path: Path = _POKEDB_PATH,
                       icon_cache_dir: Path = _ICON_CACHE_DIR,
                       persona: str = "kurepi",
                       big_logo_text: str = "AI自動実況") -> dict:
    """サムネイル生成の一連の流れ。戻り値は選ばれた瞬間の情報。"""
    manifest = load_manifest(render_dir)
    states = load_states(render_dir)

    if time_override is not None:
        moment = {"time": time_override, "score": 0.0, "reason": "manual",
                  "label": label_override or ""}
    else:
        moment = select_thumbnail_moment(manifest, states, hp_swing_threshold,
                                         allow_result_spoiler=allow_result_spoiler)
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
        tmp_path = Path(tmp)
        frame_png = tmp_path / "frame.png"
        extract_frame(video, moment["time"], frame_png)

        avatar_face_png = None
        if avatar_video is not None and Path(avatar_video).exists():
            # アバター録画はavatar_offset秒だけ先に回り始めている前提
            # （render_commentary_video.pyのアバター合成と同じ規約）
            face_time = avatar_time if avatar_time is not None else moment["time"] + avatar_offset
            candidate_png = tmp_path / "face.png"
            try:
                extract_avatar_face(Path(avatar_video), face_time, avatar_crop, candidate_png)
                avatar_face_png = candidate_png
            except Exception as exc:
                logger.warning("アバター顔の抜き出しに失敗（顔なしで続行）: %s", exc)

        roster_icon_pngs = None
        if team:
            roster_names = _collect_roster(states, team)[:_MAX_ROSTER_ICONS]
            icons = [p for p in (fetch_pokemon_icon(name, icon_cache_dir, pokedb_path)
                                 for name in roster_names) if p is not None]
            roster_icon_pngs = icons or None

        character_name = _CHARACTER_NAME_NEUTRAL if persona == "neutral" else _CHARACTER_NAME
        compose_thumbnail(frame_png, out, moment["label"],
                          avatar_face_png=avatar_face_png,
                          roster_icon_pngs=roster_icon_pngs,
                          character_name=character_name,
                          big_logo_text=big_logo_text)

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
    parser.add_argument("--allow-result-spoiler", action="store_true",
                        help="battle_end（試合結果）も候補に含める（既定は結果ネタバレ防止のため除外）")
    parser.add_argument("--team", choices=["player", "opponent"], default="player",
                        help="構築アイコンに使う陣営（既定: player=自分）")
    parser.add_argument("--no-roster-icons", action="store_true", help="構築アイコンを焼き込まない")
    parser.add_argument("--avatar-video", help="v2cアバター録画のパス（指定時は右上に顔を焼き込む）")
    parser.add_argument("--avatar-time", type=float,
                        help="アバター顔の抜き出し時刻・秒（省略時はサムネ選択時刻+avatar-offset）")
    parser.add_argument("--avatar-offset", type=float, default=0.0,
                        help="アバター録画側の頭出しオフセット・秒（render_commentary_video.pyと同じ規約）")
    parser.add_argument("--avatar-crop", default=_AVATAR_FACE_CROP_DEFAULT,
                        help=f"アバター顔のクロップ w:h:x:y（既定{_AVATAR_FACE_CROP_DEFAULT}）")
    parser.add_argument("--persona", choices=["kurepi", "neutral"], default="kurepi",
                        help="キャラクター設定（既定kurepi・neutralはバッジ下の名前表記を"
                             "「VOICEVOX：四国めたん」に切り替える。パス1と同じ値を指定すること）")
    parser.add_argument("--big-logo-text", default="AI自動実況",
                        help="--label \"\"（字幕なし）時に表示する大きなロゴテキスト"
                             "（既定\"AI自動実況\"・幅に収まらなければ自動2行折り返し）")
    args = parser.parse_args(argv)

    video = resolve_video_path(args.video) if args.video else None
    result = generate_thumbnail(
        args.render_dir, video=video, out=args.out,
        time_override=args.time, label_override=args.label,
        hp_swing_threshold=args.hp_swing_threshold,
        allow_result_spoiler=args.allow_result_spoiler,
        team=None if args.no_roster_icons else args.team,
        avatar_video=Path(args.avatar_video) if args.avatar_video else None,
        avatar_time=args.avatar_time,
        avatar_offset=args.avatar_offset,
        avatar_crop=args.avatar_crop,
        big_logo_text=args.big_logo_text,
        persona=args.persona)

    logger.info("サムネイル生成完了: %s", result["out"])
    logger.info("  選択根拠: %s（t=%.1fs, score=%.1f）", result["reason"], result["time"], result["score"])
    logger.info("  焼き込みテキスト: %s", result["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
