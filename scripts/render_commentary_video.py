"""実況動画の合成（ADR-009 パス2）。

パス1（``src/pipeline.py --render-out``）が出力した素材ディレクトリ
（manifest.jsonl + wav/）を読み、元動画に実況音声トラックを合成した
mp4 を出力する。合成のやり直しに Bedrock / VOICEVOX の再実行は不要。

処理の流れ:
  1. manifest.jsonl を読み event_time 順にソート
     （faint統合フラッシュ等で保存順と時刻順は一致しない）
  2. 近接時間内の同一実況文をデデュープ
     （faint と switch が同文を生成するケースの二重再生防止）
  3. 後ろ倒しスケジューリング: かぶったら前の実況の終了まで遅らせて
     全部言い切る。遅れが --max-delay を超えた実況のみ破棄
  4. 無音ベースの実況トラックWAVを生成（スケジュール時刻に配置）
  5. ffmpeg でゲーム音声にサイドチェインダッキングをかけつつミックス
     （映像ストリームは再エンコードなしのコピー）

使い方:
    python scripts/render_commentary_video.py renders/16-14-39
    python scripts/render_commentary_video.py renders/16-14-39 --dry-run

ffmpeg が PATH に必要（--dry-run はスケジュール確認のみで ffmpeg 不要）。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# デデュープ: この秒数以内に完全同一の実況文があれば後の方を破棄
_DEDUPE_WINDOW_SEC = 90.0
# スケジューリング: 実況と実況の間に最低限空ける間隔（秒）
_DEFAULT_GAP_SEC = 0.5
# スケジューリング: イベント時刻からこれ以上遅れる実況は破棄（秒）
_DEFAULT_MAX_DELAY_SEC = 20.0
# 実況音声の音量倍率（ゲーム音に埋もれないよう少し持ち上げる）
_DEFAULT_GAIN = 1.4
# フィラー配置: 希望時刻からこれ以上ずらさないと置けない場合は破棄（秒）
_FILLER_MAX_SHIFT_SEC = 12.0
# ダッキング（sidechaincompress）の既定値
_DEFAULT_DUCK_THRESHOLD = 0.03
_DEFAULT_DUCK_RATIO = 8.0


def load_manifest(render_dir: Path) -> list:
    """manifest.jsonl を読み込んで event_time 昇順のリストを返す。"""
    manifest_path = render_dir / "manifest.jsonl"
    entries = []
    with manifest_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: e["event_time"])
    return entries


def dedupe_entries(entries: list, window: float = _DEDUPE_WINDOW_SEC) -> tuple:
    """近接時間内の完全同一実況文を除去する。

    faint保留の早期フラッシュと直後のswitchでBedrockが同一文を返すことが
    あり（実測: 16-14-39 seq2/seq3）、そのまま合成すると同じセリフが
    2回再生される。event_time昇順で走査し、直近 ``window`` 秒以内に
    同一文があれば後の方を破棄する。

    Returns:
        (残ったエントリ, 破棄したエントリ)
    """
    kept = []
    dropped = []
    last_seen = {}  # commentary -> event_time
    for e in entries:
        text = e["commentary"]
        prev = last_seen.get(text)
        if prev is not None and e["event_time"] - prev <= window:
            dropped.append(e)
            continue
        last_seen[text] = e["event_time"]
        kept.append(e)
    return kept, dropped


def schedule_entries(entries: list, gap: float = _DEFAULT_GAP_SEC,
                     max_delay: float = _DEFAULT_MAX_DELAY_SEC) -> tuple:
    """後ろ倒しスケジューリング。

    各実況の開始時刻はイベント時刻以降で、前の実況の終了+gap より後。
    遅延が max_delay を超える実況は破棄する（ライブの割り込み停止と逆に
    「全部言い切る」方針・ADR-009）。

    Returns:
        (start/delay 付きエントリのリスト, 破棄したエントリのリスト)
    """
    scheduled = []
    dropped = []
    cursor = 0.0
    for e in sorted(entries, key=lambda e: e["event_time"]):
        start = max(float(e["event_time"]), cursor)
        delay = start - float(e["event_time"])
        if delay > max_delay:
            dropped.append(dict(e, delay=round(delay, 3)))
            continue
        item = dict(e, start=round(start, 3), delay=round(delay, 3))
        scheduled.append(item)
        cursor = start + float(e["duration"]) + gap
    return scheduled, dropped


def load_fillers(render_dir: Path) -> list:
    """fillers.jsonl（台本パスの出力・任意）を読み込む。無ければ空リスト。"""
    fillers_path = render_dir / "fillers.jsonl"
    if not fillers_path.exists():
        return []
    entries = []
    with fillers_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: e["event_time"])
    return entries


def fit_fillers(scheduled: list, fillers: list, gap: float = _DEFAULT_GAP_SEC,
                max_shift: float = _FILLER_MAX_SHIFT_SEC) -> tuple:
    """イベント実況を動かさずに、フィラーを空き区間へ収まる分だけ配置する。

    イベント実況（scheduled）が優先で、フィラーがそれを押しのけることは
    ない。希望時刻に置けない場合は後ろへずらして探すが、``max_shift`` 秒を
    超えるずれが必要なフィラーは破棄する（フィラーはその時刻の文脈で
    生成されているため、大きくずれた位置で読ませない）。

    Returns:
        (start付きフィラーのリスト, 破棄したフィラーのリスト)
    """
    occupied = sorted((e["start"], e["start"] + float(e["duration"]))
                      for e in scheduled)
    placed = []
    dropped = []
    for f in sorted(fillers, key=lambda f: f["event_time"]):
        t = float(f["event_time"])
        dur = float(f["duration"])
        start = t
        for a, b in occupied:
            if start + dur + gap <= a:
                break  # この占有区間より手前に収まる
            if start < b + gap:
                start = b + gap  # 占有区間を跨げないので直後へ
        if start - t > max_shift:
            dropped.append(dict(f, delay=round(start - t, 3)))
            continue
        item = dict(f, start=round(start, 3), delay=round(start - t, 3))
        placed.append(item)
        occupied.append((start, start + dur))
        occupied.sort()
    return placed, dropped


def resolve_video_path(raw: str) -> Path:
    """render_info.json の動画パスを実行環境のパスに解決する。

    パス1はWindows側で実行されるため ``D:\\...`` 形式で記録されている。
    WSLから合成する場合は ``/mnt/d/...`` に変換して探す。
    """
    p = Path(raw)
    if p.exists():
        return p
    if sys.platform != "win32" and len(raw) >= 3 and raw[1] == ":" and raw[2] in "\\/":
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        candidate = Path(f"/mnt/{drive}/{rest}")
        if candidate.exists():
            return candidate
    return p  # 見つからなくても呼び出し側でエラーにするため元パスを返す


def build_commentary_track(render_dir: Path, scheduled: list,
                           out_wav: Path, min_duration: float = 0.0) -> float:
    """スケジュール時刻に各WAVを配置した実況トラックWAVを生成する。

    全WAVのフォーマット（レート・チャンネル・ビット幅）は同一
    （同一speakerのVOICEVOX出力）であることを検証する。

    Args:
        min_duration: トラックの最低長（秒）。元動画と同尺まで無音で
            パディングするために使う（amixの入力長を揃えると音量の
            renormalizeが起きず、volume補償が全編で一定になる）

    Returns:
        トラック長（秒）
    """
    if not scheduled:
        raise ValueError("スケジュール済みの実況が0件です")

    clips = []
    params = None
    for item in scheduled:
        wav_path = render_dir / item["wav"]
        with wave.open(str(wav_path), "rb") as w:
            p = (w.getframerate(), w.getnchannels(), w.getsampwidth())
            if params is None:
                params = p
            elif p != params:
                raise ValueError(
                    f"WAVフォーマット不一致: {wav_path.name} {p} != {params}")
            clips.append((item["start"], w.readframes(w.getnframes())))

    rate, channels, sampwidth = params
    bytes_per_frame = channels * sampwidth
    last_end_frame = int(round(min_duration * rate))
    placed = []
    for start, data in clips:
        offset = int(round(start * rate))
        placed.append((offset, data))
        last_end_frame = max(last_end_frame, offset + len(data) // bytes_per_frame)

    buf = bytearray(last_end_frame * bytes_per_frame)
    for offset, data in placed:
        pos = offset * bytes_per_frame
        buf[pos:pos + len(data)] = data

    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(bytes(buf))
    return last_end_frame / rate


def _ass_time(sec: float) -> str:
    """秒を ASS の時刻表記（h:mm:ss.cc）に変換する。"""
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _ass_escape(text: str) -> str:
    """ASSのオーバーライドタグ・改行制御と衝突する文字を無害化する。"""
    return text.replace("{", "｛").replace("}", "｝").replace("\n", "\\N")


# 字幕フォントサイズ（スマホ視聴を考慮して帯いっぱいに大きく・太字）
_SUBTITLE_FONT_SIZE = 48
# 字幕の1行あたり最大文字数。ASS/libassの自動折り返しはスペース基準のため
# 日本語（スペースなし）では効かず、帯からはみ出す（実機フレームで確認済み）。
# 手動で\Nを挿入する。実況帯はフル幅（使用可能幅1824px ÷ フォント48px ≒ 38字）
# （最長102字＝100字フィラー+鉤括弧で3行・帯高さ230pxに収まる）
_SUBTITLE_WRAP_CHARS = 37
# 折り返し位置の直前にあれば優先して切る句読点
_WRAP_PUNCT = "。、！？!?"
# 行頭に置かない文字（禁則・前の行にぶら下げる）
_WRAP_NO_HEAD = "」。、！？!?）)"


def _wrap_jp(text: str, width: int = _SUBTITLE_WRAP_CHARS) -> str:
    """日本語テキストをwidth文字ごとに\\Nで折り返す。

    句読点があればそこで切り（読みやすさ優先）、行頭に来てはいけない
    文字（閉じ括弧・句読点）は前の行にぶら下げる（禁則処理）。
    """
    lines = []
    rest = text
    while len(rest) > width:
        cut = width
        # 20字目〜width字目の間の最後の句読点の直後で切る
        for i in range(width, max(19, width - 16), -1):
            if rest[i - 1] in _WRAP_PUNCT:
                cut = i
                break
        # 次行の行頭が禁則文字ならぶら下げる（幅は+2字まで超過を許容）
        while cut < len(rest) and rest[cut] in _WRAP_NO_HEAD and cut < width + 2:
            cut += 1
        lines.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        lines.append(rest)
    return "\\N".join(lines)


# 字幕の余韻: 音声終了後もこの秒数だけ表示を残す（次の実況開始でカット）
_SUBTITLE_LINGER_SEC = 1.5

# biim風レイアウトの配置（1920x1080・mockup_A_biim.png 準拠）
_BIIM_GAME_W, _BIIM_GAME_H = 1440, 810   # ゲーム画面の縮小サイズ
_BIIM_GAME_X, _BIIM_GAME_Y = 16, 12      # ゲーム画面の左上位置
_BIIM_BG_COLOR = "0x121627"              # 背景（ダークネイビー）
_BIIM_PANEL_COLOR = "0x1B2135"           # 右サイドパネルの下地
_BIIM_FONTS_DIR = "/mnt/c/Windows/Fonts"  # meiryo.ttc の場所（WSL）
_BIIM_FONT_FILE = f"{_BIIM_FONTS_DIR}/meiryob.ttc"


# ── v2b: 戦況パネル（右サイド・ASSベクター描画で時刻同期表示）──────────────
_PANEL_TEXT_X = 1496        # パネル内側の左端
_PANEL_RIGHT_X = 1880       # 右寄せテキストの基準X
_PANEL_BAR_W = 300          # HPバーの幅
_PANEL_BAR_H = 14           # HPバーの高さ
_PANEL_MOVES_MAX = 4        # ポケモンの技スロット数（判明分を埋める・未判明は?）
# 瞬間ログ（timeline.jsonl）の技エントリ形式: "T3:ガブリアスのドラゴンクロー"
# 名前は最初の「の」まで（技名先頭が「の」の技=のしかかり等も正しく分離できる）
_MOMENT_MOVE_RE = re.compile(r"^T[\d?]+:(.+?)の(.+)$")


def load_states(render_dir: Path) -> list:
    """states.jsonl（戦況パネル用スナップショット・任意）を読み込む。"""
    states_path = render_dir / "states.jsonl"
    if not states_path.exists():
        return []
    states = []
    with states_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                states.append(json.loads(line))
    states.sort(key=lambda s: s["time"])
    return states


def load_timeline(render_dir: Path) -> list:
    """timeline.jsonl（技検出の瞬間ログ・任意）を読み込む。無ければ空リスト。"""
    timeline_path = render_dir / "timeline.jsonl"
    if not timeline_path.exists():
        return []
    moments = []
    with timeline_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                moments.append(json.loads(line))
    moments.sort(key=lambda m: m["time"])
    return moments


def _hp_bar_color(pct: int) -> str:
    """HP%に応じたバー色（ASSのBGR表記）。ゲームのHPバー配色に準拠。"""
    if pct > 50:
        return "&H78C850&"   # 緑
    if pct > 20:
        return "&H28B4E6&"   # 黄
    return "&H3C46E6&"       # 赤


def _panel_bar(x: int, y: int, width: int, color: str) -> str:
    """塗りつぶし矩形のASS描画テキスト（HPバー用）。"""
    return (f"{{\\pos({x},{y})\\an7\\bord0\\shad0\\1c{color}\\p1}}"
            f"m 0 0 l {width} 0 l {width} {_PANEL_BAR_H} l 0 {_PANEL_BAR_H}{{\\p0}}")


def _moves_by_pokemon(moments: list, until: float) -> dict:
    """瞬間ログから時刻 until までに判明した技をポケモン別に集計する。

    技が画面に映った瞬間に「?」が埋まる時刻同期表示のためのデータ。
    値は (技名, 陣営) のリスト。陣営は瞬間ログの side フィールド
    （"自分"/"相手"・2026-07-30〜）で、無い旧データは None。
    """
    moves: dict = {}
    for m in moments:
        if float(m["time"]) > until:
            break  # momentsは時刻昇順
        match = _MOMENT_MOVE_RE.match(m.get("text", ""))
        if not match:
            continue
        name, mv = match.groups()
        lst = moves.setdefault(name, [])
        if all(mv != known for known, _ in lst):
            lst.append((mv, m.get("side")))
    return moves


def _moves_for_side(moves_map: dict, name: str, side: str) -> list:
    """パネルの陣営ブロック用に技リストを絞り込む。

    陣営タグ付きの技は一致する側にのみ表示（同名ミラーの技混ざり対策）。
    タグ無し（旧timeline・判定不能）は従来どおり両側に表示する。
    """
    return [mv for mv, s in moves_map.get(name, [])
            if s is None or s == side][:_PANEL_MOVES_MAX]


def _moves_lines(known: list) -> tuple:
    """技表示の2行（各2枠・未判明は?）を返す。例: 技:インファイト/ねこだまし"""
    slots = list(known[:_PANEL_MOVES_MAX])
    slots += ["?"] * (_PANEL_MOVES_MAX - len(slots))
    return f"技:{slots[0]}/{slots[1]}", f"　　{slots[2]}/{slots[3]}"


def _panel_dialogues(start: float, end: float, state: dict | None,
                     moves_map: dict) -> list:
    """1キーフレーム区間ぶんの戦況パネルDialogue行を生成する。"""
    t0, t1 = _ass_time(start), _ass_time(end)
    lines = []

    def dlg(layer: int, text: str) -> None:
        lines.append(f"Dialogue: {layer},{t0},{t1},Panel,,0,0,0,,{text}")

    def side_block(label: str, label_color: str, entries: list, y_label: int,
                   side: str) -> None:
        dlg(1, f"{{\\pos({_PANEL_TEXT_X},{y_label})\\fs26\\1c{label_color}}}{label}")
        if not entries:
            dlg(1, f"{{\\pos({_PANEL_TEXT_X},{y_label + 36})\\fs24\\1c&H8899AA&}}情報収集中")
            return
        for i, p in enumerate(entries[:2]):
            y = y_label + 36 + i * 116
            name = _ass_escape(p["name"])
            if p.get("status"):
                name += f"({_ass_escape(p['status'])})"
            dlg(1, f"{{\\pos({_PANEL_TEXT_X},{y})\\fs30\\1c&HFFFFFF&}}{name}")
            if p.get("hp_text"):
                dlg(1, f"{{\\pos({_PANEL_RIGHT_X},{y + 6})\\an3\\fs26\\1c&HFFFFFF&}}"
                       f"{_ass_escape(p['hp_text'])}")
            bar_y = y + 40
            dlg(1, _panel_bar(_PANEL_TEXT_X, bar_y, _PANEL_BAR_W, "&H262626&"))
            pct = p.get("hp_pct")
            if pct is not None:
                pct = max(0, min(100, int(pct)))
                fill = max(1, round(_PANEL_BAR_W * pct / 100))
                dlg(2, _panel_bar(_PANEL_TEXT_X, bar_y, fill, _hp_bar_color(pct)))
            line1, line2 = _moves_lines(_moves_for_side(moves_map, p["name"], side))
            dlg(1, f"{{\\pos({_PANEL_TEXT_X},{y + 62})\\fs22\\1c&HC8D8D8&}}{_ass_escape(line1)}")
            dlg(1, f"{{\\pos({_PANEL_TEXT_X},{y + 88})\\fs22\\1c&HC8D8D8&}}{_ass_escape(line2)}")

    if state:
        side_block("相手の場", "&HB478FF&", state.get("opponent", []), 88, side="相手")
        side_block("自分の場", "&HFFC878&", state.get("player", []), 372, side="自分")
        dlg(1, f"{{\\pos({_PANEL_TEXT_X},660)\\fs26\\1c&HFFFFFF&}}ターン {state.get('turn', '?')}")
    return lines


def build_panel_events(states: list, moments: list, video_end: float) -> list:
    """戦況スナップショットと瞬間ログから、時刻同期パネルのDialogue行を生成する。

    states / moments の時刻をキーフレームとして、各区間のパネル内容を
    まとめて描画する。技表示（技:xx/?/?/?）は瞬間ログから導出し、
    技が画面に映った時刻で ? が埋まる。
    """
    if not states:
        return []
    times = sorted({round(float(s["time"]), 3) for s in states} |
                   {round(float(m["time"]), 3) for m in moments})
    dialogues = []
    for i, t in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else max(video_end, t + 1.0)
        state = None
        for s in states:
            if s["time"] <= t:
                state = s
            else:
                break
        dialogues += _panel_dialogues(t, end, state, _moves_by_pokemon(moments, t))
    return dialogues


def build_ass(scheduled: list, out_path: Path,
              panel_events: list = None) -> None:
    """スケジュール済み実況から音声シンクロの字幕（ASS）を生成する。

    下部の実況帯（ゲーム画面の下・右パネルを避けた領域）に表示する。
    イベント実況とフィラーはスタイルを分ける（フィラーは淡色）。
    """
    # 帯の上端+14pxから表示（帯の位置はbuild_ffmpeg_command_biim側の計算と対）
    fs = _SUBTITLE_FONT_SIZE
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Event,Meiryo,{fs},&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,7,48,48,848,1
Style: Filler,Meiryo,{fs},&H00E8D8B0,&H00FFFFFF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,7,48,48,848,1
Style: Panel,Meiryo,26,&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    ordered = sorted(scheduled, key=lambda e: e["start"])
    for i, e in enumerate(ordered):
        start = float(e["start"])
        end = start + float(e["duration"]) + _SUBTITLE_LINGER_SEC
        if i + 1 < len(ordered):
            end = min(end, float(ordered[i + 1]["start"]) - 0.1)
        style = "Filler" if e["event_type"] == "filler" else "Event"
        text = _wrap_jp(_ass_escape(f"「{e['commentary']}」"))
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{text}\n")
    for d in (panel_events or []):
        lines.append(d + "\n")
    out_path.write_text("".join(lines), encoding="utf-8")


# ── v2c: アバター合成（VMC口パク録画のクロマキーワイプ・ADR-009で方式A採用）──
_AVATAR_WIDTH = 344          # アバターの表示幅（右下・縦横比は素材のまま）
_AVATAR_MARGIN = 16          # 画面右端・下端からの余白
_AVATAR_CHROMA = "0x00FF00"  # クロマキー色（グリーンバック）
_AVATAR_SIMILARITY = 0.25    # クロマキーの類似度しきい値（despillとセットで緑フリンジ対策）


def build_ffmpeg_command_biim(video: Path, track_wav: Path, out_path: Path,
                              ass_path: Path, gain: float, duck_threshold: float,
                              duck_ratio: float, avatar_video: Path = None,
                              avatar_offset: float = 0.0,
                              avatar_width: int = _AVATAR_WIDTH,
                              avatar_chroma: str = _AVATAR_CHROMA,
                              avatar_similarity: float = _AVATAR_SIMILARITY,
                              avatar_crop: str = None,
                              tail_pad: float = 0.0,
                              max_duration: float = 0.0) -> list:
    """biim風レイアウト（案A）合成のffmpegコマンドを組み立てる。

    ゲーム画面を左上に縮小配置し、右サイドパネルの下地・下部実況帯を描画、
    ASS字幕（音声シンクロの実況テキスト）を焼き込む。映像は再エンコード
    （レイアウト合成のため -c:v copy 不可）。音声チェインはplainと同一。

    avatar_video 指定時（v2c・方式A）はVMC口パク録画をクロマキーで抜いて
    右下に重ねる。avatar_offset は「録画開始→WAV再生開始」の秒数（頭合わせ・
    録画側の先頭をスキップする）。アバターが動画より短い場合は最終フレームで
    静止する（eof_action=repeat）。avatar_crop（"w:h:x:y"）指定時はスケール前に
    クロップし、全身録画から上半身だけを抜き出して拡大表示する。

    tail_pad > 0 のとき、元動画の末尾を最終フレームの静止で tail_pad 秒延長する
    （tpad=stop_mode=clone）。末尾の実況が動画終端をまたぐ場合に映像なしで音声
    だけ流れるのを防ぐ。tpadはフィルタチェーン先頭（字幕・パネル描画の前段）に
    置くため、延長区間でも字幕は時刻通りに表示・消滅する。

    max_duration > 0 のとき、出力を max_duration 秒で打ち切る（-t）。アバター録画が
    本編（動画+tail_pad）より長い場合、overlay のデフォルト挙動（eof_action=repeat と
    対称に、先に終わった側の最終フレームで静止して長い方に合わせる）により本編側の
    最終フレームが静止したまま余ったアバター秒数だけ出力が伸びてしまう
    （実機07-03-23-34-29で確認: 本編309.9秒・アバター396.8秒の組み合わせで
    出力が396.8秒に間延びし、末尾87秒が試合とは無関係な静止画+アバターだけの
    無駄な尻尾になっていた）。本編の長さに揃えて余剰分を切り捨てるための安全弁。
    """
    if avatar_video is not None and avatar_offset < 0:
        raise ValueError("avatar_offset は0以上（録画をWAV再生より先に開始する運用）")
    band_y = _BIIM_GAME_Y + _BIIM_GAME_H + 12          # 実況帯の上端（字幕3行が入る高さを確保）
    band_h = 1080 - band_y - 16                        # 実況帯の高さ（=230px）
    band_w = 1920 - _BIIM_GAME_X * 2                   # 実況帯はフル幅（右パネルの下も使う）
    panel_x = _BIIM_GAME_X + _BIIM_GAME_W + 16         # 右パネルの左端
    panel_w = 1920 - panel_x - 16
    tpad_part = (f"tpad=stop_mode=clone:stop_duration={tail_pad:.3f},"
                 if tail_pad > 0 else "")
    video_filter = (
        # ゲーム画面を縮小し、パディングで1920x1080のキャンバスに配置
        f"[0:v]{tpad_part}scale={_BIIM_GAME_W}:{_BIIM_GAME_H},"
        f"pad=1920:1080:{_BIIM_GAME_X}:{_BIIM_GAME_Y}:{_BIIM_BG_COLOR},"
        # ゲーム画面の枠線
        f"drawbox=x={_BIIM_GAME_X - 2}:y={_BIIM_GAME_Y - 2}:"
        f"w={_BIIM_GAME_W + 4}:h={_BIIM_GAME_H + 4}:color=0x35E0FF@0.9:t=2,"
        # 下部実況帯（フル幅・スマホでも読める大きさの字幕領域）
        f"drawbox=x={_BIIM_GAME_X}:y={band_y}:w={band_w}:h={band_h}:"
        f"color=black@0.85:t=fill,"
        f"drawbox=x={_BIIM_GAME_X}:y={band_y}:w={band_w}:h={band_h}:"
        f"color=white@0.7:t=2,"
        # 右サイドパネルの下地（ゲーム画面と同じ高さ・中身はv2b/v2c）
        f"drawbox=x={panel_x}:y={_BIIM_GAME_Y}:w={panel_w}:h={_BIIM_GAME_H}:"
        f"color={_BIIM_PANEL_COLOR}:t=fill,"
        f"drawtext=fontfile={_BIIM_FONT_FILE}:text='◆ 戦況':"
        f"x={panel_x + 24}:y=44:fontsize=34:fontcolor=0x66CCFF,"
        # 実況字幕（音声シンクロ）
        f"subtitles={ass_path}:fontsdir={_BIIM_FONTS_DIR}"
    )
    inputs = ["-i", str(video), "-i", str(track_wav)]
    if avatar_video is not None:
        # -ss で録画先頭（WAV再生開始前の部分）をスキップして頭を合わせる
        inputs += ["-ss", f"{avatar_offset}", "-i", str(avatar_video)]
        crop_filter = f"crop={avatar_crop}," if avatar_crop else ""
        video_filter = (
            f"{video_filter}[vsub];"
            f"[2:v]{crop_filter}scale={avatar_width}:-2,"
            f"chromakey={avatar_chroma}:{avatar_similarity}:0.08,"
            f"despill=type=green:mix=0.5:expand=0[av];"
            f"[vsub][av]overlay="
            f"main_w-overlay_w-{_AVATAR_MARGIN}:main_h-overlay_h-{_AVATAR_MARGIN}:"
            f"eof_action=repeat[vout]"
        )
    else:
        video_filter = f"{video_filter}[vout]"
    audio_filter = (
        f"[1:a]volume={gain},aresample=48000,"
        f"aformat=channel_layouts=stereo,asplit=2[sc][cm];"
        f"[0:a][sc]sidechaincompress="
        f"threshold={duck_threshold}:ratio={duck_ratio}:attack=20:release=400[duck];"
        f"[duck][cm]amix=inputs=2:duration=longest:dropout_transition=0,"
        f"volume=2[aout]"
    )
    duration_opts = ["-t", f"{max_duration:.3f}"] if max_duration > 0 else []
    return (
        ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", f"{video_filter};{audio_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
        ] + duration_opts + [
            str(out_path),
        ]
    )


def probe_duration(video: Path) -> float:
    """ffprobeで動画長（秒）を取得。ffprobeが無ければ0を返す。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def build_ffmpeg_command(video: Path, track_wav: Path, out_path: Path,
                         gain: float, duck_threshold: float,
                         duck_ratio: float) -> list:
    """ダッキング合成のffmpegコマンドを組み立てる。

    実況トラックを2分岐し、片方をサイドチェイン（ゲーム音圧縮のトリガ）、
    もう片方をミックス本体に使う。映像はコピー（再エンコードなし）。

    amixの ``normalize=0`` はffmpeg 4.4以降のためWSL標準の4.2では使えない。
    代わりにamix既定の1/2スケーリングを直後の ``volume=2`` で補償する
    （実況トラックは動画と同尺までパディング済みなので、両入力が同時に
    存続し補償が全編で一定になる）。
    """
    filter_complex = (
        f"[1:a]volume={gain},aresample=48000,"
        f"aformat=channel_layouts=stereo,asplit=2[sc][cm];"
        f"[0:a][sc]sidechaincompress="
        f"threshold={duck_threshold}:ratio={duck_ratio}:attack=20:release=400[duck];"
        f"[duck][cm]amix=inputs=2:duration=longest:dropout_transition=0,"
        f"volume=2[aout]"
    )
    return [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(track_wav),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]


def _print_schedule(scheduled: list, deduped: list, dropped: list) -> None:
    print(f"\n{'#':>3} {'イベント':<14} {'イベント時刻':>10} {'開始':>8} "
          f"{'遅延':>6} {'長さ':>6}  実況文")
    for i, e in enumerate(scheduled, 1):
        text = e["commentary"]
        if len(text) > 40:
            text = text[:40] + "…"
        print(f"{i:>3} {e['event_type']:<14} {e['event_time']:>9.1f}s "
              f"{e['start']:>7.1f}s {e['delay']:>5.1f}s {e['duration']:>5.1f}s  {text}")
    for e in deduped:
        print(f"  - デデュープ破棄: t={e['event_time']:.1f}s {e['event_type']} "
              f"（同一実況文の重複）")
    for e in dropped:
        print(f"  - 遅延超過破棄: t={e['event_time']:.1f}s {e['event_type']} "
              f"（遅延 {e['delay']:.1f}s）")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="実況動画の合成（ADR-009 パス2）")
    parser.add_argument("render_dir", help="パス1の素材ディレクトリ（renders/<動画名>）")
    parser.add_argument("--video", help="元動画パス（省略時はrender_info.jsonから）")
    parser.add_argument("--out", help="出力mp4パス（省略時は<render_dir>/<名前>_commentary.mp4）")
    parser.add_argument("--gap", type=float, default=_DEFAULT_GAP_SEC,
                        help=f"実況間の最低間隔・秒（既定{_DEFAULT_GAP_SEC}）")
    parser.add_argument("--max-delay", type=float, default=_DEFAULT_MAX_DELAY_SEC,
                        help=f"許容する最大後ろ倒し・秒（既定{_DEFAULT_MAX_DELAY_SEC}）")
    parser.add_argument("--gain", type=float, default=_DEFAULT_GAIN,
                        help=f"実況音声の音量倍率（既定{_DEFAULT_GAIN}）")
    parser.add_argument("--duck-threshold", type=float, default=_DEFAULT_DUCK_THRESHOLD,
                        help=f"ダッキング閾値（既定{_DEFAULT_DUCK_THRESHOLD}）")
    parser.add_argument("--duck-ratio", type=float, default=_DEFAULT_DUCK_RATIO,
                        help=f"ダッキング圧縮比（既定{_DEFAULT_DUCK_RATIO}）")
    parser.add_argument("--layout", choices=["plain", "biim"], default="plain",
                        help="plain=音声のみ合成（既定）/ biim=案A枠＋実況字幕帯（v2a）")
    parser.add_argument("--avatar-video", default=None,
                        help="VMC口パク録画のmp4（v2c・biim時のみ・グリーンバックをクロマキー合成）")
    parser.add_argument("--avatar-offset", type=float, default=0.0,
                        help="録画開始→WAV再生開始の秒数（頭合わせ・0以上）")
    parser.add_argument("--avatar-width", type=int, default=_AVATAR_WIDTH,
                        help=f"アバターの表示幅px（既定{_AVATAR_WIDTH}）")
    parser.add_argument("--avatar-chroma", default=_AVATAR_CHROMA,
                        help=f"クロマキー色（既定{_AVATAR_CHROMA}=緑）")
    parser.add_argument("--avatar-crop", default=None,
                        help="スケール前に録画をクロップ（ffmpeg crop式 'w:h:x:y'）。"
                             "全身録画から上半身だけを抜き出して拡大表示したい場合に指定")
    parser.add_argument("--dry-run", action="store_true",
                        help="スケジュールの表示のみ（ffmpeg不要）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    render_dir = Path(args.render_dir)
    if not (render_dir / "manifest.jsonl").exists():
        logger.error("manifest.jsonl が見つかりません: %s", render_dir)
        return 1

    entries = load_manifest(render_dir)
    kept, deduped = dedupe_entries(entries)
    scheduled, dropped = schedule_entries(kept, gap=args.gap, max_delay=args.max_delay)
    logger.info("実況 %d件 → デデュープ後 %d件 → スケジュール確定 %d件（遅延超過破棄 %d件）",
                len(entries), len(kept), len(scheduled), len(dropped))

    # 台本パスのフィラー（あれば）を空き区間へ配置（イベント実況が優先）
    fillers = load_fillers(render_dir)
    placed_fillers = []
    fillers_dropped = []
    if fillers:
        fillers_kept, fillers_deduped = dedupe_entries(fillers)
        deduped = deduped + fillers_deduped
        placed_fillers, fillers_dropped = fit_fillers(scheduled, fillers_kept,
                                                      gap=args.gap)
        logger.info("フィラー %d件 → 配置 %d件（収まらず破棄 %d件）",
                    len(fillers), len(placed_fillers), len(fillers_dropped))

    final_schedule = sorted(scheduled + placed_fillers, key=lambda e: e["start"])
    _print_schedule(final_schedule, deduped, dropped + fillers_dropped)

    # スケジュール結果を記録（検証・再現用）
    schedule_path = render_dir / "schedule.json"
    with schedule_path.open("w", encoding="utf-8") as fp:
        json.dump({"scheduled": final_schedule,
                   "deduped": deduped, "dropped": dropped,
                   "fillers_dropped": fillers_dropped},
                  fp, ensure_ascii=False, indent=2)
    logger.info("スケジュールを保存: %s", schedule_path)

    if args.dry_run:
        return 0

    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg が見つかりません（PATHに追加してください）。"
                     "--dry-run ならスケジュール確認のみ実行できます")
        return 1

    # 元動画の解決
    if args.video:
        video = Path(args.video)
    else:
        with (render_dir / "render_info.json").open(encoding="utf-8") as fp:
            info = json.load(fp)
        video = resolve_video_path(info["video"])
    if not video.exists():
        logger.error("元動画が見つかりません: %s", video)
        return 1

    # 実況トラック生成（動画と同尺までパディング＝amixのvolume補償を一定に保つ）
    video_dur = probe_duration(video)
    track_wav = render_dir / "commentary_track.wav"
    track_end = build_commentary_track(render_dir, final_schedule, track_wav,
                                       min_duration=video_dur)
    logger.info("実況トラック生成: %s（%.1f秒・動画長 %.1f秒）",
                track_wav, track_end, video_dur)
    tail_pad = max(0.0, track_end - video_dur) if video_dur else 0.0
    if tail_pad > 0:
        if args.layout == "biim":
            logger.info("実況トラック終端 %.1fs が動画長 %.1fs を超過"
                        "→末尾を最終フレーム静止で %.1f 秒延長します",
                        track_end, video_dur, tail_pad)
        else:
            logger.warning("実況トラック終端 %.1fs が動画長 %.1fs を超過"
                           "（plainは映像コピーのため延長不可・末尾の実況が"
                           "映像より長く続きます。--layout biim なら延長されます）",
                           track_end, video_dur)

    if args.layout == "biim":
        suffix = "_commentary_biim.mp4"
        ass_path = render_dir / "commentary.ass"
        states = load_states(render_dir)
        moments = load_timeline(render_dir)
        panel_events = build_panel_events(states, moments,
                                          video_dur or track_end)
        build_ass(final_schedule, ass_path, panel_events)
        logger.info("実況字幕を生成: %s（字幕%d件・パネル状態%d件・技ログ%d件）",
                    ass_path, len(final_schedule), len(states), len(moments))
        avatar_video = None
        if args.avatar_video:
            avatar_video = Path(args.avatar_video)
            if not avatar_video.exists():
                logger.error("アバター録画が見つかりません: %s", avatar_video)
                return 1
            logger.info("アバター合成: %s（offset=%.2fs・幅%dpx）",
                        avatar_video, args.avatar_offset, args.avatar_width)
        out_path = Path(args.out) if args.out else render_dir / f"{render_dir.name}{suffix}"
        # アバター録画が本編（動画+tail_pad）より長い場合の間延び防止（末尾を本編長で打ち切る）
        max_duration = (video_dur or track_end) + tail_pad if avatar_video else 0.0
        cmd = build_ffmpeg_command_biim(video, track_wav, out_path, ass_path,
                                        args.gain, args.duck_threshold, args.duck_ratio,
                                        avatar_video=avatar_video,
                                        avatar_offset=args.avatar_offset,
                                        avatar_width=args.avatar_width,
                                        avatar_chroma=args.avatar_chroma,
                                        avatar_crop=args.avatar_crop,
                                        tail_pad=tail_pad,
                                        max_duration=max_duration)
    else:
        if args.avatar_video:
            logger.error("--avatar-video は --layout biim でのみ使えます")
            return 1
        out_path = Path(args.out) if args.out else render_dir / f"{render_dir.name}_commentary.mp4"
        cmd = build_ffmpeg_command(video, track_wav, out_path,
                                   args.gain, args.duck_threshold, args.duck_ratio)
    logger.info("ffmpeg実行: %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        logger.error("ffmpegが失敗しました（exit=%d）", proc.returncode)
        return proc.returncode
    logger.info("合成完了: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
