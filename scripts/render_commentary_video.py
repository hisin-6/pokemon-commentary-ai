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
    if video_dur and track_end > video_dur:
        logger.warning("実況トラック終端 %.1fs が動画長 %.1fs を超過"
                       "（末尾の実況が映像より長く続きます）", track_end, video_dur)

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
