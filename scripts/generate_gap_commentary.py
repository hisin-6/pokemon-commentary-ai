"""台本パス: 無言区間を埋めるつなぎ実況の生成（ADR-009 パス1.5）。

パス1の素材（manifest.jsonl）からイベント実況の無言区間を検出し、
EC2 ``/api/script`` にテキストのみの1回送信でフィラー実況をまとめて生成、
VOICEVOXでWAV化して ``fillers.jsonl`` として素材ディレクトリに追加する。
パス2（render_commentary_video.py）は fillers.jsonl があれば自動でマージする。

イベント実況（manifest）には手を付けないため、フィラーの再生成・破棄は
このスクリプトの再実行（Bedrockテキスト1回・数円）だけで済む。

使い方:
    python scripts/generate_gap_commentary.py renders/16-14-39 --ec2-url http://<EC2>:5000
    （VOICEVOX起動が必要。--dry-run はギャップ検出とBedrock生成まで＝VOICEVOX不要）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

# 同ディレクトリのパス2実装からスケジューリング系を再利用
sys.path.insert(0, str(Path(__file__).parent))
from render_commentary_video import (  # noqa: E402
    dedupe_entries,
    load_manifest,
    probe_duration,
    resolve_video_path,
    schedule_entries,
)

# プロジェクトルート（src.* のインポート用）
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# この秒数を超える無言区間をフィラーの対象にする
_DEFAULT_MIN_GAP_SEC = 20.0
# ギャップ端の余白（イベント実況の直前直後にフィラーを密着させない）
_GAP_MARGIN_SEC = 2.0


def compute_gaps(scheduled: list, video_duration: float = 0.0,
                 min_gap: float = _DEFAULT_MIN_GAP_SEC,
                 margin: float = _GAP_MARGIN_SEC) -> list:
    """スケジュール済みイベント実況の間の無言区間を検出する。

    区間は前後 ``margin`` 秒を除いた「フィラーを置いてよい範囲」で返す。
    冒頭（0秒〜最初の実況）も対象。末尾は ``video_duration`` が分かる場合のみ。
    """
    gaps = []
    prev_end = 0.0
    for e in sorted(scheduled, key=lambda e: e["start"]):
        gap_start = prev_end + (margin if prev_end > 0 else 0.0)
        gap_end = e["start"] - margin
        if gap_end - gap_start >= min_gap:
            gaps.append({"start": round(gap_start, 1), "end": round(gap_end, 1)})
        prev_end = max(prev_end, e["start"] + e["duration"])
    if video_duration > 0:
        gap_start = prev_end + margin
        gap_end = video_duration - margin
        if gap_end - gap_start >= min_gap:
            gaps.append({"start": round(gap_start, 1), "end": round(gap_end, 1)})
    return gaps


def clamp_fillers_to_gaps(fillers: list, gaps: list) -> tuple:
    """フィラーの time を対応するギャップ範囲内に収める。

    Bedrockが範囲を外した time を返した場合、最も近いギャップに丸める。
    どのギャップからも大きく外れている（30秒超）ものは破棄する。
    """
    kept = []
    dropped = []
    for f in fillers:
        t = float(f["time"])
        best = None
        best_dist = None
        for g in gaps:
            if g["start"] <= t <= g["end"]:
                best, best_dist = t, 0.0
                break
            dist = min(abs(t - g["start"]), abs(t - g["end"]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = min(max(t, g["start"]), g["end"])
        if best is None or best_dist > 30.0:
            dropped.append(f)
            continue
        kept.append({"time": round(best, 1), "text": f["text"]})
    kept.sort(key=lambda f: f["time"])
    return kept, dropped


def split_gaps_by_moments(gaps: list, moments: list,
                          min_len: float = 12.0) -> list:
    """ギャップを内包する瞬間ログ（📺）の時刻で分割する。

    分割後の各サブ区間はプロンプトのタイムライン上で直前の📺より後に
    並ぶため、「★区間より下の出来事は未来」ルールが区間内の📺にも
    構造的に効く（プロンプト指示だけでは区間頭のフィラーが同区間の
    後方📺を先読みする実例があった: 2026-07-14 t=85sがt=105〜123sの
    技を予告）。min_len未満の断片はフィラー1本が収まらないため捨てる。
    """
    out = []
    for g in gaps:
        cuts = sorted(m["time"] for m in moments
                      if g["start"] < m["time"] < g["end"])
        bounds = [g["start"]] + cuts + [g["end"]]
        for a, b in zip(bounds, bounds[1:]):
            if b - a >= min_len:
                out.append({"start": round(a, 1), "end": round(b, 1)})
    return out


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


def request_fillers(ec2_url: str, events: list, gaps: list,
                    moments: list = None, timeout: float = 120.0) -> list:
    """EC2 /api/script に無言区間ごとに1回ずつ送りフィラー案を集約して受け取る。

    ネタバレ防止のため、gapごとに個別のBedrock呼び出しにして、その区間より
    未来のevents/momentsを一切見せない設計にしている（2026-07-22・従来の
    「全区間一括送信＋指示でネタバレ抑制」は指示に従い損ねる実例があった）。
    """
    events_payload = [
        {
            "time": e["event_time"],
            "event_type": e["event_type"],
            "commentary": e["commentary"],
            "context": e.get("context"),
        }
        for e in events
    ]
    moments_payload = moments or []

    fillers: list = []
    for gap in gaps:
        payload = {
            "events": events_payload,
            "gap": gap,
            "moments": moments_payload,
        }
        resp = requests.post(f"{ec2_url.rstrip('/')}/api/script", json=payload,
                             timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"/api/script 失敗: {data.get('error')} {data.get('message')}")
        usage = data.get("usage", {})
        logger.info("フィラー生成 %.0f〜%.0fs: %d件 (tokens_in=%s tokens_out=%s latency=%sms)",
                    gap["start"], gap["end"], len(data["fillers"]),
                    usage.get("input_tokens"), usage.get("output_tokens"),
                    data.get("latency_ms"))
        fillers.extend(data["fillers"])
    return fillers


def synthesize_fillers(render_dir: Path, fillers: list, voicevox_url: str,
                       speaker: int) -> list:
    """フィラーをVOICEVOXでWAV化し fillers.jsonl のエントリ列を返す。"""
    from src.output.render_sink import RenderSink
    from src.output.voicevox_client import VoicevoxClient

    client = VoicevoxClient(url=voicevox_url, speaker=speaker)
    wav_dir = render_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    entries = []
    for i, f in enumerate(fillers, 1):
        wav_bytes = client.generate_wav(f["text"])
        wav_name = f"f{i:04d}_filler.wav"
        (wav_dir / wav_name).write_bytes(wav_bytes)
        entry = {
            "seq": i,
            "event_time": f["time"],
            "event_type": "filler",
            "commentary": f["text"],
            "wav": f"wav/{wav_name}",
            "duration": round(RenderSink.wav_duration(wav_bytes), 3),
        }
        entries.append(entry)
        logger.info("[フィラー] #%d t=%.1fs (%.1f秒) %s", i, f["time"],
                    entry["duration"], f["text"][:30])
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="台本パス: ギャップフィラー生成（ADR-009）")
    parser.add_argument("render_dir", help="パス1の素材ディレクトリ（renders/<動画名>）")
    parser.add_argument("--ec2-url", required=True, help="EC2 APIのURL（http://<IP>:5000）")
    parser.add_argument("--voicevox-url", default="http://localhost:50021",
                        help="VOICEVOXのURL（既定 http://localhost:50021）")
    parser.add_argument("--speaker", type=int, default=None,
                        help="VOICEVOX話者ID（省略時はrender_info.jsonの値）")
    parser.add_argument("--min-gap", type=float, default=_DEFAULT_MIN_GAP_SEC,
                        help=f"フィラー対象とする最小無言秒数（既定{_DEFAULT_MIN_GAP_SEC}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="Bedrock生成結果の表示まで（VOICEVOX不要・fillers.jsonl未更新）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    render_dir = Path(args.render_dir)
    if not (render_dir / "manifest.jsonl").exists():
        logger.error("manifest.jsonl が見つかりません: %s", render_dir)
        return 1

    with (render_dir / "render_info.json").open(encoding="utf-8") as fp:
        info = json.load(fp)
    speaker = args.speaker if args.speaker is not None else int(info.get("speaker", 1))

    # イベント実況の確定スケジュール（パス2と同じ手順）からギャップを検出
    entries = load_manifest(render_dir)
    kept, _ = dedupe_entries(entries)
    scheduled, _ = schedule_entries(kept)
    video = resolve_video_path(info["video"])
    video_dur = probe_duration(video) if video.exists() else 0.0
    gaps = compute_gaps(scheduled, video_dur, min_gap=args.min_gap)
    if not gaps:
        logger.info("%.0f秒超の無言区間なし。フィラー生成は不要", args.min_gap)
        return 0
    logger.info("無言区間 %d箇所: %s", len(gaps),
                " / ".join(f"{g['start']:.0f}〜{g['end']:.0f}s" for g in gaps))
    moments = load_timeline(render_dir)
    if moments:
        logger.info("瞬間ログ %d件（timeline.jsonl・ライブ実況アンカーとして送信）", len(moments))
        gaps = split_gaps_by_moments(gaps, moments)
        logger.info("📺時刻でギャップを分割 → %d区間（区間内の先読みネタバレ対策）", len(gaps))
    else:
        logger.info("timeline.jsonl なし（パス1の再実行で技の瞬間ログが付きます）")

    # Bedrockでフィラー生成（テキストのみ1回）→ ギャップ範囲へクランプ
    raw_fillers = request_fillers(args.ec2_url, kept, gaps, moments)
    fillers, dropped = clamp_fillers_to_gaps(raw_fillers, gaps)
    for f in dropped:
        logger.warning("ギャップ範囲外のため破棄: t=%.1fs %s", f["time"], f["text"][:30])
    for f in fillers:
        print(f"  {f['time']:>7.1f}s  {f['text']}")

    if args.dry_run:
        logger.info("dry-run のためここまで（fillers.jsonl 未更新）")
        return 0

    # VOICEVOXでWAV化して fillers.jsonl へ（毎回作り直し）
    entries = synthesize_fillers(render_dir, fillers, args.voicevox_url, speaker)
    fillers_path = render_dir / "fillers.jsonl"
    with fillers_path.open("w", encoding="utf-8") as fp:
        for e in entries:
            fp.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info("フィラー素材出力完了: %d件 → %s", len(entries), fillers_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
