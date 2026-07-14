"""フィラーのネタバレ検査（パス1.5とパス2の間に実行する・ADR-009）。

タイムライン（イベント実況・📺技の瞬間・★フィラー）を時刻順に表示し、
フィラーが「その時点でまだ画面に映っていない技名」に言及していたら
自動でフラグを立てる。実測されたネタバレは全て未来の技名の先取りだった
（2026-07-14: ソーラービーム/ふいうち/シャカシャカほう/マジカルシャイン）
ため、このチェックで大半を機械検出できる。

使い方:
    python3 scripts/check_spoilers.py renders/<動画名>

終了コード: 0=フラグなし / 1=要確認フラグあり（内容を見て対処を判断する）

フラグが立ったら（自動検出は「疑い」なので必ず目視確認してから）:
  - fillers.jsonl の該当行の event_time を、言及している技の📺時刻より後の
    空き時間へ移動する（後ろが詰まっていて置けないなら行ごと削除）
  - テキストの編集はWAVと不一致になるため不可
  - 修正後にパス2を再実行する

なお勝敗・気絶の先取りなど技名以外のネタバレは自動検出できないため、
表示されるタイムラインの目視確認も併せて行うこと（各★Fについて
「その時刻より上の行の情報だけで書けている内容か」を見る）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 瞬間ログの技エントリ形式: "T3:ガブリアスのドラゴンクロー"
_MOMENT_MOVE_RE = re.compile(r"^T[\d?]+:(.+?)の(.+)$")
# 先読み判定の許容誤差（秒）。検知タイミングの微差での誤フラグを防ぐ
_TOLERANCE_SEC = 1.0


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def find_spoilers(fillers: list, moments: list, events: list) -> list:
    """未来の技名に言及しているフィラーを検出する。

    技名の「既知時刻」は、📺瞬間ログの初出時刻と、イベント実況テキストに
    現れた時刻の早い方。フィラーの時刻がそれより前なら先読みと判定する。

    Returns:
        [{"filler": フィラー, "move": 技名, "known_at": 既知になる時刻}]
    """
    first_seen: dict = {}
    for m in sorted(moments, key=lambda m: float(m["time"])):
        match = _MOMENT_MOVE_RE.match(m.get("text", ""))
        if not match:
            continue
        move = match.group(2)
        first_seen.setdefault(move, float(m["time"]))
    # イベント実況が先に技名に触れていたらその時刻を既知時刻とする
    for e in sorted(events, key=lambda e: float(e["event_time"])):
        for move, t in first_seen.items():
            if move in e.get("commentary", "") and float(e["event_time"]) < t:
                first_seen[move] = float(e["event_time"])

    flags = []
    for f in fillers:
        t = float(f["event_time"])
        for move, known_at in first_seen.items():
            if move in f.get("commentary", "") and t < known_at - _TOLERANCE_SEC:
                flags.append({"filler": f, "move": move, "known_at": known_at})
    return flags


def print_timeline(events: list, moments: list, fillers: list,
                   flagged: set) -> None:
    """タイムラインを時刻順に表示する（目視検査用）。"""
    rows = sorted(
        [("EV", float(e["event_time"]), e["commentary"]) for e in events] +
        [("📺", float(m["time"]), m["text"]) for m in moments] +
        [("★F", float(f["event_time"]), f["commentary"]) for f in fillers],
        key=lambda r: r[1])
    for kind, t, text in rows:
        mark = " ⚠️" if kind == "★F" and round(t, 3) in flagged else ""
        print(f" {kind} {t:>6.1f}s {text}{mark}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="フィラーのネタバレ検査")
    parser.add_argument("render_dir", help="素材ディレクトリ（renders/<動画名>）")
    args = parser.parse_args(argv)

    rd = Path(args.render_dir)
    events = _load_jsonl(rd / "manifest.jsonl")
    moments = _load_jsonl(rd / "timeline.jsonl")
    fillers = _load_jsonl(rd / "fillers.jsonl")
    if not fillers:
        print("fillers.jsonl がありません（パス1.5が未実行 or dry-runのみ）")
        return 1

    flags = find_spoilers(fillers, moments, events)
    flagged_times = {round(float(fl["filler"]["event_time"]), 3) for fl in flags}

    print(f"== タイムライン（EV=イベント実況 / 📺=技の瞬間 / ★F=フィラー） ==")
    print_timeline(events, moments, fillers, flagged_times)

    if flags:
        print(f"\n⚠️ 技名の先読み疑い {len(flags)}件（目視確認のうえ移動 or 削除）:")
        for fl in flags:
            f = fl["filler"]
            print(f"  t={float(f['event_time']):.1f}s のフィラーが「{fl['move']}」に言及"
                  f"（画面に映るのは {fl['known_at']:.1f}s）")
            print(f"    → {f['commentary'][:60]}")
        return 1
    print("\n技名の先読みなし✅（勝敗・気絶などの先取りは上のタイムラインで目視確認すること）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
