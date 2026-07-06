"""ocr_diag JSONL を BattlePhaseClassifier にリプレイしてイベント検出を検証する。

ocr_logger.py で収集した診断ログ（フレーム全体OCR）を実際の
BattlePhaseClassifier に流し込み、イベント発火数を実ターン数と突き合わせる。
時刻は JSONL の elapsed（動画内経過秒）で仮想化するため、実時間はかからない。

実パイプラインの挙動の模擬:
  - --interval 秒ごとにフレームを間引く（デフォルト 1.0 = メインOCR相当）
  - turn_start 以外のイベント発火後は --processing 秒分フレームをスキップし、
    その後 reset_after_processing() を呼ぶ（Bedrock+VOICEVOX の処理ブロック相当）

使い方（Windows venv）:
  venv\\Scripts\\python.exe scripts\\replay_phase_events.py logs\\ocr_diag_XXX.jsonl:5 ...
  「:5」は書き起こしから数えた実ターン数（省略可）。turn_start数と照合される。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.pipeline as pl  # noqa: E402


class _MarkerCounter(logging.Handler):
    """ログメッセージ中のマーカー出現数を数える（脱出弁の発火回数計測用）。"""

    def __init__(self, marker: str):
        super().__init__()
        self.marker = marker
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if self.marker in record.getMessage():
            self.count += 1


def replay(path: str, interval: float, processing: float, verbose: bool):
    """1ファイルをリプレイしてイベント数と脱出弁発火数を返す。"""
    clf = pl.BattlePhaseClassifier()
    valve = _MarkerCounter("脱出弁")
    pl.log.addHandler(valve)
    pl.log.setLevel(logging.INFO)  # 脱出弁カウンタはINFOログ経由のため常にINFO
    if not verbose:
        # コンソール・ファイルへのログ出力を抑制（カウンタはlogger直付けなので影響なし）
        logging.getLogger().handlers.clear()

    counts: dict[str, int] = {}
    next_t = 0.0
    skip_until = -1.0
    pending_reset: str | None = None
    events_detail: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = fr.get("elapsed")
            if t is None:
                continue
            if t < next_t:
                continue
            if interval > 0:
                next_t = t + interval
            pl.time.time = lambda t=t: t  # 仮想時計: 動画内経過秒をそのまま時刻に使う
            if t < skip_until:
                continue
            if pending_reset is not None:
                clf.reset_after_processing(pending_reset)
                pending_reset = None
            ocr = [{"text": r.get("text", ""), "confidence": r.get("conf", 0.0)}
                   for r in fr.get("ocr", [])]
            valve_before = valve.count
            ev = clf.detect(ocr)
            if ev:
                counts[ev] = counts.get(ev, 0) + 1
                suffix = "（脱出弁）" if valve.count > valve_before else ""
                events_detail.append(f"{t:7.1f}s {ev}{suffix}")
                if ev != "turn_start":
                    # turn_start はカウントのみの高速パス。他は実況処理ブロックを模擬
                    skip_until = t + processing
                    pending_reset = ev

    pl.log.removeHandler(valve)
    return counts, valve.count, events_detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+",
                    help="JSONLパス（末尾に :実ターン数 を付けると照合する）")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="サンプリング間隔秒（0=全フレーム・デフォルト1.0=メインOCR相当）")
    ap.add_argument("--processing", type=float, default=12.0,
                    help="イベント処理ブロックの模擬秒数（デフォルト12）")
    ap.add_argument("--verbose", action="store_true",
                    help="フェーズ遷移・イベントのINFOログとイベント時系列を表示")
    args = ap.parse_args()

    header = (f"{'ファイル':<38} {'実T':>3} {'battle':>6} {'turn':>5} "
              f"{'move':>5} {'faint':>5} {'switch':>6} {'end':>4} {'脱出弁':>6}  判定")
    print(header)
    print("-" * len(header))
    ok_all = True
    for spec in args.files:
        if ":" in Path(spec).name and not spec.endswith(".jsonl"):
            path, _, exp = spec.rpartition(":")
            expected: int | None = int(exp)
        else:
            path, expected = spec, None
        counts, valve_count, detail = replay(path, args.interval, args.processing, args.verbose)
        b = counts.get("battle_start", 0)
        ts = counts.get("turn_start", 0)
        if expected is None:
            mark = "-"
        elif ts == expected:
            mark = "OK"
        else:
            mark = f"NG (turn_start {ts} != 実T {expected})"
            ok_all = False
        print(f"{Path(path).name:<38} {expected if expected is not None else '-':>3} "
              f"{b:>6} {ts:>5} {counts.get('move_used', 0):>5} {counts.get('faint', 0):>5} "
              f"{counts.get('switch', 0):>6} {counts.get('battle_end', 0):>4} {valve_count:>6}  {mark}")
        if args.verbose:
            for d in detail:
                print("    " + d)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
