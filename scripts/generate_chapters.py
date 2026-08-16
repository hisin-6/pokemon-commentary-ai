"""実況動画のYouTubeチャプター自動生成（改善アイデア・2026-08-16）。

パス1の素材（manifest.jsonl）からターンの切り替わり・試合開始・決着のタイミングを
拾い、YouTubeの概要欄にそのまま貼れるチャプターリスト（`00:00 ラベル`形式）を
テキスト出力する。動画本体・パス2の合成には一切影響しない（読み取り専用・独立実行）。

ラベルは意図的に中身を明かさない（「ターンN」「決着」等）。チャプターは動画を
観る前に概要欄で見えてしまうため、実況内の展開（誰が倒れた・勝敗等）まで
書いてしまうと本編以上のネタバレになる。

YouTubeの仕様上の制約:
- 最初のチャプターは 00:00 から始まる必要がある
- チャプターは最低3つ必要（それ未満は概要欄に貼ってもチャプター機能が有効化されない）
- 各チャプターは最低10秒以上離れている必要がある（`--min-gap`で調整可・既定10秒）

使い方:
    python3 scripts/generate_chapters.py renders/<動画名>
    python3 scripts/generate_chapters.py renders/<動画名> --out chapters.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from render_commentary_video import load_manifest  # noqa: E402

_DEFAULT_MIN_GAP_SEC = 10.0  # YouTube仕様: チャプター間は最低10秒


def _format_timestamp(seconds: float) -> str:
    """秒数をYouTubeチャプター形式（M:SS・1時間以上ならH:MM:SS）に変換する。"""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def build_chapters(entries: list, min_gap: float = _DEFAULT_MIN_GAP_SEC) -> list[tuple[float, str]]:
    """manifest.jsonlのエントリ群からチャプターリスト（時刻, ラベル）を組み立てる。

    - 00:00は常に「オープニング」で固定（YouTube仕様で最初のチャプターは0:00必須）
    - battle_start・ターンの切り替わり・battle_endをチャプター境界として拾う
    - min_gap未満の間隔になる境界は間引く（直前のチャプターを優先・YouTube仕様対策）
    """
    if not entries:
        return [(0.0, "オープニング")]

    raw: list[tuple[float, str]] = [(0.0, "オープニング")]
    seen_turn: int | None = None
    for e in entries:
        t = e["event_time"]
        event_type = e.get("event_type")
        turn = (e.get("context") or {}).get("turn")

        if event_type == "battle_start":
            raw.append((t, "試合開始"))
            seen_turn = turn
            continue
        if event_type == "battle_end":
            raw.append((t, "決着"))
            continue
        if turn is not None and turn != seen_turn:
            raw.append((t, f"ターン{turn}"))
            seen_turn = turn

    # min_gap未満で連続するチャプターを間引く（先勝ち＝古い方のラベル・時刻を残す）
    chapters: list[tuple[float, str]] = [raw[0]]
    for t, label in raw[1:]:
        if t - chapters[-1][0] >= min_gap:
            chapters.append((t, label))
    return chapters


def format_chapters(chapters: list[tuple[float, str]]) -> str:
    return "\n".join(f"{_format_timestamp(t)} {label}" for t, label in chapters)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="実況動画のYouTubeチャプター自動生成")
    parser.add_argument("render_dir", type=Path, help="renders/<動画名> ディレクトリ")
    parser.add_argument("--out", type=Path, help="出力先テキスト（既定: <render_dir>/chapters.txt）")
    parser.add_argument("--min-gap", type=float, default=_DEFAULT_MIN_GAP_SEC,
                         help=f"チャプター間の最低秒数（既定{_DEFAULT_MIN_GAP_SEC}秒・YouTube仕様）")
    args = parser.parse_args(argv)

    entries = load_manifest(args.render_dir)
    chapters = build_chapters(entries, min_gap=args.min_gap)
    text = format_chapters(chapters)

    if len(chapters) < 3:
        print(f"⚠️ チャプターが{len(chapters)}個しかありません"
              "（YouTubeは最低3つ必要・概要欄に貼ってもチャプター機能が有効化されません）",
              file=sys.stderr)

    out_path = args.out or (args.render_dir / "chapters.txt")
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n→ {out_path} に出力しました。概要欄にそのまま貼り付けてください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
