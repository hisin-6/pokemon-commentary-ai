"""fillers.jsonl の特定行(seq)だけ文言を差し替えて、対応するWAVをVOICEVOXで再合成する。

Bedrockは呼ばない（差し替え後の文言は呼び出し側で確定済みという前提）ので追加課金なし。
「実況内容の重複が見つかったのでフィラーの言い回しだけ直したい」ような場面向け。
文言（字幕）とWAV（音声）が必ず一致した状態を保つのが目的
（commentaryフィールドだけをテキストエディタで書き換えると字幕と音声がズレるため、これは避けること）。

事前準備: VOICEVOX を起動しておくこと（実行環境: Windows Python）

使い方:
    venv\\Scripts\\python.exe scripts\\resynthesize_fillers.py renders\\<動画名> ^
        --edit 3="新しい文言その1" --edit 4="新しい文言その2"

- --edit は複数回指定できる（SEQ=新文言）
- 対象seqのwav（wav/fNNNN_filler.wav）を上書きし、fillers.jsonlのcommentary/durationを更新する
- event_time・wavパス・他の行はそのまま
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.output.render_sink import RenderSink
from src.output.voicevox_client import VoicevoxClient


def parse_edit(raw: str) -> tuple[int, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--edit は SEQ=新文言 の形式で指定してください: {raw!r}")
    seq_str, text = raw.split("=", 1)
    return int(seq_str.strip()), text


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("render_dir", help="パス1の素材ディレクトリ（renders/<動画名>）")
    parser.add_argument("--edit", type=parse_edit, action="append", required=True,
                        help="SEQ=新文言（複数指定可）")
    parser.add_argument("--voicevox-url", default="http://localhost:50021")
    parser.add_argument("--speaker", type=int, default=None,
                        help="省略時はrender_info.jsonの値")
    args = parser.parse_args(argv)

    render_dir = Path(args.render_dir)
    fillers_path = render_dir / "fillers.jsonl"
    if not fillers_path.exists():
        print(f"エラー: {fillers_path} が見つかりません", file=sys.stderr)
        return 1

    speaker = args.speaker
    if speaker is None:
        info_path = render_dir / "render_info.json"
        speaker = json.loads(info_path.read_text(encoding="utf-8")).get("speaker", 2) if info_path.exists() else 2

    edits = dict(args.edit)
    rows = load_jsonl(fillers_path)
    by_seq = {r["seq"]: r for r in rows}
    missing = [seq for seq in edits if seq not in by_seq]
    if missing:
        print(f"エラー: seq {missing} が fillers.jsonl に存在しません", file=sys.stderr)
        return 1

    client = VoicevoxClient(url=args.voicevox_url, speaker=speaker)
    wav_dir = render_dir / "wav"

    for seq, new_text in edits.items():
        row = by_seq[seq]
        old_text = row["commentary"]
        wav_bytes = client.generate_wav(new_text)
        wav_path = wav_dir / Path(row["wav"]).name
        wav_path.write_bytes(wav_bytes)
        row["commentary"] = new_text
        row["duration"] = round(RenderSink.wav_duration(wav_bytes), 3)
        print(f"[更新] seq={seq} t={row['event_time']}s ({row['duration']}秒)")
        print(f"  旧: {old_text}")
        print(f"  新: {new_text}")

    write_jsonl(fillers_path, rows)
    print(f"\n{fillers_path} を更新しました（{len(edits)}件）。パス2を再実行してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
