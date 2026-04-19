"""
OCRエリアログから「変化があった箇所」だけを抽出するスクリプト。
注釈ブロックは除外し、OCR値が前エントリから変わったエントリのみ出力する。
"""

import re
import sys
from pathlib import Path

FIELD_RE = re.compile(r'(?:>>>)?\s+([\w\s]+?)\s+:\s+(.*)')
TIMESTAMP_RE = re.compile(r'^\[(\d+:\d+\.\d+)\]')

# 常にノイズとみなすフィールド（HP・相手特性）
NOISE_FIELDS = {'hp_opp0', 'hp_opp1', 'hp_plr0', 'hp_plr1', 'ability_opp'}

# 「（なし）↔実値」の遷移だけ意味ある変化とみなすフィールド
NASHI_ONLY_FIELDS = {'name_opp0', 'name_opp1', 'name_plr0', 'name_plr1'}
NASHI = '（なし）'


def parse_log(path: Path):
    """ログを [{timestamp, fields: dict, raw_lines}] のリストに変換"""
    entries = []
    current = None

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    in_annotation = False

    for line in lines:
        stripped = line.rstrip()

        # 注釈ブロックのスキップ
        if '┌─ 注釈' in stripped:
            in_annotation = True
            if current:
                current['raw_lines'].append(stripped)
            continue
        if '└──' in stripped:
            in_annotation = False
            if current:
                current['raw_lines'].append(stripped)
            continue
        if in_annotation:
            if current:
                current['raw_lines'].append(stripped)
            continue

        # タイムスタンプ行
        m = TIMESTAMP_RE.match(stripped)
        if m:
            if current:
                entries.append(current)
            current = {'timestamp': m.group(1), 'fields': {}, 'raw_lines': [stripped]}
            continue

        if current is None:
            continue

        # フィールド行
        fm = FIELD_RE.match(stripped)
        if fm:
            key = fm.group(1).strip()
            val = fm.group(2).strip()
            current['fields'][key] = val
            current['raw_lines'].append(stripped)
        elif stripped:
            current['raw_lines'].append(stripped)

    if current:
        entries.append(current)

    return entries


def fields_changed(prev: dict, curr: dict):
    """変化したフィールド名のリストを返す"""
    changed = []
    all_keys = set(prev) | set(curr)
    for k in all_keys:
        if prev.get(k) != curr.get(k):
            changed.append(k)
    return changed


def format_entry(entry: dict, changed_keys) -> str:
    lines = [f"[{entry['timestamp']}]  ← 変化: {', '.join(changed_keys)}"]
    for line in entry['raw_lines']:
        if TIMESTAMP_RE.match(line):
            continue  # タイムスタンプは上で出したので skip
        if '┌─ 注釈' in line or '└──' in line or line.strip().startswith('│'):
            continue  # 注釈は出力しない
        lines.append(line)
    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('output_file', nargs='?')
    parser.add_argument('--from', dest='from_time', default=None,
                        help='この時刻以降のみ出力 (例: 01:09.00)')
    args = parser.parse_args()

    src = Path(args.log_file)
    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)

    out = Path(args.output_file) if args.output_file else src.with_name(src.stem + '_changes.txt')
    from_time = args.from_time

    entries = parse_log(src)
    print(f"エントリ総数: {len(entries)}")

    def ts_to_sec(ts):
        m, s = ts.split(':')
        return int(m) * 60 + float(s)

    from_sec = ts_to_sec(from_time) if from_time else 0.0

    output_blocks = []
    prev_fields: dict = {}

    for i, entry in enumerate(entries):
        curr_fields = entry['fields']
        if i == 0:
            # 最初のエントリは常に出力
            changed = list(curr_fields.keys())
        else:
            changed = fields_changed(prev_fields, curr_fields)

        # ノイズフィールドのみの変化はスキップ
        def is_meaningful(k):
            if k in NOISE_FIELDS:
                return False
            if k in NASHI_ONLY_FIELDS:
                # （なし）↔実値 の遷移だけ意味あり
                pv, cv = prev_fields.get(k, NASHI), curr_fields.get(k, NASHI)
                return (pv == NASHI) != (cv == NASHI)
            return True

        meaningful = [k for k in changed if is_meaningful(k)]
        if (i == 0 or meaningful) and ts_to_sec(entry['timestamp']) >= from_sec:
            output_blocks.append(format_entry(entry, changed))

        prev_fields = dict(curr_fields)

    result = '\n\n'.join(output_blocks)
    out.write_text(result, encoding='utf-8')
    print(f"変化エントリ数: {len(output_blocks)} / {len(entries)}")
    print(f"出力: {out}")


if __name__ == '__main__':
    main()
