"""パス1の出力（manifest/timeline/states）からレビュー用チェックリストMarkdownを生成する。

docs/manual/pass1-verification-checklist.md のA/B項目に対応する内容を、
**ターンごとにまとめて**時系列順に書き出す。動画を頭から見ながら、後戻りせずに
上から埋めていける構成。判定は「判定(OK/NG): 」に直接 OK/NG を書き込む方式
（Markdownチェックボックスはプレビューでのクリックが環境依存で不安定なため不使用）。

使い方:
    python3 scripts/generate_review_checklist.py renders/<動画名>

出力: renders/<動画名>/review_checklist.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_FIELD_EFFECT_KEYWORDS = [
    "おいかぜ", "あまごい", "にほんばれ", "すなあらし", "ゆきげしき", "ゆきふらし",
    "あめふらし", "ひでり", "すなおこし", "サンドストリーム",
    "リフレクター", "ひかりのかべ", "オーロラベール", "トリックルーム",
]

_TURN_PREFIX_RE = re.compile(r"^T(\d+):(.*)$")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _judgement(label: str) -> str:
    return f"判定(OK/NG): \n{label}（NGの場合のみ記入）: \n"


def _build_segments(states: list[dict]) -> list[dict]:
    """states.jsonlをturnごとに区切り、[start, end)の時間境界を持つセグメント列を作る。"""
    segments: list[dict] = []
    by_turn: dict[int, dict] = {}
    order: list[int] = []
    for s in states:
        turn = s.get("turn")
        if turn not in by_turn:
            by_turn[turn] = {"turn": turn, "start": s.get("time"), "snapshots": []}
            order.append(turn)
        by_turn[turn]["snapshots"].append(s)
    for i, turn in enumerate(order):
        seg = by_turn[turn]
        seg["end"] = by_turn[order[i + 1]]["start"] if i + 1 < len(order) else None
        segments.append(seg)
    return segments


def _find_segment(segments: list[dict], t: float | None) -> dict | None:
    if t is None or not segments:
        return None
    for seg in segments:
        if seg["start"] is None:
            continue
        if t >= seg["start"] and (seg["end"] is None or t < seg["end"]):
            return seg
    return segments[0] if t < segments[0]["start"] else segments[-1]


def _turn_state_summary(snapshots: list[dict]) -> tuple[dict, list[dict]]:
    """そのターン内でのポケモン初登場・HP/状態異常の変化点だけを抽出する（重複排除）。"""
    roster = {"player": [], "opponent": []}
    changes: list[dict] = []
    last: dict[tuple[str, str], tuple] = {}
    for s in snapshots:
        for side in ("player", "opponent"):
            for p in s.get(side, []):
                name = p.get("name")
                if not name:
                    continue
                if name not in roster[side]:
                    roster[side].append(name)
                key = (side, name)
                val = (p.get("hp_pct"), p.get("hp_text"), p.get("status"))
                if last.get(key) != val:
                    changes.append({
                        "time": s.get("time"), "side": side, "name": name,
                        "hp_pct": val[0], "hp_text": val[1], "status": val[2],
                    })
                    last[key] = val
    return roster, changes


def build_markdown(render_dir: Path) -> str:
    manifest = load_jsonl(render_dir / "manifest.jsonl")
    timeline = load_jsonl(render_dir / "timeline.jsonl")
    states = load_jsonl(render_dir / "states.jsonl")

    render_info = {}
    info_path = render_dir / "render_info.json"
    if info_path.exists():
        render_info = json.loads(info_path.read_text(encoding="utf-8"))

    segments = _build_segments(states)

    lines: list[str] = []
    lines.append(f"# パス1 検証チェックリスト — {render_dir.name}")
    lines.append("")
    lines.append(f"- 生成日時: {datetime.now().isoformat(timespec='seconds')}")
    if render_info.get("video"):
        lines.append(f"- 元動画: {render_info['video']}")
    if render_info.get("created_at"):
        lines.append(f"- パス1実行日時: {render_info['created_at']}")
    lines.append(f"- 参照元: `{render_dir}/manifest.jsonl` / `timeline.jsonl` / `states.jsonl`")
    lines.append("")
    lines.append("判定基準の詳細は `docs/manual/pass1-verification-checklist.md` を参照。"
                 "動画を頭から見ながら、ターンごとに上から埋めていく想定。")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not segments:
        lines.append("（states.jsonlが空のため、ターン分割できません。manifest.jsonlの内容のみ表示）")
        lines.append("")
        for m in manifest:
            lines.append(f"### #{m.get('seq')} （{m.get('event_time')}s / {m.get('event_type')}）")
            lines.append("")
            lines.append(f"> {m.get('commentary', '')}")
            lines.append("")
            lines.append(_judgement("望ましい実況内容"))
        return "\n".join(lines) + "\n"

    for seg in segments:
        turn = seg["turn"]
        start, end = seg["start"], seg["end"]
        label = "（試合開始前）" if turn == 0 else ""
        end_label = f"{end}s" if end is not None else "試合終了"
        lines.append(f"## ターン{turn} {label}（{start}s 〜 {end_label}）")
        lines.append("")

        roster, changes = _turn_state_summary(seg["snapshots"])
        lines.append("**場のポケモン**")
        lines.append(f"- 自分: {', '.join(roster['player']) or 'なし'}")
        lines.append(f"- 相手: {', '.join(roster['opponent']) or 'なし'}")
        lines.append("")

        if changes:
            lines.append("**HP/状態異常の変化**")
            for c in changes:
                status = f" [{c['status']}]" if c["status"] else ""
                lines.append(f"- {c['time']}s [{c['side']}] {c['name']}: "
                             f"{c['hp_pct']}% / {c['hp_text']}{status}")
            lines.append("")

        # このターンの技（timelineの "Tn:" 表記を優先し、無ければ時間帯で判定）
        turn_moves = []
        for t in timeline:
            if t.get("kind") != "move":
                continue
            text = t.get("text") or ""
            m = _TURN_PREFIX_RE.match(text)
            if m and int(m.group(1)) == turn:
                turn_moves.append(t)
            elif not m:
                found = _find_segment(segments, t.get("time"))
                if found is seg:
                    turn_moves.append(t)
        if turn_moves:
            lines.append("**技**")
            for t in turn_moves:
                side = t.get("side") or "?"
                mark = "⚠ " if any(k in (t.get("text") or "") for k in _FIELD_EFFECT_KEYWORDS) else ""
                lines.append(f"- {mark}{t.get('time')}s [{side}] {t.get('text')}")
            lines.append("（⚠=天候/壁/トリックルーム/おいかぜ等の場の効果。残りターン数も含めて確認）")
            lines.append("")

        # このターンで起きた switch/faint/battle_end
        events_in_turn = [m for m in manifest
                          if m.get("event_type") in ("switch", "faint", "battle_end")
                          and _find_segment(segments, m.get("event_time")) is seg]
        if events_in_turn:
            lines.append("**交代・気絶・試合終了**")
            for m in events_in_turn:
                lines.append(f"- {m.get('event_time')}s [{m.get('event_type')}]: "
                             f"{m.get('commentary', '')[:80]}")
            lines.append("")

        lines.append("**試合内容の判定**")
        lines.append("")
        lines.append(_judgement("正しい内容"))

        # このターンの実況テキスト（イベント別・個別に判定）
        commentary_in_turn = [m for m in manifest
                              if _find_segment(segments, m.get("event_time")) is seg]
        if commentary_in_turn:
            lines.append("**実況内容**")
            lines.append("")
            for m in commentary_in_turn:
                lines.append(f"#### {m.get('event_time')}s / {m.get('event_type')} (#{m.get('seq')})")
                lines.append("")
                lines.append(f"> {m.get('commentary', '')}")
                lines.append("")
                lines.append(_judgement("望ましい実況内容"))

        lines.append("---")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", help="パス1の素材ディレクトリ（renders/<動画名>）")
    parser.add_argument("--out", default=None,
                        help="出力先（省略時は <render_dir>/review_checklist.md）")
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    if not render_dir.exists():
        print(f"エラー: {render_dir} が存在しません", file=sys.stderr)
        return 1

    md = build_markdown(render_dir)
    out_path = Path(args.out) if args.out else render_dir / "review_checklist.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"生成しました: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
