#!/usr/bin/env python3
"""
ログファイルから対戦記録ひな型を生成するスクリプト。
from __future__ import annotations  # noqa: E402 (must be first in module after docstring)

ログに記録されている情報（ターン数・ボール数・OCRテキスト・HP変化）を
あらかじめ埋め込んだひな型を出力する。動画を見ながら修正・補完する用途。

Usage:
    python scripts/generate_battle_template.py logs/pipeline_20260328_172333.log
    python scripts/generate_battle_template.py logs/pipeline_20260328_172333.log -o records/20260328_172333.txt
"""

from __future__ import annotations

import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── ログ行パターン ───────────────────────────────────────────────────────────

_TIME_RE   = re.compile(r'^(\d{2}:\d{2}:\d{2})')
_TURN_RE   = re.compile(r'\[ターン\] T(\d+) 開始')
_STATE_RE  = re.compile(
    r'\[状態\] 自分: (\S+) / ボール (\d+)匹生存 \| 相手: (\S+) / ボール (\d+)匹生存 \| OCR: (.+)$'
)
_PHASE_RE  = re.compile(r'\[フェーズ\] .+? \| イベント: (.+)$')
# [戦況] T{N}(G{M}) 場(自)=... | 場(相)=... の行
_JOKYO_RE  = re.compile(r'\[戦況\] T\d+\(G\d+\) 場\(自\)=(.+?) \| 場\(相\)=(.+?)(?:\s*\|.*)?$')

# UIノイズとして除去するキーワード（OCRで拾われやすいUI要素・ポケモン以外の単語）
_UI_NOISE = {
    "たたかう", "にげる", "様子を見", "相手を見る", "もどる", "ヒシン", "B",
    "通信中", "通信中‥", "patmos",
    "タイプ", "テラスタイプ", "状態", "ひんし", "たたかえない", "いまひとつ",
    "こうかなし", "ばつぐん", "日", "あいて", "相手の",
}
_SKIP_RE  = re.compile(r'^(Lv\.?\d+|Lv50|\d+/\d+|\d+こうかあり|\d+/\d+/\d+|\d+)$')


# ─── データ構造 ───────────────────────────────────────────────────────────────

@dataclass
class StateSnapshot:
    time: str
    player_status: str
    player_balls: int
    opponent_status: str
    opponent_balls: int
    ocr_raw: str

@dataclass
class TurnInfo:
    turn_num: int
    time: str
    states: list[StateSnapshot] = field(default_factory=list)  # ターン中に記録された状態

@dataclass
class ParseResult:
    log_stem: str
    battle_start_state: StateSnapshot | None
    turns: list[TurnInfo]
    battle_result: str  # "勝利（相手降参）" / "敗北（自分降参）" / "不明"
    battle_fields: dict = field(default_factory=dict)  # game_turn → (player_names, opponent_names)


# ─── ログ解析 ─────────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> ParseResult:
    lines = log_path.read_text(encoding="utf-8").splitlines()

    turns: list[TurnInfo] = []
    current_turn: TurnInfo | None = None
    battle_start_state: StateSnapshot | None = None
    battle_result = "不明"
    in_battle = False
    battle_fields: dict = {}   # game_turn → (player_names, opponent_names)
    current_game_turn = 0      # [ターン] T{N} 開始 で更新

    for line in lines:
        time_m = _TIME_RE.match(line)
        time_str = time_m.group(1) if time_m else ""

        # フェーズイベント
        phase_m = _PHASE_RE.search(line)
        if phase_m:
            event = phase_m.group(1).strip()
            if event == "battle_start":
                in_battle = True
                current_turn = None
            elif "battle_end" in event:
                in_battle = False

        # ターン開始
        turn_m = _TURN_RE.search(line)
        if turn_m:
            current_game_turn = int(turn_m.group(1))
            current_turn = TurnInfo(turn_num=current_game_turn, time=time_str)
            turns.append(current_turn)
            continue

        # [戦況] 場の情報（ゲームターンごとの最初のスナップを使用）
        jokyo_m = _JOKYO_RE.search(line)
        if jokyo_m and current_game_turn not in battle_fields:
            player_names = _extract_field_names(jokyo_m.group(1))
            opponent_names = _extract_field_names(jokyo_m.group(2))
            if player_names or opponent_names:
                battle_fields[current_game_turn] = (player_names, opponent_names)

        # 状態ログ
        state_m = _STATE_RE.search(line)
        if state_m:
            p_status, p_balls, o_status, o_balls, ocr_raw = state_m.groups()
            snap = StateSnapshot(
                time=time_str,
                player_status=p_status,
                player_balls=int(p_balls),
                opponent_status=o_status,
                opponent_balls=int(o_balls),
                ocr_raw=ocr_raw.strip(),
            )
            if current_turn is None and in_battle:
                # battle_start 直後（T1前）の状態スナップ
                battle_start_state = snap
            elif current_turn is not None:
                current_turn.states.append(snap)

        # 勝敗判定（OCR内テキストも含めて検索）
        if ("降参が" in line and "選ばれました" in line) or "降参が / 選ばれました" in line:
            battle_result = "勝利（相手降参）"
        if "負け" in line or "まけ" in line:
            battle_result = "敗北"

    return ParseResult(
        log_stem=log_path.stem,
        battle_start_state=battle_start_state,
        turns=turns,
        battle_result=battle_result,
        battle_fields=battle_fields,
    )


# ─── OCR テキスト整形 ─────────────────────────────────────────────────────────

def _extract_field_names(field_str: str) -> list[str]:
    """「リキキリン HP:97/211 技=[...] / バドレックス HP:...」からポケモン名だけ取り出す。"""
    names = []
    for part in field_str.split(' / '):
        part = part.strip()
        if not part or part == '情報収集中':
            continue
        # スペース・HP:・(状態) の前までが名前
        name = part.split(' ')[0].split('(')[0]
        if name:
            names.append(name)
    return names


def _clean_ocr(raw: str) -> str:
    """OCRノイズを除去してポケモン名候補だけ残す。"""
    parts = [p.strip() for p in raw.split("/")]
    cleaned = []
    for p in parts:
        if not p:
            continue
        if p in _UI_NOISE:
            continue
        if _SKIP_RE.match(p):
            continue
        if any(kw in p for kw in ["たたかう", "にげる", "様子", "相手を見", "もどる", "通信中"]):
            continue
        cleaned.append(p)
    return " / ".join(cleaned)


def _extract_hp_changes(states: list[StateSnapshot]) -> list[str]:
    """状態スナップからHP変化を抽出する。
    OCR raw 文字列を直接 regex スキャンし、「名前 / HP値」パターンを抽出する。
    """
    hp_re = re.compile(r'(\S{2,12})\s*/\s*(\d+/\d+)')
    results = []
    for snap in states:
        for m in hp_re.finditer(snap.ocr_raw):
            name = m.group(1).strip()
            hp = m.group(2)
            denom = int(hp.split("/")[1])
            if denom < 50:
                continue
            if name in _UI_NOISE or _SKIP_RE.match(name) or not name:
                continue
            results.append(f"{name} {hp}")
    return list(dict.fromkeys(results))  # 重複除去・順序保持


# ─── ひな型生成 ───────────────────────────────────────────────────────────────

def format_template(result: ParseResult) -> str:
    date_str = result.log_stem.replace("pipeline_", "")

    lines: list[str] = [
        f"対戦日時：{date_str}",
        f"結果：{result.battle_result}",
        "",
    ]

    # 開始時の参考情報
    if result.battle_start_state:
        s = result.battle_start_state
        cleaned = _clean_ocr(s.ocr_raw)
        if cleaned:
            lines += [
                f"【開始時OCR】{cleaned}",
                "",
            ]

    # ターンごとのひな型（ボール数は前ターンから引き継ぎ）
    prev_p_balls: int | str = result.battle_start_state.player_balls if result.battle_start_state else "?"
    prev_o_balls: int | str = result.battle_start_state.opponent_balls if result.battle_start_state else "?"

    for t in result.turns:
        # ボール数: ターン中に最後に確認できた状態スナップを優先、なければ前ターン引き継ぎ
        if t.states:
            # 最後のスナップが最も新しいボール数
            last = t.states[-1]
            p_balls = last.player_balls
            o_balls = last.opponent_balls
            p_status = t.states[0].player_status
            o_status = t.states[0].opponent_status
        else:
            p_balls = prev_p_balls
            o_balls = prev_o_balls
            p_status = "?"
            o_status = "?"

        prev_p_balls = p_balls
        prev_o_balls = o_balls

        # HP変化リスト
        hp_changes = _extract_hp_changes(t.states)

        # OCR参考テキスト（最初のスナップだけ）
        ocr_ref = ""
        if t.states:
            ocr_ref = _clean_ocr(t.states[0].ocr_raw)

        # こちら・相手（[戦況]ログから取得。なければ前ターンの状態を引き継ぎ）
        fields = result.battle_fields.get(t.turn_num) or result.battle_fields.get(t.turn_num - 1)
        if fields:
            player_str = "、".join(fields[0]) if fields[0] else "？"
            opponent_str = "、".join(fields[1]) if fields[1] else "？"
        else:
            player_str = "？"
            opponent_str = "？"

        lines += [
            "---",
            "",
            f"T{t.turn_num}  [{t.time}]  自:{p_balls}匹({p_status})  相:{o_balls}匹({o_status})",
            f"こちら：{player_str}",
            f"相手：{opponent_str}",
            "",
        ]

        if ocr_ref:
            lines += [f"【OCR参考】{ocr_ref}", ""]

        if hp_changes:
            lines += [f"【HP検出】{' | '.join(hp_changes)}", ""]

        lines += [
            "特性・アイテム発動：",
            "",
            "行動：",
            "",
        ]

    lines += [
        "---",
        "",
        "（記録終わり）",
    ]

    return "\n".join(lines)


# ─── エントリポイント ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ログから対戦記録ひな型を生成")
    parser.add_argument("log_file", help="pipeline_YYYYMMDD_HHMMSS.log のパス")
    parser.add_argument("-o", "--output", help="出力ファイルパス（省略時は標準出力）")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"エラー: {log_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    result = parse_log(log_path)
    output = format_template(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"出力: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
