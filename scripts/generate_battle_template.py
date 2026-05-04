#!/usr/bin/env python3
"""
ログファイルから対戦記録ひな型を生成するスクリプト。

ログに記録されている情報（戦況・検出メッセージ・技ログ・特性）をターンごとに出力し、
動画と照らし合わせて差異を記録できるひな型を作る。

Usage:
    python scripts/generate_battle_template.py logs/pipeline_YYYYMMDD_HHMMSS.log
    python scripts/generate_battle_template.py logs/pipeline_YYYYMMDD_HHMMSS.log -o records/my_record.md
"""

from __future__ import annotations

import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path


# ─── ログ行パターン ───────────────────────────────────────────────────────────

_TIME_RE      = re.compile(r'^(\d{2}:\d{2}:\d{2})')
_TURN_RE      = re.compile(r'\[ターン\] T(\d+) 開始')
_STATE_RE     = re.compile(
    r'\[状態\] 自分: (\S+) / ボール (\d+)匹生存 \| 相手: (\S+) / ボール (\d+)匹生存 \| OCR: (.+)$'
)
_PHASE_RE     = re.compile(r'\[フェーズ\] .+? \| イベント: (.+)$')
_JOKYO_RE     = re.compile(r'\[戦況\] T\d+\(G\d+\) 場\(自\)=(.+?) \| 場\(相\)=(.+?)(?:\s*\|.*)?$')
_BENCH_RE     = re.compile(r'\[戦況\] 控え\(自\)=(.+?) \| 控え\(相\)=(.+?)$')
# メッセージ系: switch_in / faint / opponent_switch_in / status 等
_MSG_RE       = re.compile(r'\[メッセージ\] (\S+): (.+?) / 「(.+?)」')
# 技ログ
_MOVE_LOG_RE  = re.compile(r'\[技ログ\] 検出: (T\d+:\S+)')
# 特性/道具
_ABILITY_RE   = re.compile(r'\[特性/道具\] (.+)$')

_UI_NOISE = {
    "たたかう", "にげる", "様子を見", "相手を見る", "もどる", "ヒシン", "B",
    "通信中", "通信中‥", "通信待機中", "patmos",
    "タイプ", "テラスタイプ", "状態", "ひんし", "たたかえない", "いまひとつ",
    "こうかなし", "ばつぐん", "日", "あいて", "相手の",
}
_SKIP_RE = re.compile(r'^(Lv\.?\d+|Lv50|\d+/\d+|\d+こうかあり|\d+/\d+/\d+|\d+)$')

# 特性/道具エリアのノイズ判定（これに該当するものはひな型に出力しない）
_ABILITY_NOISE_RE = re.compile(
    r"^(\{.*'[^']{0,3}'\s*\}|"   # 値が3文字以下の dict（例: {'opp': '1'}, {'player': 'つ'}）
    r"\{.*\'\d+\'\s*\}|"          # 値が数字のみの dict
    r"\{'[^']+': '[一-龯ぁ-ん]{1,2}'\})"  # 値が漢字・かな1〜2文字
)

def _is_ability_noise(ab: str) -> bool:
    """特性/道具欄に出力すべきでないノイズか判定する。"""
    if _ABILITY_NOISE_RE.match(ab):
        return True
    # dict の全valueが3文字以下なら除外
    try:
        import ast
        d = ast.literal_eval(ab)
        if isinstance(d, dict):
            return all(len(str(v)) <= 3 for v in d.values())
    except Exception:
        pass
    return False


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
    states: list[StateSnapshot]          = field(default_factory=list)
    # 戦況（場・控え）: 最初に検出したもの
    jokyo_player: str                    = ""
    jokyo_opponent: str                  = ""
    bench_player: str                    = ""
    bench_opponent: str                  = ""
    # 検出メッセージ（重複除去済み）
    messages: list[tuple[str, str, str]] = field(default_factory=list)  # (type, name, raw)
    # 技ログ
    moves: list[str]                     = field(default_factory=list)
    # 特性/道具
    ability_msgs: list[str]              = field(default_factory=list)

@dataclass
class ParseResult:
    log_stem: str
    battle_start_state: StateSnapshot | None
    turns: list[TurnInfo]
    battle_result: str
    battle_fields: dict = field(default_factory=dict)


# ─── ログ解析 ─────────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> ParseResult:
    lines = log_path.read_text(encoding="utf-8").splitlines()

    turns: list[TurnInfo] = []
    current_turn: TurnInfo | None = None
    battle_start_state: StateSnapshot | None = None
    battle_result = "不明"
    in_battle = False
    battle_fields: dict = {}
    current_game_turn = 0

    # 重複除去用セット（ターンをまたいで同一メッセージを何度も記録しない）
    seen_messages: set[str] = set()
    seen_moves: set[str] = set()
    seen_ability: set[str] = set()

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
                seen_messages.clear()
                seen_moves.clear()
                seen_ability.clear()
            elif "battle_end" in event:
                in_battle = False

        # ターン開始
        turn_m = _TURN_RE.search(line)
        if turn_m:
            current_game_turn = int(turn_m.group(1))
            current_turn = TurnInfo(turn_num=current_game_turn, time=time_str)
            turns.append(current_turn)
            continue

        # 戦況（場）
        jokyo_m = _JOKYO_RE.search(line)
        if jokyo_m:
            if current_game_turn not in battle_fields:
                p_names = _extract_field_names(jokyo_m.group(1))
                o_names = _extract_field_names(jokyo_m.group(2))
                if p_names or o_names:
                    battle_fields[current_game_turn] = (p_names, o_names)
            if current_turn and not current_turn.jokyo_player:
                current_turn.jokyo_player = jokyo_m.group(1).strip()
                current_turn.jokyo_opponent = jokyo_m.group(2).strip()

        # 戦況（控え）
        bench_m = _BENCH_RE.search(line)
        if bench_m and current_turn and not current_turn.bench_player:
            current_turn.bench_player = bench_m.group(1).strip()
            current_turn.bench_opponent = bench_m.group(2).strip()

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
                battle_start_state = snap
            elif current_turn is not None:
                current_turn.states.append(snap)

        # 検出メッセージ（switch_in / faint / opponent_switch_in / status 等）
        msg_m = _MSG_RE.search(line)
        if msg_m and current_turn is not None:
            msg_type = msg_m.group(1)
            msg_name = msg_m.group(2).strip()
            msg_raw  = msg_m.group(3).strip()
            key = f"{msg_type}:{msg_name}"
            if key not in seen_messages:
                seen_messages.add(key)
                current_turn.messages.append((msg_type, msg_name, msg_raw))

        # 技ログ
        move_m = _MOVE_LOG_RE.search(line)
        if move_m and current_turn is not None:
            move_str = move_m.group(1)
            if move_str not in seen_moves:
                seen_moves.add(move_str)
                current_turn.moves.append(move_str)

        # 特性/道具
        ability_m = _ABILITY_RE.search(line)
        if ability_m and current_turn is not None:
            ab_str = ability_m.group(1).strip()
            if ab_str not in seen_ability:
                seen_ability.add(ab_str)
                current_turn.ability_msgs.append(ab_str)

        # 勝敗判定
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


# ─── ヘルパー ─────────────────────────────────────────────────────────────────

_JOKYO_NOISE = {"通信待機中", "情報収集中"}

def _extract_field_names(field_str: str) -> list[str]:
    names = []
    for part in field_str.split(' / '):
        part = part.strip()
        if not part or part in _JOKYO_NOISE:
            continue
        name = part.split(' ')[0].split('(')[0]
        if name and name not in _JOKYO_NOISE:
            names.append(name)
    return names


def _clean_jokyo(jokyo: str) -> str:
    """戦況文字列から通信待機中等のノイズエントリを除去する。"""
    parts = [p.strip() for p in jokyo.split(' / ')]
    cleaned = [p for p in parts if p and p.split(' ')[0] not in _JOKYO_NOISE]
    return ' / '.join(cleaned) if cleaned else '―'


def _clean_bench(bench: str) -> str:
    """控え文字列から通信待機中等のノイズエントリを除去する。"""
    parts = [p.strip() for p in bench.split(' / ')]
    cleaned = [p for p in parts if p and p.split('(')[0].strip() not in _JOKYO_NOISE]
    return ' / '.join(cleaned) if cleaned else 'なし'


def _clean_ocr(raw: str) -> str:
    parts = [p.strip() for p in raw.split("/")]
    cleaned = []
    for p in parts:
        if not p or p in _UI_NOISE:
            continue
        if _SKIP_RE.match(p):
            continue
        if any(kw in p for kw in ["たたかう", "にげる", "様子", "相手を見", "もどる", "通信中"]):
            continue
        cleaned.append(p)
    return " / ".join(cleaned)


_MSG_TYPE_LABEL = {
    "switch_in":          "自:繰り出し",
    "opponent_switch_in": "相:繰り出し",
    "faint":              "気絶",
    "switch_out":         "交代",
    "status":             "状態異常",
}


# ─── ひな型生成 ───────────────────────────────────────────────────────────────

def format_template(result: ParseResult) -> str:
    date_str = result.log_stem.replace("pipeline_", "")

    out: list[str] = [
        f"対戦日時：{date_str}",
        f"結果：{result.battle_result}",
        "",
    ]

    if result.battle_start_state:
        s = result.battle_start_state
        cleaned = _clean_ocr(s.ocr_raw)
        if cleaned:
            out += [f"【開始時OCR】{cleaned}", ""]

    prev_p_balls: int | str = result.battle_start_state.player_balls if result.battle_start_state else "?"
    prev_o_balls: int | str = result.battle_start_state.opponent_balls if result.battle_start_state else "?"

    for t in result.turns:
        if t.states:
            last = t.states[-1]
            p_balls  = last.player_balls
            o_balls  = last.opponent_balls
            p_status = t.states[0].player_status
            o_status = t.states[0].opponent_status
        else:
            p_balls  = prev_p_balls
            o_balls  = prev_o_balls
            p_status = "?"
            o_status = "?"

        prev_p_balls = p_balls
        prev_o_balls = o_balls

        out += [
            "=" * 60,
            f"T{t.turn_num}  [{t.time}]  自:{p_balls}匹({p_status})  相:{o_balls}匹({o_status})",
            "",
        ]

        # ── 検出：戦況 ──────────────────────────────────────────
        out.append("【戦況（検出）】")
        if t.jokyo_player or t.jokyo_opponent:
            out.append(f"  場(自): {_clean_jokyo(t.jokyo_player) if t.jokyo_player else '―'}")
            out.append(f"  場(相): {_clean_jokyo(t.jokyo_opponent) if t.jokyo_opponent else '―'}")
        else:
            out.append("  場: 情報収集中")
        if t.bench_player or t.bench_opponent:
            out.append(f"  控(自): {_clean_bench(t.bench_player) if t.bench_player else '―'}")
            out.append(f"  控(相): {_clean_bench(t.bench_opponent) if t.bench_opponent else '―'}")
        out.append("")

        # ── 検出：メッセージ ────────────────────────────────────
        out.append("【検出メッセージ】")
        if t.messages:
            for msg_type, name, raw in t.messages:
                label = _MSG_TYPE_LABEL.get(msg_type, msg_type)
                out.append(f"  [{label}] {name}  ←「{raw}」")
        else:
            out.append("  （なし）")
        out.append("")

        # ── 検出：技ログ ────────────────────────────────────────
        out.append("【技ログ（検出）】")
        if t.moves:
            for mv in t.moves:
                out.append(f"  {mv}")
        else:
            out.append("  （なし）")
        out.append("")

        # ── 検出：特性/道具 ─────────────────────────────────────
        valid_ab = [ab for ab in t.ability_msgs if not _is_ability_noise(ab)]
        if valid_ab:
            out.append("【特性/道具（検出）】")
            for ab in valid_ab:
                out.append(f"  {ab}")
            out.append("")

        # ── 実際の内容（手書き記入欄） ───────────────────────────
        out += [
            "【実際の内容（動画確認）】",
            "  場(自): ",
            "  場(相): ",
            "  控(自): ",
            "  控(相): ",
            "  行動(自): ",
            "  行動(相): ",
            "  差異メモ: ",
            "",
        ]

    out += [
        "=" * 60,
        "（記録終わり）",
    ]

    return "\n".join(out)


# ─── エントリポイント ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ログから対戦記録ひな型を生成")
    parser.add_argument("log_file", help="pipeline_YYYYMMDD_HHMMSS.log のパス")
    parser.add_argument("-o", "--output", help="出力ファイルパス（省略時は records/pipeline_STEM.md）")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"エラー: {log_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    result = parse_log(log_path)
    output = format_template(result)

    if args.output:
        out_path = Path(args.output)
    else:
        records_dir = log_path.parent.parent / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        out_path = records_dir / f"{log_path.stem}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"出力: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
