"""パス1無課金検証の自動一次スクリーニング（2026-08-11・目視負荷削減のため新設）。

これまでの検証（2本・NG5件）で見つかったNGパターンは、いずれも
manifest.jsonl/timeline.jsonl/states.jsonlの構造だけから機械的に検出できる
「怪しい兆候」を伴っていた。これを自動チェックし、疑いのある箇所だけに
レビュー時の目視を集中させることで、122本を全部フルで目視する負荷を減らす。

検出パターン（docs/manual/pass1-verification-ng-findings.md の実例に対応）:
  1. move_log空での技実況     … NG#3（こごえるかぜ捏造）型
  2. battle_result未検出でのbattle_end … NG#2（えらさんの勝利）型
  3. 絵文字ブロック混入        … 既知の字幕豆腐化バグ（B3）
  4. 生の保留・困惑キーワード残存 … AIグリッチ差し替え漏れ（B2）
  5. 選出画面限定登場ポケモン  … NG#5（ランクルス→オオニューラ）型
  6. ひんし判定なのにHP0%系を一度も観測していない … NG#4型
  7. 変化技のダメージ誤認疑い  … NG恒久対策フェーズ1・施策B（技効果ヒントRAG）の
                                  再発検出（2026-08-14追加）
  8. 陣営矛盾疑い              … NG恒久対策フェーズ1・施策C（is_opponentクロス
                                  チェック）の再発検出（2026-08-14追加・簡易検出）

⚠️これは「疑い」の一次検出であり確定判定ではない。フラグが立った箇所を
優先して目視確認すること。フラグゼロ＝無罪ではなく「既知パターンには
該当しなかった」という意味（未知のNGパターンは引き続き目視でしか拾えない）。

使い方:
    python3 scripts/screen_pass1.py renders/<動画名>

出力: renders/<動画名>/screening_report.md ＋ コンソールに要約
終了コード: 0=フラグなし / 1=要確認フラグあり
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# pipeline.py の _GLITCH_CAUSE_KEYWORDS と同じ内容（重い依存を避けるため複製・
# 要同期。ソースは src/pipeline.py の _GLITCH_CAUSE_KEYWORDS）
_GLITCH_CAUSE_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("矛盾", "ちぐはぐ"), "データがちぐはぐさん"),
    (("見えにく", "読み取れ"), "画面がチカチカしてた"),
    (("確定できて", "お待ち"), "情報がまだ揃ってない"),
    (("モヤモヤ", "教えてほし", "教えてもらえ", "実況できな"), "ナゾのノイズ"),
    (("了解しました", "担当させていただきます", "性格・口調の確認", "実況時の重要ルール"),
     "指示書を読みすぎちゃった"),
]

# 差し替え済み「AIグリッチ」定型文の目印（これらを含む＝既に対策済みなのでNGではない）
_GLITCH_TEMPLATE_MARKERS = [
    "くれぴの目がちょっとバグっちゃった",
    "データがぐるぐるしてる",
    "エラー発生〜！原因は",
]

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
_ROSTER_ENTRY_RE = re.compile(r"^(?:場|控え)\s*:\s*")
_NAME_RE = re.compile(r"^([^\s(（]+)")
_HINSHI_RE = re.compile(r"([^\s/()（）]+)\(ひんし\)")
_NON_NAME_TOKENS = {"なし", "情報収集中", "不明"}


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


def _extract_names(roster_str: str | None) -> set[str]:
    """context.player/opponent の表示文字列からポケモン名だけを抜き出す。"""
    if not roster_str:
        return set()
    names = set()
    for part in roster_str.split(" / "):
        part = _ROSTER_ENTRY_RE.sub("", part.strip())
        m = _NAME_RE.match(part)
        if not m:
            continue
        name = m.group(1)
        if name and name not in _NON_NAME_TOKENS:
            names.add(name)
    return names


def check_empty_move_log(manifest: list[dict]) -> list[dict]:
    """move系イベントなのにcontext.move_logが空 → 捏造疑い（NG#3型）。"""
    flags = []
    for m in manifest:
        if m.get("event_type") not in ("move_used", "move_single"):
            continue
        ctx = m.get("context") or {}
        if not ctx.get("move_log"):
            flags.append({
                "kind": "move_log空での技実況",
                "time": m.get("event_time"),
                "seq": m.get("seq"),
                "detail": f"commentary=「{m.get('commentary', '')}」",
            })
    return flags


def check_missing_battle_result(manifest: list[dict]) -> list[dict]:
    """battle_endなのにcontext.battle_resultが無い → 勝敗未検出のまま実況（NG#2型）。"""
    flags = []
    for m in manifest:
        if m.get("event_type") != "battle_end":
            continue
        ctx = m.get("context") or {}
        if not ctx.get("battle_result"):
            flags.append({
                "kind": "battle_result未検出でのbattle_end実況",
                "time": m.get("event_time"),
                "seq": m.get("seq"),
                "detail": f"commentary=「{m.get('commentary', '')}」（勝者名の捏造がないか要確認）",
            })
    return flags


def check_emoji(manifest: list[dict]) -> list[dict]:
    """絵文字ブロック（U+1F300-1FAFF）混入 → 字幕が豆腐化する既知バグ。"""
    flags = []
    for m in manifest:
        text = m.get("commentary", "") or ""
        found = _EMOJI_RE.findall(text)
        if found:
            flags.append({
                "kind": "絵文字混入",
                "time": m.get("event_time"),
                "seq": m.get("seq"),
                "detail": f"文字={''.join(sorted(set(found)))} commentary=「{text}」",
            })
    return flags


def check_leaked_glitch_keywords(manifest: list[dict]) -> tuple[list[dict], int]:
    """AIグリッチ差し替え漏れ（生の保留・困惑キーワードが残存）を検出。

    差し替え済みテンプレート文はキーワードを自己参照的に含むため誤検出になる
    （project_video_first_policyメモ既知の注意点）。テンプレート目印があれば
    「既に対策済み」として除外し、件数だけカウントする。
    """
    flags = []
    replaced_count = 0
    for m in manifest:
        text = m.get("commentary", "") or ""
        if any(marker in text for marker in _GLITCH_TEMPLATE_MARKERS):
            replaced_count += 1
            continue
        for keywords, _cause in _GLITCH_CAUSE_KEYWORDS:
            if any(kw in text for kw in keywords):
                flags.append({
                    "kind": "生の保留・困惑応答が残存（グリッチ差し替え漏れ）",
                    "time": m.get("event_time"),
                    "seq": m.get("seq"),
                    "detail": f"該当キーワード群={keywords} commentary=「{text}」"
                              "→ _GLITCH_CAUSE_KEYWORDSに未収録の新パターンの可能性",
                })
                break
    return flags, replaced_count


def check_selection_screen_only_names(states: list[dict], manifest: list[dict]) -> list[dict]:
    """turn0（選出画面）にだけ登場しturn1以降二度と出てこない名前 → 誤認識疑い（NG#5型）。"""
    flags = []
    turn0_names: dict[str, set[str]] = {"player": set(), "opponent": set()}
    later_names: dict[str, set[str]] = {"player": set(), "opponent": set()}
    for s in states:
        turn = s.get("turn")
        for side in ("player", "opponent"):
            names = {p.get("name") for p in s.get(side, []) if p.get("name")}
            if turn == 0:
                turn0_names[side] |= names
            else:
                later_names[side] |= names

    # 気絶・交代済みポケモン名（誤検出除外用）: manifestのcontext文字列から拾う
    known_fainted = set()
    for m in manifest:
        ctx = m.get("context") or {}
        for key in ("player", "opponent"):
            known_fainted |= set(_HINSHI_RE.findall(ctx.get(key) or ""))

    for side in ("player", "opponent"):
        only_in_turn0 = turn0_names[side] - later_names[side] - known_fainted
        for name in sorted(only_in_turn0):
            flags.append({
                "kind": "選出画面限定登場ポケモン（誤認識疑い）",
                "time": None,
                "seq": None,
                "detail": f"[{side}] 「{name}」はturn0（選出画面）にのみ登場し、"
                          "turn1以降・気絶記録のどちらにも出てこない",
            })
    return flags


def check_missing_hp_zero(states: list[dict], manifest: list[dict]) -> list[dict]:
    """context上「(ひんし)」判定済みなのに、states.jsonlでHP0%付近を一度も観測していない（NG#4型）。"""
    fainted_names = set()
    for m in manifest:
        ctx = m.get("context") or {}
        for key in ("player", "opponent"):
            fainted_names |= set(_HINSHI_RE.findall(ctx.get(key) or ""))

    min_hp: dict[str, int] = {}
    for s in states:
        for side in ("player", "opponent"):
            for p in s.get(side, []):
                name = p.get("name")
                hp = p.get("hp_pct")
                if not name or hp is None:
                    continue
                if name not in min_hp or hp < min_hp[name]:
                    min_hp[name] = hp

    flags = []
    for name in sorted(fainted_names):
        observed = min_hp.get(name)
        if observed is None or observed > 5:
            flags.append({
                "kind": "HP0%検出漏れ疑い",
                "time": None,
                "seq": None,
                "detail": f"「{name}」はひんし判定済みだが、states.jsonlでの最小hp_pct観測値="
                          f"{observed}（0%付近を捕捉できていない）",
            })
    return flags


_STATUS_MOVE_DAMAGE_WORDS = (
    "ダメージ", "効果ばつぐん", "こうかばつぐん", "食らった", "くらった",
    "削れ", "痛手", "命中して", "半減", "4分の1",
)


def _load_status_moves(pokedb_path: Path) -> set[str]:
    """data/pokedb.sqliteから変化技（category='変化'）の技名一覧を読む。
    DBが無い環境（CI等）ではフラグを立てずに空集合を返す。"""
    if not pokedb_path.exists():
        return set()
    import sqlite3
    conn = sqlite3.connect(pokedb_path)
    try:
        return {row[0] for row in conn.execute(
            "SELECT name_ja FROM moves WHERE category = '変化'")}
    finally:
        conn.close()


def check_status_move_damage_claim(manifest: list[dict], status_moves: set[str]) -> list[dict]:
    """変化技（ダメージを与えない技）なのにダメージ表現を伴う実況 → 技効果誤認疑い
    （2026-08-14・NG恒久対策フェーズ1「施策B: 技効果ヒントRAG新設」の再発検出用。
    RAGが機能していればこのパターンは新規発生しないはず）。"""
    flags = []
    for m in manifest:
        if m.get("event_type") not in ("move_used", "move_single", "faint"):
            continue
        text = m.get("commentary", "") or ""
        for mv in status_moves:
            if mv and mv in text and any(w in text for w in _STATUS_MOVE_DAMAGE_WORDS):
                flags.append({
                    "kind": "変化技のダメージ誤認疑い",
                    "time": m.get("event_time"),
                    "seq": m.get("seq"),
                    "detail": f"技={mv}（変化技） commentary=「{text}」",
                })
                break
    return flags


def check_side_roster_mismatch(states: list[dict], manifest: list[dict]) -> list[dict]:
    """技ログに登場する使い手の陣営と、states.jsonl上の実ロスター所属が矛盾していないか
    （2026-08-14・NG恒久対策フェーズ1「施策C: is_opponent陣営判定クロスチェック」の
    再発検出用）。states全体を通じてplayer/opponent各ロスターに一度でも出現した名前
    集合を作り、自分ロスター専属（相手ロスターには一度も出ていない）のはずの名前が
    manifest.context.opponent（相手の場/控え表示文字列）にも出現しているという
    明確な矛盾のみを拾う簡易版。⚠️完全な検出は難しいため「疑いの一次検出」の
    位置づけ（同名ミラー戦は判定不能として除外する）。"""
    ever_player: set[str] = set()
    ever_opponent: set[str] = set()
    for s in states:
        ever_player |= {p.get("name") for p in s.get("player", []) if p.get("name")}
        ever_opponent |= {p.get("name") for p in s.get("opponent", []) if p.get("name")}
    both_sides = ever_player & ever_opponent  # ミラー個体は判定不能なので除外
    player_only = ever_player - both_sides

    flags = []
    for m in manifest:
        ctx = m.get("context") or {}
        opponent_str = ctx.get("opponent") or ""
        if not opponent_str:
            continue
        for entry in ctx.get("move_log") or []:
            for name in player_only:
                if name in entry and name in opponent_str:
                    flags.append({
                        "kind": "陣営矛盾疑い（自分専用ポケモンが相手ロスターにも出現）",
                        "time": m.get("event_time"),
                        "seq": m.get("seq"),
                        "detail": f"「{name}」はstates.jsonl上は自分ロスター専属のはずだが、"
                                  f"技ログ「{entry}」の時点でcontext.opponent"
                                  f"「{opponent_str}」にも出現",
                    })
    return flags


def run_all_checks(render_dir: Path, pokedb_path: Path | None = None) -> dict:
    manifest = load_jsonl(render_dir / "manifest.jsonl")
    states = load_jsonl(render_dir / "states.jsonl")

    glitch_flags, replaced_count = check_leaked_glitch_keywords(manifest)
    if pokedb_path is None:
        pokedb_path = Path(__file__).resolve().parent.parent / "data" / "pokedb.sqlite"
    status_moves = _load_status_moves(pokedb_path)

    return {
        "move_log_empty": check_empty_move_log(manifest),
        "battle_result_missing": check_missing_battle_result(manifest),
        "emoji": check_emoji(manifest),
        "glitch_leaked": glitch_flags,
        "glitch_replaced_count": replaced_count,
        "selection_screen_only": check_selection_screen_only_names(states, manifest),
        "hp_zero_missing": check_missing_hp_zero(states, manifest),
        "status_move_damage_claim": check_status_move_damage_claim(manifest, status_moves),
        "side_roster_mismatch": check_side_roster_mismatch(states, manifest),
    }


def build_report_markdown(render_dir: Path, results: dict) -> str:
    lines = [f"# パス1 自動スクリーニング結果 — {render_dir.name}", ""]
    lines.append("⚠️これは一次検出（疑い）です。フラグが立った箇所を優先して目視確認してください。"
                 "フラグゼロは「既知パターンに該当しなかった」という意味で無罪確定ではありません。")
    lines.append("")

    total = 0
    category_labels = {
        "move_log_empty": "① move_log空での技実況",
        "battle_result_missing": "② battle_result未検出でのbattle_end",
        "emoji": "③ 絵文字混入",
        "glitch_leaked": "④ 生の保留・困惑応答の残存",
        "selection_screen_only": "⑤ 選出画面限定登場ポケモン",
        "hp_zero_missing": "⑥ HP0%検出漏れ疑い",
        "status_move_damage_claim": "⑦ 変化技のダメージ誤認疑い",
        "side_roster_mismatch": "⑧ 陣営矛盾疑い",
    }
    for key, label in category_labels.items():
        flags = results[key]
        total += len(flags)
        lines.append(f"## {label}（{len(flags)}件）")
        lines.append("")
        if not flags:
            lines.append("- フラグなし")
        else:
            for f in flags:
                t = f"{f['time']}s " if f.get("time") is not None else ""
                seq = f"#{f['seq']} " if f.get("seq") is not None else ""
                lines.append(f"- {t}{seq}{f['detail']}")
        lines.append("")

    lines.append(f"## 参考: AIグリッチ差し替え発動（正常動作・件数のみ記録）")
    lines.append("")
    lines.append(f"- {results['glitch_replaced_count']}件")
    lines.append("")

    lines.insert(2, f"**総フラグ数: {total}件**")
    lines.insert(3, "")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", help="パス1の素材ディレクトリ（renders/<動画名>）")
    parser.add_argument("--out", default=None,
                        help="出力先（省略時は <render_dir>/screening_report.md）")
    args = parser.parse_args(argv)

    render_dir = Path(args.render_dir)
    if not render_dir.exists():
        print(f"エラー: {render_dir} が存在しません", file=sys.stderr)
        return 1

    results = run_all_checks(render_dir)
    total = sum(len(v) for k, v in results.items() if isinstance(v, list))

    md = build_report_markdown(render_dir, results)
    out_path = Path(args.out) if args.out else render_dir / "screening_report.md"
    out_path.write_text(md, encoding="utf-8")

    print(f"生成しました: {out_path}")
    print(f"総フラグ数: {total}件（AIグリッチ正常差し替え: {results['glitch_replaced_count']}件・フラグ対象外）")
    if total:
        print("→ screening_report.md の該当箇所を優先して目視確認してください。")
        return 1
    print("→ 既知パターンには該当なし。軽めのスポットチェックで良さそうです。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
