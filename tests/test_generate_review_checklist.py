"""
scripts/generate_review_checklist.py（パス1レビュー用チェックリスト生成）の単体テスト

テスト対象:
  - load_jsonl           存在しないファイル・空行の扱い
  - _judgement           判定欄の書式
  - _build_segments      states.jsonl をターンごとの [start, end) に区切る
  - _find_segment        時刻→セグメント（範囲外は先頭/末尾にフォールバック）
  - _turn_state_summary  初登場ポケモン・HP/状態異常の変化点だけ抽出
  - build_markdown       render_dir（tmp_path）から Markdown 全体を組み立てる
  - main                 存在しないディレクトリ・--out 指定
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


grc = _load("generate_review_checklist")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def _poke(name, hp_pct=100, hp_text=None, status=None):
    return {"name": name, "hp_pct": hp_pct, "hp_text": hp_text, "status": status}


# ── load_jsonl ────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        assert grc.load_jsonl(tmp_path / "nothing.jsonl") == []

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
        assert grc.load_jsonl(p) == [{"a": 1}, {"a": 2}]


# ── _judgement ────────────────────────────────────────────────────────────

def test_judgement_format():
    assert grc._judgement("正しい内容") == "判定(OK/NG): \n正しい内容（NGの場合のみ記入）: \n"


# ── _build_segments ───────────────────────────────────────────────────────

class TestBuildSegments:
    def test_empty(self):
        assert grc._build_segments([]) == []

    def test_boundaries_follow_next_turn_start(self):
        states = [
            {"turn": 0, "time": 0.0},
            {"turn": 0, "time": 5.0},
            {"turn": 1, "time": 10.0},
            {"turn": 2, "time": 30.0},
            {"turn": 2, "time": 35.0},
        ]
        segs = grc._build_segments(states)
        assert [(s["turn"], s["start"], s["end"]) for s in segs] == [
            (0, 0.0, 10.0), (1, 10.0, 30.0), (2, 30.0, None),
        ]
        assert len(segs[0]["snapshots"]) == 2
        assert len(segs[2]["snapshots"]) == 2

    def test_turn_order_is_first_appearance(self):
        # 同じターン番号が後から再出現しても、最初のセグメントにまとめられる
        states = [
            {"turn": 1, "time": 10.0},
            {"turn": 2, "time": 20.0},
            {"turn": 1, "time": 25.0},
        ]
        segs = grc._build_segments(states)
        assert [s["turn"] for s in segs] == [1, 2]
        assert len(segs[0]["snapshots"]) == 2


# ── _find_segment ─────────────────────────────────────────────────────────

class TestFindSegment:
    def _segs(self):
        return grc._build_segments([
            {"turn": 1, "time": 10.0},
            {"turn": 2, "time": 20.0},
            {"turn": 3, "time": 30.0},
        ])

    def test_none_time_or_empty(self):
        assert grc._find_segment(self._segs(), None) is None
        assert grc._find_segment([], 5.0) is None

    def test_start_inclusive_end_exclusive(self):
        segs = self._segs()
        assert grc._find_segment(segs, 10.0)["turn"] == 1
        assert grc._find_segment(segs, 19.99)["turn"] == 1
        assert grc._find_segment(segs, 20.0)["turn"] == 2

    def test_last_segment_is_open_ended(self):
        assert grc._find_segment(self._segs(), 999.0)["turn"] == 3

    def test_before_first_falls_back_to_first(self):
        assert grc._find_segment(self._segs(), 1.0)["turn"] == 1


# ── _turn_state_summary ───────────────────────────────────────────────────

class TestTurnStateSummary:
    def test_roster_and_changes_dedup(self):
        snapshots = [
            {"time": 10.0, "player": [_poke("ガブリアス")], "opponent": [_poke("サーフゴー")]},
            # 変化なし → changes に出ない
            {"time": 11.0, "player": [_poke("ガブリアス")], "opponent": [_poke("サーフゴー")]},
            # 相手のHPが減り、まひになる
            {"time": 12.0, "player": [_poke("ガブリアス")],
             "opponent": [_poke("サーフゴー", hp_pct=40, status="まひ")]},
        ]
        roster, changes = grc._turn_state_summary(snapshots)
        assert roster == {"player": ["ガブリアス"], "opponent": ["サーフゴー"]}
        assert [(c["time"], c["side"], c["name"]) for c in changes] == [
            (10.0, "player", "ガブリアス"),
            (10.0, "opponent", "サーフゴー"),
            (12.0, "opponent", "サーフゴー"),
        ]
        assert changes[-1]["hp_pct"] == 40
        assert changes[-1]["status"] == "まひ"

    def test_roster_keeps_appearance_order_and_skips_nameless(self):
        snapshots = [
            {"time": 1.0, "player": [_poke("A"), {"name": None}], "opponent": []},
            {"time": 2.0, "player": [_poke("B"), _poke("A")]},  # opponentキー欠けもOK
        ]
        roster, _ = grc._turn_state_summary(snapshots)
        assert roster == {"player": ["A", "B"], "opponent": []}

    def test_hp_text_change_counts(self):
        snapshots = [
            {"time": 1.0, "player": [_poke("エルフーン", hp_text="150/150")]},
            {"time": 2.0, "player": [_poke("エルフーン", hp_text="120/150")]},
        ]
        _, changes = grc._turn_state_summary(snapshots)
        assert [c["hp_text"] for c in changes] == ["150/150", "120/150"]


# ── build_markdown ────────────────────────────────────────────────────────

def _make_render_dir(tmp_path: Path) -> Path:
    d = tmp_path / "2026-01-01_00-00-00"
    d.mkdir()
    _write_jsonl(d / "states.jsonl", [
        {"turn": 0, "time": 0.0, "player": [], "opponent": []},
        {"turn": 1, "time": 10.0, "player": [_poke("ガブリアス")], "opponent": [_poke("サーフゴー")]},
        {"turn": 2, "time": 30.0, "player": [_poke("ガブリアス")],
         "opponent": [_poke("サーフゴー", hp_pct=0)]},
    ])
    _write_jsonl(d / "timeline.jsonl", [
        {"kind": "move", "time": 12.0, "side": "player", "text": "T1:ガブリアスのじしん"},
        {"kind": "move", "time": 14.0, "side": "opponent", "text": "T1:サーフゴーのおいかぜ"},
        # "Tn:" 表記なし → 時刻(31s)からターン2と判定
        {"kind": "move", "time": 31.0, "side": "player", "text": "ガブリアスのげきりん"},
        # "T2:" 表記が時刻より優先される（時刻はターン1帯だがターン2に出る）
        {"kind": "move", "time": 15.0, "side": None, "text": "T2:テスト技"},
        {"kind": "hp", "time": 13.0, "text": "T1:無関係"},  # move以外は無視
    ])
    _write_jsonl(d / "manifest.jsonl", [
        {"seq": 1, "event_time": 2.0, "event_type": "battle_start", "commentary": "試合開始！"},
        {"seq": 2, "event_time": 12.5, "event_type": "move_used", "commentary": "じしん！"},
        {"seq": 3, "event_time": 32.0, "event_type": "faint", "commentary": "サーフゴーが倒れた！"},
    ])
    (d / "render_info.json").write_text(
        json.dumps({"video": "C:/videos/battle.mp4", "created_at": "2026-01-01T00:00:00"}),
        encoding="utf-8")
    return d


class TestBuildMarkdown:
    def test_full_render_dir(self, tmp_path):
        md = grc.build_markdown(_make_render_dir(tmp_path))

        assert md.startswith("# パス1 検証チェックリスト — 2026-01-01_00-00-00\n")
        assert "- 元動画: C:/videos/battle.mp4" in md
        assert "- パス1実行日時: 2026-01-01T00:00:00" in md

        assert "## ターン0 （試合開始前）（0.0s 〜 10.0s）" in md
        assert "## ターン1 （10.0s 〜 30.0s）" in md
        assert "## ターン2 （30.0s 〜 試合終了）" in md

        t1 = md.split("## ターン1")[1].split("## ターン2")[0]
        t2 = md.split("## ターン2")[1]

        # 技: Tn:表記で振り分け、場の効果には⚠
        assert "- 12.0s [player] T1:ガブリアスのじしん" in t1
        assert "- ⚠ 14.0s [opponent] T1:サーフゴーのおいかぜ" in t1
        assert "T2:テスト技" not in t1
        assert "- 15.0s [?] T2:テスト技" in t2
        assert "- 31.0s [player] ガブリアスのげきりん" in t2
        assert "無関係" not in md

        # 場のポケモン・HP変化
        assert "- 自分: ガブリアス" in t1
        assert "- 30.0s [opponent] サーフゴー: 0% / None" in t2

        # 交代・気絶はターン2、move_usedは実況内容のみ
        assert "**交代・気絶・試合終了**" in t2
        assert "- 32.0s [faint]: サーフゴーが倒れた！" in t2
        assert "**交代・気絶・試合終了**" not in t1
        assert "#### 12.5s / move_used (#2)" in t1
        assert "> じしん！" in t1

        # ターン0（ポケモンなし）
        t0 = md.split("## ターン0")[1].split("## ターン1")[0]
        assert "- 自分: なし" in t0
        assert "#### 2.0s / battle_start (#1)" in t0

        # ターンごとに「試合内容の判定」が1つずつ
        assert md.count("**試合内容の判定**") == 3
        assert md.endswith("\n")

    def test_without_states_lists_manifest_only(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        _write_jsonl(d / "manifest.jsonl", [
            {"seq": 1, "event_time": 2.0, "event_type": "battle_start", "commentary": "開始"},
        ])
        md = grc.build_markdown(d)
        assert "states.jsonlが空のため" in md
        assert "### #1 （2.0s / battle_start）" in md
        assert "> 開始" in md
        assert "## ターン" not in md
        assert "元動画" not in md  # render_info.json なし

    def test_long_event_commentary_truncated_to_80(self, tmp_path):
        d = tmp_path / "long"
        d.mkdir()
        _write_jsonl(d / "states.jsonl", [{"turn": 1, "time": 0.0}])
        long_text = "あ" * 100
        _write_jsonl(d / "manifest.jsonl", [
            {"seq": 1, "event_time": 1.0, "event_type": "switch", "commentary": long_text},
        ])
        md = grc.build_markdown(d)
        assert f"- 1.0s [switch]: {'あ' * 80}\n" in md
        assert f"> {long_text}" in md  # 実況内容欄は全文


# ── main ──────────────────────────────────────────────────────────────────

class TestMain:
    def test_missing_dir_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "nope")])
        assert grc.main() == 1
        assert "存在しません" in capsys.readouterr().err

    def test_writes_default_output(self, tmp_path, monkeypatch):
        d = _make_render_dir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["x", str(d)])
        assert grc.main() == 0
        assert (d / "review_checklist.md").read_text(encoding="utf-8").startswith("# パス1")

    def test_out_option(self, tmp_path, monkeypatch):
        d = _make_render_dir(tmp_path)
        out = tmp_path / "custom.md"
        monkeypatch.setattr(sys, "argv", ["x", str(d), "--out", str(out)])
        assert grc.main() == 0
        assert out.exists()
        assert not (d / "review_checklist.md").exists()
