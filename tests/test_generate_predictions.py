"""
generate_predictions.py（予測→回収パス・2026-08-21新設）の単体テスト

テスト対象:
  - find_prediction_candidates  場のコンディションを確立したmove_singleの検出
  - find_decisive_event         回収アンカー（最後のfaint/battle_end）の検出
  - determine_battle_result     最終battle_resultの取得
  - judge_hit                   予測の的中/外れ判定
"""

import importlib.util
from pathlib import Path

# scripts/ はパッケージではないためファイルパスから直接ロードする
_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_predictions.py"
_spec = importlib.util.spec_from_file_location("generate_predictions", _SCRIPT)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def _event(event_time, event_type="move_single", commentary="実況", **ctx):
    entry = {"event_time": event_time, "event_type": event_type, "commentary": commentary}
    if ctx:
        entry["context"] = ctx
    return entry


class TestFindPredictionCandidates:
    def test_player_condition_hint_detected(self):
        entries = [
            _event(100.0, condition_hint=None),
            _event(249.4, condition_hint="自分側におい風（あと4ターン・素早さ2倍）",
                   move_log=["T3:ペリッパーのおいかぜ"]),
        ]
        candidates = gp.find_prediction_candidates(entries)
        assert len(candidates) == 1
        assert candidates[0] == {
            "time": 249.4, "side": "player",
            "move_text": "T3:ペリッパーのおいかぜ",
            "hint": "自分側におい風（あと4ターン・素早さ2倍）",
        }

    def test_opponent_condition_hint_detected(self):
        entries = [_event(50.0, condition_hint="相手側にリフレクター（あと5ターン）",
                          move_log=["T1:ジャラランガのリフレクター"])]
        candidates = gp.find_prediction_candidates(entries)
        assert candidates[0]["side"] == "opponent"

    def test_max_one_candidate_per_side(self):
        """同じ陣営で条件が更新され続けても（残りターン数の変化等）候補は1件だけ。"""
        entries = [
            _event(100.0, condition_hint="自分側におい風（あと4ターン・素早さ2倍）",
                   move_log=["T3:ペリッパーのおいかぜ"]),
            _event(120.0, condition_hint="自分側におい風（あと3ターン・素早さ2倍）",
                   move_log=["T3:ペリッパーのおいかぜ"]),
        ]
        candidates = gp.find_prediction_candidates(entries)
        assert len(candidates) == 1
        assert candidates[0]["time"] == 100.0

    def test_up_to_two_candidates_both_sides(self):
        entries = [
            _event(100.0, condition_hint="自分側におい風（あと4ターン・素早さ2倍）",
                   move_log=["T3:ペリッパーのおいかぜ"]),
            _event(150.0, condition_hint="相手側にリフレクター（あと5ターン）",
                   move_log=["T4:ジャラランガのリフレクター"]),
        ]
        candidates = gp.find_prediction_candidates(entries)
        assert [c["side"] for c in candidates] == ["player", "opponent"]

    def test_non_move_single_events_ignored_as_source_but_update_prev_hint(self):
        """move_used等はそれ自体は候補にならないが、直後のmove_singleとの
        「新規出現」比較には影響する（同一hintの重複検出を防ぐ）。"""
        entries = [
            _event(90.0, event_type="move_used",
                   condition_hint="自分側におい風（あと4ターン・素早さ2倍）"),
            _event(100.0, event_type="move_single",
                   condition_hint="自分側におい風（あと4ターン・素早さ2倍）",
                   move_log=["T3:ペリッパーのおいかぜ"]),
        ]
        candidates = gp.find_prediction_candidates(entries)
        assert candidates == []  # move_usedの時点で既出扱いになり新規と見なされない

    def test_no_condition_hint_returns_empty(self):
        entries = [_event(100.0), _event(120.0, event_type="faint")]
        assert gp.find_prediction_candidates(entries) == []


class TestFindDecisiveEvent:
    def test_last_faint_preferred(self):
        entries = [
            _event(100.0, event_type="faint", commentary="1匹目が倒れた"),
            _event(400.0, event_type="faint", commentary="最後の1匹が倒れた"),
            _event(450.0, event_type="battle_end", commentary="試合終了"),
        ]
        decisive = gp.find_decisive_event(entries)
        assert decisive["event_time"] == 400.0
        assert decisive["commentary"] == "最後の1匹が倒れた"

    def test_falls_back_to_battle_end_without_faint(self):
        """降参決着等でfaintが1件も無いケース。"""
        entries = [_event(50.0, event_type="move_used"),
                  _event(200.0, event_type="battle_end", commentary="降参で終了")]
        decisive = gp.find_decisive_event(entries)
        assert decisive["event_type"] == "battle_end"

    def test_none_when_neither_exists(self):
        entries = [_event(50.0, event_type="move_used")]
        assert gp.find_decisive_event(entries) is None


class TestDetermineBattleResult:
    def test_finds_result_from_last_event_carrying_it(self):
        entries = [
            _event(100.0, battle_result=None),
            _event(400.0, event_type="battle_end", battle_result="勝ち"),
        ]
        assert gp.determine_battle_result(entries) == "勝ち"

    def test_none_when_unset(self):
        entries = [_event(100.0), _event(400.0, event_type="battle_end")]
        assert gp.determine_battle_result(entries) is None


class TestJudgeHit:
    def test_player_prediction_wins_on_win(self):
        assert gp.judge_hit("player", "勝ち") is True

    def test_player_prediction_misses_on_loss(self):
        assert gp.judge_hit("player", "負け") is False

    def test_opponent_prediction_hits_on_player_loss(self):
        assert gp.judge_hit("opponent", "負け") is True

    def test_opponent_prediction_misses_on_player_win(self):
        assert gp.judge_hit("opponent", "勝ち") is False
