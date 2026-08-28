"""
generate_predictions.py（予測→回収パス・2026-08-21新設）の単体テスト

テスト対象:
  - find_prediction_candidates  場のコンディションを確立したmove_singleの検出
  - find_decisive_event         回収アンカー（最後のfaint/battle_end）の検出
  - determine_battle_result     最終battle_resultの取得
  - judge_hit                   予測の的中/外れ判定
  - find_selection_prediction_candidate  選出予想の予測ポイント検出（2026-08-24新設）
  - find_battle_start_event              選出予想の回収アンカー検出（2026-08-24新設）
  - judge_selection_hit                  選出予想の的中/外れ判定（2026-08-24新設）
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pokedb.team_preview import save_team_preview

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

    def test_surrendered_battle_end_preferred_over_earlier_faint(self):
        """降参決着では、時系列的に後でもfaintではなくbattle_endを優先する
        （2026-08-29新設の回帰ガード）。

        降参は最後のfaintより前に起きることがあり（試合を決めたのはKOでは
        なく相手の意思決定）、その場合manifest.jsonl上の「最後のfaint」は
        決着とは無関係な内容になり得る（実機2026-08-28_21-52-34で確認:
        相手はムクホークを残したまま降参しており、「最後のfaint」に見える
        エントリはメタグロスの登場を伝える内容で決着の42秒以上前だった）。"""
        entries = [
            _event(100.0, event_type="faint", commentary="1匹目が倒れた"),
            _event(200.0, event_type="faint", commentary="相手が入れ替わった"),
            _event(370.0, event_type="battle_end", commentary="降参で勝利",
                   battle_surrendered=True),
        ]
        decisive = gp.find_decisive_event(entries)
        assert decisive["event_type"] == "battle_end"
        assert decisive["event_time"] == 370.0

    def test_non_surrendered_battle_end_does_not_override_last_faint(self):
        """通常決着（KOで終了）では従来通り最後のfaintを優先する
        （battle_surrendered=trueでない限り挙動を変えない回帰ガード）。"""
        entries = [
            _event(100.0, event_type="faint", commentary="1匹目が倒れた"),
            _event(400.0, event_type="faint", commentary="最後の1匹が倒れた"),
            _event(450.0, event_type="battle_end", commentary="試合終了",
                   battle_surrendered=False),
        ]
        decisive = gp.find_decisive_event(entries)
        assert decisive["event_type"] == "faint"
        assert decisive["event_time"] == 400.0


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


class TestFindSelectionPredictionCandidate:
    """選出予想の予測ポイント検出（2026-08-24新設）。"""

    def test_returns_candidate_when_team_preview_present(self, tmp_path):
        save_team_preview(tmp_path, ["コノヨザル"], ["リザードン", "オオニューラ"])
        entries = [_event(46.5, event_type="battle_start")]
        candidate = gp.find_selection_prediction_candidate(tmp_path, entries)
        assert candidate is not None
        assert candidate["opponent_team"] == ["リザードン", "オオニューラ"]
        assert "リザードン" in candidate["hint"]

    def test_time_stays_before_first_event(self, tmp_path):
        """短い試合（最初のイベントが早い）でも予測時刻はそれより前に収まる。"""
        save_team_preview(tmp_path, [], ["リザードン"])
        entries = [_event(1.0, event_type="battle_start")]
        candidate = gp.find_selection_prediction_candidate(tmp_path, entries)
        assert 0.0 <= candidate["time"] < 1.0

    def test_none_without_team_preview_file(self, tmp_path):
        entries = [_event(46.5, event_type="battle_start")]
        assert gp.find_selection_prediction_candidate(tmp_path, entries) is None

    def test_none_when_opponent_team_empty(self, tmp_path):
        """自分の構築しか入力されていない（相手が未入力）場合は予想しない。"""
        save_team_preview(tmp_path, ["コノヨザル"], [])
        entries = [_event(46.5, event_type="battle_start")]
        assert gp.find_selection_prediction_candidate(tmp_path, entries) is None

    def test_none_when_no_entries(self, tmp_path):
        save_team_preview(tmp_path, [], ["リザードン"])
        assert gp.find_selection_prediction_candidate(tmp_path, []) is None


class TestFindBattleStartEvent:
    def test_finds_earliest_battle_start(self):
        entries = [
            _event(50.0, event_type="move_used"),
            _event(46.5, event_type="battle_start", commentary="開幕"),
        ]
        result = gp.find_battle_start_event(entries)
        assert result["event_time"] == 46.5

    def test_none_when_absent(self):
        entries = [_event(50.0, event_type="move_used")]
        assert gp.find_battle_start_event(entries) is None


class TestExtractFieldNames:
    def test_extracts_names_with_hp_and_moves(self):
        s = "場: ペリッパー HP:167/167 技=[ぼうふう] / ラグラージ / 控え: コノヨザル"
        assert gp._extract_field_names(s) == ["ペリッパー", "ラグラージ"]

    def test_extracts_names_with_status(self):
        s = "場: コノヨザル(まひ) / ガオガエン"
        assert gp._extract_field_names(s) == ["コノヨザル", "ガオガエン"]

    def test_empty_for_unknown_placeholder(self):
        assert gp._extract_field_names("情報収集中") == []

    def test_empty_for_empty_string(self):
        assert gp._extract_field_names("") == []


class TestJudgeSelectionHit:
    def test_hit_when_predicted_name_in_actual_leads(self):
        entry = {"context": {"opponent": "場: リザードン HP:100% / オオニューラ"}}
        hit, leads = gp.judge_selection_hit(
            "リザードンとガブリアスが来ると予想！", ["リザードン", "オオニューラ", "ガブリアス"], entry)
        assert hit is True
        assert leads == ["リザードン", "オオニューラ"]

    def test_miss_when_no_predicted_name_matches(self):
        entry = {"context": {"opponent": "場: リザードン / オオニューラ"}}
        hit, leads = gp.judge_selection_hit(
            "ガブリアスが来そう", ["リザードン", "オオニューラ", "ガブリアス"], entry)
        assert hit is False
        assert leads == ["リザードン", "オオニューラ"]

    def test_miss_when_actual_leads_unavailable(self):
        entry = {"context": {"opponent": "情報収集中"}}
        hit, leads = gp.judge_selection_hit("リザードンが来そう", ["リザードン"], entry)
        assert hit is False
        assert leads == []
