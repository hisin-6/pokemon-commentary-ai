"""
screen_pass1.py（パス1無課金検証の自動一次スクリーニング）の単体テスト

テスト対象:
  - check_empty_move_log             move_log空での技実況
  - check_missing_battle_result      battle_result未検出でのbattle_end実況
  - check_emoji                      絵文字ブロック混入
  - check_leaked_glitch_keywords     生の保留・困惑応答の残存（差し替え済みは除外）
  - check_selection_screen_only_names 選出画面限定登場ポケモン
  - check_missing_hp_zero            HP0%検出漏れ疑い
"""

import importlib.util
from pathlib import Path

# scripts/ はパッケージではないためファイルパスから直接ロードする
_SCRIPT = Path(__file__).parent.parent / "scripts" / "screen_pass1.py"
_spec = importlib.util.spec_from_file_location("screen_pass1", _SCRIPT)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def _manifest(event_time, event_type, commentary, context=None):
    return {"seq": 1, "event_time": event_time, "event_type": event_type,
            "commentary": commentary, "wav": "wav/x.wav", "duration": 5.0,
            "context": context or {}}


def _state(time, turn, player=None, opponent=None):
    return {"time": time, "turn": turn, "player": player or [], "opponent": opponent or []}


def _mon(name, hp_pct=None):
    return {"name": name, "hp_pct": hp_pct, "hp_text": None, "status": None}


class TestCheckEmptyMoveLog:
    def test_flags_move_event_with_empty_move_log(self):
        manifest = [_manifest(126.0, "move_used", "捏造された実況", {"move_log": []})]
        flags = sp.check_empty_move_log(manifest)
        assert len(flags) == 1
        assert flags[0]["time"] == 126.0

    def test_does_not_flag_when_move_log_populated(self):
        manifest = [_manifest(126.0, "move_used", "正常な実況",
                              {"move_log": ["T1:イダイトウのシャドーボール"]})]
        assert sp.check_empty_move_log(manifest) == []

    def test_ignores_non_move_events(self):
        manifest = [_manifest(80.0, "battle_start", "試合開始！", {"move_log": []})]
        assert sp.check_empty_move_log(manifest) == []


class TestCheckMissingBattleResult:
    def test_flags_battle_end_without_result(self):
        manifest = [_manifest(162.1, "battle_end", "えらさんの勝利だ♪", {})]
        flags = sp.check_missing_battle_result(manifest)
        assert len(flags) == 1

    def test_does_not_flag_when_result_present(self):
        manifest = [_manifest(337.8, "battle_end", "わあ、勝ったぁ～！",
                              {"battle_result": "勝ち"})]
        assert sp.check_missing_battle_result(manifest) == []


class TestCheckEmoji:
    def test_flags_emoji_block_chars(self):
        manifest = [_manifest(144.7, "move_used", "攻めていくクレぴ！💖✨")]
        flags = sp.check_emoji(manifest)
        assert len(flags) == 1

    def test_does_not_flag_allowed_symbols(self):
        manifest = [_manifest(1.0, "move_used", "頑張るよ♪♡")]
        assert sp.check_emoji(manifest) == []


class TestCheckLeakedGlitchKeywords:
    def test_flags_raw_hesitation_text(self):
        manifest = [_manifest(200.0, "move_used", "データが矛盾していて実況できません")]
        flags, replaced = sp.check_leaked_glitch_keywords(manifest)
        assert len(flags) == 1
        assert replaced == 0

    def test_does_not_flag_already_replaced_template(self):
        """テンプレ差し替え後の文はキーワードを自己参照的に含むため誤検出しない。"""
        manifest = [_manifest(200.0, "move_used",
                              "あれれ？データがちぐはぐさんで、くれぴの目がちょっとバグっちゃったかも…！次いくよ次〜♪")]
        flags, replaced = sp.check_leaked_glitch_keywords(manifest)
        assert flags == []
        assert replaced == 1

    def test_does_not_flag_normal_commentary(self):
        manifest = [_manifest(1.0, "move_used", "イダイトウの攻撃が炸裂だ！")]
        flags, replaced = sp.check_leaked_glitch_keywords(manifest)
        assert flags == [] and replaced == 0


class TestCheckSelectionScreenOnlyNames:
    def test_flags_name_seen_only_in_turn0(self):
        states = [
            _state(57.3, 0, player=[_mon("ランクルス")]),
            _state(87.3, 1, player=[_mon("オオニューラ")]),
        ]
        flags = sp.check_selection_screen_only_names(states, manifest=[])
        assert len(flags) == 1
        assert "ランクルス" in flags[0]["detail"]

    def test_does_not_flag_name_persisting_into_turn1(self):
        states = [
            _state(57.3, 0, player=[_mon("イダイトウ")]),
            _state(87.3, 1, player=[_mon("イダイトウ")]),
        ]
        assert sp.check_selection_screen_only_names(states, manifest=[]) == []

    def test_does_not_flag_legitimately_fainted_pokemon(self):
        """turn0限定でも、その後ひんし記録があれば誤認識ではなく正当な気絶なので除外。"""
        states = [_state(57.3, 0, opponent=[_mon("ユキメノコ")])]
        manifest = [_manifest(150.0, "faint", "ユキメノコが倒れた",
                              {"opponent": "控え: ユキメノコ(ひんし)"})]
        assert sp.check_selection_screen_only_names(states, manifest) == []


class TestCheckMissingHpZero:
    def test_flags_fainted_pokemon_with_no_low_hp_observed(self):
        states = [_state(80.0, 0, opponent=[_mon("ユキメノコ", hp_pct=100)])]
        manifest = [_manifest(150.0, "faint", "ユキメノコが倒れた",
                              {"opponent": "控え: ユキメノコ(ひんし)"})]
        flags = sp.check_missing_hp_zero(states, manifest)
        assert len(flags) == 1
        assert "ユキメノコ" in flags[0]["detail"]

    def test_does_not_flag_when_zero_percent_observed(self):
        states = [_state(80.0, 0, opponent=[_mon("ユキメノコ", hp_pct=0)])]
        manifest = [_manifest(150.0, "faint", "ユキメノコが倒れた",
                              {"opponent": "控え: ユキメノコ(ひんし)"})]
        assert sp.check_missing_hp_zero(states, manifest) == []

    def test_ignores_non_fainted_pokemon(self):
        states = [_state(80.0, 0, player=[_mon("イダイトウ", hp_pct=50)])]
        manifest = [_manifest(80.0, "move_used", "普通の実況", {"player": "場: イダイトウ HP:50%"})]
        assert sp.check_missing_hp_zero(states, manifest) == []


class TestCheckStatusMoveDamageClaim:
    """check_status_move_damage_claim: 2026-08-14新設（NG恒久対策フェーズ1・
    施策B「技効果ヒントRAG新設」の再発検出用）。"""

    def test_flags_status_move_with_damage_wording(self):
        manifest = [_manifest(121.6, "move_single",
                              "めいそうでダメージを与えていくよ！",
                              {"move_log": ["T1:フシギバナのめいそう"]})]
        flags = sp.check_status_move_damage_claim(manifest, status_moves={"めいそう"})
        assert len(flags) == 1
        assert "めいそう" in flags[0]["detail"]

    def test_does_not_flag_status_move_without_damage_wording(self):
        manifest = [_manifest(121.6, "move_single", "めいそうで自分を強化するのね♪",
                              {"move_log": ["T1:フシギバナのめいそう"]})]
        assert sp.check_status_move_damage_claim(manifest, status_moves={"めいそう"}) == []

    def test_does_not_flag_damage_move_with_damage_wording(self):
        """物理/特殊技（変化技リストに無い）はダメージ表現があっても正常なのでフラグしない。"""
        manifest = [_manifest(121.6, "move_single", "じしんで大ダメージ！",
                              {"move_log": ["T1:ガブリアスのじしん"]})]
        assert sp.check_status_move_damage_claim(manifest, status_moves={"めいそう"}) == []

    def test_ignores_non_move_events(self):
        manifest = [_manifest(80.0, "battle_start", "めいそうでダメージが入った気がする",
                              {"move_log": []})]
        assert sp.check_status_move_damage_claim(manifest, status_moves={"めいそう"}) == []

    def test_empty_status_moves_flags_nothing(self):
        """DB未検出等でstatus_movesが空集合の場合はフラグを立てない（安全側）。"""
        manifest = [_manifest(121.6, "move_single", "めいそうでダメージを与えていくよ！",
                              {"move_log": ["T1:フシギバナのめいそう"]})]
        assert sp.check_status_move_damage_claim(manifest, status_moves=set()) == []


class TestCheckSideRosterMismatch:
    """check_side_roster_mismatch: 2026-08-14新設（NG恒久対策フェーズ1・
    施策C「is_opponent陣営判定クロスチェック」の再発検出用）。"""

    def test_flags_player_only_name_appearing_in_opponent_context(self):
        states = [_state(80.0, 1, player=[_mon("ガブリアス")], opponent=[_mon("リザードン")])]
        manifest = [_manifest(90.0, "move_single", "相手のガブリアスのじしん！",
                              {"move_log": ["T1:ガブリアスのじしん"],
                               "opponent": "場: ガブリアス / 控え: なし"})]
        flags = sp.check_side_roster_mismatch(states, manifest)
        assert len(flags) == 1
        assert "ガブリアス" in flags[0]["detail"]

    def test_does_not_flag_consistent_roster(self):
        states = [_state(80.0, 1, player=[_mon("ガブリアス")], opponent=[_mon("リザードン")])]
        manifest = [_manifest(90.0, "move_single", "ガブリアスのじしん！",
                              {"move_log": ["T1:ガブリアスのじしん"],
                               "opponent": "場: リザードン / 控え: なし"})]
        assert sp.check_side_roster_mismatch(states, manifest) == []

    def test_mirror_name_in_both_rosters_not_flagged(self):
        """同名ミラー戦（両陣営に登場）は判定不能として除外し、フラグしない。"""
        states = [_state(80.0, 1, player=[_mon("ガブリアス")], opponent=[_mon("ガブリアス")])]
        manifest = [_manifest(90.0, "move_single", "ガブリアスのじしん！",
                              {"move_log": ["T1:ガブリアスのじしん"],
                               "opponent": "場: ガブリアス / 控え: なし"})]
        assert sp.check_side_roster_mismatch(states, manifest) == []

    def test_no_opponent_context_not_flagged(self):
        states = [_state(80.0, 1, player=[_mon("ガブリアス")])]
        manifest = [_manifest(90.0, "move_single", "ガブリアスのじしん！",
                              {"move_log": ["T1:ガブリアスのじしん"]})]
        assert sp.check_side_roster_mismatch(states, manifest) == []


class TestExtractNames:
    def test_parses_field_and_bench(self):
        roster = "場: オオニューラ HP:98/155 技=[ねこだまし] / イダイトウ HP:7/201★ピンチ 技=[だくりゅう] / 控え: なし"
        names = sp._extract_names(roster)
        assert names == {"オオニューラ", "イダイトウ"}

    def test_parses_fainted_bench_entries(self):
        roster = "場: 情報収集中 / 控え: ロトム(ひんし) / オオニューラ / ユキメノコ(ひんし)"
        names = sp._extract_names(roster)
        assert names == {"ロトム", "オオニューラ", "ユキメノコ"}

    def test_empty_string_returns_empty_set(self):
        assert sp._extract_names(None) == set()
        assert sp._extract_names("") == set()
