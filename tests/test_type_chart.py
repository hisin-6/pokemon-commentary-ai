"""
src/pokedb/type_chart.py（タイプ相性表・戦況推論強化 2026-08-04）の単体テスト
"""

from src.pokedb.type_chart import describe_matchup, effectiveness_multiplier


class TestEffectivenessMultiplier:
    def test_super_effective_single_type(self):
        assert effectiveness_multiplier("みず", ["ほのお"]) == 2.0

    def test_not_very_effective_single_type(self):
        assert effectiveness_multiplier("ほのお", ["みず"]) == 0.5

    def test_neutral_when_no_chart_entry(self):
        assert effectiveness_multiplier("ノーマル", ["みず"]) == 1.0

    def test_immunity_is_zero(self):
        assert effectiveness_multiplier("じめん", ["ひこう"]) == 0.0

    def test_dual_type_multiplies(self):
        # はがね(steel)技 vs フェアリー単タイプ = 2倍
        assert effectiveness_multiplier("はがね", ["フェアリー"]) == 2.0
        # かくとう(fighting)技 vs はがね/エスパー = 2.0 * 0.5 = 1.0
        assert effectiveness_multiplier("かくとう", ["はがね", "エスパー"]) == 1.0

    def test_quad_weakness(self):
        # こおり(ice)技 vs じめん/ひこう(ground/flying) = 2.0 * 2.0 = 4.0
        assert effectiveness_multiplier("こおり", ["じめん", "ひこう"]) == 4.0

    def test_dual_type_immunity_short_circuits_to_zero(self):
        # どく(poison)技 vs はがね/くさ = 0.0 * 2.0 = 0.0
        assert effectiveness_multiplier("どく", ["はがね", "くさ"]) == 0.0

    def test_empty_defender_types_is_neutral(self):
        assert effectiveness_multiplier("ほのお", []) == 1.0

    def test_none_defender_types_is_neutral(self):
        assert effectiveness_multiplier("ほのお", None) == 1.0

    def test_unknown_move_type_is_neutral(self):
        assert effectiveness_multiplier("未知タイプ", ["みず"]) == 1.0


class TestDescribeMatchup:
    def test_super_effective_label(self):
        assert describe_matchup("みず", ["ほのお"]) == "バツグン"

    def test_quad_super_effective_label(self):
        assert describe_matchup("こおり", ["じめん", "ひこう"]) == "4倍バツグン"

    def test_not_very_effective_label(self):
        assert describe_matchup("ほのお", ["みず"]) == "いまひとつ"

    def test_quarter_effective_label(self):
        assert describe_matchup("かくとう", ["ひこう", "エスパー"]) == "4分の1"

    def test_no_effect_label(self):
        assert describe_matchup("でんき", ["じめん"]) == "こうかなし"

    def test_neutral_label(self):
        assert describe_matchup("ノーマル", ["みず"]) == "等倍"
