"""
src/pokedb/mega_forms.py（メガシンカのタイプ変化・2026-08-04新規）の単体テスト
"""

from src.pokedb.mega_forms import get_mega_types


class TestGetMegaTypes:
    def test_charizard_x_type_change(self):
        assert get_mega_types("リザードン") == ["ほのお", "ドラゴン"]

    def test_mewtwo_x_type_change(self):
        assert get_mega_types("ミュウツー") == ["エスパー", "かくとう"]

    def test_unregistered_pokemon_returns_none(self):
        assert get_mega_types("ピカチュウ") is None

    def test_unknown_pokemon_returns_none(self):
        assert get_mega_types("存在しないポケモン12345") is None
