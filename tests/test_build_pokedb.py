"""
scripts/build_pokedb.py の単体テスト（2026-08-14新設・技効果ヒントRAG対策）

テスト対象:
  - _pick_move_effect_text()  flavor_text_entries からのバージョングループ優先度選択・
                               「この技は使えません」プレースホルダー除外
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from build_pokedb import _pick_move_effect_text  # noqa: E402


def _entry(vg: str, text: str, lang: str = "ja") -> dict:
    return {
        "language": {"name": lang},
        "version_group": {"name": vg},
        "flavor_text": text,
    }


class TestPickMoveEffectText:

    def test_prefers_sword_shield_when_available(self):
        entries = [
            _entry("x-y", "旧世代の説明文"),
            _entry("sword-shield", "最新の説明文"),
        ]
        assert _pick_move_effect_text(entries) == "最新の説明文"

    def test_falls_back_to_lower_priority_version_group(self):
        """sword-shieldが無い場合は優先度リストの次点にフォールバックする。"""
        entries = [_entry("x-y", "X・Y世代の説明文")]
        assert _pick_move_effect_text(entries) == "X・Y世代の説明文"

    def test_falls_back_to_any_entry_when_no_priority_match(self):
        """優先度リストに無いバージョングループしか無くても何かしら返す。"""
        entries = [_entry("red-blue", "初代の説明文")]
        assert _pick_move_effect_text(entries) == "初代の説明文"

    def test_placeholder_text_excluded(self):
        """SWSHで技マシン等から削除された技の「この技は使えません」ダミー説明文を除外する
        （実測: sword-shield 826件中147件が該当・bide/barrier等）。"""
        entries = [
            _entry("sword-shield", "この技は使えません思い出すことができなくなりますが"),
            _entry("x-y", "実際の効果説明文"),
        ]
        assert _pick_move_effect_text(entries) == "実際の効果説明文"

    def test_all_placeholder_returns_none(self):
        entries = [_entry("sword-shield", "この技は使えません")]
        assert _pick_move_effect_text(entries) is None

    def test_non_japanese_entries_ignored(self):
        entries = [_entry("sword-shield", "English text", lang="en")]
        assert _pick_move_effect_text(entries) is None

    def test_empty_entries_returns_none(self):
        assert _pick_move_effect_text([]) is None

    def test_fullwidth_space_and_newline_stripped(self):
        entries = [_entry("sword-shield", "激しく　吹きあれる\n風の渦")]
        assert _pick_move_effect_text(entries) == "激しく吹きあれる風の渦"
