"""src/pokedb/team_preview.py の単体テスト（2026-08-24新設）。

選出前チームプレビュー（自分・相手それぞれ6匹の種族名のみ）の保存・読み込み・
プロンプト用ヒント整形・自分の構築プリセットの保存/読込。
"""

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pokedb.team_preview import (
    TEAM_PREVIEW_FILENAME,
    format_team_preview_hint,
    load_own_team_presets,
    load_team_preview,
    save_own_team_preset,
    save_team_preview,
)


class TestSaveLoadTeamPreview:
    def test_round_trip(self, tmp_path):
        save_team_preview(tmp_path, ["コノヨザル", "ガオガエン"], ["リザードン", "オオニューラ"])
        data = load_team_preview(tmp_path)
        assert data == {
            "own_team": ["コノヨザル", "ガオガエン"],
            "opponent_team": ["リザードン", "オオニューラ"],
        }

    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "not_yet_created"
        save_team_preview(target, ["コノヨザル"], [])
        assert (target / TEAM_PREVIEW_FILENAME).exists()

    def test_blank_entries_excluded(self, tmp_path):
        """GUIで空欄のまま保存した場合、空文字は構築リストから除外する
        （6匹揃っていなくても保存できる仕様）。"""
        save_team_preview(tmp_path, ["コノヨザル", "", "ガオガエン"], ["", "", ""])
        data = load_team_preview(tmp_path)
        assert data["own_team"] == ["コノヨザル", "ガオガエン"]
        assert data["opponent_team"] == []

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_team_preview(tmp_path) is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        (tmp_path / TEAM_PREVIEW_FILENAME).write_text("not json{{{", encoding="utf-8")
        assert load_team_preview(tmp_path) is None

    def test_load_non_dict_json_returns_none(self, tmp_path):
        (tmp_path / TEAM_PREVIEW_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
        assert load_team_preview(tmp_path) is None

    def test_includes_created_at_timestamp(self, tmp_path):
        save_team_preview(tmp_path, ["コノヨザル"], [])
        raw = json.loads((tmp_path / TEAM_PREVIEW_FILENAME).read_text(encoding="utf-8"))
        assert "created_at" in raw


class TestFormatTeamPreviewHint:
    def test_both_sides_present(self):
        hint = format_team_preview_hint(
            {"own_team": ["コノヨザル", "ガオガエン"], "opponent_team": ["リザードン"]})
        assert hint == (
            "自分の構築（選出前・種族のみ）: コノヨザル / ガオガエン ／ "
            "相手の構築（選出前・種族のみ）: リザードン"
        )

    def test_only_opponent_present(self):
        hint = format_team_preview_hint({"own_team": [], "opponent_team": ["リザードン"]})
        assert "自分の構築" not in hint
        assert "相手の構築（選出前・種族のみ）: リザードン" in hint

    def test_both_empty_returns_empty_string(self):
        assert format_team_preview_hint({"own_team": [], "opponent_team": []}) == ""

    def test_missing_keys_treated_as_empty(self):
        assert format_team_preview_hint({}) == ""


class TestOwnTeamPresets:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "own_team_presets.json"
        save_own_team_preset("いつもの構築", ["コノヨザル", "ガオガエン"], path=path)
        presets = load_own_team_presets(path=path)
        assert presets == {"いつもの構築": ["コノヨザル", "ガオガエン"]}

    def test_multiple_presets_coexist(self, tmp_path):
        path = tmp_path / "own_team_presets.json"
        save_own_team_preset("構築A", ["コノヨザル"], path=path)
        save_own_team_preset("構築B", ["ガオガエン"], path=path)
        presets = load_own_team_presets(path=path)
        assert set(presets.keys()) == {"構築A", "構築B"}

    def test_overwrites_same_name(self, tmp_path):
        path = tmp_path / "own_team_presets.json"
        save_own_team_preset("構築A", ["コノヨザル"], path=path)
        save_own_team_preset("構築A", ["ガオガエン"], path=path)
        presets = load_own_team_presets(path=path)
        assert presets == {"構築A": ["ガオガエン"]}

    def test_load_missing_file_returns_empty_dict(self, tmp_path):
        assert load_own_team_presets(path=tmp_path / "nope.json") == {}

    def test_load_corrupt_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / "own_team_presets.json"
        path.write_text("not json{{{", encoding="utf-8")
        assert load_own_team_presets(path=path) == {}
