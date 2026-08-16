"""
generate_chapters.py（実況動画のYouTubeチャプター自動生成）の単体テスト

テスト対象:
  - build_chapters       manifest.jsonlエントリからチャプター境界を組み立てる
  - format_chapters      (時刻, ラベル)リストをYouTube貼り付け用テキストに変換
  - _format_timestamp    秒数→M:SS / H:MM:SS
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_chapters.py"
_spec = importlib.util.spec_from_file_location("generate_chapters", _SCRIPT)
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def _entry(event_time, event_type, turn=None):
    return {"event_time": event_time, "event_type": event_type, "context": {"turn": turn}}


class TestFormatTimestamp:
    def test_under_a_minute(self):
        assert gc._format_timestamp(9) == "0:09"

    def test_minutes_and_seconds(self):
        assert gc._format_timestamp(125) == "2:05"

    def test_over_an_hour(self):
        assert gc._format_timestamp(3725) == "1:02:05"


class TestBuildChapters:
    def test_empty_entries_returns_opening_only(self):
        assert gc.build_chapters([]) == [(0.0, "オープニング")]

    def test_first_chapter_always_zero(self):
        entries = [_entry(53.5, "battle_start", turn=0)]
        chapters = gc.build_chapters(entries)
        assert chapters[0] == (0.0, "オープニング")

    def test_battle_start_and_turn_transitions_labeled(self):
        entries = [
            _entry(53.5, "battle_start", turn=0),
            _entry(70.0, "move_single", turn=1),
            _entry(121.9, "move_single", turn=2),
        ]
        chapters = gc.build_chapters(entries)
        labels = [label for _, label in chapters]
        assert labels == ["オープニング", "試合開始", "ターン1", "ターン2"]

    def test_same_turn_events_do_not_duplicate_chapter(self):
        """同じターン内の複数イベントはチャプターを増やさない（ターンの切り替わりのみ拾う）。"""
        entries = [
            _entry(53.5, "battle_start", turn=0),
            _entry(70.0, "move_single", turn=1),
            _entry(80.0, "move_single", turn=1),
            _entry(90.0, "switch", turn=1),
        ]
        chapters = gc.build_chapters(entries)
        assert [label for _, label in chapters] == ["オープニング", "試合開始", "ターン1"]

    def test_battle_end_labeled_without_revealing_result(self):
        """battle_endは「決着」とだけ表示し、勝敗（ネタバレ）を含めない。"""
        entries = [
            _entry(53.5, "battle_start", turn=0),
            _entry(500.0, "battle_end", turn=8),
        ]
        chapters = gc.build_chapters(entries)
        assert chapters[-1][1] == "決着"

    def test_min_gap_merges_close_chapters(self):
        """YouTube仕様: チャプターは最低10秒（既定）離れている必要がある。"""
        entries = [
            _entry(53.5, "battle_start", turn=0),
            _entry(56.0, "move_single", turn=1),  # battle_startから2.5秒後→間引かれる
            _entry(200.0, "move_single", turn=2),
        ]
        chapters = gc.build_chapters(entries, min_gap=10.0)
        labels = [label for _, label in chapters]
        assert "ターン1" not in labels
        assert labels == ["オープニング", "試合開始", "ターン2"]

    def test_custom_min_gap(self):
        entries = [
            _entry(53.5, "battle_start", turn=0),
            _entry(56.0, "move_single", turn=1),
        ]
        chapters = gc.build_chapters(entries, min_gap=1.0)
        assert [label for _, label in chapters] == ["オープニング", "試合開始", "ターン1"]


class TestFormatChapters:
    def test_joins_with_newlines(self):
        text = gc.format_chapters([(0.0, "オープニング"), (53.5, "試合開始")])
        assert text == "0:00 オープニング\n0:53 試合開始"
