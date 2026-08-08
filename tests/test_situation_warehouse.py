"""
src/analytics/situation_warehouse.py（データウェアハウスの箱・2026-08-04新規）の単体テスト
"""

import sqlite3

import pytest

from src.analytics.situation_warehouse import (
    backfill_outcome,
    clear_match,
    count_situations,
    record_situation,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_situations.sqlite"


class TestRecordSituation:
    def test_creates_db_and_inserts_row(self, db_path):
        record_situation({"match_id": "m1", "turn": "1", "event_type": "move_used"}, db_path=db_path)
        assert db_path.exists()
        assert count_situations(db_path=db_path) == 1

    def test_missing_match_id_raises(self, db_path):
        with pytest.raises(ValueError):
            record_situation({"turn": "1"}, db_path=db_path)

    def test_multiple_inserts_accumulate(self, db_path):
        for i in range(3):
            record_situation({"match_id": "m1", "turn": str(i)}, db_path=db_path)
        assert count_situations(db_path=db_path) == 3

    def test_unspecified_columns_are_null(self, db_path):
        record_situation({"match_id": "m1"}, db_path=db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT weather, outcome FROM situations").fetchone()
        conn.close()
        assert row == (None, None)

    def test_extra_keys_in_snapshot_are_ignored(self, db_path):
        record_situation({"match_id": "m1", "not_a_real_column": "x"}, db_path=db_path)
        assert count_situations(db_path=db_path) == 1


class TestBackfillOutcome:
    def test_updates_all_rows_for_match(self, db_path):
        for i in range(3):
            record_situation({"match_id": "m1", "turn": str(i)}, db_path=db_path)
        record_situation({"match_id": "m2", "turn": "0"}, db_path=db_path)

        updated = backfill_outcome("m1", "勝ち", db_path=db_path)
        assert updated == 3

        conn = sqlite3.connect(db_path)
        outcomes = [r[0] for r in conn.execute(
            "SELECT outcome FROM situations WHERE match_id = 'm1'")]
        other_outcome = conn.execute(
            "SELECT outcome FROM situations WHERE match_id = 'm2'").fetchone()[0]
        conn.close()
        assert outcomes == ["勝ち", "勝ち", "勝ち"]
        assert other_outcome is None

    def test_unknown_match_id_updates_nothing(self, db_path):
        record_situation({"match_id": "m1"}, db_path=db_path)
        assert backfill_outcome("does-not-exist", "勝ち", db_path=db_path) == 0


class TestClearMatch:
    """2026-08-08追加: 同じ動画（match_id）の再実行で新旧スナップショットが
    混在する事故（RenderSinkの「前回素材の自動クリア」と同種の問題）への対策。"""

    def test_removes_only_rows_for_given_match(self, db_path):
        for i in range(3):
            record_situation({"match_id": "m1", "turn": str(i)}, db_path=db_path)
        record_situation({"match_id": "m2", "turn": "0"}, db_path=db_path)

        removed = clear_match("m1", db_path=db_path)

        assert removed == 3
        assert count_situations(db_path=db_path) == 1
        conn = sqlite3.connect(db_path)
        remaining = conn.execute("SELECT match_id FROM situations").fetchall()
        conn.close()
        assert remaining == [("m2",)]

    def test_unknown_match_id_removes_nothing(self, db_path):
        record_situation({"match_id": "m1"}, db_path=db_path)
        assert clear_match("does-not-exist", db_path=db_path) == 0
        assert count_situations(db_path=db_path) == 1

    def test_clear_then_reinsert_leaves_only_new_rows(self, db_path):
        """再実行のシナリオ: クリア→新しいスナップショットで入れ直す。"""
        record_situation({"match_id": "m1", "turn": "0", "weather": "旧データ"}, db_path=db_path)
        clear_match("m1", db_path=db_path)
        record_situation({"match_id": "m1", "turn": "0", "weather": "新データ"}, db_path=db_path)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT weather FROM situations WHERE match_id = 'm1'").fetchall()
        conn.close()
        assert rows == [("新データ",)]


class TestCountSituations:
    def test_returns_zero_for_nonexistent_db(self, tmp_path):
        assert count_situations(db_path=tmp_path / "no_such.sqlite") == 0

    def test_returns_accurate_count(self, db_path):
        record_situation({"match_id": "m1"}, db_path=db_path)
        record_situation({"match_id": "m2"}, db_path=db_path)
        assert count_situations(db_path=db_path) == 2
