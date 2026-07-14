"""
generate_gap_commentary.py（台本パス・ADR-009 パス1.5）の単体テスト

テスト対象:
  - compute_gaps           イベント実況スケジュールからの無言区間検出
  - clamp_fillers_to_gaps  Bedrock出力のギャップ範囲への丸め・範囲外破棄
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/ はパッケージではないためファイルパスから直接ロードする
# （モジュール自身が scripts/ を sys.path に足して render_commentary_video を import する）
_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_gap_commentary.py"
_spec = importlib.util.spec_from_file_location("generate_gap_commentary", _SCRIPT)
ggc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ggc)


def _sched(start, duration=10.0):
    return {"start": start, "duration": duration}


class TestComputeGaps:
    def test_leading_gap_detected_without_margin_at_zero(self):
        """冒頭ギャップは0秒から（margin適用なし）。"""
        gaps = ggc.compute_gaps([_sched(63.0)], min_gap=20.0, margin=2.0)
        assert gaps[0] == {"start": 0.0, "end": 61.0}

    def test_between_events_gap_has_margin_on_both_sides(self):
        gaps = ggc.compute_gaps([_sched(10.0, 10.0), _sched(100.0, 10.0)],
                                min_gap=20.0, margin=2.0)
        # 前イベント終了20.0 + 2.0 〜 次イベント開始100.0 - 2.0
        assert {"start": 22.0, "end": 98.0} in gaps

    def test_short_gap_ignored(self):
        gaps = ggc.compute_gaps([_sched(10.0, 10.0), _sched(30.0, 10.0)],
                                min_gap=20.0, margin=2.0)
        # 20〜30秒の間は実質6秒 → 対象外
        assert gaps == []

    def test_trailing_gap_only_when_video_duration_known(self):
        scheduled = [_sched(10.0, 10.0)]
        assert ggc.compute_gaps(scheduled, video_duration=0.0, min_gap=20.0) == []
        gaps = ggc.compute_gaps(scheduled, video_duration=100.0,
                                min_gap=20.0, margin=2.0)
        assert gaps == [{"start": 22.0, "end": 98.0}]


class TestClampFillersToGaps:
    _gaps = [{"start": 0.0, "end": 61.0}, {"start": 78.0, "end": 131.0}]

    def test_in_range_kept_as_is(self):
        kept, dropped = ggc.clamp_fillers_to_gaps(
            [{"time": 30.0, "text": "実況"}], self._gaps)
        assert kept == [{"time": 30.0, "text": "実況"}]
        assert dropped == []

    def test_slightly_out_of_range_clamped_to_nearest_gap(self):
        """ギャップの少し外の time は最寄りのギャップ端に丸められる。"""
        kept, _ = ggc.clamp_fillers_to_gaps(
            [{"time": 70.0, "text": "実況"}], self._gaps)
        assert len(kept) == 1
        assert kept[0]["time"] in (61.0, 78.0)

    def test_far_out_of_range_dropped(self):
        kept, dropped = ggc.clamp_fillers_to_gaps(
            [{"time": 300.0, "text": "実況"}], self._gaps)
        assert kept == []
        assert len(dropped) == 1

    def test_result_sorted_by_time(self):
        kept, _ = ggc.clamp_fillers_to_gaps(
            [{"time": 100.0, "text": "後"}, {"time": 30.0, "text": "先"}],
            self._gaps)
        assert [f["text"] for f in kept] == ["先", "後"]
