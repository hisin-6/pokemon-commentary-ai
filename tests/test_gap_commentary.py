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
        """冒頭ギャップは0秒から（margin適用なし）・is_introが付く（2026-08-15）。"""
        gaps = ggc.compute_gaps([_sched(63.0)], min_gap=20.0, margin=2.0)
        assert gaps[0] == {"start": 0.0, "end": 61.0, "is_intro": True}

    def test_between_events_gap_has_margin_on_both_sides(self):
        # 冒頭区間の混入を避けるためintro_min_gapを無効化（0）してイベント間だけ見る
        gaps = ggc.compute_gaps([_sched(10.0, 10.0), _sched(100.0, 10.0)],
                                min_gap=20.0, margin=2.0, intro_min_gap=20.0)
        # 前イベント終了20.0 + 2.0 〜 次イベント開始100.0 - 2.0
        assert {"start": 22.0, "end": 98.0} in gaps

    def test_short_gap_ignored(self):
        gaps = ggc.compute_gaps([_sched(10.0, 10.0), _sched(30.0, 10.0)],
                                min_gap=20.0, margin=2.0, intro_min_gap=20.0)
        # 20〜30秒の間は実質6秒 → 対象外（intro_min_gap=min_gapで冒頭8秒も同基準）
        assert gaps == []

    def test_trailing_gap_only_when_video_duration_known(self):
        scheduled = [_sched(10.0, 10.0)]
        assert ggc.compute_gaps(scheduled, video_duration=0.0, min_gap=20.0,
                                intro_min_gap=20.0) == []
        gaps = ggc.compute_gaps(scheduled, video_duration=100.0,
                                min_gap=20.0, margin=2.0, intro_min_gap=20.0)
        assert gaps == [{"start": 22.0, "end": 98.0}]

    def test_intro_gap_uses_shorter_threshold_than_normal_gaps(self):
        """開始時挨拶を確実に入れるため、冒頭だけはintro_min_gap（通常のmin_gapより
        短い）で対象化される（2026-08-15新設）。"""
        # 冒頭6秒だけならmin_gap=20では対象外だが、intro_min_gap=5なら対象
        gaps = ggc.compute_gaps([_sched(8.0, 10.0)], min_gap=20.0, margin=2.0,
                                intro_min_gap=5.0)
        assert gaps == [{"start": 0.0, "end": 6.0, "is_intro": True}]

    def test_only_leading_gap_gets_is_intro_flag(self):
        """2件目以降のギャップ（イベント間・末尾）にはis_introが付かない。"""
        gaps = ggc.compute_gaps(
            [_sched(5.0, 10.0), _sched(50.0, 10.0)],
            video_duration=100.0, min_gap=20.0, margin=2.0, intro_min_gap=1.0)
        assert gaps[0]["is_intro"] is True
        assert all("is_intro" not in g for g in gaps[1:])


class TestSplitGapsByMoments:
    def test_gap_split_at_moment_times(self):
        """区間内の📺時刻で分割される（先読みネタバレの構造対策）。"""
        gaps = [{"start": 76.0, "end": 131.0}]
        moments = [{"time": 105.0, "kind": "move", "text": "A"},
                   {"time": 111.0, "kind": "move", "text": "B"}]
        result = ggc.split_gaps_by_moments(gaps, moments, min_len=5.0)
        assert result == [{"start": 76.0, "end": 105.0},
                          {"start": 105.0, "end": 111.0},
                          {"start": 111.0, "end": 131.0}]

    def test_short_fragments_dropped(self):
        """min_len未満の断片は捨てられる（フィラー1本が収まらない）。"""
        gaps = [{"start": 150.0, "end": 255.0}]
        moments = [{"time": 222.3, "kind": "move", "text": "A"},
                   {"time": 227.3, "kind": "move", "text": "B"}]
        result = ggc.split_gaps_by_moments(gaps, moments, min_len=12.0)
        # 222.3-227.3（5秒）は落ちる
        assert result == [{"start": 150.0, "end": 222.3},
                          {"start": 227.3, "end": 255.0}]

    def test_no_moments_keeps_gaps(self):
        gaps = [{"start": 0.0, "end": 61.0}]
        assert ggc.split_gaps_by_moments(gaps, []) == gaps

    def test_moment_outside_gap_ignored(self):
        gaps = [{"start": 76.0, "end": 131.0}]
        moments = [{"time": 140.0, "kind": "move", "text": "A"}]
        assert ggc.split_gaps_by_moments(gaps, moments) == gaps


class TestLoadTimeline:
    def test_missing_file_returns_empty(self, tmp_path):
        assert ggc.load_timeline(tmp_path) == []

    def test_sorted_by_time(self, tmp_path):
        import json as _json
        lines = [_json.dumps({"time": 230.0, "kind": "move", "text": "B"}),
                 _json.dumps({"time": 90.0, "kind": "move", "text": "A"})]
        (tmp_path / "timeline.jsonl").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        moments = ggc.load_timeline(tmp_path)
        assert [m["text"] for m in moments] == ["A", "B"]


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


class TestRequestFillers:
    """gapごとに1回ずつ /api/script を呼び、結果を集約する（2026-07-22・ネタバレ抑制の根本強化）。"""

    _events = [{"event_time": 63.0, "event_type": "battle_start", "commentary": "開幕だ！"}]
    _gaps = [{"start": 0.0, "end": 61.0}, {"start": 78.0, "end": 131.0}]

    def test_calls_once_per_gap_with_matching_gap_payload(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(json)
            gap = json["gap"]
            resp = MockResponse({
                "success": True,
                "fillers": [{"time": gap["start"] + 1, "text": f"filler-{gap['start']}"}],
                "usage": {},
            })
            return resp

        monkeypatch.setattr(ggc.requests, "post", fake_post)
        fillers = ggc.request_fillers("http://ec2", self._events, self._gaps)

        assert len(calls) == 2
        assert [c["gap"] for c in calls] == self._gaps
        assert len(fillers) == 2
        assert fillers[0]["text"] == "filler-0.0"
        assert fillers[1]["text"] == "filler-78.0"

    def test_raises_on_api_failure(self, monkeypatch):
        def fake_post(url, json=None, timeout=None):
            return MockResponse({"success": False, "error": "bedrock_error", "message": "NG"})

        monkeypatch.setattr(ggc.requests, "post", fake_post)
        with pytest.raises(RuntimeError):
            ggc.request_fillers("http://ec2", self._events, self._gaps)

    def test_default_persona_is_kurepi(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(json)
            return MockResponse({"success": True, "fillers": [], "usage": {}})

        monkeypatch.setattr(ggc.requests, "post", fake_post)
        ggc.request_fillers("http://ec2", self._events, self._gaps)

        assert all(c["persona"] == "kurepi" for c in calls)

    def test_persona_neutral_is_forwarded_to_payload(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(json)
            return MockResponse({"success": True, "fillers": [], "usage": {}})

        monkeypatch.setattr(ggc.requests, "post", fake_post)
        ggc.request_fillers("http://ec2", self._events, self._gaps, persona="neutral")

        assert all(c["persona"] == "neutral" for c in calls)


class MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload
