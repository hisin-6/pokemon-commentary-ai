"""
render_commentary_video.py（実況動画合成・ADR-009 パス2）の単体テスト

テスト対象:
  - dedupe_entries          近接時間内の同一実況文の除去
  - schedule_entries        後ろ倒しスケジューリング（かぶり解決・遅延超過破棄）
  - resolve_video_path      Windowsパス→WSLパス変換
  - build_commentary_track  実況トラックWAVの配置・長さ
"""

import importlib.util
import io
import json
import wave
from pathlib import Path

import pytest

# scripts/ はパッケージではないためファイルパスから直接ロードする
_SCRIPT = Path(__file__).parent.parent / "scripts" / "render_commentary_video.py"
_spec = importlib.util.spec_from_file_location("render_commentary_video", _SCRIPT)
rcv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcv)


def _entry(event_time, duration=10.0, commentary="実況", event_type="move_used",
           wav="wav/0001_move_used.wav", seq=1):
    return {"seq": seq, "event_time": event_time, "event_type": event_type,
            "commentary": commentary, "wav": wav, "duration": duration}


class TestDedupeEntries:
    def test_identical_text_within_window_dropped(self):
        """近接時間の同一実況文は後の方が破棄される（faint/switch同文の実測ケース）。"""
        entries = [_entry(133.2, commentary="同じ実況", event_type="faint"),
                   _entry(144.0, commentary="同じ実況", event_type="switch")]
        kept, dropped = rcv.dedupe_entries(entries)
        assert len(kept) == 1
        assert kept[0]["event_time"] == 133.2  # 早い方が残る
        assert len(dropped) == 1
        assert dropped[0]["event_type"] == "switch"

    def test_identical_text_outside_window_kept(self):
        """時間が離れた同一文（定型実況など）は両方残る。"""
        entries = [_entry(10.0, commentary="同じ実況"),
                   _entry(10.0 + rcv._DEDUPE_WINDOW_SEC + 1, commentary="同じ実況")]
        kept, dropped = rcv.dedupe_entries(entries)
        assert len(kept) == 2
        assert dropped == []

    def test_different_text_kept(self):
        entries = [_entry(10.0, commentary="実況A"), _entry(12.0, commentary="実況B")]
        kept, dropped = rcv.dedupe_entries(entries)
        assert len(kept) == 2
        assert dropped == []


class TestScheduleEntries:
    def test_no_overlap_keeps_event_time(self):
        """かぶりが無ければイベント時刻ちょうどに配置される。"""
        entries = [_entry(10.0, duration=5.0), _entry(30.0, duration=5.0)]
        scheduled, dropped = rcv.schedule_entries(entries)
        assert [e["start"] for e in scheduled] == [10.0, 30.0]
        assert all(e["delay"] == 0.0 for e in scheduled)
        assert dropped == []

    def test_overlap_pushed_back_with_gap(self):
        """かぶったら前の実況の終了+gapまで後ろ倒しされる。"""
        entries = [_entry(10.0, duration=10.0), _entry(15.0, duration=5.0)]
        scheduled, _ = rcv.schedule_entries(entries, gap=0.5)
        assert scheduled[1]["start"] == pytest.approx(20.5)
        assert scheduled[1]["delay"] == pytest.approx(5.5)

    def test_chained_pushback(self):
        """後ろ倒しが連鎖する（3件が数珠つなぎ）。"""
        entries = [_entry(10.0, duration=10.0), _entry(11.0, duration=10.0),
                   _entry(12.0, duration=10.0)]
        scheduled, _ = rcv.schedule_entries(entries, gap=0.5)
        assert scheduled[1]["start"] == pytest.approx(20.5)
        assert scheduled[2]["start"] == pytest.approx(31.0)

    def test_exceeding_max_delay_dropped(self):
        """遅延がmax_delayを超える実況は破棄され、後続はかぶらなければ定刻。"""
        entries = [_entry(10.0, duration=30.0), _entry(12.0, duration=5.0),
                   _entry(50.0, duration=5.0)]
        scheduled, dropped = rcv.schedule_entries(entries, gap=0.5, max_delay=20.0)
        assert len(dropped) == 1
        assert dropped[0]["event_time"] == 12.0
        assert dropped[0]["delay"] == pytest.approx(28.5)
        # 破棄されたエントリはカーソルを進めない
        assert [e["start"] for e in scheduled] == [10.0, 50.0]

    def test_unsorted_input_sorted_by_event_time(self):
        """manifest保存順（faintフラッシュで時刻順と不一致）でも時刻順に整列。"""
        entries = [_entry(144.0, event_type="switch"), _entry(133.2, event_type="faint")]
        scheduled, _ = rcv.schedule_entries(entries)
        assert [e["event_time"] for e in scheduled] == [133.2, 144.0]


def _filler(event_time, duration=8.0, commentary="フィラー", seq=1):
    return {"seq": seq, "event_time": event_time, "event_type": "filler",
            "commentary": commentary, "wav": f"wav/f{seq:04d}_filler.wav",
            "duration": duration}


class TestFitFillers:
    def test_placed_at_desired_time_when_free(self):
        """空いていれば希望時刻ちょうどに置かれる。"""
        scheduled = [dict(_entry(10.0, duration=10.0), start=10.0),
                     dict(_entry(100.0, duration=10.0), start=100.0)]
        placed, dropped = rcv.fit_fillers(scheduled, [_filler(50.0)])
        assert len(placed) == 1
        assert placed[0]["start"] == 50.0
        assert dropped == []

    def test_pushed_past_event_when_shift_within_max(self):
        """イベント実況と重なるフィラーはイベントを動かさず後ろへずれる
        （ずれがmax_shift以内なら採用）。"""
        scheduled = [dict(_entry(50.0, duration=10.0), start=50.0)]
        placed, _ = rcv.fit_fillers(scheduled, [_filler(48.0)],
                                    gap=0.5, max_shift=15.0)
        assert placed[0]["start"] == pytest.approx(60.5)

    def test_dropped_when_shift_exceeds_max(self):
        """max_shiftを超えるずれが必要なフィラーは破棄される
        （既定12秒: 長いイベント実況を飛び越える配置は基本却下）。"""
        scheduled = [dict(_entry(50.0, duration=13.0), start=50.0)]
        placed, dropped = rcv.fit_fillers(scheduled, [_filler(48.0)], gap=0.5)
        assert placed == []
        assert len(dropped) == 1
        assert dropped[0]["delay"] == pytest.approx(15.5)

    def test_fillers_do_not_overlap_each_other(self):
        """フィラー同士も重ならない（先に置いた方が占有区間になる）。"""
        placed, _ = rcv.fit_fillers([], [_filler(10.0, duration=8.0, seq=1),
                                         _filler(12.0, duration=8.0, seq=2)],
                                    gap=0.5)
        assert placed[0]["start"] == 10.0
        assert placed[1]["start"] == pytest.approx(18.5)

    def test_event_before_filler_not_pushed(self):
        """フィラーの後ろ倒しでも後続イベント実況は動かない。"""
        scheduled = [dict(_entry(10.0, duration=10.0), start=10.0),
                     dict(_entry(30.0, duration=5.0), start=30.0)]
        # 28秒に8秒フィラー → 30秒のイベントとぶつかる → イベント後の35.5へ
        placed, _ = rcv.fit_fillers(scheduled, [_filler(28.0, duration=8.0)],
                                    gap=0.5)
        assert placed[0]["start"] == pytest.approx(35.5)
        # イベント側は不変
        assert [e["start"] for e in scheduled] == [10.0, 30.0]


class TestLoadFillers:
    def test_missing_file_returns_empty(self, tmp_path):
        assert rcv.load_fillers(tmp_path) == []

    def test_sorted_by_event_time(self, tmp_path):
        lines = [json.dumps(_filler(100.0, seq=2), ensure_ascii=False),
                 json.dumps(_filler(50.0, seq=1), ensure_ascii=False)]
        (tmp_path / "fillers.jsonl").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
        fillers = rcv.load_fillers(tmp_path)
        assert [f["event_time"] for f in fillers] == [50.0, 100.0]


class TestResolveVideoPath:
    def test_windows_drive_path_converted_to_wsl(self, tmp_path, monkeypatch):
        """存在しないWindowsパスは /mnt/<drive>/ 形式に変換して探す。"""
        monkeypatch.setattr(rcv.sys, "platform", "linux")
        # /mnt/x/... は存在しないので変換候補も見つからず元パスが返る
        result = rcv.resolve_video_path("X:\\foo\\bar.mp4")
        assert str(result) == "X:\\foo\\bar.mp4"

    def test_existing_path_returned_as_is(self, tmp_path):
        video = tmp_path / "battle.mp4"
        video.write_bytes(b"")
        assert rcv.resolve_video_path(str(video)) == video


def _write_wav(path, seconds, rate=24000, value=1000):
    """一定振幅のモノラル16bit WAVを生成する。"""
    frames = int(seconds * rate)
    data = value.to_bytes(2, "little", signed=True) * frames
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


class TestBuildCommentaryTrack:
    def test_clips_placed_at_scheduled_time(self, tmp_path):
        """各WAVがstart秒の位置に配置され、間は無音になる。"""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        _write_wav(wav_dir / "0001_a.wav", 1.0)
        _write_wav(wav_dir / "0002_b.wav", 1.0)
        scheduled = [
            {"start": 2.0, "wav": "wav/0001_a.wav"},
            {"start": 5.0, "wav": "wav/0002_b.wav"},
        ]
        out = tmp_path / "track.wav"
        track_len = rcv.build_commentary_track(tmp_path, scheduled, out)
        assert track_len == pytest.approx(6.0)

        with wave.open(str(out), "rb") as w:
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())

        def sample_at(sec):
            i = int(sec * rate) * 2
            return int.from_bytes(raw[i:i + 2], "little", signed=True)

        assert sample_at(1.0) == 0       # 配置前は無音
        assert sample_at(2.5) == 1000    # クリップ1の途中
        assert sample_at(4.0) == 0       # クリップ間は無音
        assert sample_at(5.5) == 1000    # クリップ2の途中

    def test_min_duration_pads_track(self, tmp_path):
        """min_duration指定でトラックが動画長まで無音パディングされる。"""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        _write_wav(wav_dir / "0001_a.wav", 1.0)
        scheduled = [{"start": 0.0, "wav": "wav/0001_a.wav"}]
        out = tmp_path / "track.wav"
        track_len = rcv.build_commentary_track(tmp_path, scheduled, out,
                                               min_duration=10.0)
        assert track_len == pytest.approx(10.0)

    def test_format_mismatch_raises(self, tmp_path):
        """WAVフォーマット不一致は明示的にエラーにする。"""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        _write_wav(wav_dir / "0001_a.wav", 1.0, rate=24000)
        _write_wav(wav_dir / "0002_b.wav", 1.0, rate=48000)
        scheduled = [
            {"start": 0.0, "wav": "wav/0001_a.wav"},
            {"start": 2.0, "wav": "wav/0002_b.wav"},
        ]
        with pytest.raises(ValueError, match="フォーマット不一致"):
            rcv.build_commentary_track(tmp_path, scheduled, tmp_path / "track.wav")

    def test_empty_schedule_raises(self, tmp_path):
        with pytest.raises(ValueError, match="0件"):
            rcv.build_commentary_track(tmp_path, [], tmp_path / "track.wav")


class TestLoadManifest:
    def test_sorted_by_event_time(self, tmp_path):
        """保存順が時刻順と違っても event_time 昇順で返る。"""
        lines = [
            json.dumps(_entry(144.0, seq=2), ensure_ascii=False),
            json.dumps(_entry(133.2, seq=3), ensure_ascii=False),
            json.dumps(_entry(63.0, seq=1), ensure_ascii=False),
        ]
        (tmp_path / "manifest.jsonl").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        entries = rcv.load_manifest(tmp_path)
        assert [e["event_time"] for e in entries] == [63.0, 133.2, 144.0]
