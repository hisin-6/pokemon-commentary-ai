"""
render_sink.py（実況動画レンダリング素材出力・ADR-009 パス1）の単体テスト

テスト対象:
  - RenderSink            WAV保存・マニフェストJSONL追記・メタ情報
  - Pipeline._speak_async レンダリングモード分岐（保存して再生しない）
"""

import io
import json
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# プロジェクトルートを sys.path に追加（pytest がルートから実行されない場合の保険）
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.output.render_sink import RenderSink
from src.pipeline import Pipeline


def _make_wav(duration: float = 0.5, rate: int = 24000) -> bytes:
    """指定秒数の無音WAVバイト列を生成する。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * duration))
    return buf.getvalue()


class TestRenderSink:
    def test_add_writes_wav_and_manifest(self, tmp_path):
        sink = RenderSink(tmp_path / "out")
        wav = _make_wav(0.5)

        entry = sink.add(12.345, "move_used", "ソーラービームが炸裂！", wav)

        assert entry["seq"] == 1
        assert entry["event_time"] == 12.345
        assert entry["event_type"] == "move_used"
        assert entry["commentary"] == "ソーラービームが炸裂！"
        assert entry["wav"] == "wav/0001_move_used.wav"
        assert entry["duration"] == pytest.approx(0.5, abs=0.01)
        # WAV 実体が保存されている
        assert (tmp_path / "out" / "wav" / "0001_move_used.wav").read_bytes() == wav
        # マニフェストに 1 行、内容が entry と一致
        lines = (tmp_path / "out" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == entry

    def test_seq_increments_and_manifest_appends(self, tmp_path):
        sink = RenderSink(tmp_path)
        sink.add(1.0, "battle_start", "試合開始！", _make_wav(0.2))
        entry2 = sink.add(30.0, "faint", "リザードン倒れた！", _make_wav(0.3))

        assert entry2["seq"] == 2
        assert entry2["wav"] == "wav/0002_faint.wav"
        assert sink.count == 2
        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event_type"] == "battle_start"
        assert json.loads(lines[1])["event_type"] == "faint"

    def test_write_info(self, tmp_path):
        sink = RenderSink(tmp_path)
        sink.write_info({"video": "C:/rec/battle.mp4", "sample_fps": 2.0})

        info = json.loads((tmp_path / "render_info.json").read_text(encoding="utf-8"))
        assert info["video"] == "C:/rec/battle.mp4"
        assert info["sample_fps"] == 2.0
        assert "created_at" in info

    def test_wav_duration(self):
        assert RenderSink.wav_duration(_make_wav(1.5)) == pytest.approx(1.5, abs=0.01)

    def test_context_stored_when_provided(self, tmp_path):
        """context指定時のみマニフェストに記録される（台本パス用）。"""
        sink = RenderSink(tmp_path)
        ctx = {"turn": 3, "player": "場: イダイトウ", "move_log": ["T1:だくりゅう"]}
        entry_with = sink.add(10.0, "move_used", "実況A", _make_wav(0.2), context=ctx)
        entry_without = sink.add(20.0, "switch", "実況B", _make_wav(0.2))

        assert entry_with["context"] == ctx
        assert "context" not in entry_without
        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["context"] == ctx
        assert "context" not in json.loads(lines[1])

    def test_add_moment_appends_timeline(self, tmp_path):
        """瞬間ログがtimeline.jsonlに時刻付きで追記される（台本パス用）。"""
        sink = RenderSink(tmp_path)
        sink.add_moment(200.5, "move", "T3:ガブリアスのドラゴンクロー")
        sink.add_moment(230.0, "move", "T3:ペリッパーのぼうふう")

        lines = (tmp_path / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {"time": 200.5, "kind": "move",
                         "text": "T3:ガブリアスのドラゴンクロー"}

    def test_rerun_clears_previous_materials(self, tmp_path):
        """同じ出力先での再実行は前回のmanifest・wav・fillers・timelineをクリアする
        （追記のままだと新旧混在＋連番WAV同名衝突が起きる・2026-07-14実発生）。"""
        sink1 = RenderSink(tmp_path)
        sink1.add(10.0, "battle_start", "旧実況", _make_wav(0.2))
        sink1.add_moment(20.0, "move", "T1:旧技")
        (tmp_path / "fillers.jsonl").write_text('{"seq": 1}\n', encoding="utf-8")

        sink2 = RenderSink(tmp_path)
        assert not (tmp_path / "manifest.jsonl").exists()
        assert not (tmp_path / "fillers.jsonl").exists()
        assert not (tmp_path / "timeline.jsonl").exists()
        assert list((tmp_path / "wav").glob("*.wav")) == []

        entry = sink2.add(30.0, "switch", "新実況", _make_wav(0.2))
        assert entry["seq"] == 1  # 連番も最初から
        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["commentary"] == "新実況"


def _make_render_pipeline(tmp_path, video_now: float) -> Pipeline:
    """__init__ を通さずに _speak_async に必要な属性だけ持つ Pipeline を作る。"""
    p = Pipeline.__new__(Pipeline)
    p._video_now = video_now
    p._voicevox = MagicMock()
    p._voicevox.generate_wav.return_value = _make_wav(0.4)
    p._player = MagicMock()
    p._render_sink = RenderSink(tmp_path)
    p._speech_thread = None
    return p


class TestSpeakAsyncRenderMode:
    def test_saves_wav_and_skips_playback(self, tmp_path):
        p = _make_render_pipeline(tmp_path, video_now=42.5)

        p._speak_async("いくぞー！", event_type="move_used")
        p._speech_thread.join(timeout=5)

        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "move_used"
        assert entry["event_time"] == 42.5
        assert (tmp_path / entry["wav"]).exists()
        # レンダリングモードでは再生しない（停止もしない）
        p._player.play.assert_not_called()
        p._player.stop.assert_not_called()

    def test_event_time_captured_at_call_not_at_synthesis(self, tmp_path):
        """event_time は呼び出し時点の動画内時刻で確定する（合成中に動画が進んでも変わらない）。"""
        p = _make_render_pipeline(tmp_path, video_now=42.5)

        p._speak_async("いくぞー！", event_type="move_used")
        p._video_now = 99.0  # 合成スレッド実行中に動画が先へ進んだ想定
        p._speech_thread.join(timeout=5)

        entry = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
        assert entry["event_time"] == 42.5

    def test_explicit_event_time_overrides_clock(self, tmp_path):
        """faint統合フラッシュのように検知時刻を明示指定した場合はそちらを使う。"""
        p = _make_render_pipeline(tmp_path, video_now=100.0)

        p._speak_async("倒れた！", event_type="faint", event_time=77.7)
        p._speech_thread.join(timeout=5)

        entry = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
        assert entry["event_time"] == 77.7
        assert entry["event_type"] == "faint"

    def test_normal_mode_still_plays(self, tmp_path):
        """レンダリングモードでなければ従来通り再生する（リグレッションガード）。"""
        p = _make_render_pipeline(tmp_path, video_now=10.0)
        p._render_sink = None

        p._speak_async("通常再生", event_type="move_used")
        p._speech_thread.join(timeout=5)

        p._player.stop.assert_called_once()
        p._player.play.assert_called_once()
        assert not (tmp_path / "manifest.jsonl").exists()
