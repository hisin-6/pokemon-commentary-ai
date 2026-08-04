"""
render_sink.py（実況動画レンダリング素材出力・ADR-009 パス1）の単体テスト

テスト対象:
  - RenderSink                        WAV保存・マニフェストJSONL追記・メタ情報
  - Pipeline._speak_async             レンダリングモード分岐（保存して再生しない）
  - Pipeline._dispatch_commentary     後付け生成バッファ／ライブ経路の分岐（ADR-009追記）
  - Pipeline._generate_posthoc_commentary  動画モードの後付け実況生成（ADR-009追記）
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

import src.pipeline as pipeline_module
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

    def test_moment_side_field(self, tmp_path):
        """陣営タグ（2026-07-30・同名ミラー対策）。side=Noneならフィールド自体を
        書かない（旧形式と同一＝後方互換の検証を兼ねる）。"""
        sink = RenderSink(tmp_path)
        sink.add_moment(10.0, "move", "T1:イダイトウのだくりゅう", side="自分")
        sink.add_moment(20.0, "move", "T1:イダイトウのおはかまいり", side=None)

        lines = (tmp_path / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["side"] == "自分"
        assert "side" not in json.loads(lines[1])

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


class TestAddState:
    def test_state_appended_with_time(self, tmp_path):
        """スナップショットが時刻付きでstates.jsonlに追記される。"""
        sink = RenderSink(tmp_path)
        sink.add_state(120.5, {"turn": 3, "player": [], "opponent": [],
                               "alive_player": 2, "alive_opponent": 2})
        rec = json.loads((tmp_path / "states.jsonl").read_text(encoding="utf-8").strip())
        assert rec["time"] == 120.5
        assert rec["turn"] == 3

    def test_identical_state_deduped(self, tmp_path):
        """内容が同一なら追記されない（時刻だけ違っても書かない）。"""
        sink = RenderSink(tmp_path)
        state = {"turn": 3, "player": [{"name": "A", "hp_pct": 50}]}
        sink.add_state(100.0, dict(state))
        sink.add_state(105.0, dict(state))
        sink.add_state(110.0, {"turn": 4, "player": [{"name": "A", "hp_pct": 50}]})
        lines = (tmp_path / "states.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_rerun_clears_states(self, tmp_path):
        sink1 = RenderSink(tmp_path)
        sink1.add_state(10.0, {"turn": 1})
        RenderSink(tmp_path)
        assert not (tmp_path / "states.jsonl").exists()


def _make_posthoc_pipeline(tmp_path) -> Pipeline:
    """__init__ を通さずに _dispatch_commentary に必要な属性だけ持つ Pipeline を作る
    （posthoc_mode=True・動画モードの後付け生成経路）。"""
    p = Pipeline.__new__(Pipeline)
    p._posthoc_mode = True
    p._pending_render_events = []
    p._render_sink = RenderSink(tmp_path)
    p._commentary_history = []
    p._video_now = 10.0
    return p


class TestDispatchCommentaryPosthocMode:
    """posthoc_mode=True では Bedrock/VOICEVOX を呼ばずバッファに積むだけ（ADR-009追記）。"""

    def test_buffers_event_without_network_or_synthesis(self, tmp_path, monkeypatch):
        p = _make_posthoc_pipeline(tmp_path)
        p._voicevox = MagicMock()
        p._render_context = MagicMock(return_value={"turn": 1, "move_log": []})
        bedrock_called = MagicMock()
        monkeypatch.setattr(pipeline_module, "_call_bedrock_vision", bedrock_called)

        game_state = {"ocr_text": "テスト"}
        battle_context = {"player_pokemon": "ガブリアス"}
        p._dispatch_commentary("move_used", None, game_state, battle_context,
                                ["T1:じしん"], attempt_bedrock=True, event_time=12.3)

        assert len(p._pending_render_events) == 1
        ev = p._pending_render_events[0]
        assert ev == {
            "event_time": 12.3,
            "event_type": "move_used",
            "game_state": game_state,
            "battle_context": battle_context,
            "move_log": ["T1:じしん"],
            "render_context": {"turn": 1, "move_log": []},
        }
        bedrock_called.assert_not_called()
        p._voicevox.generate_wav.assert_not_called()
        # マニフェストにはまだ何も書かれない（生成は run() 完了後）
        assert not (tmp_path / "manifest.jsonl").exists()

    def test_event_time_defaults_to_now_when_omitted(self, tmp_path):
        p = _make_posthoc_pipeline(tmp_path)
        p._render_context = MagicMock(return_value=None)

        p._dispatch_commentary("battle_start", None, {"ocr_text": ""}, {}, [],
                                attempt_bedrock=True)

        assert p._pending_render_events[0]["event_time"] == 10.0  # p._video_now


class TestDispatchCommentaryLiveMode:
    """posthoc_mode=False（ライブ経路）は従来どおり即座にBedrock/Phi-3→再生する
    （既存の即時実況経路は無変更・リグレッションガード）。"""

    def _make_pipeline(self):
        p = Pipeline.__new__(Pipeline)
        p._posthoc_mode = False
        p._render_sink = None
        p._commentary_history = []
        p._video_now = 5.0
        p._ec2_url = "http://ec2.example"
        p._classifier = None
        p._move_log = []
        p._phi3 = MagicMock()
        p._voicevox = MagicMock()
        p._voicevox.generate_wav.return_value = _make_wav(0.3)
        p._player = MagicMock()
        p._speech_thread = None
        p._render_context = MagicMock(return_value=None)
        return p

    def test_uses_bedrock_result_and_speaks(self, monkeypatch):
        p = self._make_pipeline()
        monkeypatch.setattr(pipeline_module, "_call_bedrock_vision",
                             lambda *a, **k: ("ガブリアスのじしん炸裂！", "状況説明"))

        p._dispatch_commentary("move_used", None, {"ocr_text": ""}, {}, [], attempt_bedrock=True)
        p._speech_thread.join(timeout=5)

        assert p._commentary_history == ["ガブリアスのじしん炸裂！"]
        p._player.play.assert_called_once()

    def test_falls_back_to_phi3_when_bedrock_returns_none(self, monkeypatch):
        p = self._make_pipeline()
        monkeypatch.setattr(pipeline_module, "_call_bedrock_vision", lambda *a, **k: (None, None))
        p._phi3.generate_commentary.return_value = "フォールバック実況！"

        p._dispatch_commentary("move_used", None, {"ocr_text": "hint"}, {}, [], attempt_bedrock=True)
        p._speech_thread.join(timeout=5)

        assert p._commentary_history == ["フォールバック実況！"]
        p._player.play.assert_called_once()

    def test_attempt_bedrock_false_skips_bedrock_call(self, monkeypatch):
        p = self._make_pipeline()
        bedrock_called = MagicMock()
        monkeypatch.setattr(pipeline_module, "_call_bedrock_vision", bedrock_called)
        p._phi3.generate_commentary.return_value = "フォールバックのみ！"

        p._dispatch_commentary("move_used", None, {"ocr_text": "hint"}, {}, [], attempt_bedrock=False)
        p._speech_thread.join(timeout=5)

        bedrock_called.assert_not_called()
        assert p._commentary_history == ["フォールバックのみ！"]

    def test_glitch_response_replaced_before_speak(self, monkeypatch):
        """保留・困惑応答はライブ経路でもAIグリッチ定型文に差し替えてから合成・再生する。"""
        p = self._make_pipeline()
        monkeypatch.setattr(pipeline_module, "_call_bedrock_vision",
                             lambda *a, **k: ("状況がモヤモヤしていて判断できません", "分析"))

        p._dispatch_commentary("move_used", None, {"ocr_text": ""}, {}, [], attempt_bedrock=True)
        p._speech_thread.join(timeout=5)

        assert len(p._commentary_history) == 1
        expected = [t.format(cause="ナゾのノイズ")
                    for t in pipeline_module._GLITCH_TEMPLATES]
        assert p._commentary_history[0] in expected
        # 音声も差し替え後テキストで合成される
        synthesized = p._voicevox.generate_wav.call_args[0][0]
        assert synthesized == p._commentary_history[0]


class TestGeneratePosthocCommentary:
    """run() 完了後にバッファから実況を一括生成する処理（ADR-009追記）。"""

    def _make_pipeline(self, tmp_path):
        p = Pipeline.__new__(Pipeline)
        p._render_sink = RenderSink(tmp_path)
        p._ec2_url = "http://ec2.example"
        p._classifier = None
        p._phi3 = MagicMock()
        p._voicevox = MagicMock()
        p._voicevox.generate_wav.return_value = _make_wav(0.3)
        p._pending_render_events = []
        # 戦況ウェアハウス（2026-08-04）: 実データ（data/battle_situations.sqlite）を
        # 汚さないようテスト専用のtmp_pathに向ける
        p._situation_db_path = tmp_path / "test_situations.sqlite"
        return p

    def test_generates_manifest_entries_in_event_time_order(self, tmp_path, monkeypatch):
        p = self._make_pipeline(tmp_path)
        p._pending_render_events = [
            {"event_time": 1.0, "event_type": "battle_start", "game_state": {"ocr_text": ""},
             "battle_context": {}, "move_log": [], "render_context": None},
            {"event_time": 30.0, "event_type": "move_used", "game_state": {"ocr_text": ""},
             "battle_context": {}, "move_log": ["T1:じしん"], "render_context": None},
        ]
        responses = iter([("バトル開始だよ！", "a"), ("じしんが炸裂！", "b")])
        monkeypatch.setattr(pipeline_module, "_call_bedrock_text",
                             lambda *a, **k: next(responses))

        p._generate_posthoc_commentary()

        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        entries = [json.loads(line) for line in lines]
        assert entries[0]["commentary"] == "バトル開始だよ！"
        assert entries[0]["event_time"] == 1.0
        assert entries[1]["commentary"] == "じしんが炸裂！"
        assert entries[1]["event_time"] == 30.0

    def test_history_grows_sequentially(self, tmp_path, monkeypatch):
        """直前の実況の繰り返し防止用historyが、ライブ経路と同様に発生順に蓄積される。"""
        p = self._make_pipeline(tmp_path)
        p._pending_render_events = [
            {"event_time": 1.0, "event_type": "battle_start", "game_state": {"ocr_text": ""},
             "battle_context": {}, "move_log": [], "render_context": None},
            {"event_time": 2.0, "event_type": "move_used", "game_state": {"ocr_text": ""},
             "battle_context": {}, "move_log": [], "render_context": None},
        ]
        seen_histories = []

        def fake_call(ec2_url, game_state, event_type, history, battle_context, classifier, move_log):
            seen_histories.append(list(history))
            return (f"実況{len(seen_histories)}", "a")

        monkeypatch.setattr(pipeline_module, "_call_bedrock_text", fake_call)

        p._generate_posthoc_commentary()

        assert seen_histories[0] == []
        assert seen_histories[1] == ["実況1"]

    def test_falls_back_to_phi3_on_bedrock_failure(self, tmp_path, monkeypatch):
        p = self._make_pipeline(tmp_path)
        p._pending_render_events = [
            {"event_time": 5.0, "event_type": "faint", "game_state": {"ocr_text": "倒れた"},
             "battle_context": {}, "move_log": [], "render_context": None},
        ]
        monkeypatch.setattr(pipeline_module, "_call_bedrock_text", lambda *a, **k: (None, None))
        p._phi3.generate_commentary.return_value = "フォールバック実況！"

        p._generate_posthoc_commentary()

        entry = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
        assert entry["commentary"] == "フォールバック実況！"

    def test_skips_event_on_voicevox_error_without_raising(self, tmp_path, monkeypatch):
        p = self._make_pipeline(tmp_path)
        p._pending_render_events = [
            {"event_time": 1.0, "event_type": "battle_start", "game_state": {"ocr_text": ""},
             "battle_context": {}, "move_log": [], "render_context": None},
        ]
        monkeypatch.setattr(pipeline_module, "_call_bedrock_text", lambda *a, **k: ("実況", "a"))
        p._voicevox.generate_wav.side_effect = Exception("voicevox down")

        p._generate_posthoc_commentary()  # 例外を投げずにスキップすること

        assert not (tmp_path / "manifest.jsonl").exists()

    def test_glitch_response_replaced_before_manifest(self, tmp_path, monkeypatch):
        """保留・困惑応答はmanifest.jsonlに書き込まれる前にAIグリッチ定型文へ差し替え、
        差し替え後のテキストでVOICEVOX合成する（2026-07-29恒久対策・07-00-19/08-15-22で
        保留応答がそのまま実況として合成された問題への対策）。"""
        p = self._make_pipeline(tmp_path)
        p._pending_render_events = [
            {"event_time": 262.2, "event_type": "move_used", "game_state": {"ocr_text": ""},
             "battle_context": {}, "move_log": [], "render_context": None},
        ]
        glitch = "データが矛盾していて実況できません。次のフレーム更新を待ちます。"
        monkeypatch.setattr(pipeline_module, "_call_bedrock_text",
                             lambda *a, **k: (glitch, "a"))

        p._generate_posthoc_commentary()

        entry = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
        expected = [t.format(cause="データがちぐはぐさん")
                    for t in pipeline_module._GLITCH_TEMPLATES]
        assert entry["commentary"] in expected
        assert "矛盾" not in entry["commentary"]
        # 音声も差し替え後テキストで合成される（字幕と音声の不一致防止）
        synthesized = p._voicevox.generate_wav.call_args[0][0]
        assert synthesized == entry["commentary"]
