"""
scripts/resynthesize_fillers.py（フィラー文言差し替え＋WAV再合成）の単体テスト

VOICEVOX は呼ばない（VoicevoxClient をフェイクに差し替える）。

テスト対象:
  - parse_edit   SEQ=新文言 のパース（"=" 無し・文言内の "="・前後空白）
  - load_jsonl / write_jsonl  空行スキップ・日本語のまま書き出し・往復一致
  - main         正常系（対象行だけ更新・WAV上書き・duration再計算・他行不変）
                 fillers.jsonl 無し／存在しない seq／speaker の決定順
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rf = _load("resynthesize_fillers")


def _make_wav(seconds: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class _FakeClient:
    """VoicevoxClient の代わり。文字数×0.1秒のWAVを返し、呼び出しを記録する。"""
    instances: list["_FakeClient"] = []

    def __init__(self, url, speaker):
        self.url = url
        self.speaker = speaker
        self.texts: list[str] = []
        _FakeClient.instances.append(self)

    def generate_wav(self, text: str) -> bytes:
        self.texts.append(text)
        return _make_wav(len(text) * 0.1)


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.instances = []
    monkeypatch.setattr(rf, "VoicevoxClient", _FakeClient)
    return _FakeClient


def _rows():
    return [
        {"seq": 0, "event_time": 1.5, "commentary": "旧その0", "wav": "wav/f0000_filler.wav", "duration": 1.0},
        {"seq": 1, "event_time": 12.0, "commentary": "旧その1", "wav": "wav/f0001_filler.wav", "duration": 2.0},
        {"seq": 2, "event_time": 30.25, "commentary": "旧その2", "wav": "wav/f0002_filler.wav", "duration": 3.0},
    ]


@pytest.fixture
def render_dir(tmp_path):
    (tmp_path / "wav").mkdir()
    for r in _rows():
        (tmp_path / r["wav"]).write_bytes(b"OLD")
    rf.write_jsonl(tmp_path / "fillers.jsonl", _rows())
    return tmp_path


# ── parse_edit ────────────────────────────────────────────────

def test_parse_edit_basic():
    assert rf.parse_edit("3=新しい文言") == (3, "新しい文言")


def test_parse_edit_strips_seq_but_keeps_text():
    assert rf.parse_edit(" 12 = 前後の空白は文言側だけ残る ") == (12, " 前後の空白は文言側だけ残る ")


def test_parse_edit_text_may_contain_equals():
    assert rf.parse_edit("4=HP=半分くらい") == (4, "HP=半分くらい")


def test_parse_edit_without_equals_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        rf.parse_edit("3 新しい文言")


def test_parse_edit_non_int_seq_raises():
    with pytest.raises(ValueError):
        rf.parse_edit("abc=文言")


# ── load_jsonl / write_jsonl ─────────────────────────────────

def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"seq": 0}\n\n   \n{"seq": 1}\n', encoding="utf-8")
    assert rf.load_jsonl(p) == [{"seq": 0}, {"seq": 1}]


def test_write_jsonl_keeps_japanese_and_roundtrips(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = _rows()
    rf.write_jsonl(p, rows)
    text = p.read_text(encoding="utf-8")
    assert "旧その1" in text          # ensure_ascii=False（\uXXXX にならない）
    assert text.count("\n") == len(rows)
    assert rf.load_jsonl(p) == rows


# ── main ─────────────────────────────────────────────────────

def test_main_updates_only_target_rows(render_dir, fake_client, capsys):
    rc = rf.main([str(render_dir), "--edit", "1=あたらしい", "--edit", "2=ことば"])
    assert rc == 0

    rows = rf.load_jsonl(render_dir / "fillers.jsonl")
    before = _rows()
    # 対象外の行は丸ごと不変
    assert rows[0] == before[0]
    # 対象行: commentary と duration だけ変わる
    assert rows[1]["commentary"] == "あたらしい"
    assert rows[1]["duration"] == pytest.approx(0.5)
    assert rows[2]["commentary"] == "ことば"
    assert rows[2]["duration"] == pytest.approx(0.3)
    for new, old in zip(rows[1:], before[1:]):
        assert new["event_time"] == old["event_time"]
        assert new["wav"] == old["wav"]
        assert new["seq"] == old["seq"]

    # WAV: 対象だけ上書き、対象外は元のまま
    assert (render_dir / "wav/f0000_filler.wav").read_bytes() == b"OLD"
    for name, sec in [("f0001_filler.wav", 0.5), ("f0002_filler.wav", 0.3)]:
        data = (render_dir / "wav" / name).read_bytes()
        assert data != b"OLD"
        with wave.open(io.BytesIO(data)) as w:
            assert w.getnframes() / w.getframerate() == pytest.approx(sec)

    assert fake_client.instances[0].texts == ["あたらしい", "ことば"]
    out = capsys.readouterr().out
    assert "旧: 旧その1" in out and "新: あたらしい" in out
    assert "（2件）" in out


def test_main_same_seq_twice_last_wins(render_dir, fake_client):
    rc = rf.main([str(render_dir), "--edit", "0=さいしょ", "--edit", "0=さいご"])
    assert rc == 0
    rows = rf.load_jsonl(render_dir / "fillers.jsonl")
    assert rows[0]["commentary"] == "さいご"
    assert fake_client.instances[0].texts == ["さいご"]   # 合成は1回だけ


def test_main_missing_fillers_returns_1(tmp_path, fake_client, capsys):
    rc = rf.main([str(tmp_path), "--edit", "0=x"])
    assert rc == 1
    assert "見つかりません" in capsys.readouterr().err
    assert fake_client.instances == []   # VOICEVOX に接続しにいかない


def test_main_unknown_seq_returns_1_and_changes_nothing(render_dir, fake_client, capsys):
    original = (render_dir / "fillers.jsonl").read_text(encoding="utf-8")
    rc = rf.main([str(render_dir), "--edit", "1=ok", "--edit", "99=ng"])
    assert rc == 1
    assert "[99]" in capsys.readouterr().err
    # 一部だけ書き換わる事故が無いこと
    assert (render_dir / "fillers.jsonl").read_text(encoding="utf-8") == original
    assert (render_dir / "wav/f0001_filler.wav").read_bytes() == b"OLD"
    assert fake_client.instances == []


def test_main_edit_is_required(render_dir, fake_client):
    with pytest.raises(SystemExit):
        rf.main([str(render_dir)])


@pytest.mark.parametrize("render_info, cli, expected", [
    (None, [], 2),                          # render_info.json 無し → 2
    ({"speaker": 8}, [], 8),                # render_info.json の値
    ({"other": 1}, [], 2),                  # speaker キー無し → 2
    ({"speaker": 8}, ["--speaker", "3"], 3),  # CLI 指定が最優先
])
def test_main_speaker_resolution(render_dir, fake_client, render_info, cli, expected):
    if render_info is not None:
        (render_dir / "render_info.json").write_text(json.dumps(render_info), encoding="utf-8")
    rc = rf.main([str(render_dir), "--edit", "0=x", *cli])
    assert rc == 0
    assert fake_client.instances[0].speaker == expected


def test_main_passes_voicevox_url(render_dir, fake_client):
    rf.main([str(render_dir), "--edit", "0=x", "--voicevox-url", "http://example:1234"])
    assert fake_client.instances[0].url == "http://example:1234"
