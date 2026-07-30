"""
voicevox_client.py の単体テスト（TTS読み置換）

conftest.py が src.output.voicevox_client を sys.modules でモックしている
（Python 3.8 互換のため）ので、importlib でファイルから直接ロードして
実物をテストする。sys.modules には登録しない（モックを壊さない）。
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).parent.parent / "src" / "output" / "voicevox_client.py"
_spec = importlib.util.spec_from_file_location("voicevox_client_real", _PATH)
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


class TestTtsReadings:
    """「花圓」の誤読（はなえん）対策: 合成時だけ読み仮名に置換する。"""

    def test_kaen_reading_replaced(self):
        assert (vc.apply_tts_readings("やあやあ、花圓くれぴだよ♪")
                == "やあやあ、はなまるくれぴだよ♪")

    def test_text_without_entries_unchanged(self):
        text = "ガブリアスのじしんが炸裂！"
        assert vc.apply_tts_readings(text) == text

    def test_generate_wav_synthesizes_with_reading(self):
        """generate_wav は置換後テキストで audio_query を作る（元テキストは
        呼び出し側の字幕・manifest 用にそのまま残る）。"""
        client = vc.VoicevoxClient()
        captured = {}

        def fake_query(text):
            captured["text"] = text
            return {"q": 1}

        client._create_audio_query = fake_query
        client._synthesize = lambda q: b"wav"

        assert client.generate_wav("花圓くれぴだよ") == b"wav"
        assert captured["text"] == "はなまるくれぴだよ"
