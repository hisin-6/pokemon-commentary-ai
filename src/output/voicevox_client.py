"""
VOICEVOX 音声合成クライアント
VOICEVOX API を呼び出してテキストを WAV 音声に変換する。

実行環境: Windows Python (venv/Scripts/python.exe)
VOICEVOX URL: http://localhost:50021
事前準備: VOICEVOX を起動しておくこと
"""

import io
import requests


VOICEVOX_URL = "http://localhost:50021"
DEFAULT_SPEAKER = 1  # ずんだもん（ノーマル）
# 話者一覧は GET /speakers で確認できる


class VoicevoxClient:
    def __init__(
        self,
        url: str = VOICEVOX_URL,
        speaker: int = DEFAULT_SPEAKER,
        timeout: int = 30,
    ):
        self.url = url
        self.speaker = speaker
        self.timeout = timeout

    def generate_wav(self, text: str) -> bytes:
        """
        テキストを WAV 音声データに変換する。

        Args:
            text: 読み上げるテキスト

        Returns:
            WAV 形式の音声データ（bytes）
        """
        audio_query = self._create_audio_query(text)
        return self._synthesize(audio_query)

    def save_wav(self, text: str, path: str) -> str:
        """
        テキストを WAV ファイルに保存する。

        Args:
            text: 読み上げるテキスト
            path: 保存先ファイルパス

        Returns:
            保存したファイルパス
        """
        wav_bytes = self.generate_wav(text)
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return path

    def _create_audio_query(self, text: str) -> dict:
        """VOICEVOX の audio_query を生成する"""
        response = requests.post(
            f"{self.url}/audio_query",
            params={"text": text, "speaker": self.speaker},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _synthesize(self, audio_query: dict) -> bytes:
        """audio_query から WAV を合成する"""
        response = requests.post(
            f"{self.url}/synthesis",
            params={"speaker": self.speaker},
            json=audio_query,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content


if __name__ == "__main__":
    import tempfile
    import os

    client = VoicevoxClient()

    test_text = "ガブリアスのじしんが炸裂！サーフゴーはピンチです！"
    print(f"VOICEVOX 接続テスト中...")
    print(f"テキスト: {test_text}")

    try:
        wav_bytes = client.generate_wav(test_text)
        print(f"音声生成成功: {len(wav_bytes)} bytes")

        # 一時ファイルに保存して確認
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        print(f"WAV 保存先: {tmp_path}")
        print("再生確認後、不要なら削除してください。")

    except requests.exceptions.ConnectionError:
        print("エラー: VOICEVOX が起動していません。VOICEVOX を起動してから再実行してください。")
    except Exception as e:
        print(f"エラー: {e}")
