"""
音声再生モジュール
VOICEVOX が生成した WAV データを指定デバイスに再生する。

実行環境: Windows Python (venv/Scripts/python.exe)
依存パッケージ: pip install sounddevice soundfile

デバイス指定について:
  - device=None (デフォルト): システムのデフォルト出力（スピーカー）に再生
  - device="CABLE Input (VB-Audio Virtual Cable)": バーチャルモーションキャプチャーの
    リップシンクに使用する仮想デバイスに再生（視聴者には音が聞こえない点に注意）
  ※ 音声出力の二重化については ADR-006 TODO を参照
"""

import io

import sounddevice as sd
import soundfile as sf


DEFAULT_DEVICE = None  # None = システムデフォルト（スピーカー）
# CABLE Input に流す場合は以下を使用:
# DEFAULT_DEVICE = "CABLE Input (VB-Audio Virtual Cable)"

# --- このマシンの主要デバイス番号メモ（list_devices() で確認済み）---
#  6  DELL S2721HSX (NVIDIA)          - モニター出力（システムデフォルト）
#  7  CABLE Input (VB-Audio)          - バーチャルモーションキャプチャー口パク用
# 10  スピーカー (2- USB Audio Device) - USB スピーカー
# 12  ヘッドホン (ATH-S220BT)          - Bluetooth ヘッドホン
# 20  CABLE Input (Windows DirectSound) - 同上の別 API
# 使い方例: AudioPlayer(device=10)


class AudioPlayer:
    def __init__(self, device: str | int | None = DEFAULT_DEVICE):
        """
        Args:
            device: 出力デバイス名またはインデックス。None でシステムデフォルト。
                    利用可能なデバイス一覧は list_devices() で確認できる。
        """
        self.device = device

    def play(self, wav_bytes: bytes) -> None:
        """
        WAV バイトデータを再生する（再生完了まで待機）。

        Args:
            wav_bytes: WAV 形式の音声データ
        """
        with io.BytesIO(wav_bytes) as f:
            data, samplerate = sf.read(f)
        sd.play(data, samplerate, device=self.device)
        sd.wait()

    def play_file(self, path: str) -> None:
        """
        WAV ファイルを再生する（再生完了まで待機）。

        Args:
            path: WAV ファイルパス
        """
        data, samplerate = sf.read(path)
        sd.play(data, samplerate, device=self.device)
        sd.wait()

    @staticmethod
    def list_devices() -> None:
        """利用可能なオーディオデバイス一覧を表示する"""
        print(sd.query_devices())


if __name__ == "__main__":
    import sys

    print("=== 利用可能なオーディオデバイス ===")
    AudioPlayer.list_devices()
    print()

    # WAV ファイルパスを引数で受け取るか、VOICEVOX で生成してテスト
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        print(f"再生: {wav_path}")
        player = AudioPlayer()
        player.play_file(wav_path)
        print("再生完了")
    else:
        # VOICEVOX と組み合わせたテスト
        try:
            import requests
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
            from voicevox_client import VoicevoxClient

            text = "ガブリアスのじしんが炸裂！サーフゴーはピンチです！"
            print(f"VOICEVOX + AudioPlayer 連携テスト")
            print(f"テキスト: {text}")

            voicevox = VoicevoxClient()
            wav_bytes = voicevox.generate_wav(text)
            print(f"音声生成成功: {len(wav_bytes)} bytes")

            player = AudioPlayer()
            print("再生中...")
            player.play(wav_bytes)
            print("再生完了")

        except requests.exceptions.ConnectionError:
            print("エラー: VOICEVOX が起動していません。")
        except Exception as e:
            print(f"エラー: {e}")
