"""
テスト共通設定 - 重いGPU依存モジュールをモックして高速にテストを実行する。

pipeline.py のモジュールレベルのインポート順:
  import cv2 / numpy / requests          → venv にある → OK
  from src.capture.screen_capture import → easyocr が必要 → モック
  from src.capture.yolo_detector import  → cv2/numpy のみ → OK
  from src.commentary.phi3_client import → requests のみ → OK
  from src.output.audio_player import    → sounddevice 等 → モック
  from src.output.voicevox_client import → requests のみ → OK
  from src.pokedb.classifier import      → rapidfuzz/sqlite3 → OK

server.py:
  import boto3 / botocore                → AWS SDK → モック
"""

import sys
import logging
from unittest.mock import MagicMock


# ── 1. easyocr をモック（GPU 初期化を防ぐ） ──────────────────────────────────
sys.modules.setdefault("easyocr", MagicMock())


# ── 2. boto3 / botocore をモック（AWS 認証不要にする） ───────────────────────
class _MockClientError(Exception):
    """botocore.exceptions.ClientError の代替。"""
    def __init__(self, *args, **kwargs):
        self.response = {"Error": {"Code": "MockError", "Message": "mock"}}
        super().__init__(*args, **kwargs)


class _MockReadTimeoutError(Exception):
    """botocore.exceptions.ReadTimeoutError の代替。"""
    pass


_mock_botocore_exceptions = MagicMock()
_mock_botocore_exceptions.ClientError = _MockClientError
_mock_botocore_exceptions.ReadTimeoutError = _MockReadTimeoutError

sys.modules.setdefault("boto3", MagicMock())
sys.modules.setdefault("botocore", MagicMock())
sys.modules.setdefault("botocore.config", MagicMock())
sys.modules.setdefault("botocore.exceptions", _mock_botocore_exceptions)


# ── 3. sounddevice / soundfile をモック（オーディオデバイス不要にする） ─────
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())


# ── 4. Python 3.10+ 記法（str | None）を使う src モジュールをモック ──────────
# phi3_client.py / voicevox_client.py は str | None 型ヒントがあり
# Python 3.8 でパースエラーになるため、ファイルを読まれる前にモックする。
# pipeline.py がこれらを from-import する前に sys.modules に入れておく必要がある。
sys.modules["src.commentary.phi3_client"] = MagicMock()
sys.modules["src.output.voicevox_client"] = MagicMock()
sys.modules["src.output.audio_player"]    = MagicMock()


# ── 4. ログファイル生成を抑制（pipeline.py モジュールレベルの _setup_logging） ──
# NullHandler を返すことでファイル生成を防ぐ
class _NullFileHandler(logging.NullHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()

logging.FileHandler = _NullFileHandler  # type: ignore[misc]
