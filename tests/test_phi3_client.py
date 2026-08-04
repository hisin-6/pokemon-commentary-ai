"""
src/commentary/phi3_client.py（ローカルLLMフォールバック・2026-08-04にgemma2:9b化）の単体テスト

テスト対象:
  - Phi3Client._build_prompt   状況（game_state/bedrock_analysis/battle_context）からの
                                プロンプト組み立て
  - Phi3Client.generate_commentary  Ollama /api/generate 呼び出し（requestsをモック）
  - 履歴管理（_add_history/clear_history）

⚠️ conftest.py はpipeline.py用に `src.commentary.phi3_client` を丸ごとMagicMockに差し替えて
いるため、このテストファイルではその影響を受けないよう素のPythonインポートを使う
（pipeline.py経由ではなく直接importするだけなので、conftestのsys.modules差し替えは
このテストの実行順序次第で有効/無効になり得る。確実に実体をテストするため、
テスト冒頭でsys.modulesから該当キーを削除してから読み直す）。
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.modules.pop("src.commentary.phi3_client", None)
from src.commentary.phi3_client import Phi3Client  # noqa: E402


@pytest.fixture
def client():
    return Phi3Client()


class TestDefaultModel:
    def test_default_model_is_gemma2(self, client):
        """2026-08-04: Phi-3 miniからgemma2:9bへ切り替え（改善ロードマップ⑦第一歩）。"""
        assert client.model == "gemma2:9b"


class TestBuildPrompt:
    def test_contains_character_and_output_rules(self, client):
        prompt = client._build_prompt({"event_type": "move_used", "status": "なし"}, None, None)
        assert "花圓くれぴ" in prompt
        assert "了解しました" in prompt  # 指示エコー対策ルール

    def test_battle_context_fields_included(self, client):
        battle_context = {
            "turn": 2, "player_pokemon": "場: メタグロス / 控え: なし",
            "opponent_pokemon": "場: コータス / 控え: なし",
            "event_log": "T2:メタグロスのアイアンヘッド",
        }
        prompt = client._build_prompt({"event_type": "move_single"}, None, battle_context)
        assert "場: メタグロス / 控え: なし" in prompt
        assert "場: コータス / 控え: なし" in prompt
        assert "T2:メタグロスのアイアンヘッド" in prompt

    def test_missing_battle_context_falls_back_gracefully(self, client):
        prompt = client._build_prompt({"event_type": "battle_start"}, None, None)
        assert "battle_start" in prompt

    def test_battle_result_instruction_included_when_present(self, client):
        battle_context = {"turn": 3, "battle_result": "勝ち"}
        prompt = client._build_prompt({"event_type": "battle_end"}, None, battle_context)
        assert "自分の勝ち" in prompt
        assert "締めの実況で必ず勝敗に触れること" in prompt

    def test_bedrock_analysis_included_as_supplement(self, client):
        prompt = client._build_prompt({"event_type": "move_used"}, "画面には炎が見える", None)
        assert "画面の補足情報: 画面には炎が見える" in prompt

    def test_undetected_bedrock_analysis_excluded(self, client):
        prompt = client._build_prompt({"event_type": "move_used"}, "（テキスト未検出）", None)
        assert "画面の補足情報" not in prompt

    def test_status_included_when_present(self, client):
        prompt = client._build_prompt({"event_type": "move_used", "status": "まひ"}, None, None)
        assert "状態異常: まひ" in prompt

    def test_history_included_after_generation(self, client):
        client._add_history("さっきの実況文")
        prompt = client._build_prompt({"event_type": "move_used"}, None, None)
        assert "直前の実況（繰り返さないこと）: さっきの実況文" in prompt


class TestGenerateCommentary:
    def test_success_returns_stripped_text_and_updates_history(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "  炸裂したよ！  "}
        mock_response.raise_for_status.return_value = None
        with patch("src.commentary.phi3_client.requests.post", return_value=mock_response) as mock_post:
            result = client.generate_commentary({"event_type": "move_used"})
        assert result == "炸裂したよ！"
        assert client._history == ["炸裂したよ！"]
        assert mock_post.call_args.kwargs["json"]["model"] == "gemma2:9b"

    def test_history_trimmed_to_history_size(self):
        client = Phi3Client(history_size=2)
        for text in ["a", "b", "c"]:
            client._add_history(text)
        assert client._history == ["b", "c"]

    def test_clear_history_resets(self, client):
        client._add_history("x")
        client.clear_history()
        assert client._history == []

    def test_connection_error_propagates(self, client):
        with patch("src.commentary.phi3_client.requests.post",
                  side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(requests.exceptions.ConnectionError):
                client.generate_commentary({"event_type": "move_used"})
