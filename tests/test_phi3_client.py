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


class TestAttributionError:
    """_has_attribution_error: 2026-08-04・phi3.5が「我がイッカネズミ」（相手のポケモンを
    自分呼ばわり）した実機事故を受けて追加した簡易検証（PokéLLMon論文のconsistent action
    generationに着想）。"""

    _CONTEXT = {"player_names": ["メタグロス"], "opponent_names": ["イッカネズミ"]}

    def test_opponent_pokemon_called_self_is_error(self):
        assert Phi3Client._has_attribution_error("我がイッカネズミが頑張る！", self._CONTEXT)

    def test_player_pokemon_called_opponent_is_error(self):
        assert Phi3Client._has_attribution_error("相手のメタグロスが炸裂！", self._CONTEXT)

    def test_correct_attribution_is_not_error(self):
        text = "我がメタグロスと相手のイッカネズミの対決だ！"
        assert not Phi3Client._has_attribution_error(text, self._CONTEXT)

    def test_no_battle_context_is_not_error(self):
        assert not Phi3Client._has_attribution_error("我がイッカネズミが頑張る！", None)

    def test_missing_names_fields_is_not_error(self):
        assert not Phi3Client._has_attribution_error("我がイッカネズミが頑張る！", {"turn": 1})


class TestSamplesVoting:
    """generate_commentary(samples=N): 帰属エラーの無いサンプルを採用する多数決
    （2026-08-04・改善ロードマップ「戦況推論強化」フェーズ3）。"""

    _CONTEXT = {"player_names": ["メタグロス"], "opponent_names": ["イッカネズミ"]}

    def _mock_response(self, text):
        r = MagicMock()
        r.json.return_value = {"response": text}
        r.raise_for_status.return_value = None
        return r

    def test_default_samples_is_one_call(self, client):
        with patch("src.commentary.phi3_client.requests.post",
                  return_value=self._mock_response("正常な実況文")) as mock_post:
            client.generate_commentary({"event_type": "move_used"}, battle_context=self._CONTEXT)
        assert mock_post.call_count == 1

    def test_first_clean_sample_is_returned_without_extra_calls(self, client):
        with patch("src.commentary.phi3_client.requests.post",
                  return_value=self._mock_response("我がメタグロスが炸裂！")) as mock_post:
            result = client.generate_commentary(
                {"event_type": "move_used"}, battle_context=self._CONTEXT, samples=3)
        assert result == "我がメタグロスが炸裂！"
        assert mock_post.call_count == 1  # 1発目がクリーンなので追加生成しない

    def test_retries_until_clean_sample_found(self, client):
        responses = [
            self._mock_response("我がイッカネズミが頑張る！"),  # 帰属エラー
            self._mock_response("相手のメタグロスが来た！"),      # 帰属エラー
            self._mock_response("我がメタグロスが炸裂！"),        # クリーン
        ]
        with patch("src.commentary.phi3_client.requests.post", side_effect=responses) as mock_post:
            result = client.generate_commentary(
                {"event_type": "move_used"}, battle_context=self._CONTEXT, samples=3)
        assert result == "我がメタグロスが炸裂！"
        assert mock_post.call_count == 3

    def test_all_samples_dirty_falls_back_to_first(self, client):
        responses = [
            self._mock_response("我がイッカネズミが頑張る！"),
            self._mock_response("相手のメタグロスが来た！"),
        ]
        with patch("src.commentary.phi3_client.requests.post", side_effect=responses):
            result = client.generate_commentary(
                {"event_type": "move_used"}, battle_context=self._CONTEXT, samples=2)
        assert result == "我がイッカネズミが頑張る！"  # 先頭のサンプルを採用
