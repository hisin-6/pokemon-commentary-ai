"""
compare_local_llm_commentary.py（ローカルLLM候補モデル比較・改善ロードマップ⑦）の単体テスト

テスト対象:
  - _build_kurepi_prompt         状況dictからくれぴプロンプトを組み立て
  - _default_scenario            組み込みサンプル状況
  - _scenario_from_manifest_entry  manifest.jsonlの1件から状況/参照実況文を抽出
  - load_scenario                 render_dir指定有無での分岐
  - call_ollama                   Ollama /api/generate 呼び出し（requestsをモック）
  - run_comparison                複数モデルへの一括呼び出し
  - format_results                比較結果の整形表示
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

_SCRIPT = Path(__file__).parent.parent / "scripts" / "compare_local_llm_commentary.py"
_spec = importlib.util.spec_from_file_location("compare_local_llm_commentary", _SCRIPT)
clc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clc)


class TestBuildKurepiPrompt:
    def test_contains_character_name_and_situation_fields(self):
        situation = {"event_type": "move_single", "turn": 2,
                     "player": "メタグロス HP:91/157", "opponent": "コータス HP:64%",
                     "move_log": ["T1:コータスのふんか"]}
        prompt = clc._build_kurepi_prompt(situation)
        assert "花圓くれぴ" in prompt
        assert "メタグロス HP:91/157" in prompt
        assert "コータス HP:64%" in prompt
        assert "move_single" in prompt
        assert "T1:コータスのふんか" in prompt

    def test_contains_anti_echo_rule(self):
        """2026-08-04の「了解しました」事故対策ルールが全モデル共通で入っていること。"""
        prompt = clc._build_kurepi_prompt(clc._default_scenario())
        assert "了解しました" in prompt

    def test_missing_move_log_does_not_crash(self):
        situation = {"event_type": "battle_start", "turn": 0, "player": "情報収集中",
                     "opponent": "情報収集中"}
        prompt = clc._build_kurepi_prompt(situation)
        assert "battle_start" in prompt

    def test_tone_examples_included(self):
        prompt = clc._build_kurepi_prompt(clc._default_scenario())
        assert "口調のイメージ例" in prompt

    def test_no_recap_past_turns_rule_included(self):
        """2026-08-04: gemma2:9bが前ターンの出来事を今起きたかのように話した実例を
        受けて追加したルール。全モデル共通プロンプトに入っていること。"""
        prompt = clc._build_kurepi_prompt(clc._default_scenario())
        assert "今起きたかのように振り返らない" in prompt

    def test_battle_result_instruction_included_when_present(self):
        situation = {**clc._default_scenario(), "event_type": "battle_end", "battle_result": "勝ち"}
        prompt = clc._build_kurepi_prompt(situation)
        assert "自分の勝ち" in prompt
        assert "締めの実況で必ず勝敗に触れること" in prompt

    def test_battle_result_instruction_omitted_when_absent(self):
        prompt = clc._build_kurepi_prompt(clc._default_scenario())
        assert "締めの実況で必ず勝敗に触れること" not in prompt


class TestDefaultScenario:
    def test_has_expected_keys(self):
        scenario = clc._default_scenario()
        assert set(scenario.keys()) >= {"event_type", "turn", "player", "opponent", "move_log"}


class TestScenarioFromManifestEntry:
    def test_extracts_situation_and_reference(self):
        entry = {
            "event_type": "move_used",
            "commentary": "メタグロスのアイアンヘッド炸裂〜！",
            "context": {"turn": 1, "player": "メタグロス", "opponent": "コータス",
                       "move_log": ["T0:コーチング"]},
        }
        situation, reference = clc._scenario_from_manifest_entry(entry)
        assert situation["player"] == "メタグロス"
        assert situation["move_log"] == ["T0:コーチング"]
        assert reference == "メタグロスのアイアンヘッド炸裂〜！"

    def test_missing_context_falls_back_to_defaults(self):
        entry = {"event_type": "battle_start", "commentary": "バトル開始だ〜！"}
        situation, reference = clc._scenario_from_manifest_entry(entry)
        assert situation["player"] == "情報収集中"
        assert situation["opponent"] == "情報収集中"
        assert situation["move_log"] == []
        assert reference == "バトル開始だ〜！"

    def test_battle_result_extracted_when_present(self):
        entry = {"event_type": "battle_end", "commentary": "勝った！",
                 "context": {"turn": 2, "player": "A", "opponent": "B", "battle_result": "勝ち"}}
        situation, _ = clc._scenario_from_manifest_entry(entry)
        assert situation["battle_result"] == "勝ち"

    def test_battle_result_absent_when_not_in_context(self):
        entry = {"event_type": "battle_end", "commentary": "降参で終了",
                 "context": {"turn": 2, "player": "A", "opponent": "B"}}
        situation, _ = clc._scenario_from_manifest_entry(entry)
        assert "battle_result" not in situation


class TestLoadScenario:
    def test_no_render_dir_returns_default_scenario(self):
        situation, reference = clc.load_scenario(None, 0)
        assert situation == clc._default_scenario()
        assert reference is None

    def test_render_dir_loads_specified_event(self, tmp_path):
        manifest_path = tmp_path / "manifest.jsonl"
        entries = [
            {"event_time": 10.0, "event_type": "move_used", "commentary": "1件目",
             "context": {"turn": 0, "player": "A", "opponent": "B", "move_log": []}},
            {"event_time": 20.0, "event_type": "faint", "commentary": "2件目",
             "context": {"turn": 1, "player": "C", "opponent": "D", "move_log": []}},
        ]
        manifest_path.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
        situation, reference = clc.load_scenario(tmp_path, 1)
        assert situation["player"] == "C"
        assert reference == "2件目"

    def test_out_of_range_event_index_raises(self, tmp_path):
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({"event_time": 1.0, "event_type": "move_used", "commentary": "x",
                       "context": {}}, ensure_ascii=False),
            encoding="utf-8")
        with pytest.raises(IndexError):
            clc.load_scenario(tmp_path, 5)


class TestCallOllama:
    def test_success_returns_text_and_elapsed(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "  炸裂したよ！  "}
        mock_response.raise_for_status.return_value = None
        with patch("compare_local_llm_commentary.requests.post", return_value=mock_response):
            result = clc.call_ollama("qwen2.5:7b", "プロンプト")
        assert result["model"] == "qwen2.5:7b"
        assert result["text"] == "炸裂したよ！"
        assert result["error"] is None
        assert result["elapsed"] >= 0

    def test_connection_error_returns_error_message(self):
        with patch("compare_local_llm_commentary.requests.post",
                  side_effect=requests.exceptions.ConnectionError()):
            result = clc.call_ollama("gemma2:9b", "プロンプト")
        assert result["text"] is None
        assert "Ollama" in result["error"]

    def test_generic_exception_captured_without_raising(self):
        with patch("compare_local_llm_commentary.requests.post", side_effect=ValueError("boom")):
            result = clc.call_ollama("phi3.5", "プロンプト")
        assert result["text"] is None
        assert "boom" in result["error"]


class TestRunComparison:
    def test_calls_each_model_and_collects_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "実況テキスト"}
        mock_response.raise_for_status.return_value = None
        with patch("compare_local_llm_commentary.requests.post", return_value=mock_response) as mock_post:
            results = clc.run_comparison(["qwen2.5:7b", "gemma2:9b"], clc._default_scenario())
        assert [r["model"] for r in results] == ["qwen2.5:7b", "gemma2:9b"]
        assert all(r["text"] == "実況テキスト" for r in results)
        assert mock_post.call_count == 2


class TestFormatResults:
    def test_includes_reference_when_provided(self):
        results = [{"model": "qwen2.5:7b", "text": "テキスト", "elapsed": 1.2, "error": None}]
        out = clc.format_results(results, reference_commentary="参考の実況文")
        assert "参考の実況文" in out
        assert "qwen2.5:7b" in out
        assert "テキスト" in out

    def test_shows_error_tag_on_failure(self):
        results = [{"model": "phi3.5", "text": None, "elapsed": 0.1, "error": "接続失敗"}]
        out = clc.format_results(results)
        assert "[エラー]" in out
        assert "接続失敗" in out
