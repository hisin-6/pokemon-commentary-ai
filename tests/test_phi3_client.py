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

    def test_contains_target_defense_side_uncertainty_rules(self, client):
        """2026-08-14: kurepi_persona.OUTPUT_RULES_LINESに追加した対象/防御結果/陣営の
        断定回避ルールがPhi3Client側にもそのまま反映されること（単一ソース経由）。"""
        prompt = client._build_prompt({"event_type": "move_used", "status": "なし"}, None, None)
        assert "対象ポケモン名を" in prompt and "決め打ちしない" in prompt
        assert "まもる等の防御が成功したかどうか" in prompt
        assert "陣営（自分/相手）が状況情報から特定できない場合" in prompt

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

    def test_switch_focus_instruction_for_switch_event(self, client):
        """交代ヒント（2026-08-15・server.py側と同じ文言で配線）。"""
        game_state = {"event_type": "switch", "switch_focus": "自分のペリッパー"}
        prompt = client._build_prompt(game_state, None, None)
        assert "実際に繰り出されたのは「自分のペリッパー」" in prompt

    def test_switch_focus_info_line_for_move_used(self, client):
        game_state = {"event_type": "move_used", "switch_focus": "自分のブリジュラス"}
        prompt = client._build_prompt(game_state, None, None)
        assert "自分のブリジュラス" in prompt
        assert "必ずこれに従うこと" in prompt

    def test_move_used_turn_transition_framing(self, client):
        """2026-08-15: move_usedを戦況全体フレーミングに変更（server.py側と同じ趣旨）。"""
        prompt = client._build_prompt({"event_type": "move_used"}, None, None)
        assert "新しいターンの攻防が始まる場面" in prompt

    def test_move_target_hint_included_when_present(self, client):
        """技の対象ヒント（2026-08-15・server.py側と同じ文言で配線）。"""
        battle_context = {
            "turn": 3,
            "move_target_hint": "ライチュウ（相手側）は攻撃から身を守った＝この技は防がれた",
        }
        prompt = client._build_prompt({"event_type": "move_single"}, None, battle_context)
        assert "身を守った＝この技は防がれた" in prompt
        assert "必ず上記の観測に従うこと" in prompt

    def test_move_target_hint_omitted_when_absent(self, client):
        prompt = client._build_prompt({"event_type": "move_single"}, None, {"turn": 3})
        assert "観測された変化" not in prompt

    def test_surrender_instruction_included_when_present(self, client):
        """降参決着（2026-08-15）: 気絶による全滅と捏造しない指示を出す。"""
        battle_context = {"turn": 3, "battle_result": "負け", "battle_surrendered": True}
        prompt = client._build_prompt({"event_type": "battle_end"}, None, battle_context)
        assert "降参（ギブアップ）によって終了した" in prompt
        assert "自分が降参を選んだ" in prompt

    def test_surrender_instruction_without_result(self, client):
        """勝敗未検出の降参決着ではどちらの降参か断定させない。"""
        battle_context = {"turn": 3, "battle_surrendered": True}
        prompt = client._build_prompt({"event_type": "battle_end"}, None, battle_context)
        assert "どちらの降参かは不明" in prompt

    def test_no_surrender_instruction_when_absent(self, client):
        prompt = client._build_prompt({"event_type": "battle_end"}, None, {"turn": 3})
        assert "降参（ギブアップ）" not in prompt

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

    def test_move_single_hint_embeds_move_focus(self, client):
        """技ごとの実況（move_single）: move_focus に積んだ「陣営の＋ポケモン＋の＋技」を
        プロンプトに埋め込み、この技1つだけに焦点を絞らせる。

        2026-08-08発見: server.py（Bedrock）側は元々move_focusを使っていたが、
        Phi3Client（ローカルLLMフォールバック）側だけ配線が漏れており、move_singleの
        実況が直近の技ログ全体から古い技を拾って話してしまうバグがあった。"""
        game_state = {"event_type": "move_single", "move_focus": "自分のガブリアスのじしん"}
        prompt = client._build_prompt(game_state, None, None)
        assert "自分のガブリアスのじしん" in prompt
        assert "1つだけに反応する" in prompt

    def test_move_single_hint_handles_missing_move_focus(self, client):
        """move_focus が無くても例外にならない（指示行を追加しないだけ）。"""
        prompt = client._build_prompt({"event_type": "move_single"}, None, None)
        assert "1つだけに反応する" not in prompt

    def test_non_move_single_event_has_no_move_focus_hint(self, client):
        """move_single以外のイベントではmove_focusが渡っていても指示行を追加しない。"""
        game_state = {"event_type": "move_used", "move_focus": "自分のガブリアスのじしん"}
        prompt = client._build_prompt(game_state, None, None)
        assert "1つだけに反応する" not in prompt

    def test_type_hint_included_when_present(self, client):
        """2026-08-08: server.py（Bedrock）側には既にあったタイプ相性ヒントの配線が
        Phi3Client側だけ漏れていたバグの修正確認。"""
        battle_context = {"type_hint": "メタグロスの技はコータスにバツグン"}
        prompt = client._build_prompt({"event_type": "move_used"}, None, battle_context)
        assert "メタグロスの技はコータスにバツグン" in prompt
        assert "信頼して有利不利の実況に使ってよい" in prompt

    def test_type_hint_omitted_when_absent(self, client):
        prompt = client._build_prompt({"event_type": "move_used"}, None, {"turn": 1})
        assert "タイプ相性ヒント" not in prompt

    def test_move_effect_hint_included_when_present(self, client):
        """2026-08-14: 技効果ヒントRAG（server.py側と同じ文言でPhi3Client側にも配線。
        最頻NGパターン「技の効果に関する事実誤認」対策）。"""
        battle_context = {"move_effect_hint": "おいかぜ: 味方全員の素早さをあげる。"}
        prompt = client._build_prompt({"event_type": "move_used"}, None, battle_context)
        assert "おいかぜ: 味方全員の素早さをあげる。" in prompt
        assert "ダメージを与えない変化技" in prompt

    def test_move_effect_hint_omitted_when_absent(self, client):
        prompt = client._build_prompt({"event_type": "move_used"}, None, {"turn": 1})
        assert "ダメージを与えない変化技" not in prompt

    def test_condition_hint_included_when_present(self, client):
        """2026-08-08: 天候「にほんばれ」下でウェザーボールが炎技になることを
        ローカルLLMが知らず「水技」と誤って実況したバグ（renders/2026-06-07_12-48-22の
        実機検証で発見）の根本原因＝condition_hintの配線漏れの修正確認。"""
        battle_context = {"condition_hint": "にほんばれが5ターン継続中"}
        prompt = client._build_prompt({"event_type": "move_single"}, None, battle_context)
        assert "にほんばれが5ターン継続中" in prompt
        assert "信頼して有利不利の実況に使ってよい" in prompt

    def test_condition_hint_omitted_when_absent(self, client):
        prompt = client._build_prompt({"event_type": "move_used"}, None, {"turn": 1})
        assert "場のコンディション" not in prompt

    def test_no_conditions_active_explicitly_stated_when_absent(self, client):
        """2026-08-16: server.py側と同じ「いずれも発生していない」の明示。
        condition_hintが無い場合にLLMが独自の知識で天候/おいかぜ等を捏造する事故
        （実機2026-08-14_20-52-59）への対策をPhi3Client側にも配線。"""
        prompt = client._build_prompt({"event_type": "move_used"}, None, {"turn": 1})
        assert "いずれも発生していない" in prompt

    def test_condition_hint_present_forbids_inventing_unlisted_conditions(self, client):
        battle_context = {"condition_hint": "あめが3ターン継続中"}
        prompt = client._build_prompt({"event_type": "move_used"}, None, battle_context)
        assert "独自の知識や一般的な戦術の連想で" in prompt

    def test_move_effect_hint_weather_power_caution(self, client):
        """2026-08-16: server.py側と同じ天候ボーナス捏造禁止の注記
        （実機2026-08-14_20-52-59: ぼうふうを「あまごい下で威力絶大」と誤実況）。"""
        battle_context = {"move_effect_hint": "ぼうふう: 強烈な風で相手を包みこんで攻撃する。"}
        prompt = client._build_prompt({"event_type": "move_used"}, None, battle_context)
        assert "天候による技の威力アップ" in prompt

    def test_faint_focus_embeds_inferred_faint_hint(self, client):
        """合成faint（ボール数減少推定）: 画面に0%表示が無いため、確定済みの対象を
        直接指示する。server.py側と同時配線（move_focus/type_hintで過去2回あった
        片側配線漏れの再発防止）。"""
        game_state = {"event_type": "faint", "faint_focus": "相手のリキキリン"}
        prompt = client._build_prompt(game_state, None, None)
        assert "相手のリキキリン" in prompt
        assert "蓄積した戦況データから確定" in prompt

    def test_faint_without_focus_has_no_inferred_hint(self, client):
        """通常のfaint（OCRの0%表示由来）ではfaint_focusが無く、指示行を追加しない。"""
        prompt = client._build_prompt({"event_type": "faint"}, None, None)
        assert "蓄積した戦況データから確定" not in prompt

    def test_non_faint_event_has_no_faint_focus_hint(self, client):
        game_state = {"event_type": "move_used", "faint_focus": "相手のリキキリン"}
        prompt = client._build_prompt(game_state, None, None)
        assert "蓄積した戦況データから確定" not in prompt

    def test_faint_context_included_when_present(self, client):
        """faint→move_used統合時の直前気絶情報（faint_context）の配線確認。"""
        game_state = {"event_type": "move_used",
                      "faint_context": "場(自)=ガブリアス | 場(相)=リキキリン"}
        prompt = client._build_prompt(game_state, None, None)
        assert "場(自)=ガブリアス | 場(相)=リキキリン" in prompt


class TestPersonaSwitch:
    """2026-08-14新設: 3Dモデル一時差し替え検証用のpersona="neutral"切り替え
    （--persona neutral）。デフォルトは"kurepi"のまま従来動作を維持する。"""

    def test_default_persona_is_kurepi(self):
        client = Phi3Client()
        prompt = client._build_prompt({"event_type": "move_used", "status": "なし"}, None, None)
        assert "花圓くれぴ" in prompt

    def test_neutral_persona_excludes_kurepi_name(self):
        client = Phi3Client(persona="neutral")
        prompt = client._build_prompt({"event_type": "move_used", "status": "なし"}, None, None)
        assert "くれぴ" not in prompt
        assert "花圓" not in prompt

    def test_neutral_persona_still_contains_output_rules(self):
        """SLANG_GLOSSARY_LINES/OUTPUT_RULES_LINESはキャラ非依存のため両ペルソナで
        共用されること（差し替え不要な部分が誤って消えていないことの確認）。"""
        client = Phi3Client(persona="neutral")
        prompt = client._build_prompt({"event_type": "move_used", "status": "なし"}, None, None)
        assert "了解しました" in prompt  # OUTPUT_RULES_LINES由来
        assert "集中攻撃" in prompt  # SLANG_GLOSSARY_LINES由来


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
