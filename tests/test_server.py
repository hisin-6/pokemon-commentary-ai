"""
src/api/server.py の単体テスト

テスト対象:
  - _parse_commentary()      【実況】【状況】セクション抽出
  - _build_vision_prompt()   Bedrock プロンプト構築
  - GET  /health             死活確認エンドポイント
  - POST /api/vision         バリデーション（Bedrock 呼び出しはモック）
  - POST /api/log            バリデーション（S3 呼び出しはモック）
"""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# conftest.py で boto3/botocore がモック済みなのでインポートできる
import src.api.server as server_module
from src.api.server import _build_vision_prompt, _parse_commentary, app


# ─── テスト用の小さな PNG（1x1 pixel 透過）──────────────────────────────────
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_commentary
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseCommentary:

    def test_extracts_jikkyou_section(self):
        text = "【状況】HPが減っている\n【実況】ピカチュウが技を放った！"
        _, commentary = _parse_commentary(text)
        assert "ピカチュウが技を放った" in commentary
        assert "状況" not in commentary

    def test_extracts_jokyo_section(self):
        text = "【状況】両者のHPが減っている\n【実況】激しいバトルだ"
        analysis, _ = _parse_commentary(text)
        assert "両者のHPが減っている" in analysis

    def test_fallback_when_no_markers(self):
        """マーカーがない場合は全文をそのまま返す。"""
        text = "何らかの実況テキスト"
        analysis, commentary = _parse_commentary(text)
        assert commentary == text

    def test_jikkyou_trims_whitespace(self):
        text = "【実況】  スペースあり  "
        _, commentary = _parse_commentary(text)
        assert not commentary.startswith(" ")
        assert not commentary.endswith(" ")

    def test_multiple_brackets_stops_at_next(self):
        """【実況】の後に別の【セクション】が来たら、そこで切る。"""
        text = "【実況】技を使った！【補足】これは補足"
        _, commentary = _parse_commentary(text)
        assert "補足" not in commentary

    def test_empty_string(self):
        analysis, commentary = _parse_commentary("")
        assert analysis == ""
        assert commentary == ""

    def test_markdown_heading_normalized_to_brackets(self):
        """Bedrockが【】の代わりに# 見出しで返すケース（2026-07-13発見バグ）。"""
        text = "# 状況\n両者HPが減っている\n# 実況\nピカチュウが技を放った！"
        analysis, commentary = _parse_commentary(text)
        assert "ピカチュウが技を放った" in commentary
        assert "#" not in commentary
        assert "状況" not in commentary

    def test_markdown_heading_with_hash_levels(self):
        text = "## 状況\n説明\n### 実況\n実況テキスト"
        _, commentary = _parse_commentary(text)
        assert commentary == "実況テキスト"


# ═══════════════════════════════════════════════════════════════════════════════
# _build_vision_prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildVisionPrompt:

    def _context(self, event_type="move_used", **kwargs):
        base = {
            "event_type": event_type,
            "ocr_text": "バツグンだ",
            "hp_values": ["150/200"],
            "name_candidates_player": ["ピカチュウ"],
            "name_candidates_opponent": ["エルフーン"],
        }
        base.update(kwargs)
        return base

    def _battle_state(self, **kwargs):
        base = {
            "turn": 3,
            "player_field": "ピカチュウ",
            "player_bench": "エースバーン",
            "opponent_field": "エルフーン",
            "opponent_bench": "ガオガエン",
        }
        base.update(kwargs)
        return base

    def test_prompt_contains_event_type(self):
        prompt = _build_vision_prompt(self._context("move_used"), [], self._battle_state())
        assert "move_used" in prompt

    def test_prompt_contains_battle_start_hint(self):
        prompt = _build_vision_prompt(self._context("battle_start"), [], self._battle_state())
        assert "バトル開始" in prompt

    def test_prompt_contains_faint_hint(self):
        prompt = _build_vision_prompt(self._context("faint"), [], self._battle_state())
        assert "倒れた" in prompt or "HP=0" in prompt

    def test_faint_focus_switches_to_inferred_hint(self):
        """合成faint（ボール数減少推定）: 画面に0%表示が無いため「HP=0のポケモンを
        特定」ではなく確定済みの対象を直接指示するevent_hintに差し替える。"""
        prompt = _build_vision_prompt(
            self._context("faint", faint_focus="相手のリキキリン"), [], self._battle_state())
        assert "相手のリキキリン" in prompt
        assert "残りポケモン数の減少から確定" in prompt
        assert "HP=0のポケモンを特定すること" not in prompt

    def test_faint_focus_ignored_for_other_events(self):
        """faint以外のイベントではevent_hintを差し替えない（状況セクションへの
        faint_focus行追加は行われるため、event_hint固有の文言で判定する）。"""
        prompt = _build_vision_prompt(
            self._context("move_used", faint_focus="相手のリキキリン"), [], self._battle_state())
        assert "倒れたことに今気づいた体で" not in prompt
        assert "今ターンで使われた技とその効果を実況する" in prompt

    def test_faint_context_included_when_present(self):
        """faint→move_used統合時のfaint_contextはpipeline側から送信されていたのに
        プロンプトで一度も使われていなかった配線漏れ（2026-08-10発見）の修正確認。"""
        prompt = _build_vision_prompt(
            self._context("move_used", faint_context="場(自)=ピカチュウ | 場(相)=エルフーン"),
            [], self._battle_state())
        assert "場(自)=ピカチュウ | 場(相)=エルフーン" in prompt

    def test_prompt_contains_ocr_text(self):
        prompt = _build_vision_prompt(self._context(ocr_text="ピカチュウのかみなり"), [], self._battle_state())
        assert "ピカチュウのかみなり" in prompt

    def test_battle_result_rule_when_present(self):
        """battle_endで勝敗が確定していれば明言指示を出す（2026-07-30視聴fb#4）。"""
        prompt = _build_vision_prompt(
            self._context("battle_end", battle_result="勝ち"), [], self._battle_state())
        assert "自分の勝ち" in prompt
        assert "勝敗に触れること" in prompt

    def test_no_battle_result_rule_when_absent(self):
        prompt = _build_vision_prompt(self._context("battle_end"), [], self._battle_state())
        assert "勝敗に触れること" not in prompt

    def test_target_uncertainty_rule_present(self):
        """2026-08-14: 技の対象不明時に対象ポケモン名を決め打ちしないよう指示する
        安全策（ダブルバトル対象取り違え対策）。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "対象ポケモン名を" in prompt and "決め打ちしない" in prompt

    def test_defense_result_uncertainty_rule_present(self):
        """2026-08-14: 直前ターンの防御成功有無が不明な場合にダメージ命中を
        断定しないよう指示する安全策（まもる無視/捏造対策）。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "まもる等の防御が成功したかどうか" in prompt

    def test_side_uncertainty_rule_present(self):
        """2026-08-14: 技を使った陣営（自分/相手）が不明な場合に断定しないよう
        指示する安全策（陣営取り違え対策）。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "陣営（自分/相手）が状況情報から特定できない場合" in prompt

    def test_move_effect_hint_included_when_present(self):
        """2026-08-14: 技効果ヒントRAG（最頻NGパターン「技の効果に関する事実誤認」対策）。"""
        prompt = _build_vision_prompt(
            self._context(), [], self._battle_state(
                move_effect_hint="おいかぜ: 味方全員の素早さをあげる。"))
        assert "おいかぜ: 味方全員の素早さをあげる。" in prompt
        assert "ダメージを与えない変化技" in prompt

    def test_move_effect_hint_omitted_when_absent(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "ダメージを与えない変化技" not in prompt

    def test_effectiveness_tag_usage_rule(self):
        """2026-08-04: パイプライン側でバツグンタグを検出できるようになったため、
        『絶対に使うな』の全面禁止から『タグがあれば信頼して使ってよい』に変更した。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "「（バツグン）」の注記がある技は、実際に検出された" in prompt
        assert "信頼して有利不利の実況に使ってよい" in prompt
        assert "絶対に実況文に含めてはいけない" not in prompt

    def test_type_hint_included_when_present(self):
        """2026-08-04: pipeline.py側で計算したタイプ相性ヒント（Cicero型アーキテクチャ）
        が battle_state.type_hint 経由で渡された場合、プロンプトに含める。"""
        prompt = _build_vision_prompt(
            self._context(), [], self._battle_state(type_hint="メタグロスの技はコータスにバツグン"))
        assert "メタグロスの技はコータスにバツグン" in prompt
        assert "信頼して有利不利の実況に使ってよい" in prompt

    def test_type_hint_omitted_when_absent(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "タイプ相性ヒント" not in prompt

    def test_condition_hint_included_when_present(self):
        """2026-08-04: 天候/壁/速度操作ヒント（Cicero型アーキテクチャ）が
        battle_state.condition_hint 経由で渡された場合、プロンプトに含める。"""
        prompt = _build_vision_prompt(
            self._context(), [],
            self._battle_state(condition_hint="あめが3ターン継続中 / トリックルーム中"))
        assert "あめが3ターン継続中 / トリックルーム中" in prompt
        assert "信頼して有利不利の実況に使ってよい" in prompt

    def test_condition_hint_omitted_when_absent(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "場のコンディション" not in prompt

    def test_condition_hint_overrides_transient_ocr_text(self):
        """2026-08-07: renders/07-03-23-34-29_condition_checkの実機検証で、
        condition_hintが「壁が張られている」と言ってるのにBedrockが「壁が消えた」と
        矛盾する実況をする不具合を発見。生OCRの一時的な演出テキストより
        condition_hintを優先する指示を追加したことを確認する。"""
        prompt = _build_vision_prompt(
            self._context(), [],
            self._battle_state(condition_hint="相手側にリフレクターが張られている（あと3ターン）"))
        assert "画面テキストに「壁が消えた」等の記述や過去の発動演出が見えても" in prompt
        assert "それより必ずこちらを優先すること" in prompt

    def test_persona_self_name_and_perspective(self):
        """自称くれぴ（花圓の誤読対策）と自分側視点の固定（2026-07-30視聴fb#1・#3）。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "「花圓」という漢字表記は実況文に書かないこと" in prompt
        assert "自分側を応援する立場" in prompt
        assert "今この瞬間に起きたこと" in prompt  # 視聴fb#5: 過去振り返り抑制

    def test_prompt_contains_turn_info(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state(turn=5))
        assert "5" in prompt

    def test_prompt_contains_history(self):
        prompt = _build_vision_prompt(
            self._context(),
            ["前回の実況テキスト"],
            self._battle_state(),
        )
        assert "前回の実況テキスト" in prompt

    def test_prompt_contains_rag_info(self):
        context = self._context()
        context["rag_pokemon_info"] = ["ピカチュウ: でんき / せいでんき"]
        prompt = _build_vision_prompt(context, [], self._battle_state())
        assert "ピカチュウ: でんき" in prompt

    def test_prompt_no_rag_info_when_empty(self):
        context = self._context()
        context["rag_pokemon_info"] = []
        prompt = _build_vision_prompt(context, [], self._battle_state())
        assert "ポケモン図鑑情報" not in prompt

    def test_prompt_contains_jikkyou_marker(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "【実況】" in prompt

    def test_prompt_contains_jokyo_marker(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "【状況】" in prompt

    def test_prompt_player_field_and_bench_shown(self):
        prompt = _build_vision_prompt(
            self._context(),
            [],
            self._battle_state(player_field="ピカチュウ", player_bench="エースバーン"),
        )
        assert "ピカチュウ" in prompt
        assert "エースバーン" in prompt

    def test_prompt_contains_kurepi_persona(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "くれぴ" in prompt

    def test_prompt_contains_turn_history_when_present(self):
        prompt = _build_vision_prompt(
            self._context(), [],
            self._battle_state(turn_history="T1: 自分=ピカチュウ80% / 相手=リザードン65%"),
        )
        assert "ピカチュウ80%" in prompt

    def test_prompt_turn_history_defaults_to_nashi(self):
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "ターン推移: なし" in prompt

    def test_has_image_false_omits_image_dependent_instructions(self):
        """動画モードの後付け生成（画像なし・ADR-009追記）では、画像の目視判断を
        求める指示（HPバー位置・画像からの技名読み取り等）を含めてはいけない。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state(), has_image=False)
        assert "画像を直接見て" not in prompt
        assert "画像のバトルメッセージ" not in prompt
        assert "画像に状態異常アイコンが見えたら" not in prompt

    def test_has_image_true_keeps_image_dependent_instructions(self):
        """デフォルト（has_image省略時）はライブ経路の従来プロンプトのまま。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "画像を直接見て、HPバーの位置から自分と相手のポケモンを判断すること" in prompt

    def test_has_image_false_still_references_ocr_detected_moves(self):
        prompt = _build_vision_prompt(
            self._context(), [], self._battle_state(), has_image=False,
        )
        assert "OCRで検出した使用技" in prompt

    def test_move_single_hint_embeds_move_focus(self):
        """技ごとの実況（move_single）: move_focus に積んだ「陣営の＋ポケモン＋の＋技」を
        プロンプトのイベント指示に埋め込み、この技1つだけに焦点を絞らせる。"""
        prompt = _build_vision_prompt(
            self._context("move_single", move_focus="自分のガブリアスのじしん"),
            [], self._battle_state(),
        )
        assert "自分のガブリアスのじしん" in prompt
        assert "1つだけに反応する" in prompt

    def test_move_single_hint_handles_missing_move_focus(self):
        """move_focus が無くても例外にならない（空文字扱い）。"""
        prompt = _build_vision_prompt(self._context("move_single"), [], self._battle_state())
        assert "1つだけに反応する" in prompt

    def test_includes_slang_glossary(self):
        """改善ロードマップ④（口調・知識改善）: 対戦スラング用語集が注入されている。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "集中攻撃" in prompt
        assert "積み" in prompt

    def test_includes_tone_examples(self):
        """口調のfew-shot例文が注入されている。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "この試合とは無関係な架空例" in prompt
        assert "狙うはただ1匹" in prompt

    def test_includes_doubles_tactics_knowledge(self):
        """改善ロードマップ④続き: ダブルバトルの技・特性・戦術知識が注入されている。"""
        prompt = _build_vision_prompt(self._context(), [], self._battle_state())
        assert "ねこだまし＝必ずひるませる" in prompt
        assert "いかく＝場に出た瞬間敵2体の攻撃を下げる" in prompt
        assert "トリックルームパ" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_health_returns_timestamp(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "timestamp" in data


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/vision
# ═══════════════════════════════════════════════════════════════════════════════

def _valid_vision_payload(**overrides):
    base = {
        "image_base64": _TINY_PNG_B64,
        "context": {
            "event_type": "move_used",
            "ocr_text": "バツグンだ",
        },
        "history": [],
        "battle_state": {"turn": 1},
    }
    base.update(overrides)
    return base


class TestVisionEndpoint:

    def test_missing_json_returns_400(self, client):
        resp = client.post("/api/vision", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_image_calls_bedrock_without_image_block(self, client):
        """image_base64 は任意（動画モードの後付け生成・ADR-009追記）。
        画像なしでも200で応答し、Bedrockへは画像ブロックを含めずテキストのみ送る。"""
        payload = _valid_vision_payload(image_base64="")
        mock_response_body = {
            "content": [{"text": "【実況】テスト実況"}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
        mock_bedrock_response = {
            "body": MagicMock(read=MagicMock(return_value=json.dumps(mock_response_body).encode())),
        }

        with patch.object(server_module.bedrock, "invoke_model",
                           return_value=mock_bedrock_response) as mock_invoke:
            resp = client.post("/api/vision", json=payload)

        assert resp.status_code == 200
        assert resp.get_json()["success"] is True
        sent_body = json.loads(mock_invoke.call_args.kwargs["body"])
        content = sent_body["messages"][0]["content"]
        assert all(block["type"] != "image" for block in content)
        assert any(block["type"] == "text" for block in content)

    def test_missing_context_returns_400(self, client):
        payload = _valid_vision_payload(context={})
        resp = client.post("/api/vision", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "missing_context"

    def test_invalid_event_type_returns_400(self, client):
        payload = _valid_vision_payload(
            context={"event_type": "INVALID_EVENT", "ocr_text": "test"}
        )
        resp = client.post("/api/vision", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "invalid_event_type"

    def test_invalid_base64_returns_400(self, client):
        payload = _valid_vision_payload(image_base64="not-valid-base64!!!")
        resp = client.post("/api/vision", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "invalid_image"

    def test_successful_vision_call(self, client):
        """Bedrock をモックして正常レスポンスを確認。"""
        mock_response_body = {
            "content": [{"text": "【状況】技が命中\n【実況】ピカチュウが技を使った！"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        mock_bedrock_response = {
            "body": MagicMock(
                read=MagicMock(return_value=json.dumps(mock_response_body).encode())
            )
        }

        with patch.object(server_module.bedrock, "invoke_model", return_value=mock_bedrock_response):
            resp = client.post("/api/vision", json=_valid_vision_payload())

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "commentary" in data
        assert "usage" in data

    def test_bedrock_timeout_returns_504(self, client):
        """Bedrock タイムアウト時は 504 を返す。"""
        from botocore.exceptions import ReadTimeoutError as _ReadTimeoutError

        with patch.object(server_module.bedrock, "invoke_model", side_effect=_ReadTimeoutError()):
            resp = client.post("/api/vision", json=_valid_vision_payload())

        assert resp.status_code == 504
        assert resp.get_json()["error"] == "bedrock_timeout"

    def test_all_valid_event_types_accepted(self, client):
        """全イベント種別が 400 にならないことを確認（Bedrock はモック）。"""
        valid_events = ["battle_start", "move_used", "move_single", "switch", "faint", "battle_end"]
        mock_response_body = {
            "content": [{"text": "【実況】テスト実況"}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
        mock_bedrock_response = {
            "body": MagicMock(
                read=MagicMock(return_value=json.dumps(mock_response_body).encode())
            )
        }
        with patch.object(server_module.bedrock, "invoke_model", return_value=mock_bedrock_response):
            for event in valid_events:
                payload = _valid_vision_payload(
                    context={"event_type": event, "ocr_text": "test"}
                )
                resp = client.post("/api/vision", json=payload)
                assert resp.status_code != 400, f"event_type={event} が 400 になった"


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/log
# ═══════════════════════════════════════════════════════════════════════════════

def _valid_log_payload(**overrides):
    base = {
        "session_id": "test-session-123",
        "turn": 1,
        "commentary": "ピカチュウが技を使った！",
        "event_type": "move_used",
        "context": {},
    }
    base.update(overrides)
    return base


class TestLogEndpoint:

    def test_missing_json_returns_400(self, client):
        resp = client.post("/api/log", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_session_id_returns_400(self, client):
        payload = _valid_log_payload(session_id="")
        resp = client.post("/api/log", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "missing_session_id"

    def test_missing_commentary_returns_400(self, client):
        payload = _valid_log_payload(commentary="")
        resp = client.post("/api/log", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "missing_commentary"

    def test_s3_not_configured_returns_500(self, client):
        """S3_BUCKET が未設定の場合は 500 を返す。"""
        import src.api.server as sv
        original = sv.S3_BUCKET
        sv.S3_BUCKET = ""
        try:
            resp = client.post("/api/log", json=_valid_log_payload())
            assert resp.status_code == 500
            assert resp.get_json()["error"] == "s3_not_configured"
        finally:
            sv.S3_BUCKET = original

    def test_successful_log_save(self, client):
        """S3 をモックして正常保存を確認。"""
        import src.api.server as sv
        sv.S3_BUCKET = "test-bucket"
        try:
            with patch.object(sv.s3, "put_object", return_value={}):
                resp = client.post("/api/log", json=_valid_log_payload())
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            assert "s3_log_path" in data
        finally:
            sv.S3_BUCKET = ""

    def test_log_path_contains_session_id_and_turn(self, client):
        """S3 パスに session_id とターン番号が含まれる。"""
        import src.api.server as sv
        sv.S3_BUCKET = "test-bucket"
        try:
            with patch.object(sv.s3, "put_object", return_value={}):
                resp = client.post("/api/log", json=_valid_log_payload(turn=7))
            data = resp.get_json()
            assert "test-session-123" in data["s3_log_path"]
            assert "007" in data["s3_log_path"]
        finally:
            sv.S3_BUCKET = ""

    def test_image_saved_when_provided(self, client):
        """image_base64 が渡された場合、スクリーンショットも保存される。"""
        import src.api.server as sv
        sv.S3_BUCKET = "test-bucket"
        try:
            call_count = {"n": 0}

            def mock_put_object(**kwargs):
                call_count["n"] += 1
                return {}

            with patch.object(sv.s3, "put_object", side_effect=mock_put_object):
                payload = _valid_log_payload(image_base64=_TINY_PNG_B64)
                resp = client.post("/api/log", json=payload)
            assert resp.status_code == 200
            assert call_count["n"] == 2  # JSON + 画像の 2 回
            data = resp.get_json()
            assert data["s3_image_path"] is not None
        finally:
            sv.S3_BUCKET = ""


# ═══════════════════════════════════════════════════════════════════════════════
# /api/script（台本パス・ADR-009）
# ═══════════════════════════════════════════════════════════════════════════════

from src.api.server import _build_script_prompt, _gap_filler_count, _parse_script_fillers


def _valid_script_payload(**overrides):
    payload = {
        "events": [
            {"time": 63.0, "event_type": "battle_start", "commentary": "開幕だ！",
             "context": {"turn": 0, "player": "場: イダイトウ", "opponent": "場: リザードン",
                         "move_log": ["T1:イダイトウのだくりゅう"]}},
            {"time": 133.2, "event_type": "faint", "commentary": "イダイトウが倒れた！"},
        ],
        "gap": {"start": 78.0, "end": 131.0},
    }
    payload.update(overrides)
    return payload


class TestBuildScriptPrompt:

    def test_script_persona_self_name_and_perspective(self):
        """台本パスにも自称くれぴ・視点固定が入る（2026-07-30視聴fb#1・#3）。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "「花圓」という漢字表記は実況文に書かないこと" in prompt
        assert "自分側を応援する立場" in prompt
        assert "推測で断定しない" in prompt

    def test_moment_side_tag_rendered(self):
        """瞬間ログのsideフィールドは📺行に【〜側】タグとして描画される
        （同名ミラーの視点誤り対策・2026-07-30）。無い場合はタグなし（後方互換）。"""
        payload = _valid_script_payload()
        moments = [
            {"time": 40.0, "kind": "move", "text": "T1:イダイトウのおはかまいり",
             "side": "相手"},
            {"time": 50.0, "kind": "move", "text": "T1:オオニューラのまもる"},
        ]
        prompt = _build_script_prompt(payload["gap"], payload["events"], moments)
        assert "📺40.0秒 画面: 【相手側】T1:イダイトウのおはかまいり" in prompt
        assert "📺50.0秒 画面: T1:オオニューラのまもる" in prompt

    def test_contains_visible_timeline_and_gap(self):
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "63.0秒" in prompt
        assert "開幕だ！" in prompt
        assert "★78.0秒 〜 131.0秒" in prompt

    def test_contains_no_spoiler_rule(self):
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "ネタバレ禁止" in prompt
        assert "先取り" in prompt

    def test_target_and_defense_uncertainty_rules_present(self):
        """2026-08-14: 台本パス（フィラー生成）にも対象不明時の決め打ち禁止・
        防御結果不明時の命中断定禁止の安全策を追加（ビジョンパスと同種の対策）。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "対象ポケモン名を" in prompt and "決め打ちしない" in prompt
        assert "まもる等の防御が成功したかどうか" in prompt

    def test_context_rendered_when_present(self):
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "イダイトウのだくりゅう" in prompt
        assert "T0" in prompt

    def test_pre_battle_generic_talk_rule_uses_first_event_time(self):
        """最初のイベント時刻より前は汎用トークにする指示が入る。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "63秒より前" in prompt

    def test_moments_interleaved_in_timeline(self):
        """区間開始以前の瞬間ログが📺付きで時系列に混ぜ込まれる（ライブ実況アンカー）。"""
        payload = _valid_script_payload()
        moments = [{"time": 70.0, "kind": "move", "text": "T1:イダイトウのだくりゅう"}]
        prompt = _build_script_prompt(payload["gap"], payload["events"], moments)
        assert "📺70.0秒 画面: T1:イダイトウのだくりゅう" in prompt
        assert prompt.index("63.0秒") < prompt.index("📺70.0秒")

    def test_gap_line_contains_filler_count(self):
        """★区間の行に区間長に応じた件数指定が入る（53秒区間→1件）。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "★78.0秒 〜 131.0秒 = 無言区間（ここにフィラーを1件）" in prompt  # 53秒÷40秒/件

    def test_live_commentary_persona(self):
        """録画感を出さないライブ実況指示が入る。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "ライブ実況" in prompt

    def test_contains_kurepi_persona(self):
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "くれぴ" in prompt

    def test_future_events_excluded_from_prompt(self):
        """区間より未来のeventsはプロンプトに一切含まれない（ネタバレ構造対策・最重要）。"""
        payload = _valid_script_payload()  # gap=78〜131 / battle_start@63(過去) / faint@133.2(未来)
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "開幕だ" in prompt
        assert "倒れた" not in prompt
        assert "133.2" not in prompt

    def test_future_moments_excluded_from_prompt(self):
        """区間より未来の瞬間ログ（📺）もプロンプトに一切含まれない。"""
        payload = _valid_script_payload()
        moments = [
            {"time": 70.0, "kind": "move", "text": "T1:過去の技"},
            {"time": 90.0, "kind": "move", "text": "T2:未来の技"},
        ]
        prompt = _build_script_prompt(payload["gap"], payload["events"], moments)
        assert "過去の技" in prompt
        assert "未来の技" not in prompt

    def test_first_gap_has_no_visible_events(self):
        """試合開始前の区間はvisible eventsが空でも壊れない。"""
        payload = _valid_script_payload(gap={"start": 0.0, "end": 61.0})
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "開幕だ" not in prompt
        assert "倒れた" not in prompt
        assert "★0.0秒 〜 61.0秒" in prompt

    def test_includes_slang_glossary(self):
        """改善ロードマップ④（口調・知識改善）: 対戦スラング用語集が注入されている。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "集中攻撃" in prompt
        assert "積み" in prompt

    def test_includes_tone_examples(self):
        """口調のfew-shot例文が注入されている。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "この試合とは無関係な架空例" in prompt
        assert "狙うはただ1匹" in prompt

    def test_includes_doubles_tactics_knowledge(self):
        """改善ロードマップ④続き: ダブルバトルの技・特性・戦術知識が注入されている。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "ねこだまし＝必ずひるませる" in prompt
        assert "トリックルームパ" in prompt


class TestGapFillerCount:

    def test_scales_with_gap_length(self):
        # 2026-07-30続き「あ、あ、が耳につく・フィラーを減らして実況を活かしたい」で
        # 40秒/件・上限3に再々調整
        assert _gap_filler_count(0.0, 15.0) == 1     # 短い区間は最低1件
        assert _gap_filler_count(0.0, 65.0) == 1
        assert _gap_filler_count(0.0, 100.0) == 2
        assert _gap_filler_count(0.0, 300.0) == 3    # 上限3件


class TestParseScriptFillers:

    def test_parses_plain_json_array(self):
        text = '[{"time": 30.0, "text": "さあ始まるぞ"}, {"time": 100.0, "text": "考察タイム"}]'
        fillers = _parse_script_fillers(text)
        assert len(fillers) == 2
        assert fillers[0] == {"time": 30.0, "text": "さあ始まるぞ"}

    def test_parses_json_with_code_fence_and_preamble(self):
        text = 'はい、生成します。\n```json\n[{"time": 30, "text": "実況"}]\n```'
        fillers = _parse_script_fillers(text)
        assert fillers == [{"time": 30.0, "text": "実況"}]

    def test_invalid_items_skipped(self):
        text = '[{"time": "abc", "text": "NG"}, {"time": 30, "text": ""}, {"time": 40, "text": "OK"}]'
        fillers = _parse_script_fillers(text)
        assert fillers == [{"time": 40.0, "text": "OK"}]

    def test_no_json_returns_none(self):
        assert _parse_script_fillers("JSONを生成できませんでした") is None

    def test_broken_json_returns_none(self):
        assert _parse_script_fillers('[{"time": 30, "text": "途中で切れ') is None


class TestScriptEndpoint:

    def test_missing_json_returns_400(self, client):
        resp = client.post("/api/script", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_events_returns_400(self, client):
        resp = client.post("/api/script", json=_valid_script_payload(events=[]))
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "missing_events"

    def test_missing_gap_returns_400(self, client):
        resp = client.post("/api/script", json=_valid_script_payload(gap=None))
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "missing_gap"

    def test_malformed_gap_returns_400(self, client):
        resp = client.post("/api/script", json=_valid_script_payload(gap={"start": 0.0}))
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "missing_gap"

    def test_successful_script_call(self, client):
        mock_response_body = {
            "content": [{"text": '[{"time": 30.0, "text": "さあ試合開始が近いぞ"}]'}],
            "usage": {"input_tokens": 500, "output_tokens": 80},
        }
        mock_bedrock_response = {
            "body": MagicMock(
                read=MagicMock(return_value=json.dumps(mock_response_body).encode())
            )
        }
        with patch.object(server_module.bedrock_script, "invoke_model", return_value=mock_bedrock_response):
            resp = client.post("/api/script", json=_valid_script_payload(
                moments=[{"time": 90.0, "kind": "move", "text": "T1:だくりゅう"}]))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["fillers"] == [{"time": 30.0, "text": "さあ試合開始が近いぞ"}]

    def test_unparseable_bedrock_output_returns_502(self, client):
        mock_response_body = {
            "content": [{"text": "JSONではない自由文の応答"}],
            "usage": {},
        }
        mock_bedrock_response = {
            "body": MagicMock(
                read=MagicMock(return_value=json.dumps(mock_response_body).encode())
            )
        }
        with patch.object(server_module.bedrock_script, "invoke_model", return_value=mock_bedrock_response):
            resp = client.post("/api/script", json=_valid_script_payload())
        assert resp.status_code == 502
        assert resp.get_json()["error"] == "bedrock_parse_error"
