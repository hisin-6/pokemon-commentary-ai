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

    def test_prompt_contains_ocr_text(self):
        prompt = _build_vision_prompt(self._context(ocr_text="ピカチュウのかみなり"), [], self._battle_state())
        assert "ピカチュウのかみなり" in prompt

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
        valid_events = ["battle_start", "move_used", "switch", "faint", "battle_end"]
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
        """★区間の行に区間長に応じた件数指定が入る（53秒区間→2件）。"""
        payload = _valid_script_payload()
        prompt = _build_script_prompt(payload["gap"], payload["events"])
        assert "★78.0秒 〜 131.0秒 = 無言区間（ここにフィラーを2件）" in prompt

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


class TestGapFillerCount:

    def test_scales_with_gap_length(self):
        assert _gap_filler_count(0.0, 15.0) == 1     # 短い区間は最低1件
        assert _gap_filler_count(0.0, 53.0) == 2
        assert _gap_filler_count(0.0, 100.0) == 5
        assert _gap_filler_count(0.0, 300.0) == 5    # 上限5件


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
