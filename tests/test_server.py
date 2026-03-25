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

    def test_missing_image_returns_400(self, client):
        payload = _valid_vision_payload(image_base64="")
        resp = client.post("/api/vision", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "missing_image"

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
