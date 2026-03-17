"""
EC2 Flask API サーバー
ポケモン対戦実況AIのクラウド処理を担当する。

エンドポイント:
  GET  /health      — 死活確認
  POST /api/vision  — Bedrock Claude Haiku で画面Vision分析
  POST /api/log     — S3 に実況ログ・スクリーンショットを保存

実行環境: EC2 (IAMロール経由でBedrock・S3にアクセス)
ポート: 5000
"""

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request

# ─── 設定 ────────────────────────────────────────────────────────────────────

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-2")
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_TIMEOUT_SEC = 5
IMAGE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "ap-southeast-2")

VALID_EVENT_TYPES = {"turn_end", "switch", "faint"}

# ─── Flask・AWS クライアント初期化 ────────────────────────────────────────────

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
s3 = boto3.client("s3", region_name=S3_REGION)

# ─── ヘルパー ────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_vision_prompt(context: dict, history: list[str]) -> str:
    """Bedrock に送るプロンプトを組み立てる"""
    lines = [
        "あなたはポケモン対戦の実況者です。",
        "画面を見て、以下の形式で日本語で出力してください。",
        "",
        "【状況】",
        "（画面から読み取れる対戦状況を1〜2文で説明。ポケモン名・技名・HP等を正確に）",
        "",
        "【実況】",
        "（テンポよく興奮感のある実況を1〜2文。ポケモン名・技名はそのまま使う。HPが低い時は緊張感を出す。鉤括弧は使わない）",
        "",
        "【参考情報（OCR・YOLO取得）】",
        f"画面テキスト: {context.get('ocr_text', '不明')}",
        f"自分の状態異常: {context.get('status_player', 'なし')}",
        f"相手の状態異常: {context.get('status_opponent', 'なし')}",
        f"残りボール (自分/相手): {context.get('balls_remaining_player', '?')} / {context.get('balls_remaining_opponent', '?')}",
        f"イベント種別: {context.get('event_type', '不明')}",
    ]
    if history:
        lines.append(f"直前の実況: {history[-1]}")
    return "\n".join(lines)


def _parse_commentary(text: str) -> tuple[str, str]:
    """
    Haiku の出力から【状況】と【実況】を抽出する。
    Returns: (analysis, commentary)
    """
    analysis = text
    commentary = text  # フォールバック: 全文を実況に使う

    if "【実況】" in text:
        parts = text.split("【実況】")
        commentary = parts[1].strip().split("【")[0].strip()

    if "【状況】" in text:
        parts = text.split("【状況】")
        analysis_raw = parts[1].split("【")[0].strip()
        analysis = analysis_raw if analysis_raw else text

    return analysis, commentary


# ─── エンドポイント ──────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return jsonify({"status": "ok", "timestamp": _now_iso()})


@app.post("/api/vision")
def vision():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json", "message": "リクエストボディがJSONではありません"}), 400

    # バリデーション
    image_b64: str = data.get("image_base64", "")
    context: dict = data.get("context", {})
    history: list = data.get("history", [])

    if not image_b64:
        return jsonify({"success": False, "error": "missing_image", "message": "image_base64 が必要です"}), 400

    if not context:
        return jsonify({"success": False, "error": "missing_context", "message": "context が必要です"}), 400

    event_type = context.get("event_type", "")
    if event_type not in VALID_EVENT_TYPES:
        return jsonify({
            "success": False,
            "error": "invalid_event_type",
            "message": f"event_type は {VALID_EVENT_TYPES} のいずれかにしてください",
        }), 400

    # 画像サイズチェック（Base64デコード前に文字数で概算）
    if len(image_b64) > IMAGE_MAX_BYTES * 4 // 3 + 100:
        return jsonify({"success": False, "error": "image_too_large", "message": "画像サイズが上限（5MB）を超えています"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"success": False, "error": "invalid_image", "message": "Base64デコードに失敗しました"}), 400

    if len(image_bytes) > IMAGE_MAX_BYTES:
        return jsonify({"success": False, "error": "image_too_large", "message": "画像サイズが上限（5MB）を超えています"}), 400

    # Bedrock 呼び出し
    prompt_text = _build_vision_prompt(context, history)
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    }

    start_ms = time.monotonic()
    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if "Timeout" in code or "ThrottlingException" == code:
            logger.warning("Bedrock タイムアウト / スロットリング: %s", e)
            return jsonify({"success": False, "error": "bedrock_timeout", "message": f"Bedrock APIエラー: {code}"}), 504
        logger.error("Bedrock ClientError: %s", e)
        return jsonify({"success": False, "error": "bedrock_error", "message": str(e)}), 502
    except Exception as e:
        logger.error("Bedrock 予期しないエラー: %s", e)
        return jsonify({"success": False, "error": "bedrock_error", "message": str(e)}), 502

    latency_ms = int((time.monotonic() - start_ms) * 1000)

    result = json.loads(response["body"].read())
    raw_text = result["content"][0]["text"].strip()
    usage = result.get("usage", {})

    analysis, commentary = _parse_commentary(raw_text)

    logger.info("Vision分析完了 latency=%dms tokens_in=%s tokens_out=%s", latency_ms, usage.get("input_tokens"), usage.get("output_tokens"))
    logger.info("実況文: %s", commentary)

    return jsonify({
        "success": True,
        "analysis": analysis,
        "commentary": commentary,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "latency_ms": latency_ms,
    })


@app.post("/api/log")
def log_save():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json", "message": "リクエストボディがJSONではありません"}), 400

    session_id: str = data.get("session_id", "")
    turn: int = data.get("turn", 0)
    commentary: str = data.get("commentary", "")

    if not session_id:
        return jsonify({"success": False, "error": "missing_session_id", "message": "session_id が必要です"}), 400
    if not commentary:
        return jsonify({"success": False, "error": "missing_commentary", "message": "commentary が必要です"}), 400
    if not S3_BUCKET:
        return jsonify({"success": False, "error": "s3_not_configured", "message": "S3_BUCKET 環境変数が未設定です"}), 500

    # S3 保存パス
    log_key = f"logs/{session_id}/turn_{turn:03d}.json"
    image_key = f"screenshots/{session_id}/turn_{turn:03d}.png"

    # ログ JSON 保存
    log_payload = {
        "session_id": session_id,
        "turn": turn,
        "timestamp": data.get("timestamp", _now_iso()),
        "event_type": data.get("event_type", ""),
        "context": data.get("context", {}),
        "analysis": data.get("analysis", ""),
        "commentary": commentary,
    }

    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=log_key,
            Body=json.dumps(log_payload, ensure_ascii=False),
            ContentType="application/json",
        )
    except ClientError as e:
        logger.error("S3 ログ保存失敗: %s", e)
        return jsonify({"success": False, "error": "s3_error", "message": str(e)}), 502

    response_body: dict = {
        "success": True,
        "s3_log_path": f"s3://{S3_BUCKET}/{log_key}",
        "s3_image_path": None,
    }

    # スクリーンショット保存（任意）
    image_b64: str = data.get("image_base64", "")
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=image_key,
                Body=image_bytes,
                ContentType="image/png",
            )
            response_body["s3_image_path"] = f"s3://{S3_BUCKET}/{image_key}"
        except Exception as e:
            # 画像保存失敗はログ保存成功後でも警告のみ
            logger.warning("S3 画像保存失敗（ログは保存済み）: %s", e)

    logger.info("ログ保存完了 session=%s turn=%03d", session_id, turn)
    return jsonify(response_body)


# ─── 起動 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Flask API 起動 port=%d", port)
    app.run(host="0.0.0.0", port=port)
