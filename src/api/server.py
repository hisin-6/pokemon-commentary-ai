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
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from flask import Flask, jsonify, request

# ─── 設定 ────────────────────────────────────────────────────────────────────

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-2")
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_TIMEOUT_SEC = 5
IMAGE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "ap-southeast-2")

VALID_EVENT_TYPES = {"battle_start", "move_used", "switch", "faint", "battle_end"}

# ─── Flask・AWS クライアント初期化 ────────────────────────────────────────────

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(
        connect_timeout=5,
        read_timeout=BEDROCK_TIMEOUT_SEC,
        retries={"max_attempts": 0},
    ),
)
s3 = boto3.client("s3", region_name=S3_REGION)

# ─── ヘルパー ────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_vision_prompt(context: dict, history: list[str], battle_state: dict) -> str:
    """Bedrock に送るプロンプトを組み立てる"""

    # イベント別の実況指示
    event_type = context.get("event_type", "")
    event_hint = {
        "battle_start": "バトル開始！両者のポケモンを紹介して試合への期待感を高める実況をする",
        "move_used":    "今ターンで使われた技とその効果を実況する",
        "switch":       "ポケモンの交代について実況する",
        "faint":        "ポケモンが倒れた瞬間を実況する（HP=0のポケモンを特定すること）",
        "battle_end":   "試合終了を締めくくる実況をする",
    }.get(event_type, "状況を実況する")

    lines = [
        "あなたはポケモンSVダブルバトルの熱狂的な実況者です。",
        "",
        "【ダブルバトルの基本知識】",
        "- 各プレイヤーが2匹ずつ場に出す（合計4匹が同時に戦う）",
        "- 技名（テラクラスター・アストラルビット・フレアドライブ等）はポケモン名ではなく技の名前",
        "- みがわり・めいそう・テラスタル・アンコール・かなしばりは戦略的な行動",
        "- バツグン・いまひとつ・こうかなし・こうかあり・こうかばつぐん はダメージ結果テキスト。絶対に実況文に含めてはいけない",
        "- 特性発動メッセージ（〜のわざわいのつるぎ・〜のこだいかっせい等）は技名ではない。技名と混同しないこと",
        "- トレーナー名（英数字の名前）はポケモン名ではない",
        "",
        "【出力ルール】",
        f"- 今回のイベント: {event_type} → {event_hint}",
        "- 【実況】に実況文を1〜2文で書く",
        "- 必ず下記の情報にあるポケモン名・HP のみを使う（創作禁止）",
        "- 技名は必ず画像のバトルメッセージ（〜のXXを使った！等）から直接読み取ること",
        "- OCRテキストに変な表記（例: すいゆゆうれんだ）があっても無視すること。技名は画像から正確に読み取ること",
        "- 画像のバトルメッセージで技名が確認できない場合は、絶対に技名を実況しないこと（ポケモンの知識から推測した技名も使用禁止）",
        "- 画像を直接見て、HPバーの位置から自分と相手のポケモンを判断すること",
        "  （画面左上/左下のHPバー＝相手のポケモン、画面右下/右上のHPバー＝自分のポケモン）",
        "- 【蓄積された戦況】の「自分のポケモン」「相手のポケモン」リストを最優先で参照すること",
        "  （このリストは複数ターン分の情報で正確。OCR名前候補より信頼度が高い）",
        "- 画像に状態異常アイコンが見えたら必ず言及すること",
        "  （まひ=黄色、やけど=橙、どく=紫、ねむり=黒、こおり=水色）",
        "- HPが残り30%未満の時は緊張感を出す",
        "- 鉤括弧（「」）は使わない",
        "",
        "【蓄積された戦況（複数ターン分の確定情報）】",
        f"ターン数: {battle_state.get('turn', '不明')}",
        f"自分の場: {battle_state.get('player_field', battle_state.get('player_pokemon', '情報収集中'))}",
        f"自分の控え: {battle_state.get('player_bench', 'なし')}",
        f"相手の場: {battle_state.get('opponent_field', battle_state.get('opponent_pokemon', '情報収集中'))}",
        f"相手の控え: {battle_state.get('opponent_bench', 'なし')}",
        f"直近のイベント履歴: {battle_state.get('event_log', 'なし')}",
        "※ 「場」のポケモンが現在戦闘中。「控え」は場にいない（交代前の控え・ひんし含む）。",
        "※ (ひんし) とマークされたポケモンはすでに倒れており絶対に言及しないこと。",
        "",
        "【現在フレームのOCR情報（ヒント・画像と矛盾する場合は画像優先）】",
        "※ 名前候補はy座標の仮分類で、技名・選出画面の手持ち・OCR誤読が混入する。画像のHPバーで必ず確認すること。",
        f"画面テキスト: {context.get('ocr_text', '不明')}",
        f"HP値: {context.get('hp_values', '不明')}",
        f"自分側のポケモン名候補（不正確・参考のみ）: {context.get('name_candidates_player', '不明')}",
        f"相手側のポケモン名候補（不正確・参考のみ）: {context.get('name_candidates_opponent', '不明')}",
        f"自分の状態異常: {context.get('status_player', 'なし')}",
        f"相手の状態異常: {context.get('status_opponent', 'なし')}",
        f"OCRで検出した使用技（〜のXX形式・信頼度高）: {context.get('detected_moves', 'なし')}",
        "  ↑ このターンで実際に使われた技として最優先で参照すること",
    ]
    rag_info: list = context.get("rag_pokemon_info", [])
    if rag_info:
        lines += [
            "",
            "【ポケモン図鑑情報（DB参照・信頼度高）】",
            "（OCR認識したポケモン名に基づくタイプ・特性・代表技。実況の参考に必ず活用すること）",
        ]
        lines += [f"- {entry}" for entry in rag_info]
    if history:
        lines.append(f"直前の実況（繰り返さないこと）: {history[-1]}")
    lines += [
        "",
        "【状況】",
        "（1文で状況説明）",
        "",
        "【実況】",
        "（1〜2文の実況文）",
    ]
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
    battle_state: dict = data.get("battle_state", {})
    prompt_text = _build_vision_prompt(context, history, battle_state)
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
    except ReadTimeoutError as e:
        logger.warning("Bedrock 読み取りタイムアウト: %s", e)
        return jsonify({"success": False, "error": "bedrock_timeout", "message": "Bedrock タイムアウト"}), 504
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            logger.warning("Bedrock スロットリング: %s", e)
            return jsonify({"success": False, "error": "bedrock_timeout", "message": f"Bedrock スロットリング: {code}"}), 504
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
