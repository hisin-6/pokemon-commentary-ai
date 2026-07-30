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

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from flask import Flask, jsonify, request

# ─── 設定 ────────────────────────────────────────────────────────────────────

# Claude 3 Haiku は2026-09-10 EOL（ADR-001参照）。後継のClaude Haiku 4.5に移行。
# ap-southeast-2はIn-Region直接呼び出し非対応のため、AU推論プロファイルIDを使用。
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-2")
BEDROCK_MODEL_ID = "au.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_TIMEOUT_SEC = 5
# 台本パス（/api/script）用の読み取りタイムアウト。テキストのみだが生成量が
# 多く（フィラー最大4件/区間×60〜100字・max_tokens 3000）5秒では足りない。
# オフライン処理なのでライブ用の短いタイムアウトを適用しない。
# ⚠️ gunicornのworker timeout（既定30秒）も超える場合はsystemdユニット側の調整が必要
BEDROCK_SCRIPT_TIMEOUT_SEC = 60
IMAGE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "ap-southeast-2")

VALID_EVENT_TYPES = {"battle_start", "move_used", "switch", "faint", "battle_end"}

# ─── Flask・AWS クライアント初期化 ────────────────────────────────────────────

app = Flask(__name__)
# リクエストボディの上限（Flask/Werkzeugがこの値を超えるボディを早期拒否する）。
# 画像は最大 IMAGE_MAX_BYTES だが、Base64化・JSON全体のオーバーヘッド分の余裕を持たせる。
app.config["MAX_CONTENT_LENGTH"] = IMAGE_MAX_BYTES * 2
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
# 台本パス用（読み取りタイムアウトのみ長い・他は同一設定）
bedrock_script = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(
        connect_timeout=5,
        read_timeout=BEDROCK_SCRIPT_TIMEOUT_SEC,
        retries={"max_attempts": 0},
    ),
)
s3 = boto3.client("s3", region_name=S3_REGION)

# ─── ヘルパー ────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_vision_prompt(context: dict, history: list[str], battle_state: dict,
                          has_image: bool = True) -> str:
    """Bedrock に送るプロンプトを組み立てる。

    has_image=False の場合（動画モードの後付け生成・ADR-009追記）は画像を渡さないため、
    画像依存の指示（HPバー位置の目視判断・画像からの技名読み取り等）を、構造化データを
    正確な事実として扱わせる指示に差し替える。
    """

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
        "あなたは、ポケモン対戦実況AIVTuber「花圓くれぴ（はなまるくれぴ）」です。",
        "性格は元気で甘えん坊、でもポケモン知識はガチ勢。",
        "口調はアイドル・かわいい系（語尾に♪を適度に使う・タメ口・テンション高め・"
        "かわいい褒め言葉多め）で実況しつつ、技名やHP等の情報は正確に伝えてください。",
        "自称は「くれぴ」（ひらがな）。「花圓」という漢字表記は実況文に書かないこと"
        "（音声合成が正しく読めないため）。",
        "あなたは「自分」側（プレイヤー側）の視点で、自分側を応援する立場で実況します。"
        "「自分の」「相手の」の帰属は下記の戦況リストに厳密に従い、"
        "相手のポケモンを自分のもののように（またはその逆に）語らないこと。",
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
        "- 実況は「今この瞬間に起きたこと」を最優先。数十秒前・過去のターンの出来事を"
        "今起きたかのように振り返らないこと",
    ]
    battle_result = context.get("battle_result", "")
    if battle_result:
        lines += [
            f"- この試合の勝敗は「自分の{battle_result}」で確定している。"
            "締めの実況で必ず勝敗に触れること（勝ちなら喜び・負けなら悔しさ＋前向きな一言）",
        ]
    if has_image:
        lines += [
            "- 技名は必ず画像のバトルメッセージ（〜のXXを使った！等）から直接読み取ること",
            "- OCRテキストに変な表記（例: すいゆゆうれんだ）があっても無視すること。技名は画像から正確に読み取ること",
            "- 画像のバトルメッセージで技名が確認できない場合は、絶対に技名を実況しないこと（ポケモンの知識から推測した技名も使用禁止）",
            "- 画像を直接見て、HPバーの位置から自分と相手のポケモンを判断すること",
            "  （画面左上/左下のHPバー＝相手のポケモン、画面右下/右上のHPバー＝自分のポケモン）",
            "- 【蓄積された戦況】の「自分のポケモン」「相手のポケモン」リストを最優先で参照すること",
            "  （このリストは複数ターン分の情報で正確。OCR名前候補より信頼度が高い）",
            "- 画像に状態異常アイコンが見えたら必ず言及すること",
            "  （まひ=黄色、やけど=橙、どく=紫、ねむり=黒、こおり=水色）",
        ]
    else:
        lines += [
            "- 画像は渡されていない。以下の【蓄積された戦況】【現在フレームのOCR情報】に書かれた情報のみを"
            "正確な事実として扱うこと（画像で確認・上書きすることはできない）",
            "- 技名は必ず【OCRで検出した使用技】の情報のみを根拠にすること。記載がない技は絶対に実況しないこと"
            "（ポケモンの知識から推測した技名も使用禁止）",
            "- 【蓄積された戦況】の「自分のポケモン」「相手のポケモン」リストを最優先で参照すること",
            "  （このリストは複数ターン分の情報で正確。OCR名前候補より信頼度が高い）",
            "- 自分の状態異常／相手の状態異常に記載があれば必ず言及すること",
            "  （まひ=黄色、やけど=橙、どく=紫、ねむり=黒、こおり=水色）",
        ]
    lines += [
        "- HPが残り30%未満の時は緊張感を出す",
        "- 鉤括弧（「」）は使わない",
        "- 見出しは【状況】【実況】の全角鍵括弧形式のみを使う（Markdown見出し#は使わない）",
        "",
        "【蓄積された戦況（複数ターン分の確定情報）】",
        f"ターン数: {battle_state.get('turn', '不明')}",
        f"自分の場: {battle_state.get('player_field', battle_state.get('player_pokemon', '情報収集中'))}",
        f"自分の控え: {battle_state.get('player_bench', 'なし')}",
        f"相手の場: {battle_state.get('opponent_field', battle_state.get('opponent_pokemon', '情報収集中'))}",
        f"相手の控え: {battle_state.get('opponent_bench', 'なし')}",
        f"ターン推移: {battle_state.get('turn_history', 'なし')}",
        f"直近のイベント履歴: {battle_state.get('event_log', 'なし')}",
        "※ 「場」のポケモンが現在戦闘中。「控え」は場にいない（交代前の控え・ひんし含む）。",
        "※ (ひんし) とマークされたポケモンはすでに倒れており絶対に言及しないこと。",
        "※ 「控え」のポケモンに言及する場合は必ず「控えの」等を付けて控えだと分かる言い方にすること。"
        "「場」にいるかのような表現（現在戦っている前提の言い回し）をしてはいけない。",
        "",
        "【現在フレームのOCR情報（ヒント・画像と矛盾する場合は画像優先）】" if has_image
        else "【現在フレームのOCR情報（蓄積された戦況を優先し、矛盾する場合はそちらを信じること）】",
        "※ 名前候補はy座標の仮分類で、技名・選出画面の手持ち・OCR誤読が混入する。画像のHPバーで必ず確認すること。" if has_image
        else "※ 名前候補はy座標の仮分類で、技名・選出画面の手持ち・OCR誤読が混入する。【蓄積された戦況】を優先すること。",
        f"画面テキスト: {context.get('ocr_text', '不明')}",
        f"HP値: {context.get('hp_values', '不明')}",
        f"自分側のポケモン名候補（不正確・参考のみ）: {context.get('name_candidates_player', '不明')}",
        f"相手側のポケモン名候補（不正確・参考のみ）: {context.get('name_candidates_opponent', '不明')}",
        f"自分の状態異常: {context.get('status_player', 'なし')}",
        f"相手の状態異常: {context.get('status_opponent', 'なし')}",
        f"OCRで検出した使用技（〜のXX形式・信頼度高。「（推定）」付きは使い手ポケモンが未確定の仮推定）: {context.get('detected_moves', 'なし')}",
        "  ↑ このターンで実際に使われた技として最優先で参照すること",
        "  （「（推定）」付きの場合、技自体は使われたが使い手の名前は断定せず「相手」等に留めること）",
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
    # Bedrockが【状況】【実況】の代わりにMarkdown見出し（# 状況 等）で返す
    # ケースがある（2026-07-13発見）。抽出前に正規化して両形式に対応する。
    text = re.sub(r'^#{1,3}\s*(状況|実況)\s*$', r'【\1】', text, flags=re.MULTILINE)

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


def _gap_filler_count(start: float, end: float) -> int:
    """無言区間の長さから生成するフィラーの目安件数を決める（約30秒に1件・1〜3件）。

    2026-07-14に「とてもしゃべらせたい」で25秒→18秒へ増量したが、
    2026-07-30の視聴フィードバック「フィラーが多い」で30秒/件・上限3へ減量。
    """
    return max(1, min(3, int((end - start) // 30)))


def _build_script_prompt(gap: dict, events: list, moments: list = None) -> str:
    """台本パス（ギャップフィラー生成）のプロンプトを組み立てる。

    録画解析済みの実況タイムライン・瞬間ログ（技が画面に映った動画内時刻）
    のうち、対象の無言区間 ``gap`` の開始時刻より前のものだけを使って
    プロンプトを組み立てる（ADR-009 台本パス）。

    ⚠️ 呼び出し元は無言区間ごとに1回呼ぶこと（1プロンプト=1区間）。
    未来の情報を「見せた上で使うな」と指示するだけでは指示に従い損ねて
    ネタバレする実例があったため（2026-07-14）、そもそも未来の events/moments
    をプロンプトに含めない構造的対策にしている。
    """
    first_event_time = min(float(e.get("time", 0)) for e in events)
    gap_start = float(gap["start"])
    gap_end = float(gap["end"])

    lines = [
        "あなたは、ポケモン対戦実況AIVTuber「花圓くれぴ（はなまるくれぴ）」です。",
        "性格は元気で甘えん坊、でもポケモン知識はガチ勢。口調はアイドル・かわいい系",
        "（語尾に♪を適度に使う・タメ口・テンション高め・かわいい褒め言葉多め）。",
        "自称は「くれぴ」（ひらがな）。「花圓」という漢字表記は実況文に書かないこと"
        "（音声合成が正しく読めないため）。",
        "あなたは「自分」側（プレイヤー側）の視点で、自分側を応援する立場で実況します。",
        "ポケモンの「自分の」「相手の」の帰属はタイムラインの記載（【自分側】【相手側】タグ・",
        "戦況の場/控え情報）に厳密に従い、タグや記載がなく判別できない場合は",
        "「自分の/相手の」を付けずにポケモン名だけで実況すること（推測で断定しない）。",
        "録画された試合に後から実況を吹き込みますが、視聴者には生放送の",
        "ライブ実況に聞こえるようにしてください（後から見返している・録画といった言い方は禁止）。",
        "主要イベントの実況音声はすでに収録済みです。",
        "イベント間の「無言区間」を埋めるつなぎ実況（フィラー）を生成してください。",
        "",
        "【最重要ルール: ネタバレ禁止】",
        "- 各フィラーは、その time 時点までに起きた出来事の情報だけを使うこと",
        "- time より後の出来事や試合の勝敗を絶対に先取りして言及しないこと",
        "- 「次に〜が来たら怖い」のような予想はOK（断定はしない・外れてもよい）",
        f"- {first_event_time:.0f}秒より前の区間（試合開始前）は、ポケモン名などの具体情報を使わず、",
        "  挨拶・意気込み・観戦ポイントなどの汎用トークにすること",
        "",
        "【フィラーの内容（バリエーションを持たせる）】",
        "- 【最優先】直前に画面に映った技（📺印）への反応・実況。区間の開始時刻が📺と",
        "  ほぼ同じ場合、最初のフィラーは区間開始時刻に置き、その技への反応から始めること",
        "  （例: 📺200.0秒 ドラゴンクロー → 200.5秒 ガブリアスのドラゴンクローが炸裂！...）",
        "- 直前の展開の振り返り・戦況の整理（HP・残り頭数）",
        "- タイプ相性や特性をふまえた考察",
        "- 次の展開の予想",
        "",
        "【出力形式】",
        "- JSON配列のみを出力すること（前置き・説明文・コードフェンスは書かない）",
        '- 形式: [{"time": 秒数の数値, "text": "実況文"}, ...]',
        "- text は60〜100文字程度（読み上げ約10〜18秒）",
        "- time は必ず無言区間の範囲内の数値にすること。各フィラーは20秒以上離すこと",
        "- 鉤括弧（「」）は使わない",
        "",
        "【タイムライン（時刻順・ここまでに起きた出来事のみ。📺は画面に技が映った瞬間）】",
        "※ 収録済み実況と重複しない内容にすること。",
    ]
    # events・momentsのうち、この無言区間の開始時刻以前のものだけを見せる
    # （それより後は「まだ起きていない」ではなく、そもそもプロンプトに含めない）
    visible_events = [e for e in events if float(e.get("time", 0)) <= gap_start]
    visible_moments = [m for m in (moments or []) if float(m.get("time", 0)) <= gap_start]
    timeline = [("event", float(e.get("time", 0)), e) for e in visible_events]
    timeline += [("moment", float(m.get("time", 0)), m) for m in visible_moments]
    timeline.sort(key=lambda item: item[1])
    for kind, _, item in timeline:
        if kind == "moment":
            # 陣営タグ（2026-07-30〜のtimelineに付与・同名ミラーの視点誤りを防ぐ）
            side = item.get("side")
            side_txt = f"【{side}側】" if side else ""
            lines.append(f"- 📺{float(item['time']):.1f}秒 画面: {side_txt}{item.get('text', '')}")
            continue
        e = item
        lines.append(f"- {float(e['time']):.1f}秒 [{e.get('event_type', '?')}] {e.get('commentary', '')}")
        ctx = e.get("context") or {}
        if ctx:
            parts = []
            if ctx.get("turn") is not None:
                parts.append(f"T{ctx['turn']}")
            if ctx.get("player"):
                parts.append(f"自分={ctx['player']}")
            if ctx.get("opponent"):
                parts.append(f"相手={ctx['opponent']}")
            if ctx.get("move_log"):
                parts.append(f"技ログ={' / '.join(ctx['move_log'])}")
            if parts:
                lines.append(f"    （戦況: {'・'.join(parts)}）")

    n = _gap_filler_count(gap_start, gap_end)
    lines.append(f"- ★{gap_start:.1f}秒 〜 {gap_end:.1f}秒 = 無言区間（ここにフィラーを{n}件）")
    return "\n".join(lines)


def _parse_script_fillers(text: str):
    """台本パスのBedrock出力からフィラーのJSON配列を抽出する。

    コードフェンスや前置きが混入しても最初の ``[`` 〜 最後の ``]`` を
    JSONとして読む。形式不正なら None を返す（要素単位では time/text を
    持つものだけ採用）。
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list):
        return None
    fillers = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        time_val = item.get("time")
        text_val = item.get("text")
        if isinstance(time_val, (int, float)) and isinstance(text_val, str) and text_val.strip():
            fillers.append({"time": float(time_val), "text": text_val.strip()})
    return fillers


# ─── エンドポイント ──────────────────────────────────────────────────────────


# @app.get/@app.post ショートカットは Flask 2.0+ 専用のため、
# ローカル検証環境（Flask 1.x）でも動く @app.route 形式で書く
@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": _now_iso()})


@app.route("/api/vision", methods=["POST"])
def vision():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json", "message": "リクエストボディがJSONではありません"}), 400

    # バリデーション
    # image_base64 は任意（動画モードの後付け生成・ADR-009追記では画像を渡さない）
    image_b64: str = data.get("image_base64", "")
    context: dict = data.get("context", {})
    history: list = data.get("history", [])

    if not context:
        return jsonify({"success": False, "error": "missing_context", "message": "context が必要です"}), 400

    event_type = context.get("event_type", "")
    if event_type not in VALID_EVENT_TYPES:
        return jsonify({
            "success": False,
            "error": "invalid_event_type",
            "message": f"event_type は {VALID_EVENT_TYPES} のいずれかにしてください",
        }), 400

    image_bytes: bytes | None = None
    if image_b64:
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
    prompt_text = _build_vision_prompt(context, history, battle_state, has_image=image_bytes is not None)
    content: list = []
    if image_bytes is not None:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        })
    content.append({"type": "text", "text": prompt_text})
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": content,
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
        return jsonify({"success": False, "error": "bedrock_error", "message": "Bedrock呼び出しでエラーが発生しました"}), 502
    except Exception as e:
        logger.error("Bedrock 予期しないエラー: %s", e)
        return jsonify({"success": False, "error": "bedrock_error", "message": "Bedrock呼び出しでエラーが発生しました"}), 502

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


@app.route("/api/script", methods=["POST"])
def script():
    """台本パス（ADR-009）: 解析済み実況タイムラインの無言区間1つ分を埋める
    フィラー実況をテキストのみのBedrock呼び出しで生成する。

    無言区間ごとに1回呼び出す契約（ネタバレ防止のため、この区間より
    未来のevents/momentsはプロンプトに含めない構造にしている）。"""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json", "message": "リクエストボディがJSONではありません"}), 400

    events: list = data.get("events", [])
    gap = data.get("gap")
    moments: list = data.get("moments", [])
    if not events:
        return jsonify({"success": False, "error": "missing_events", "message": "events が必要です"}), 400
    if not isinstance(gap, dict) or "start" not in gap or "end" not in gap:
        return jsonify({"success": False, "error": "missing_gap", "message": "gap（start/endを持つオブジェクト）が必要です"}), 400

    prompt_text = _build_script_prompt(gap, events, moments)
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ],
    }

    start_ms = time.monotonic()
    try:
        response = bedrock_script.invoke_model(
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
        return jsonify({"success": False, "error": "bedrock_error", "message": "Bedrock呼び出しでエラーが発生しました"}), 502
    except Exception as e:
        logger.error("Bedrock 予期しないエラー: %s", e)
        return jsonify({"success": False, "error": "bedrock_error", "message": "Bedrock呼び出しでエラーが発生しました"}), 502

    latency_ms = int((time.monotonic() - start_ms) * 1000)

    result = json.loads(response["body"].read())
    raw_text = result["content"][0]["text"].strip()
    usage = result.get("usage", {})

    fillers = _parse_script_fillers(raw_text)
    if fillers is None:
        logger.error("台本パス: Bedrock出力のJSON解析に失敗: %s", raw_text[:200])
        return jsonify({"success": False, "error": "bedrock_parse_error", "message": "Bedrock出力をJSONとして解析できませんでした"}), 502

    logger.info("台本生成完了 latency=%dms fillers=%d tokens_in=%s tokens_out=%s",
                latency_ms, len(fillers), usage.get("input_tokens"), usage.get("output_tokens"))

    return jsonify({
        "success": True,
        "fillers": fillers,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "latency_ms": latency_ms,
    })


@app.route("/api/log", methods=["POST"])
def log_save():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json", "message": "リクエストボディがJSONではありません"}), 400

    session_id: str = data.get("session_id", "")
    turn = data.get("turn", 0)
    commentary: str = data.get("commentary", "")

    if not session_id:
        return jsonify({"success": False, "error": "missing_session_id", "message": "session_id が必要です"}), 400
    if not commentary:
        return jsonify({"success": False, "error": "missing_commentary", "message": "commentary が必要です"}), 400
    if not isinstance(turn, int) or isinstance(turn, bool):
        return jsonify({"success": False, "error": "invalid_turn", "message": "turn は整数である必要があります"}), 400
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
        return jsonify({"success": False, "error": "s3_error", "message": "S3への保存でエラーが発生しました"}), 502

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
