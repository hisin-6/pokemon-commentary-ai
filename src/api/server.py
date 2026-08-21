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

VALID_EVENT_TYPES = {"battle_start", "move_used", "move_single", "switch", "faint", "battle_end"}

# ─── Flask・AWS クライアント初期化 ────────────────────────────────────────────

app = Flask(__name__)
# リクエストボディの上限（Flask/Werkzeugがこの値を超えるボディを早期拒否する）。
# 画像は最大 IMAGE_MAX_BYTES だが、Base64化・JSON全体のオーバーヘッド分の余裕を持たせる。
app.config["MAX_CONTENT_LENGTH"] = IMAGE_MAX_BYTES * 2
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Bedrockに実際に送るプロンプト全文をログに出す（RAGヒント等の確認用・2026-08-21）。
# CLIが無いFlask/gunicorn運用のため環境変数で切り替える（起動前に export DEBUG_PROMPTS=1、
# もしくはsystemdユニットに Environment=DEBUG_PROMPTS=1 を追加して要再起動）。
if os.environ.get("DEBUG_PROMPTS") == "1":
    logger.setLevel(logging.DEBUG)
    logger.info("DEBUG_PROMPTS=1: Bedrockプロンプト全文のログ出力を有効化しました")

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

# 改善ロードマップ④（実況の口調・知識改善）用語集注入（2026-08-03策定）。
# 対戦勢が実際に使う口語スラングのうち、ダブルバトルの実況で自然に使えるものだけを
# ユーザーと相談の上で厳選（テラスタル等チャンピオンズ未解禁の要素は解禁後に追加予定。
# 「対面」「起点」はシングルバトル寄りの概念のため不採用）。vision/script両プロンプトで
# 共通利用する。
_SLANG_GLOSSARY_LINES = [
    "【対戦実況で使える言い回し（自然に使ってOK・無理に全部使わなくてよい）】",
    "- 基本戦術: 積み/積み技・展開/崩す・受け/受ける・通す・縛る・ごり押し・詰み・圧をかける・選出",
    "- 役割: アタッカー・エース・サイクル・削り要員",
    "- ダブルバトル特有の技分類:",
    "  - 集中攻撃＝味方2匹で相手1匹に攻撃を集中させること"
    "（倒せる可能性が上がる一方、まもる等で1ターン無駄になるリスクもある）",
    "  - 範囲攻撃＝相手2匹に同時に効果が及ぶ技（相手が2匹とも場にいる場合、威力は3/4程度に減衰する）",
    "  - 全体攻撃＝範囲攻撃のうち味方も巻き込んでしまう技",
    "- 試合展開: 見せ合い/選出画面・詰めていく",
]

# 改善ロードマップ④続き（2026-08-03・ユーザー提供のチャンピオンズDBダブルバトル
# 基礎知識wiki https://champions.pokewiki.net/ダブルバトル/基礎知識 より抜粋・
# ユーザー確認済み）。技・特性の効果とダブルバトル特有の戦術知識（実況の考察の
# 根拠に使ってよい・スラングのような言い回し集ではなく事実知識）。
_DOUBLES_TACTICS_LINES = [
    "【ダブルバトルの技・特性・戦術知識（実況の考察に活用してよい）】",
    "- 技: ねこだまし＝必ずひるませる優先技・いわなだれ＝全体攻撃+ひるみ30%・"
    "こごえるかぜ＝敵2体の素早さ低下・ワイドガード＝全体技を防ぐ・"
    "まもる/みきり＝1ターン攻撃を完全防御・トリックルーム＝5ターン素早さの順番が逆になる",
    "- 特性: いかく＝場に出た瞬間敵2体の攻撃を下げる・ひらいしん＝電気技を無効化し特攻上昇・"
    "せいぎのこころ＝悪技を受けると攻撃上昇・テレパシー＝味方からの全体攻撃を無効化・"
    "かげふみ＝敵の交換を封じる・ひでり/あめふらし＝天候を変えて炎/水技を強化・"
    "グラスメイカー/サイコメイカー＝フィールドを展開する",
    "- 戦術: てだすけコンボ＝てだすけで味方の技（威力が下がりがちな全体攻撃等）を強化する・"
    "トリックルームパ＝トリックルームで鈍足高火力を通す戦術・"
    "いばるコンボ＝いばる等を絡めて混乱を無効化しつつ攻撃を上げる戦術・"
    "積み技＋誘導技＝誘導技で敵の攻撃を引き受けつつ味方が安全に積む戦術・"
    "じゃくてんほけん活用＝弱点を突かれると発動する持ち物で攻撃/特攻を上げる戦術・"
    "無効/吸収特性活用＝ちょすい等の無効/吸収特性を持つ味方を置いて全体攻撃の巻き込みを防ぐ戦術",
]

# 改善ロードマップ④続き: 口調・用語集の使い方をイメージさせるfew-shot例文
# （2026-08-03・ユーザー確認済み）。(状況, 実況文) のタプル。実在の試合とは無関係な
# 架空例のため、vision/script両プロンプトとも「この試合とは無関係」と明示して使う。
_TONE_EXAMPLES: list[tuple[str, str]] = [
    ("自分の場の2匹が、相手の1匹に技を集中させた",
     "狙うはただ1匹！2匹がかりの集中攻撃、これはキマってほしいっ！"),
    ("自分のポケモンがつるぎのまいで能力を上げた",
     "ここで積んできた〜！攻撃力アップで一気に展開のチャンスが広がったね♪"),
    ("相手の範囲技が自分の場の2匹に着弾した",
     "うわっ範囲技が両方に直撃！？でもまだ大丈夫、ここからしっかり受けていくよ！"),
]

# 2026-08-14: persona="neutral"用（3Dモデル一時差し替え検証用・花圓くれぴの名前・
# 自称・タメ口・♪を含まない中立実況）。src/commentary/kurepi_persona.pyの
# NEUTRAL_CHARACTER_INTRO/NEUTRAL_TONE_EXAMPLESと手動同期すること
# （このファイルはboto3/flask依存でkurepi_persona.pyをimportできないため）。
_NEUTRAL_CHARACTER_INTRO_LINES = [
    "あなたは、ポケモン対戦実況AIVTuberです。",
    "テンション高めだが落ち着いた、中立的な実況口調で話してください"
    "（キャラクターとしての自己紹介・名乗り・一人称のキャラ付けはしないこと）。",
    "あなたは「自分」側（プレイヤー側）の視点で、自分側を応援する立場で実況します。"
    "「自分の」「相手の」の帰属は下記の戦況リストに厳密に従い、"
    "相手のポケモンを自分のもののように（またはその逆に）語らないこと。",
]

_NEUTRAL_TONE_EXAMPLES: list[tuple[str, str]] = [
    ("自分の場の2匹が、相手の1匹に技を集中させた",
     "狙うは1匹！2匹がかりの集中攻撃、これは決まってほしいところです！"),
    ("自分のポケモンがつるぎのまいで能力を上げた",
     "ここで積んできました。攻撃力アップで一気に展開のチャンスが広がります"),
    ("相手の範囲技が自分の場の2匹に着弾した",
     "範囲技が両方に直撃！しかしまだ大丈夫、ここからしっかり受けていきます"),
]


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
    if event_type == "move_single":
        # 技ごとの実況（timeline.jsonlの瞬間ログをトリガーにした新パス）: ターン全体では
        # なく「今まさに使われた技1つ」だけに焦点を絞らせる。move_focus はこのイベント用に
        # pipeline側で組み立てた「陣営の＋ポケモン名＋の＋技名」の文字列
        event_hint = (
            f"今まさに使われた技「{context.get('move_focus', '')}」1つだけに反応する"
            "短い実況をする（ターン全体のまとめや他の技には触れず、この技への即時リアクションに徹すること）"
        )
    elif event_type == "faint" and context.get("faint_focus"):
        # 合成faint（ボール数減少推定）: 画面にHP=0表示は映っていないため、
        # 「HP=0のポケモンを特定」ではなく確定済みの対象を直接指示する
        event_hint = (
            f"「{context.get('faint_focus', '')}」が倒れたことが蓄積した戦況データから確定した"
            "（画面にHP=0の表示は映っていない）。倒れたことに今気づいた体で、"
            "このポケモンが倒れたことだけを実況する"
        )
    elif event_type == "switch" and context.get("switch_focus"):
        # 交代ヒント（2026-08-15）: switchイベントは交代選択画面の時点で発火するため、
        # 実際に繰り出されたポケモンをパイプライン側で確定させて直接指示する
        # （これが無いとLLMが直前の別の交代を今起きたかのように実況する）
        event_hint = (
            f"ポケモンの交代・繰り出しの場面。実際に繰り出されたのは「{context.get('switch_focus', '')}」"
            "（画面の繰り出しメッセージから確定・信頼度高）。この繰り出しだけを実況し、"
            "それより前の交代を今起きたかのように語らないこと"
        )
    else:
        event_hint = {
            "battle_start": "バトル開始！両者のポケモンを紹介して試合への期待感を高める実況をする",
            # move_used=通信終了＝新しいターンの攻防が始まる瞬間に発火する。個別の技は
            # move_singleが都度実況するため、ここでは戦況全体に徹させる（2026-08-15:
            # 従来の「今ターンで使われた技と効果を実況」は発火時点でまだ技が出ておらず、
            # 前ターンの技ログを今起きたかのように語る誤実況の温床だった）
            "move_used":    "コマンドが確定して新しいターンの攻防が始まる場面。戦況全体"
                            "（HP状況・残り数・有利不利）とこのターンの注目ポイントを実況する。"
                            "イベント履歴・技ログにある過去の技や交代を今起きたかのように"
                            "実況し直さないこと",
            "switch":       "ポケモンの交代について実況する",
            "faint":        "ポケモンが倒れた瞬間を実況する（HP=0のポケモンを特定すること）",
            "battle_end":   "試合終了を締めくくる実況をする",
        }.get(event_type, "状況を実況する")

    # 2026-08-14: persona="neutral"（3Dモデル一時差し替え検証用）は花圓くれぴの
    # 名前・自称・口調を含まない中立版のキャラ設定ブロックに差し替える
    persona_mode = context.get("persona", "kurepi")
    if persona_mode == "neutral":
        intro_lines = [*_NEUTRAL_CHARACTER_INTRO_LINES, ""]
    else:
        intro_lines = [
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
        ]

    lines = [
        *intro_lines,
        "【ダブルバトルの基本知識】",
        "- 各プレイヤーが2匹ずつ場に出す（合計4匹が同時に戦う）",
        "- 技名（テラクラスター・アストラルビット・フレアドライブ等）はポケモン名ではなく技の名前",
        "- みがわり・めいそう・テラスタル・アンコール・かなしばりは戦略的な行動",
        "- いまひとつ・こうかなし・こうかあり・こうかばつぐん はダメージ結果テキスト。"
        "ポケモン名・技名と混同しないこと（実況文にそのまま使わない）",
        "- 【OCRで検出した使用技】に「（バツグン）」の注記がある技は、実際に検出された"
        "効果なので信頼して有利不利の実況に使ってよい（例:「そうたいバツグンだ！」）。"
        "注記が無い技については自分でタイプ相性を推測して断定しないこと",
        "- 特性発動メッセージ（〜のわざわいのつるぎ・〜のこだいかっせい等）は技名ではない。技名と混同しないこと",
        "- トレーナー名（画面に映る英数字のIDやハンドルネーム）はポケモン名ではない。"
        "実況でトレーナー名をそのまま呼ばないこと（動画は公開されるため個人が特定できる"
        "名前は伏せる）。相手プレイヤーを指す時は「相手」「お相手」「対戦相手」等の"
        "表現にとどめること",
        "",
        *_DOUBLES_TACTICS_LINES,
        "",
        *_SLANG_GLOSSARY_LINES,
        "",
        "【出力ルール】",
        f"- 今回のイベント: {event_type} → {event_hint}",
        "- 「了解しました/いたしました」「担当させていただきます」「スタンバイ完了」等、",
        "  言い回しを問わず指示への相槌・確認・自己紹介・準備完了報告は絶対に書かない。",
        "  設定の復唱もしない。【状況】【実況】の中身以外は一切出力しないこと",
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
    # 降参による決着（2026-08-15）。kurepi_persona.battle_surrendered_line()と手動同期
    # （server.pyはboto3/flask依存でkurepi_persona.pyをimportできないため）
    if context.get("battle_surrendered"):
        if battle_result == "勝ち":
            _surrender_who = "相手が降参を選んだ"
        elif battle_result == "負け":
            _surrender_who = "自分が降参を選んだ"
        else:
            _surrender_who = "どちらの降参かは不明・断定しない"
        lines += [
            f"- この試合は降参（ギブアップ）によって終了した（{_surrender_who}）。ポケモンが倒れて"
            "決着したわけではないので、最後のポケモンが倒れた・全滅したという描写を"
            "絶対にしないこと",
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
        "- 「あ、あ、」のような相槌を毎回の書き出しに使わないこと。実況ごとに書き出し方を"
        "変える（相槌なしで内容から直接入る・ポケモン名から入る・感嘆詞を変える等）",
        "- 鉤括弧（「」）は使わない",
        "- 見出しは【状況】【実況】の全角鍵括弧形式のみを使う（Markdown見出し#は使わない）",
        "- 技の対象（誰に当たったか）が状況情報から確定できない場合、対象ポケモン名を"
        "勝手に決め打ちしないこと（ダブルバトルでは技ログに対象情報が無いことがあるため、"
        "その場合は「相手の場のポケモン」等ぼかした表現に留める）",
        "- 直前ターンでまもる等の防御が成功したかどうかが状況情報にない場合、攻撃が命中して"
        "ダメージが入ったかのように断定しないこと（逆に、まもるを使っていないのに"
        "使った体で実況することも禁止）",
        "- 技を使った陣営（自分/相手）が状況情報から特定できない場合、断定的に陣営を"
        "言い切らないこと",
        "- 「★ピンチ」はHPが少ないことを示す表示であり、そのポケモンはまだ倒れていない"
        "（気絶した場合はfaintイベントとして別途明示される）。この表示だけを根拠に"
        "「倒れた」「落ちた」「もういない」等の気絶を意味する表現を使わないこと",
        # 2026-08-21: 「事実の説明だけの実況」から「事実＋意見」への改善
        # （kurepi_persona.OUTPUT_RULES_LINESと同じ文言で手動同期）。上記の断定禁止
        # ルール群はあくまで事実（対象・タイプ相性・陣営等）に関するものであり、感情・
        # 短期予想はその対象外であることを明示する
        "- 出来事の説明だけで終わらせず、驚き・期待・不安等の感情や、次の一手がどうなりそうか"
        "という短い予想を一言添えてよい（例:「これで流れが変わるかもしれない」"
        "「次はどう出てくるだろうか」）。ただし技の対象・タイプ相性・天候や場の状態等、"
        "上記の事実に関するルールは変わらず厳守すること（感情・予想はあくまで実況者自身の"
        "見立てとして語り、事実であるかのように断定しないこと）",
        "",
        "【蓄積された戦況（複数ターン分の確定情報）】",
        f"ターン数: {battle_state.get('turn', '不明')}",
        f"自分の場: {battle_state.get('player_field', battle_state.get('player_pokemon', '情報収集中'))}",
        f"自分の控え: {battle_state.get('player_bench', 'なし')}",
        f"相手の場: {battle_state.get('opponent_field', battle_state.get('opponent_pokemon', '情報収集中'))}",
        f"相手の控え: {battle_state.get('opponent_bench', 'なし')}",
        f"ターン推移: {battle_state.get('turn_history', 'なし')}",
        f"直近のイベント履歴: {battle_state.get('event_log', 'なし')}",
    ]
    type_hint = battle_state.get("type_hint")
    if type_hint:
        lines += [
            f"タイプ相性ヒント（Python側で計算済みの確定結果。信頼して有利不利の実況に使ってよい）: {type_hint}",
        ]
    move_effect_hint = battle_state.get("move_effect_hint")
    if move_effect_hint:
        lines += [
            f"直近で使われた技の効果（信頼して事実として扱ってよい）: {move_effect_hint}",
            "※ ダメージを与えない変化技（能力変化・状態異常付与等）の場合、"
            "「ダメージ」「効果ばつぐん」「〜に効いた」等の攻撃結果を表す言葉を使わないこと",
            "※ 上記が「1ターン目は攻撃せず、2ターン目に攻撃が発動する」等の溜め技である"
            "旨を含む場合、それが今1ターン目なのか2ターン目なのかは戦況情報から判断し、"
            "1ターン目（例外的な即時発動条件に該当しない限り）はまだ攻撃していないので"
            "「炸裂」「大ダメージ」等の攻撃結果を表す言葉を使わないこと（2026-08-20追加）",
            "※ 天候による技の威力アップは水/ほのお/こおり等一部タイプの技に限られる特殊ルールで、"
            "上記の効果テキストにその記載が無い限り、天候を理由に「威力が上がる/絶大」等と"
            "独自の知識で付け足さないこと（命中率アップ等、記載の無い天候効果も同様）",
        ]
    # 技の対象ヒント（2026-08-15・move_single対象誤認対策）: 技の直後に画面から
    # 観測されたHP減少・状態異常付与・まもる成功に加え、技そのものの対象範囲
    # （自分自身/相手全体等・2026-08-16追加）も含む。対象の推測誤りが最頻NGだったため、
    # 情報がある場合はそれに厳密に従わせる
    move_target_hint = battle_state.get("move_target_hint")
    if move_target_hint:
        lines += [
            f"この技の対象・結果に関する確定情報（技の仕様＋画面から観測された変化。"
            f"Python側で照合済み）: {move_target_hint}",
            "※ この技の対象・結果は必ず上記に従うこと。そこに登場しない"
            "ポケモンをこの技の対象として実況しないこと",
        ]
    condition_hint = battle_state.get("condition_hint")
    if condition_hint:
        lines += [
            f"場のコンディション（天候・壁・素早さ操作。Python側で計算済みの確定結果。"
            f"信頼して有利不利の実況に使ってよい）: {condition_hint}",
            "※ 上記が壁・天候・トリックルーム・おいかぜの最新の正確な継続状況。"
            "画面テキストに「壁が消えた」等の記述や過去の発動演出が見えても、"
            "それより必ずこちらを優先すること（張り直し演出やテキスト誤読で"
            "矛盾が生じやすいため、上記に無い『消えた/終わった』は実況しないこと）。"
            "同様に、上記に記載の無い天候・壁・トリックルーム・おいかぜは発生していない"
            "ものとして扱い、独自の知識や一般的な戦術の連想で「あまごいとおいかぜが"
            "残っている」等と勝手に補って言及しないこと",
        ]
    else:
        lines += [
            "※ 天候・壁・トリックルーム・おいかぜは現在いずれも発生していない"
            "（Python側で確認済み）。これらに言及しないこと",
        ]
    lines += [
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
    if context.get("faint_focus"):
        lines.append(
            f"倒れたことが確定したポケモン（蓄積した戦況データから確定・信頼度高）: {context['faint_focus']}"
        )
    if context.get("switch_focus"):
        lines += [
            f"直近で実際に繰り出されたポケモン（画面の繰り出しメッセージから確定・信頼度高）: {context['switch_focus']}",
            "※ 交代・繰り出しに言及する場合は必ず上記に従うこと",
        ]
    if context.get("faint_context"):
        lines.append(
            f"直前に起きた気絶の時点の戦況（この直後に下記の技が使われた）: {context['faint_context']}"
        )
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
        "【口調のイメージ例（この試合とは無関係な架空例・内容は真似しなくてよい）】",
    ]
    tone_examples = _TONE_EXAMPLES if persona_mode == "kurepi" else _NEUTRAL_TONE_EXAMPLES
    for situation, commentary in tone_examples:
        lines += ["【状況】", f"（{situation}）", "【実況】", commentary, ""]
    lines += [
        "【状況】",
        "（1文で状況説明）",
        "",
        "【実況】",
        "（1〜2文の実況文）",
    ]
    prompt = "\n".join(lines)
    # Bedrockに実際に送るプロンプト全文をデバッグログに出す（2026-08-21新設・
    # RAGヒント等が実際どう組み込まれたかを事後確認できるように）。既定ではroot
    # ロガーがINFOのため出力されない。確認時はログレベルをDEBUGに上げること。
    logger.debug("[Bedrockプロンプト]\n%s", prompt)
    return prompt


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
    """無言区間の長さから生成するフィラーの目安件数を決める（約20秒に1件・1〜5件）。

    2026-07-14に「とてもしゃべらせたい」で25秒→18秒へ増量したが、
    2026-07-30の視聴フィードバック「フィラーが多い」で30秒/件・上限3へ減量。
    その後「もう少し増やしたい」で25秒/件・上限4へ再調整。
    さらに視聴フィードバック「あ、あ、が耳につく・フィラーを減らして実況を
    活かしたい」で40秒/件・上限3に再々調整（2026-07-30続き）。
    **2026-08-15訂正**: 上記「あ、あ、」fbは実際には言葉遣い（相槌の書き出し）への
    指摘であり、無言埋め自体の生成頻度を絞る話ではなかったとユーザーから訂正あり。
    相槌対策は別途「書き出しのバリエーション」指示で対応済みのため、無言埋めは
    20秒/件・上限5へ積極的に戻す（無言を減らし、隙間を実況・雑談で積極的に埋める方針）。
    """
    return max(1, min(5, int((end - start) // 20)))


def _build_script_prompt(gap: dict, events: list, moments: list = None,
                          persona: str = "kurepi", mode: str = "filler") -> str:
    """台本パス（ギャップフィラー生成／予測）のプロンプトを組み立てる。

    録画解析済みの実況タイムライン・瞬間ログ（技が画面に映った動画内時刻）
    のうち、対象の区間 ``gap`` の開始時刻より前のものだけを使って
    プロンプトを組み立てる（ADR-009 台本パス）。

    ⚠️ 呼び出し元は区間ごとに1回呼ぶこと（1プロンプト=1区間）。
    未来の情報を「見せた上で使うな」と指示するだけでは指示に従い損ねて
    ネタバレする実例があったため（2026-07-14）、そもそも未来の events/moments
    をプロンプトに含めない構造的対策にしている。

    persona: "kurepi"（デフォルト・花圓くれぴ）/"neutral"（3Dモデル一時差し替え
    検証用・2026-08-14）。
    mode: "filler"（デフォルト・無言区間埋め）/"predict"（2026-08-21新設・
    「予測→回収」実況の予測側。gap は候補時刻の0秒幅区間 {start: T, end: T} を
    渡すことでフィラー1件だけ生成させる想定。events/momentsの「gap_start以前だけ
    見せる」構造的ネタバレ対策はfillerと共通でそのまま効く。回収側は
    _build_payoff_prompt()を使う）。
    """
    first_event_time = min(float(e.get("time", 0)) for e in events)
    gap_start = float(gap["start"])
    gap_end = float(gap["end"])
    # 動画冒頭の無言区間か（generate_gap_commentary.pyのcompute_gapsが付与）。
    # 開始時挨拶を確実に入れるための2026-08-15新設フラグ
    is_intro = bool(gap.get("is_intro"))

    if persona == "neutral":
        intro_lines = [
            "あなたは、ポケモン対戦実況AIVTuberです。",
            "テンション高めだが落ち着いた、中立的な実況口調で話してください"
            "（キャラクターとしての自己紹介・名乗り・一人称のキャラ付けはしないこと）。",
            "あなたは「自分」側（プレイヤー側）の視点で、自分側を応援する立場で実況します。",
        ]
    else:
        intro_lines = [
            "あなたは、ポケモン対戦実況AIVTuber「花圓くれぴ（はなまるくれぴ）」です。",
            "性格は元気で甘えん坊、でもポケモン知識はガチ勢。口調はアイドル・かわいい系",
            "（語尾に♪を適度に使う・タメ口・テンション高め・かわいい褒め言葉多め）。",
            "自称は「くれぴ」（ひらがな）。「花圓」という漢字表記は実況文に書かないこと"
            "（音声合成が正しく読めないため）。",
            "あなたは「自分」側（プレイヤー側）の視点で、自分側を応援する立場で実況します。",
        ]

    tone_examples = _TONE_EXAMPLES if persona == "kurepi" else _NEUTRAL_TONE_EXAMPLES

    lines = [
        *intro_lines,
        "ポケモンの「自分の」「相手の」の帰属はタイムラインの記載（【自分側】【相手側】タグ・",
        "戦況の場/控え情報）に厳密に従い、タグや記載がなく判別できない場合は",
        "「自分の/相手の」を付けずにポケモン名だけで実況すること（推測で断定しない）。",
        "録画された試合に後から実況を吹き込みますが、視聴者には生放送の",
        "ライブ実況に聞こえるようにしてください（後から見返している・録画といった言い方は禁止）。",
        "主要イベントの実況音声はすでに収録済みです。",
        *(
            [
                "ここでは、ここまでの試合展開を踏まえて、この後の試合の行方について",
                "実況者らしい短い予想を1つだけ述べてください（当たっても外れても構いません。",
                "あくまで実況者自身の見立てとして語ること。断定はしないこと）。",
            ]
            if mode == "predict"
            else [
                "イベントとイベントの間の時間も、視聴者を楽しませる実況・雑談として積極的に",
                "語ってください。「無言を埋めるための最低限の一言」ではなく、内容のあるトーク",
                "（考察・戦況の掘り下げ・キャラとしての感想等）を心がけること。",
            ]
        ),
        "トレーナー名（画面に映る英数字のIDやハンドルネーム）が情報に含まれていても",
        "実況でそのまま呼ばないこと（動画は公開されるため個人が特定できる名前は伏せる）。"
        "相手プレイヤーを指す時は「相手」「お相手」「対戦相手」等の表現にとどめること。",
        "",
        *_DOUBLES_TACTICS_LINES,
        "",
        *_SLANG_GLOSSARY_LINES,
        "",
        "【口調のイメージ例（この試合とは無関係な架空例・内容は真似しなくてよい）】",
        *[f"- {commentary}" for _situation, commentary in tone_examples],
        "",
    ]
    if is_intro and mode == "filler":
        if persona == "neutral":
            # 2026-08-15実機発見: neutralペルソナの「自己紹介・名乗り・キャラ付け禁止」
            # ルールと素朴に併記すると、LLMが挨拶自体まで自重してしまい弱い書き出し
            # （「それでは対戦を始めていきます」等）になっていた実例あり。視聴者への
            # 挨拶は名乗り・キャラ付けとは別物だと明示して矛盾を解消する
            lines += [
                "【最優先・動画冒頭の挨拶】",
                "- この区間は動画が始まって最初の無言区間です。time が一番早いフィラーは",
                "  必ず視聴者への挨拶（「みなさん、こんにちは」「今日の対戦をお届けします」",
                "  等の呼びかけ）から始めること。この挨拶は自己紹介・一人称のキャラ付けには",
                "  当たらないため、上記の名乗り禁止ルールとは別に必ず行うこと（名前を",
                "  名乗る必要はない・挨拶の言葉自体は省略しないこと）。今日の対戦への",
                "  意気込みや見どころ紹介を続けてよい",
                "",
            ]
        else:
            lines += [
                "【最優先・動画冒頭の挨拶】",
                "- この区間は動画が始まって最初の無言区間です。time が一番早いフィラーは",
                "  必ず視聴者への挨拶（「みんな、こんにちは/こんばんは」「今日も一緒に",
                "  見ていくよ」等の呼びかけ）から始めること。今日の対戦への意気込みや",
                "  見どころ紹介を続けてよい（自己紹介的な名乗りも1回まではOK）",
                "",
            ]
    lines += [
        "【最重要ルール: ネタバレ禁止】",
        "- 各フィラーは、その time 時点までに起きた出来事の情報だけを使うこと",
        "- time より後の出来事や試合の勝敗を絶対に先取りして言及しないこと",
        "- 「次に〜が来たら怖い」のような予想はOK（断定はしない・外れてもよい）",
        f"- {first_event_time:.0f}秒より前の区間（試合開始前）は、ポケモン名などの具体情報を使わず、",
        "  挨拶・意気込み・観戦ポイントなどの汎用トークにすること",
        "",
        *(
            [
                "【予想の内容】",
                "- 直前までの展開（技・交代・気絶・場の状態等）を踏まえた、この後の試合の",
                "  行方についての短い予想を1つ述べること（例:「ここでおいかぜを展開したのは"
                "大きいかもしれない」）",
                "- タイプ相性や特性をふまえた考察を交えてよい",
                "",
            ]
            if mode == "predict"
            else [
                "【フィラーの内容（バリエーションを持たせる）】",
                "- 【最優先】直前に画面に映った技（📺印）への反応・実況。区間の開始時刻が📺と",
                "  ほぼ同じ場合、最初のフィラーは区間開始時刻に置き、その技への反応から始めること",
                "  （例: 📺200.0秒 ドラゴンクロー → 200.5秒 ガブリアスのドラゴンクローが炸裂！...）",
                "- 直前の展開の振り返り・戦況の整理（HP・残り頭数）",
                "- タイプ相性や特性をふまえた考察",
                "- 次の展開の予想",
                "",
            ]
        ),
        "【書き出しのバリエーション】",
        "- 「あ、あ、」のような相槌を毎回の書き出しに使わないこと。フィラーごとに",
        "  書き出し方を変える（相槌なしで内容から直接入る・ポケモン名から入る・",
        "  感嘆詞を変える等）",
        "",
        "【出力形式】",
        "- JSON配列のみを出力すること（前置き・説明文・コードフェンスは書かない）",
        '- 形式: [{"time": 秒数の数値, "text": "実況文"}, ...]',
        # 2026-08-21新設: 無言区間の長さに応じてフィラーの目標文字数を可変にする
        # （区間実測で5〜19秒の無言が最多帯と判明したのに、従来は60〜100文字固定＝
        # 読み上げ10〜18秒だったため短い区間に収まらずfit_fillers()で破棄されていた）。
        # predictモードは常に単発なのでこの区間長スケーリングは対象外
        (
            "- text は60〜100文字程度（読み上げ約10〜18秒）" if mode == "predict"
            else "- text は15〜25文字程度（読み上げ約3〜5秒・短い間なので手短に）" if (gap_end - gap_start) < 10
            else "- text は40〜60文字程度（読み上げ約7〜10秒）" if (gap_end - gap_start) < 20
            else "- text は60〜100文字程度（読み上げ約10〜18秒）"
        ),
        (
            f"- 要素は1件だけにすること。time は {gap_start:.1f} にすること"
            if mode == "predict"
            else "- time は必ず無言区間の範囲内の数値にすること。各フィラーは20秒以上離すこと"
        ),
        "- 鉤括弧（「」）は使わない",
        "- 技の対象（誰に当たったか）がタイムラインから確定できない場合、対象ポケモン名を"
        "勝手に決め打ちしないこと（ダブルバトルでは対象情報が無いことがあるため、"
        "その場合は「相手の場のポケモン」等ぼかした表現に留める）",
        "- 直前ターンでまもる等の防御が成功したかどうかがタイムラインにない場合、攻撃が"
        "命中してダメージが入ったかのように断定しないこと（逆に、まもるを使っていないのに"
        "使った体で実況することも禁止）",
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

    if mode == "predict":
        lines.append(f"- ★{gap_start:.1f}秒 = ここで実況者としての短い予想を1件")
    else:
        n = _gap_filler_count(gap_start, gap_end)
        lines.append(f"- ★{gap_start:.1f}秒 〜 {gap_end:.1f}秒 = 無言区間（ここにフィラーを{n}件）")
    return "\n".join(lines)


def _build_payoff_prompt(prediction_text: str, hit: bool, outcome_summary: str,
                          payoff_time: float, persona: str = "kurepi") -> str:
    """「予測→回収」実況の回収側のプロンプトを組み立てる（2026-08-21新設）。

    的中/外れの判定自体は呼び出し元（scripts/generate_predictions.py）が
    最終battle_resultから機械的に確定させ、事実として渡す。LLMには演技
    （実況者らしい反応）だけをさせ、判定自体を推測させない。
    _build_script_prompt（predict側）と違い、events/momentsのタイムラインは
    渡さない軽量プロンプト（回収は「さっきの予想はどうだったか」の一言だけで
    完結するため、ネタバレ対策としての未来情報フィルタも不要）。
    """
    if persona == "neutral":
        intro_lines = [
            "あなたは、ポケモン対戦実況AIVTuberです。",
            "テンション高めだが落ち着いた、中立的な実況口調で話してください"
            "（キャラクターとしての自己紹介・名乗り・一人称のキャラ付けはしないこと）。",
        ]
    else:
        intro_lines = [
            "あなたは、ポケモン対戦実況AIVTuber「花圓くれぴ（はなまるくれぴ）」です。",
            "性格は元気で甘えん坊、でもポケモン知識はガチ勢。口調はアイドル・かわいい系",
            "（語尾に♪を適度に使う・タメ口・テンション高め・かわいい褒め言葉多め）。",
            "自称は「くれぴ」（ひらがな）。「花圓」という漢字表記は実況文に書かないこと"
            "（音声合成が正しく読めないため）。",
        ]
    result_word = "的中" if hit else "外れ"
    lines = [
        *intro_lines,
        "録画された試合に後から実況を吹き込みますが、視聴者には生放送のライブ実況に"
        "聞こえるようにしてください（後から見返している・録画といった言い方は禁止）。",
        "",
        "【状況】",
        f"あなたはさっき「{prediction_text}」という予想を口にしていました。",
        f"その後の展開: {outcome_summary}",
        f"結果、あなたの予想は{result_word}でした（この判定は確定事実として扱うこと）。",
        "",
        "【指示】",
        f"この結果を受けて、実況者らしく一言で反応してください（{result_word}であることが"
        "伝わるように。的中なら誇らしげ/嬉しそうに、外れなら驚き・悔しさ・自虐等の中から"
        "自然な反応を選ぶこと。さっきの予想の内容に軽く触れてよい）。",
        "- 鉤括弧（「」）は使わない",
        "- 「了解しました」等、指示への相槌・確認は書かない",
        "",
        "【出力形式】",
        "- JSON配列のみを出力すること（前置き・説明文・コードフェンスは書かない）",
        '- 形式: [{"time": 秒数の数値, "text": "実況文"}]（要素は1件だけ）',
        f"- time は {payoff_time:.1f} にすること",
        "- text は60〜100文字程度（読み上げ約10〜18秒）",
    ]
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
    フィラー実況、または「予測→回収」実況（2026-08-21新設）をテキストのみの
    Bedrock呼び出しで生成する。

    無言区間・予測ポイントごとに1回呼び出す契約（ネタバレ防止のため、この
    時点より未来のevents/momentsはプロンプトに含めない構造にしている）。

    mode（既定"filler"）:
    - "filler"/"predict": events（gap開始時刻以前だけ使われる）・gap（predictは
      {start,end}に同じ候補時刻を渡す）が必要。_build_script_prompt を使う
    - "payoff": prediction_text/hit/outcome_summary/time が必要（events/gap不要）。
      _build_payoff_prompt を使う（的中/外れ判定は呼び出し元が確定済みの事実として渡す）
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json", "message": "リクエストボディがJSONではありません"}), 400

    mode: str = data.get("mode", "filler")
    # 2026-08-14: 3Dモデル一時差し替え検証用（"kurepi"デフォルト/"neutral"）
    persona: str = data.get("persona", "kurepi")

    if mode == "payoff":
        prediction_text = data.get("prediction_text")
        hit = data.get("hit")
        outcome_summary = data.get("outcome_summary")
        payoff_time = data.get("time")
        if not prediction_text or hit is None or not outcome_summary or payoff_time is None:
            return jsonify({
                "success": False, "error": "missing_payoff_fields",
                "message": "prediction_text/hit/outcome_summary/time が必要です",
            }), 400
        prompt_text = _build_payoff_prompt(
            prediction_text, bool(hit), outcome_summary, float(payoff_time), persona=persona)
    else:
        events: list = data.get("events", [])
        gap = data.get("gap")
        moments: list = data.get("moments", [])
        if not events:
            return jsonify({"success": False, "error": "missing_events", "message": "events が必要です"}), 400
        if not isinstance(gap, dict) or "start" not in gap or "end" not in gap:
            return jsonify({"success": False, "error": "missing_gap", "message": "gap（start/endを持つオブジェクト）が必要です"}), 400
        prompt_text = _build_script_prompt(gap, events, moments, persona=persona, mode=mode)
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
