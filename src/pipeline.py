"""
Sprint 5: パイプライン統合スクリプト
--------------------------------------------------
各コンポーネントをつなぎ、ポケモン対戦をリアルタイム実況する。

処理フロー（ADR-007 イベント駆動アーキテクチャ）:
  1. OBS仮想カメラからフレームキャプチャ（1秒ごと）
  2. YOLOv8 でアイコン検出（毎フレーム）
  3. OpenCV 差分検出でイベント判定
  4. イベント検知時: EasyOCR でテキスト取得
  5. Phi-3 mini (Ollama) で実況文生成
  6. VOICEVOX で音声合成 → 再生

オプションで EC2 API 経由の Bedrock Vision 分析も利用可能（ターン切替・交代・気絶時）。

実行例:
  venv\\Scripts\\python.exe src/pipeline.py --camera 3 --model runs/detect/train4/weights/best.pt

事前起動:
  - OBS仮想カメラ ON（カメラ番号 3）
  - Ollama 起動（ollama serve）
  - VOICEVOX 起動（localhost:50021）
"""

from __future__ import annotations

import argparse
import base64
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

# プロジェクトルートを sys.path に追加
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.capture.screen_capture import DiffDetector, init_reader, run_ocr
from src.capture.yolo_detector import BattleState, YoloDetector
from src.commentary.phi3_client import Phi3Client
from src.output.audio_player import AudioPlayer
from src.output.voicevox_client import VoicevoxClient
from src.pokedb.classifier import CATEGORY_POKEMON, PokeClassifier


def _setup_logging() -> Path:
    """コンソールとファイル両方にログを出力する。ログファイルのパスを返す。"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()  # import 時に他モジュールが追加したハンドラーを除去
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)

    return log_path

log_file_path = _setup_logging()
log = logging.getLogger(__name__)
log.info(f"ログファイル: {log_file_path}")

# Bedrock を呼ぶイベント種別
BEDROCK_EVENTS = {"battle_start", "move_used", "switch", "faint", "battle_end"}


# ─── OCR 結果からゲーム状態を構築 ─────────────────────────────────────────────

# バトル中画面を示すキーワード（これが OCR に含まれていない場合はスキップ）
_BATTLE_KEYWORDS = {
    "HP", "hp", "わざ", "技", "どうする", "たたかう", "もちもの",
    "にげる", "テラスタル", "交代", "こうたい",
}

# バトル外画面を示すキーワード（含まれている場合はスキップ）
# 注意: 「通信中」はバトル中のアニメーション画面でも出るため除外している
_NON_BATTLE_KEYWORDS = {
    "オフライン", "ユニオンサークル", "テラレイドバトル", "通信交換",
    "マジカル交換", "通信対戦", "バトルスタジアム", "ランクバトル選択",
    "レンタル", "てもち", "チーム", "マスターボール級", "RANK MAX",
    "せいせき", "はんえい", "ごほうび", "リーグペイ",
    "受けとりました", "おめでとう",
}

OCR_MAX_CHARS = 120  # Phi-3 に渡す OCR テキストの最大文字数

# 画面上半分（y < この値）= 相手エリア、下半分 = 自分エリア（1080p 基準）
_PLAYER_Y_THRESHOLD = 500

# コマンドメニュー（y > この値）= 技選択UI → ポケモン名候補から除外（1080p 基準）
_COMMAND_Y_MIN = 700

# UI ノイズ（ポケモン名・技名ではないテキスト）
_UI_WORDS = {
    "たたかう", "ポケモン", "にげる", "相手を見る", "様子を見る",
    "もちもの", "こうたい", "テラスタル", "どうする",
}

# バトル結果テキスト（name_candidates に混入しないようにフィルター）
_BATTLE_RESULT_WORDS = {
    "バツグンだ", "いまひとつ", "こうかは", "こうかなし", "こうかがない",
    "効果は", "今ひとつ", "のようだ", "こうか", "効果",
    "あまり", "ない", "技", "わざ", "もどる",
}

# 技名・特性名（バトルメッセージ/コマンドエリアに出やすくポケモン名と混同される）
_MOVE_ABILITY_WORDS = {
    "みがわり", "まもる", "めいそう", "こわいかお", "いかく", "きんちょうかん",
    "ひかりのかべ", "リフレクター", "おいかぜ", "トリックルーム",
    "アンコール", "かなしばり", "テラバースト", "こうごうせい", "ちょすい",
    "ふゆう", "はやあし", "すてみ", "じしん", "ほのおのうず",
}

# 「相手を見る」UI オーバーレイ・システムテキスト（ポケモン名でないもの）
_UI_OVERLAY_WORDS = {
    "状態", "戦闘中", "タイプ", "テラスタイプ", "オンライン", "通信中",
    "ロノマル", "待機中", "ヒシン", "日", "ガラル", "アローラ",
}


def _ocr_results_to_text(
    ocr_results: list[dict],
    classifier: PokeClassifier | None = None,
) -> str:
    """OCR 結果を読みやすいテキスト文字列にまとめる（最大 OCR_MAX_CHARS 文字）。

    classifier が渡された場合は技名・特性名・アイテム名（単語として独立したもの）を除外する。
    助詞・助動詞を含む文脈付きテキスト（バトルメッセージ）は除外しない。
    これにより技選択画面の技一覧が ocr_text に混入するのを防ぐ。
    """
    lines = []
    for r in ocr_results:
        if r["confidence"] < 0.4:
            continue
        text = r["text"]
        # 技名・特性名・アイテム名（単語として独立したもの）を除外
        # 助詞・助動詞・感嘆符を含む場合はバトルメッセージなので残す
        # 「の」で終わるテキストはバトルメッセージの一部（「トルネロスの」等）なので残す
        # 「の」で終わらない場合は「キノコのほうし」等の技名・アイテム名も除外対象にする
        if (classifier is not None
                and len(text) >= 2
                and not any(c in text for c in "はがをにからでと！？")
                and not text.endswith("の")):
            cat = classifier.classify(text).category
            if cat in ("move", "ability", "item"):
                continue
        lines.append(text)
    text = " / ".join(lines) if lines else "（テキスト未検出）"
    return text[:OCR_MAX_CHARS]


def _extract_structured_info(
    ocr_results: list[dict],
    classifier: PokeClassifier | None = None,
) -> dict:
    """
    OCR 結果から HP 値・ポケモン名候補を抽出して構造化する。

    classifier（PokeClassifier）が渡された場合は DB 照合でポケモン名のみを抽出し、
    技名・特性名・アイテム名・UIノイズを自動除外する。
    渡されない場合は従来の手動フィルターで動作する（フォールバック）。
    """
    # HP 抽出用（分母と分子を別グループで取得）
    hp_pattern = re.compile(r'\b(\d{1,3})/(\d{1,3})\b')
    # PP 最大値は 40 以下なので分母 >= 50 のみ HP として採用（PP 値との混同を防ぐ）
    _HP_MIN_DENOM = 50

    hp_values: list[str] = []
    hp_values_player: list[str] = []    # y >= _PLAYER_Y_THRESHOLD: 自分側HP
    hp_values_opponent: list[str] = []  # y < _PLAYER_Y_THRESHOLD:  相手側HP
    name_candidates_player: list[str] = []    # 下半分 = 自分のポケモンエリア
    name_candidates_opponent: list[str] = []  # 上半分 = 相手のポケモンエリア

    # 「相手を見る」状態確認パネル検出
    # "戦闘中" はこのパネル専用テキスト。ここではポケモンが画面上に並ぶため
    # y 座標ベースの自分/相手分類が信頼できない → 名前候補の収集をスキップする
    all_texts = {r["text"] for r in ocr_results if r["confidence"] >= 0.4}
    is_status_panel = any("戦闘中" in t for t in all_texts)
    if is_status_panel:
        log.debug("「相手を見る」パネル検出 → ポケモン名候補収集をスキップ")

    for r in ocr_results:
        if r["confidence"] < 0.4:
            continue
        text = r["text"].strip()

        # HP 値を抽出（分母 < 50 は PP 値のため除外し continue で名前候補にも入れない）
        m = hp_pattern.search(text)
        if m:
            denom = int(m.group(2))
            if denom >= _HP_MIN_DENOM:
                hp_str = f"{m.group(1)}/{m.group(2)}"
                hp_values.append(hp_str)
                # y座標で自分/相手側に分類（HP値を場のポケモンに紐付けるため）
                bbox_hp = r.get("bbox", [])
                if bbox_hp:
                    cy = (bbox_hp[0][1] + bbox_hp[2][1]) / 2
                    if cy < _PLAYER_Y_THRESHOLD:
                        hp_values_opponent.append(hp_str)
                    elif cy < _COMMAND_Y_MIN:
                        hp_values_player.append(hp_str)
                else:
                    hp_values_player.append(hp_str)  # bbox なし → 自分側に追加
            continue

        # 状態確認パネル中は名前候補収集しない（HP値は上で収集済み）
        if is_status_panel:
            continue

        # 共通の軽量フィルター（DB 照合前に除外）
        if (text.startswith("Lv") or re.match(r'^[\d\s/]+$', text)
                or text in _UI_WORDS or text in _BATTLE_RESULT_WORDS
                or any(kw in text for kw in _BATTLE_RESULT_WORDS)
                or (re.match(r'^[A-Za-z0-9\s]+$', text) and len(text) < 4)
                or text.endswith("の") or text.endswith("」") or text.endswith("!")):
            continue

        # タイプ表示テキスト除外（「タイプ」→「タイプ：ヌル」への誤 fuzzy マッチ防止）
        if "タイプ" in text or "テラスタイプ" in text:
            continue

        # bbox の中心 y 座標で自分/相手エリアを判定
        bbox = r.get("bbox", [])
        if bbox:
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            if center_y > _COMMAND_Y_MIN:
                continue  # コマンドメニュー内は除外
        else:
            center_y = 999  # bbox なし → 自分側扱い

        # ── PokeClassifier で分類 ─────────────────────────────────────────
        if classifier is not None:
            result = classifier.classify(text)
            if result.category != CATEGORY_POKEMON:
                # 相手エリアで認識不能かつ「中国語ポケモン名」らしいテキストはそのまま登録
                # （相手プレイヤーのゲームが中国語設定の場合の対応）
                if (center_y < _PLAYER_Y_THRESHOLD
                        and result.category == "unknown"
                        and 2 <= len(text) <= 8):
                    has_cjk  = any('\u4e00' <= c <= '\u9fff' for c in text)
                    has_kana = any('\u3040' <= c <= '\u30ff' for c in text)
                    if has_cjk and not has_kana:
                        log.debug("中国語ポケモン名候補: %s", text)
                        name_candidates_opponent.append(text)
                # ポケモン名でなければ除外（技・特性・アイテム・不明）
                continue
            # 正規化された名前（OCR 誤読を補正）を使う
            canonical = result.canonical_ja
            log.debug("PokeDB分類: %s → %s (score=%.1f)", text, canonical, result.score)
        else:
            # フォールバック: 手動フィルター（DB 未使用時）
            if text in _MOVE_ABILITY_WORDS or text in _UI_OVERLAY_WORDS:
                continue
            canonical = text

        # 自分/相手エリアに振り分け
        if center_y < _PLAYER_Y_THRESHOLD:
            name_candidates_opponent.append(canonical)
        else:
            name_candidates_player.append(canonical)

    return {
        "hp_values":          hp_values,
        "hp_values_player":   hp_values_player[:2],    # ダブルバトル: 最大2匹分
        "hp_values_opponent": hp_values_opponent[:2],
        "name_candidates_player":   name_candidates_player[:5],
        "name_candidates_opponent": name_candidates_opponent[:5],
    }


def _is_battle_screen(ocr_results: list[dict]) -> bool:
    """
    OCR テキストからバトル中の画面かどうかを簡易判定する。
    バトル外キーワードが含まれていたら False、
    OCR が 0 件の場合は判定できないので True（通過させる）。
    """
    if not ocr_results:
        return True
    texts = {r["text"] for r in ocr_results}
    for kw in _NON_BATTLE_KEYWORDS:
        if any(kw in t for t in texts):
            return False
    return True


def _build_game_state(
    ocr_results: list[dict],
    yolo_state: BattleState,
    event_type: str,
    prev_yolo: BattleState | None,
    classifier: PokeClassifier | None = None,
) -> dict:
    """
    OCR + YOLO 結果から Phi-3 に渡す game_state を組み立てる。
    HP・ポケモン名の精密なパースは難しいため、OCR 生テキストを
    実況文生成の追加コンテキストとして渡す。
    """
    # YOLO から状態異常・ボール数を取得
    status_text = yolo_state.player_status or "なし"
    if yolo_state.opponent_status:
        status_text += f" / 相手: {yolo_state.opponent_status}"

    p_balls = yolo_state.player_balls.alive
    o_balls = yolo_state.opponent_balls.alive

    structured = _extract_structured_info(ocr_results, classifier)

    return {
        "pokemon_player":   "（OCR参照）",
        "hp_player":        "?",
        "pokemon_opponent": "（OCR参照）",
        "hp_opponent":      "?",
        "last_move":        "（OCR参照）",
        "status":           status_text,
        "balls_remaining":  [p_balls, o_balls] if (p_balls or o_balls) else [],
        "event_type":       event_type,
        "ocr_text":         _ocr_results_to_text(ocr_results, classifier),
        "hp_values":          structured["hp_values"],
        "hp_values_player":   structured["hp_values_player"],
        "hp_values_opponent": structured["hp_values_opponent"],
        "name_candidates_player":   structured["name_candidates_player"],
        "name_candidates_opponent": structured["name_candidates_opponent"],
    }


# ─── バトルフェーズ分類 + イベント検知 ───────────────────────────────────────

class BattlePhaseClassifier:
    """
    OCR テキストから現在のバトルフェーズを分類し、
    フェーズ遷移からイベントを検知する。

    フェーズ:
      command_select  ─ コマンド選択中（たたかう表示）
      switch_select   ─ 交代選択中
      animation       ─ 技アニメーション中（ダメージテキスト出現）
      faint           ─ HP=0 検知
      battle_end      ─ 勝敗決定
      unknown         ─ 判定不能（演出中など）

    イベント（フェーズ遷移）:
      battle_start  ─ 初回 command_select 出現
      move_used     ─ command_select → それ以外
      switch        ─ switch_select 出現
      faint         ─ faint フェーズ出現
      battle_end    ─ battle_end フェーズ出現
    """

    # "ゆけつ" = 行けっ！（ポケモン繰り出し）/ "いけつ" は OCR 誤読バリアント
    # これらはバトル最初のコマンド選択画面より前に出るため battle_start の早期検知に使う
    _COMMAND_KW    = {"たたかう", "どうする", "ゆけつ", "いけつ"}
    _SWITCH_KW     = {"こうたい", "ポケモンをえらんで"}
    _ANIM_KW       = {"バツグンだ", "いまひとつ", "こうかなし", "きゅうしょ", "急所", "ひんし"}
    _END_KW        = {"勝負に勝", "勝負に負", "降参が選ばれ", "通信エラー", "切断されました"}
    # 選出画面キーワード（この画面中はバトルイベントを発火させない）
    _SELECTION_KW  = {"ポケモンを選んで", "選出", "きめる", "リーダー", "選出順"}
    # L50競技ポケモンの最低HPは約50以上なので、分母が50未満は除外
    _HP_ZERO_RE    = re.compile(r'\b0/([5-9]\d|\d{3})\b')

    def __init__(self, debounce_seconds: float = 10.0):
        self._debounce = debounce_seconds
        self._last_event_time: dict[str, float] = {}
        self._prev_phase = "unknown"
        self._battle_started = False
        self._is_processing = False

    def set_processing(self, v: bool) -> None:
        self._is_processing = v

    def classify(self, ocr_results: list[dict]) -> str:
        """OCR 結果から現在のフェーズを判定する（優先度順）。"""
        texts = {r["text"] for r in ocr_results if r["confidence"] >= 0.4}

        if any(kw in t for kw in self._END_KW for t in texts):
            return "battle_end"
        # 選出画面: バトル前のポケモン選択画面（ここでのイベント発火を防ぐ）
        if any(kw in t for kw in self._SELECTION_KW for t in texts):
            return "selection_screen"
        if any(self._HP_ZERO_RE.search(r["text"]) for r in ocr_results if r["confidence"] >= 0.4):
            return "faint"
        if any(kw in t for kw in self._SWITCH_KW for t in texts):
            return "switch_select"
        if any(kw in t for kw in self._COMMAND_KW for t in texts):
            return "command_select"
        if any(kw in t for kw in self._ANIM_KW for t in texts):
            return "animation"
        return "unknown"

    def detect(self, ocr_results: list[dict]) -> str | None:
        """フェーズ遷移からイベントを返す。イベントなし or 処理中 は None。
        ただし battle_end は処理中でも割り込み検知する（実況中の試合終了を見逃さないため）。
        """
        curr = self.classify(ocr_results)
        prev = self._prev_phase
        self._prev_phase = curr

        # 処理中でも battle_end だけは割り込み検知
        if self._is_processing:
            if curr == "battle_end" and prev != "battle_end":
                self._battle_started = False
                log.info(f"[フェーズ] {prev} → {curr} | イベント: battle_end (実況中割り込み)")
                return "battle_end"
            return None

        event: str | None = None

        # 選出画面中は battle_started をリセット（誤発火対策）
        if curr == "selection_screen":
            self._battle_started = False

        if curr == "command_select" and not self._battle_started:
            self._battle_started = True
            event = "battle_start"
        elif prev == "command_select" and curr != "command_select":
            # unknown = アニメーション中（特殊テキストなし）でも move_used を発火させる
            event = "switch" if curr == "switch_select" else "move_used"
        elif curr == "faint" and prev != "faint":
            event = "faint"
        elif curr == "switch_select" and prev not in ("switch_select", "command_select"):
            event = "switch"
        elif curr == "battle_end" and prev != "battle_end":
            event = "battle_end"
            self._battle_started = False  # 次の試合の battle_start を正しく検知するためリセット

        if curr != prev:
            log.info(f"[フェーズ] {prev} → {curr}" + (f" | イベント: {event}" if event else ""))

        if event:
            now = time.time()
            no_debounce = {"battle_start", "battle_end"}
            last = self._last_event_time.get(event, 0.0)
            if event not in no_debounce and now - last < self._debounce:
                log.debug(f"デバウンス中のためスキップ: {event} (残り {self._debounce-(now-last):.1f}s)")
                return None
            self._last_event_time[event] = now

        return event


# ─── 戦況トラッカー ────────────────────────────────────────────────────────────

@dataclass
class FieldPokemon:
    """1匹のポケモンの戦況スロット（ダブルバトル対応）。"""
    name: str
    hp: str | None = None                        # "176/176" 形式（最新HP）
    status: str | None = None                    # まひ / やけど / どく / ひんし
    moves_used: list[str] = field(default_factory=list)  # このポケモンが使った技リスト
    on_field: bool = False                        # 現在場にいるか
    fainted: bool = False                         # 気絶済みフラグ
    confidence: int = 0                           # 検出回数（信頼度）
    last_seen_turn: int = 0                       # 最後に検出されたターン番号


class BattleStateTracker:
    """
    試合全体の戦況を蓄積するクラス（ダブルバトル対応）。

    自分・相手それぞれ最大 4 スロットでポケモンを管理する。
    場のポケモン（on_field=True）は最大 2 匹まで（ダブルバトル制約）。
    HP はy座標分類で側ごとに紐付け、技はポケモンごとに記録する。
    """

    MAX_SLOTS       = 4    # 試合全体での最大登録数（4匹パーティ）
    MAX_ON_FIELD    = 2    # ダブルバトル: 同時に場に出せる最大数
    MAX_EVENTS      = 8
    # on_field=True でこのターン数以上不検出なら場にいないと判断
    _ON_FIELD_MISS_THRESHOLD = 5
    _HP_RE = re.compile(r'(\d{1,3})/(\d{1,3})')

    def __init__(self):
        self.turn = 0
        self._player:   list[FieldPokemon] = []  # 自分の最大4匹
        self._opponent: list[FieldPokemon] = []  # 相手の最大4匹
        self._event_log: list[str] = []

    # ── 内部ヘルパー ─────────────────────────────────────────────────────────

    def _get_or_create(self, slots: list[FieldPokemon], name: str) -> FieldPokemon | None:
        """名前でスロットを検索。なければ新規作成（MAX_SLOTS を超えたら None）。"""
        for s in slots:
            if s.name == name:
                return s
        if len(slots) < self.MAX_SLOTS:
            slot = FieldPokemon(name=name)
            slots.append(slot)
            return slot
        return None  # 4匹超過は無視

    def _cap_on_field(self, slots: list[FieldPokemon]) -> None:
        """ダブルバトル制約: 場に出せるのは最大 MAX_ON_FIELD 匹。超えた分は confidence が低い方を除外。"""
        on_field = [s for s in slots if s.on_field]
        if len(on_field) > self.MAX_ON_FIELD:
            sorted_on = sorted(on_field, key=lambda s: -s.confidence)
            for s in sorted_on[self.MAX_ON_FIELD:]:
                s.on_field = False

    def _assign_hp_to_on_field(self, slots: list[FieldPokemon], hp_list: list[str]) -> None:
        """on_field のポケモンに HP 値をインデックス順に割り当てる。
        HP=0での即気絶はしない（誤分類で無実のポケモンが気絶扱いになるのを防ぐ）。
        気絶判定は faint イベント時のみ行う。
        """
        on_field = [s for s in slots if s.on_field]
        for i, hp in enumerate(hp_list):
            if i < len(on_field):
                on_field[i].hp = hp

    # ── メイン更新 ───────────────────────────────────────────────────────────

    def update(self, game_state: dict, event_type: str) -> None:
        """1 イベントごとに呼び出して戦況を更新する。"""
        self.turn += 1

        current_player_names   = set(game_state.get("name_candidates_player", []))
        current_opponent_names = set(game_state.get("name_candidates_opponent", []))

        # ── ポケモン名の蓄積・on_field 更新（自分側） ──────────────────────
        for name in current_player_names:
            if any(s.name == name for s in self._opponent):
                continue  # 相手側に登録済みは混入させない
            slot = self._get_or_create(self._player, name)
            if slot:
                slot.confidence += 1
                slot.last_seen_turn = self.turn
                if not slot.fainted:
                    slot.on_field = True  # 現フレームで見えた → 場にいる

        # 長期間不検出のポケモンを場から降ろす（OCRノイズで一時的に消える場合は維持）
        for slot in self._player:
            if slot.on_field and not slot.fainted:
                if self.turn - slot.last_seen_turn > self._ON_FIELD_MISS_THRESHOLD:
                    slot.on_field = False
                    log.info(f"[戦況] {slot.name} が{self._ON_FIELD_MISS_THRESHOLD}ターン不検出 → 場から降ろす")

        # ── ポケモン名の蓄積・on_field 更新（相手側） ──────────────────────
        for name in current_opponent_names:
            if any(s.name == name for s in self._player):
                continue
            slot = self._get_or_create(self._opponent, name)
            if slot:
                slot.confidence += 1
                slot.last_seen_turn = self.turn
                if not slot.fainted:
                    slot.on_field = True

        for slot in self._opponent:
            if slot.on_field and not slot.fainted:
                if self.turn - slot.last_seen_turn > self._ON_FIELD_MISS_THRESHOLD:
                    slot.on_field = False
                    log.info(f"[戦況] {slot.name} が{self._ON_FIELD_MISS_THRESHOLD}ターン不検出 → 場から降ろす")

        # ダブルバトル制約: 場のポケモンは最大 2 匹
        self._cap_on_field(self._player)
        self._cap_on_field(self._opponent)

        # ── HP 値を場のポケモンに紐付け ──────────────────────────────────
        # y座標で側分類した HP（player=自分側、opponent=相手側）を優先使用
        hp_player   = game_state.get("hp_values_player", [])
        hp_opponent = game_state.get("hp_values_opponent", [])
        if not hp_player and not hp_opponent:
            # フォールバック: 全HP値を均等分配（前後2個ずつ）
            all_hp = game_state.get("hp_values", [])
            hp_player   = all_hp[:2]
            hp_opponent = all_hp[2:4]

        self._assign_hp_to_on_field(self._player,   hp_player)
        self._assign_hp_to_on_field(self._opponent, hp_opponent)

        # ── 状態異常の更新（YOLO 由来） ─────────────────────────────────────
        status_raw = game_state.get("status", "")
        if status_raw and status_raw != "なし":
            parts = status_raw.split(" / 相手: ")
            p_status = parts[0] if parts[0] != "なし" else None
            o_status = parts[1] if len(parts) > 1 else None
            on_field_p = [s for s in self._player   if s.on_field]
            on_field_o = [s for s in self._opponent if s.on_field]
            if p_status and on_field_p:
                on_field_p[0].status = p_status
            if o_status and on_field_o:
                on_field_o[0].status = o_status

        # ── 気絶検知（faintイベント時: HP=0 に加えて明示的なマーク） ────────
        if event_type == "faint":
            for side in (self._player, self._opponent):
                for slot in side:
                    if slot.hp and slot.hp.startswith("0/") and not slot.fainted:
                        slot.fainted = True
                        slot.on_field = False
                        log.info(f"[戦況] {slot.name} が気絶（faintイベント）")

        # ── イベントログ追記 ─────────────────────────────────────────────────
        ocr_snip = game_state.get("ocr_text", "")[:25]
        self._event_log.append(f"T{self.turn}:{event_type}[{ocr_snip}]")
        if len(self._event_log) > self.MAX_EVENTS:
            self._event_log.pop(0)

    def update_move(self, pokemon_name: str, move_name: str) -> None:
        """ポケモンが技を使ったことを記録する（per-pokemon 技リスト更新）。"""
        for side in (self._player, self._opponent):
            for slot in side:
                if slot.name == pokemon_name:
                    if move_name not in slot.moves_used:
                        slot.moves_used.append(move_name)
                        if len(slot.moves_used) > 4:
                            slot.moves_used.pop(0)
                    return

    def set_not_on_field(self, pokemon_name: str) -> bool:
        """指定ポケモンを場から降ろす（交代・とんぼがえり検出時に呼ぶ）。
        見つかった場合は True を返す。
        """
        # fuzzy マッチ（OCR誤読でポケモン名が少し違う場合も対応）
        for side in (self._player, self._opponent):
            for slot in side:
                # 完全一致 or 片方がもう一方に含まれる（OCR部分読み対応）
                if slot.name == pokemon_name or slot.name in pokemon_name or pokemon_name in slot.name:
                    if slot.on_field:
                        slot.on_field = False
                        return True
        return False

    # ── コンテキスト生成 ─────────────────────────────────────────────────────

    def _format_pokemon(self, p: FieldPokemon) -> str:
        """場にいるポケモンの詳細フォーマット（HP・状態異常・使用技を含む）。"""
        s = p.name
        if p.status:
            s += f"({p.status})"
        if p.hp:
            s += f" HP:{p.hp}"
            m = self._HP_RE.match(p.hp)
            if m and int(m.group(2)) > 0:
                pct = int(m.group(1)) / int(m.group(2)) * 100
                if pct <= 25:
                    s += "★ピンチ"
        if p.moves_used:
            s += f" 技=[{', '.join(p.moves_used[-4:])}]"
        return s

    def _format_bench(self, p: FieldPokemon) -> str:
        """控えポケモンの簡略フォーマット。"""
        if p.fainted:
            return f"{p.name}(ひんし)"
        s = p.name
        if p.status:
            s += f"({p.status})"
        return s

    def to_context(self) -> dict:
        """Bedrock に渡す戦況サマリーを返す。"""
        on_field_p = [p for p in self._player   if p.on_field and not p.fainted]
        on_field_o = [p for p in self._opponent if p.on_field and not p.fainted]
        bench_p    = [p for p in self._player   if not p.on_field]
        bench_o    = [p for p in self._opponent if not p.on_field]

        player_field_str  = " / ".join(self._format_pokemon(p) for p in on_field_p)  or "情報収集中"
        opponent_field_str = " / ".join(self._format_pokemon(p) for p in on_field_o) or "情報収集中"
        player_bench_str  = " / ".join(self._format_bench(p) for p in bench_p)       or "なし"
        opponent_bench_str = " / ".join(self._format_bench(p) for p in bench_o)      or "なし"

        # RAG 用: 蓄積済み全ポケモン名リスト（信頼度順）
        player_names   = [p.name for p in sorted(self._player,   key=lambda p: -p.confidence)]
        opponent_names = [p.name for p in sorted(self._opponent, key=lambda p: -p.confidence)]

        return {
            "turn":             self.turn,
            "player_field":     player_field_str,
            "player_bench":     player_bench_str,
            "opponent_field":   opponent_field_str,
            "opponent_bench":   opponent_bench_str,
            "event_log":        " | ".join(self._event_log[-5:]),
            # server.py 互換フィールド（player_pokemon / opponent_pokemon）
            "player_pokemon":   f"場: {player_field_str} / 控え: {player_bench_str}",
            "opponent_pokemon": f"場: {opponent_field_str} / 控え: {opponent_bench_str}",
            "player_names":     player_names,    # RAG 用
            "opponent_names":   opponent_names,  # RAG 用
        }


# ─── OCR デバッグ画像保存 ────────────────────────────────────────────────────

_HP_RE_DEBUG = re.compile(r'\d{1,3}/\d{1,3}')

def _save_ocr_debug_image(frame: np.ndarray, ocr_results: list[dict], turn: int) -> None:
    """
    OCR 結果を frame 上に描画して debug/ に保存する。
    色分け:
      緑  = 自分側ポケモン名候補（y >= _PLAYER_Y_THRESHOLD）
      赤  = 相手側ポケモン名候補（y < _PLAYER_Y_THRESHOLD）
      青  = HP 値
      灰  = フィルター済み / 低信頼度
    """
    img = frame.copy()
    for r in ocr_results:
        bbox = r.get("bbox")
        if not bbox:
            continue
        text = r["text"].strip()
        conf = r["confidence"]
        pts = np.array(bbox, dtype=np.int32)
        center_y = (bbox[0][1] + bbox[2][1]) / 2

        # 色決定
        if conf < 0.4:
            color = (120, 120, 120)   # 灰: 低信頼度
        elif center_y > _COMMAND_Y_MIN:
            color = (0, 165, 255)     # 橙: コマンドメニュー除外エリア
        elif _HP_RE_DEBUG.search(text):
            color = (255, 100, 0)     # 青: HP 値
        elif (text.startswith("Lv") or re.match(r'^[\d\s/]+$', text)
              or text in _UI_WORDS or text in _BATTLE_RESULT_WORDS
              or text in _MOVE_ABILITY_WORDS or text in _UI_OVERLAY_WORDS
              or any(kw in text for kw in _BATTLE_RESULT_WORDS)
              or (re.match(r'^[A-Za-z0-9\s]+$', text) and len(text) < 4)
              or text.endswith("の") or text.endswith("」") or text.endswith("!")):
            color = (120, 120, 120)   # 灰: フィルター済み
        elif center_y < _PLAYER_Y_THRESHOLD:
            color = (0, 60, 220)      # 赤: 相手側
        else:
            color = (0, 180, 0)       # 緑: 自分側

        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        label = f"{text} ({conf:.0%})"
        cv2.putText(img, label, (pts[0][0], pts[0][1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 分割ライン（相手/自分）
    cv2.line(img, (0, _PLAYER_Y_THRESHOLD), (img.shape[1], _PLAYER_Y_THRESHOLD),
             (0, 255, 255), 1)
    cv2.putText(img, f"y={_PLAYER_Y_THRESHOLD} opponent/player split",
                (10, _PLAYER_Y_THRESHOLD - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # コマンドメニュー除外ライン
    cv2.line(img, (0, _COMMAND_Y_MIN), (img.shape[1], _COMMAND_Y_MIN),
             (0, 165, 255), 1)
    cv2.putText(img, f"y={_COMMAND_Y_MIN} command menu (excluded)",
                (10, _COMMAND_Y_MIN - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / f"ocr_turn_{turn:03d}.png"), img)
    log.info(f"OCRデバッグ画像を保存: debug/ocr_turn_{turn:03d}.png")


# ─── 実況文クリーンアップ ────────────────────────────────────────────────────

def _clean_commentary(text: str) -> str:
    """
    Phi-3 mini が出力するゴミ（プロンプトの漏れ・追跡質問など）を除去する。
    - "---" / "【" 以降を切り捨て
    - "指示" / "質問" / "注:" を含む行以降を切り捨て
    - 各行頭の "- " "・ " を除去
    - 鉤括弧「」を除去
    - 最初の 2 文だけ残す（。！？で区切る）
    """
    # "---" 以降を除去
    text = text.split("---")[0]

    # "【" 以降を除去（「【画面分析】...」が漏れてくる場合）
    text = text.split("【")[0]

    # "指示" "質問" "注:" を含む行以降を除去
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if any(kw in line for kw in ["指示", "質問", "注:"]):
            break
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    # 各行頭の "- " "・ " を除去してから結合
    text = " ".join(
        re.sub(r"^[-・]\s*", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    )

    # 鉤括弧を除去
    text = text.replace("「", "").replace("」", "")

    # 最初の 2 文だけ残す（。！？で区切る）
    sentences = re.split(r"(?<=[。！？])", text)
    text = "".join(sentences[:2]).strip()

    return text


# ─── Bedrock Vision 呼び出し（EC2 API 経由・オプション） ─────────────────────

def _call_bedrock_vision(
    ec2_url: str,
    frame: np.ndarray,
    game_state: dict,
    event_type: str,
    commentary_history: list[str],
    battle_context: dict | None = None,
    classifier=None,
    move_log: list[str] | None = None,
) -> str | None:
    """
    EC2 API に画像と状況を送り、Bedrock Vision 分析結果を受け取る。
    失敗してもパイプラインを止めない（None を返す）。
    """
    try:
        # 縮小してから PNG エンコード（nginx の 5MB 制限対策）
        small = cv2.resize(frame, (800, 450), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".png", small)
        image_b64 = base64.b64encode(buf.tobytes()).decode()

        # server.py の /api/vision は context.event_type でバリデーションする
        status_parts = (game_state.get("status", "") or "").split(" / 相手: ")
        status_player   = status_parts[0] if status_parts[0] != "なし" else "なし"
        status_opponent = status_parts[1] if len(status_parts) > 1 else "なし"

        balls = game_state.get("balls_remaining", [])
        hp_values = game_state.get("hp_values", [])
        names_player   = game_state.get("name_candidates_player", [])
        names_opponent = game_state.get("name_candidates_opponent", [])

        # RAG: 蓄積済みポケモン名（battle_context）を優先、なければ現フレームの候補を使用
        # battle_context には複数ターン分の蓄積があり現フレームより信頼度が高い
        rag_names: list[str] = []
        if battle_context:
            rag_names += battle_context.get("player_names", [])
            rag_names += battle_context.get("opponent_names", [])
        if not rag_names:
            rag_names = names_player + names_opponent  # フォールバック

        rag_info: list[str] = []
        if classifier:
            seen: set[str] = set()
            for name in rag_names:
                if name in seen:
                    continue
                seen.add(name)
                info = classifier.get_pokemon_info(name)
                if info:
                    abilities_str = " / ".join(info["abilities"]) if info["abilities"] else "不明"
                    # 代表技は渡さない（Bedrockが「使った技」として創作するのを防ぐため）
                    rag_info.append(
                        f"{info['name_ja']}({info['name_en']}): タイプ={info['type']} / 特性={abilities_str}"
                    )

        payload = {
            "image_base64": image_b64,
            "context": {
                "status_player":            status_player,
                "status_opponent":          status_opponent,
                "balls_remaining_player":   balls[0] if len(balls) > 0 else "?",
                "balls_remaining_opponent": balls[1] if len(balls) > 1 else "?",
                "event_type":               event_type,
                "ocr_text":                 game_state.get("ocr_text", ""),
                "hp_values":                " / ".join(hp_values) if hp_values else "不明",
                "name_candidates_player":   " / ".join(names_player)   if names_player   else "不明",
                "name_candidates_opponent": " / ".join(names_opponent) if names_opponent else "不明",
                "rag_pokemon_info":         rag_info,
                "detected_moves":           " / ".join(move_log) if move_log else "なし",
            },
            "history": commentary_history[-3:],
            "battle_state": battle_context or {},
        }
        ctx = payload["context"]
        bs  = payload.get("battle_state", {})
        log.info(
            "[Bedrock送信] event=%s | 自分=%s | 相手=%s | HP=%s | 技ログ=%s | RAG=%s",
            ctx["event_type"],
            bs.get("player_pokemon", "不明"),
            bs.get("opponent_pokemon", "不明"),
            ctx["hp_values"],
            ctx["detected_moves"],
            " / ".join(ctx["rag_pokemon_info"]) if ctx["rag_pokemon_info"] else "なし",
        )
        resp = requests.post(f"{ec2_url}/api/vision", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            log.debug(f"Bedrock tokens: in={data.get('usage',{}).get('input_tokens')} out={data.get('usage',{}).get('output_tokens')} latency={data.get('latency_ms')}ms")
        # commentary（実況文）と analysis（状況説明）を両方返す
        return data.get("commentary"), data.get("analysis")
    except Exception as e:
        log.warning(f"Bedrock Vision 呼び出しスキップ: {e}")
        return None, None


# ─── メインパイプライン ────────────────────────────────────────────────────────

class Pipeline:
    def __init__(
        self,
        camera_index: int,
        model_path: str | None,
        interval: float,
        speaker: int,
        gpu: bool,
        conf: float,
        ec2_url: str | None,
        audio_device: int | None,
    ):
        log.info("=== パイプライン初期化 ===")

        log.info("EasyOCR 初期化中...")
        self._reader = init_reader(gpu=gpu)

        log.info("YoloDetector 初期化中...")
        self._yolo = YoloDetector(model_path=model_path, conf=conf)

        log.info("Phi-3 クライアント初期化...")
        self._phi3 = Phi3Client()

        log.info("VOICEVOX クライアント初期化...")
        self._voicevox = VoicevoxClient(speaker=speaker)

        log.info("AudioPlayer 初期化...")
        self._player = AudioPlayer(device=audio_device)

        self._camera_index = camera_index
        self._interval = interval
        self._ec2_url = ec2_url
        self._diff_detector = DiffDetector()             # 静止フレームのスキップ用
        self._phase_classifier = BattlePhaseClassifier() # フェーズ分類 + イベント検知
        self._last_ocr_time: float = 0.0                  # 定期OCR用タイマー
        self._PERIODIC_OCR_INTERVAL_BATTLE = 1.5         # バトル中: 終了画面を取りこぼさないよう短め
        self._PERIODIC_OCR_INTERVAL_IDLE   = 3.0         # バトル外: 重くならないよう長め
        self._battle_tracker = BattleStateTracker()       # 戦況累積
        self._battle_active = False  # battle_start〜battle_end の間のみ True
        self._prev_yolo: BattleState | None = None
        self._commentary_history: list[str] = []
        self._move_log: list[str] = []   # OCRから検出した「使われた技」のリングバッファ
        self._MAX_MOVE_LOG = 8

        # PokeDB 分類器（DB がなければ None でフォールバック動作）
        log.info("PokeClassifier 初期化中...")
        try:
            self._classifier: PokeClassifier | None = PokeClassifier()
        except FileNotFoundError as e:
            log.warning("PokeDB が見つからないため手動フィルターで動作: %s", e)
            self._classifier = None

        log.info("=== 初期化完了 ===")

    def run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            log.error(f"カメラ {self._camera_index} を開けませんでした（OBS仮想カメラが起動中か確認）")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        for _ in range(10):
            cap.read()

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"カメラ {self._camera_index} オープン: {w}x{h}")
        log.info("パイプライン開始（Ctrl+C で終了）")

        turn = 0
        try:
            while True:
                loop_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    log.warning("フレーム取得失敗。再試行します...")
                    time.sleep(0.5)
                    continue

                # ── YOLO 検出（毎フレーム）──────────────────────────────────
                yolo_state = self._yolo.detect(frame)
                if yolo_state.detections:
                    log.debug(f"[YOLO] {yolo_state.summary()}")

                # ── 差分検出（静止フレームの OCR スキップ用）────────────────
                diff_changed, diff_score = self._diff_detector.detect(frame)
                now = time.perf_counter()
                _interval = self._PERIODIC_OCR_INTERVAL_BATTLE if self._battle_active else self._PERIODIC_OCR_INTERVAL_IDLE
                periodic_ocr = (now - self._last_ocr_time) >= _interval

                if diff_changed or periodic_ocr:
                    # ── OCR（差分あり or 定期実行）──────────────────────────
                    reason = "diff" if diff_changed else "periodic"
                    t_ocr = time.perf_counter()
                    ocr_results = run_ocr(self._reader, frame)
                    self._last_ocr_time = time.perf_counter()
                    ocr_texts = [r["text"] for r in ocr_results if r["confidence"] >= 0.4]
                    log.info(f"OCR({reason}): {len(ocr_results)} 件 | {' / '.join(ocr_texts[:10])}")

                    # 定期OCR時はデバッグ画像を保存（終了画面など未検知フェーズの診断用）
                    if periodic_ocr and ocr_results:
                        _save_ocr_debug_image(frame, ocr_results, turn * 1000 + int(now) % 1000)

                    # ── 技使用・交代メッセージの検出（バトル中は常時監視）──────
                    if self._battle_active:
                        self._update_move_log(ocr_results)
                        self._update_switch_out(ocr_results)

                    # ── フェーズ分類 + イベント検知 ─────────────────────────
                    event_type = self._phase_classifier.detect(ocr_results)

                    if event_type:
                        turn += 1
                        log.info(f"[ターン {turn}] イベント検知 (diff={diff_score:.1f}, type={event_type}, phase={self._phase_classifier._prev_phase})")
                        self._phase_classifier.set_processing(True)
                        try:
                            self._process_event(frame, yolo_state, ocr_results, event_type, turn)
                        finally:
                            self._phase_classifier.set_processing(False)

                self._prev_yolo = yolo_state

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, self._interval - elapsed))

        except KeyboardInterrupt:
            log.info(f"終了します（総ターン数: {turn}）")
        finally:
            cap.release()

    def _process_event(
        self,
        frame: np.ndarray,
        yolo_state: BattleState,
        ocr_results: list[dict],
        event_type: str,
        turn: int,
    ) -> None:
        """イベント発生時の一連の処理（game_state 構築 → Phi-3 / Bedrock → VOICEVOX → 再生）。"""
        log.info(f"OCR: {len(ocr_results)} 件")

        # ── バトル外画面はスキップ（battle_end は終了画面なので除外しない）────
        if event_type != "battle_end" and not _is_battle_screen(ocr_results):
            log.info("バトル外の画面を検知 → スキップ")
            return

        # OCR 件数が少なすぎる場合はスキップ（battle_end は例外）
        if event_type != "battle_end" and len(ocr_results) < 2:
            log.info(f"OCR 件数が少なすぎる（{len(ocr_results)} 件）→ スキップ")
            return

        # ── game_state 構築 ───────────────────────────────────────────────────
        game_state = _build_game_state(ocr_results, yolo_state, event_type, self._prev_yolo, self._classifier)
        log.info(f"[状態] {yolo_state.summary()} | OCR: {game_state['ocr_text']}")
        log.info(f"[構造化] HP={game_state['hp_values']} | 自分={game_state['name_candidates_player']} | 相手={game_state['name_candidates_opponent']}")
        _save_ocr_debug_image(frame, ocr_results, turn)

        # ── 戦況トラッカー更新 ────────────────────────────────────────────────
        if event_type == "battle_start":
            # バトル開始: トラッカーをリセットしてアクティブ化
            self._battle_tracker = BattleStateTracker()
            self._battle_active = True
            self._commentary_history = []
            self._move_log = []
            log.info("[戦況] バトル開始 → トラッカーリセット")

        if self._battle_active:
            self._battle_tracker.update(game_state, event_type)

        battle_context = self._battle_tracker.to_context()
        log.info(
            "[戦況] T%s 場(自)=%s | 場(相)=%s",
            battle_context["turn"],
            battle_context["player_field"],
            battle_context["opponent_field"],
        )
        log.info(
            "[戦況] 控え(自)=%s | 控え(相)=%s",
            battle_context["player_bench"],
            battle_context["opponent_bench"],
        )

        # ── Bedrock Vision（バトル中のみ・対象イベントのみ・EC2 URL が設定されている場合）──
        # _battle_active = False の間（選出画面等）は Bedrock を呼ばない
        bedrock_commentary: str | None = None
        bedrock_analysis: str | None = None
        if self._ec2_url and event_type in BEDROCK_EVENTS and self._battle_active:
            log.info("Bedrock Vision 呼び出し中...")
            if self._move_log:
                log.info(f"[技ログ] {' / '.join(self._move_log[-5:])}")
            t0 = time.perf_counter()
            bedrock_commentary, bedrock_analysis = _call_bedrock_vision(
                self._ec2_url, frame, game_state, event_type,
                self._commentary_history, battle_context, self._classifier,
                self._move_log[-5:],
            )
            if bedrock_commentary:
                log.info(f"Bedrock 完了 ({time.perf_counter()-t0:.2f}s): 「{bedrock_commentary}」")

        # ── 実況文の決定 ──────────────────────────────────────────────────────
        # Bedrock が実況文を返してくれた場合はそれを優先（Phi-3 スキップ）
        if bedrock_commentary:
            commentary = _clean_commentary(bedrock_commentary)
            log.info(f"Bedrock 実況文を使用: 「{commentary}」")
        else:
            # フォールバック: Phi-3 で生成
            phi3_context = bedrock_analysis or game_state["ocr_text"]
            log.info("Phi-3 実況文生成中（フォールバック）...")
            t0 = time.perf_counter()
            try:
                commentary = self._phi3.generate_commentary(game_state, bedrock_analysis=phi3_context)
                commentary = _clean_commentary(commentary)
                log.info(f"Phi-3 実況文生成完了 ({time.perf_counter()-t0:.2f}s): 「{commentary}」")
            except requests.exceptions.ConnectionError:
                log.error("Ollama が起動していません。`ollama serve` を実行してください。")
                return
            except Exception as e:
                log.error(f"Phi-3 エラー: {e}")
                return

        if not commentary:
            log.warning("実況文が空のためスキップ")
            return

        self._commentary_history.append(commentary)
        if len(self._commentary_history) > 5:
            self._commentary_history.pop(0)

        # ── VOICEVOX 音声合成 ─────────────────────────────────────────────────
        log.info("VOICEVOX 音声合成中...")
        t0 = time.perf_counter()
        try:
            wav_bytes = self._voicevox.generate_wav(commentary)
            log.info(f"音声合成完了 ({time.perf_counter()-t0:.2f}s): {len(wav_bytes)} bytes")
        except requests.exceptions.ConnectionError:
            log.error("VOICEVOX が起動していません。VOICEVOX を起動してください。")
            return
        except Exception as e:
            log.error(f"VOICEVOX エラー: {e}")
            return

        # ── 音声再生 ──────────────────────────────────────────────────────────
        log.info("音声再生中...")
        t0 = time.perf_counter()
        try:
            self._player.play(wav_bytes)
            log.info(f"再生完了 ({time.perf_counter()-t0:.2f}s)")
        except Exception as e:
            log.error(f"音声再生エラー: {e}")

        # バトル終了後にアクティブフラグをリセット（Bedrock呼び出し後）
        if event_type == "battle_end":
            self._battle_active = False
            log.info("[戦況] バトル終了 → トラッカー非アクティブ化")

        # デバッグ用スクリーンショット保存
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"pipeline_turn_{turn:03d}.png"), frame)

    def _update_switch_out(self, ocr_results: list[dict]) -> None:
        """「〜は戻っていく」テキストを検出してポケモンを場から降ろす。

        とんぼがえり・Uターン等の交代技ではフェーズが switch_select を経由しないため、
        テキストパターンで交代を検出して on_field=False を即時反映する。
        例: "ゴリランダーは / ともの元へ / 戻っていく"
        """
        texts = [r["text"].strip() for r in ocr_results if r["confidence"] >= 0.3]
        # 「戻っていく」「戻つていく」（OCR誤読）が含まれているか確認
        has_return_text = any("戻" in t or "もど" in t for t in texts)
        if not has_return_text:
            return

        # 「〜は」で終わるテキストからポケモン名を取得
        # 例: "ゴリランダーは" → "ゴリランダー"
        for text in texts:
            if text.endswith("は") and len(text) >= 3:
                pokemon_name_candidate = text[:-1]  # 末尾の「は」を除去
                # tracker で self._player / self._opponent 両方を検索してon_field=False
                found = self._battle_tracker.set_not_on_field(pokemon_name_candidate)
                if found:
                    log.info(f"[交代検知] {pokemon_name_candidate} が場から退いた（「戻っていく」テキスト検出）")

    def _update_move_log(self, ocr_results: list[dict]) -> None:
        """OCR 結果から「〜の → 技名」パターンを検出して _move_log に追記する。

        例: ["トルネロスの", "ハリケーン"] → "トルネロスのハリケーン" として蓄積。
        classifier で技名確認済みのものだけを登録（OCR誤読対策）。
        """
        # ポケモン名として無効な「〜の」プレフィックス（UIテキスト・バトルメッセージ等）
        _INVALID_POKEMON_PREFIXES = {"相手", "あいて", "とも", "自分", "じぶん", "ともの"}

        texts = [r["text"].strip() for r in ocr_results if r["confidence"] >= 0.4]
        for i, text in enumerate(texts):
            if not text.endswith("の"):
                continue
            if i + 1 >= len(texts):
                continue
            pokemon_name = text[:-1]  # 末尾の「の」を除去（事前チェック用）
            # 「相手の」「あいての」等はポケモン名ではない
            if pokemon_name in _INVALID_POKEMON_PREFIXES:
                continue
            next_text = texts[i + 1].rstrip("！!」")
            # 3文字未満は OCR の断片ノイズ（"こう"等が WRatio 部分マッチで誤検出される）
            if not next_text or len(next_text) < 3:
                continue
            if self._classifier:
                result = self._classifier.classify(next_text)
                # 技名でない（特性・アイテム等）は除外
                if result.category != "move":
                    continue
                # スコア90未満（confidentでない）は OCR 誤読の可能性が高いため除外
                # 例: "ぼうぎよ"（防御の誤読）→ "ぼうぎょしれい" にマッチするケース
                if not result.confident:
                    continue
                # OCR テキストが技名の末尾/中間にしか一致しない部分マッチを除外
                # 例: "こうげき"(4文字) → "ヘドロこうげき"(7文字): WRatio は部分一致で100点になるが
                #     これは "がむしゃらこうげき" の OCR 分割ノイズ
                # canonical が OCR テキストの 1.5 倍超なら部分マッチアーティファクトと判断
                if len(result.canonical_ja) > len(next_text) * 1.5:
                    continue
                move_name = result.canonical_ja or next_text
            else:
                # classifier なし: 技名フィルターは行わずそのまま登録
                move_name = next_text
            # pokemon_name はループ上部で既に取得済み（text[:-1]）
            entry = f"{pokemon_name}の{move_name}"
            if not self._move_log or self._move_log[-1] != entry:
                self._move_log.append(entry)
                if len(self._move_log) > self._MAX_MOVE_LOG:
                    self._move_log.pop(0)
                log.info(f"[技ログ] 検出: {entry}")
                # ポケモンごとの技リストも更新
                if self._battle_active:
                    self._battle_tracker.update_move(pokemon_name, move_name)


# ─── エントリポイント ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ポケモン対戦実況AI パイプライン（Sprint 5）")
    parser.add_argument("--camera",  type=int,   default=3,
                        help="OBS仮想カメラのデバイス番号（デフォルト: 3）")
    parser.add_argument("--model",   default=None,
                        help="YOLOv8 カスタムモデルのパス（例: runs/detect/train4/weights/best.pt）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="キャプチャ間隔（秒、デフォルト: 1.0）")
    parser.add_argument("--speaker", type=int,   default=1,
                        help="VOICEVOX 話者 ID（デフォルト: 1 = ずんだもん）")
    parser.add_argument("--cpu",     action="store_true",
                        help="EasyOCR を CPU モードで実行（GPU 無効）")
    parser.add_argument("--conf",    type=float, default=0.5,
                        help="YOLO 信頼度閾値（デフォルト: 0.5）")
    parser.add_argument("--ec2-url", default=None,
                        help="EC2 API の URL（例: http://<EC2-IP>:5000）。指定時に Bedrock Vision を使用。")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="音声出力デバイス番号（省略でシステムデフォルト）")

    args = parser.parse_args()

    pipeline = Pipeline(
        camera_index=args.camera,
        model_path=args.model,
        interval=args.interval,
        speaker=args.speaker,
        gpu=not args.cpu,
        conf=args.conf,
        ec2_url=args.ec2_url,
        audio_device=args.audio_device,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
