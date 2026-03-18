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
_NON_BATTLE_KEYWORDS = {
    "オフライン", "ユニオンサークル", "テラレイドバトル", "通信交換",
    "マジカル交換", "通信対戦", "バトルスタジアム", "ランクバトル選択",
    "レンタル", "てもち", "チーム", "マスターボール級", "RANK MAX",
    "せいせき", "はんえい", "通信中", "ごほうび", "リーグペイ",
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


def _ocr_results_to_text(ocr_results: list[dict]) -> str:
    """OCR 結果を読みやすいテキスト文字列にまとめる（最大 OCR_MAX_CHARS 文字）。"""
    lines = [r["text"] for r in ocr_results if r["confidence"] >= 0.4]
    text = " / ".join(lines) if lines else "（テキスト未検出）"
    return text[:OCR_MAX_CHARS]


def _extract_structured_info(ocr_results: list[dict]) -> dict:
    """
    OCR 結果から HP 値・技名候補を抽出して構造化する。
    Haiku に「画面から読めた事実」を明示的に渡すことで創作を抑制する。
    """
    hp_pattern = re.compile(r'\b(\d{1,3}/\d{1,3})\b')
    hp_values = []
    name_candidates_player: list[str] = []    # 下半分 = 自分のポケモンエリア
    name_candidates_opponent: list[str] = []  # 上半分 = 相手のポケモンエリア

    for r in ocr_results:
        if r["confidence"] < 0.4:
            continue
        text = r["text"].strip()
        # HP 値を抽出
        m = hp_pattern.search(text)
        if m:
            hp_values.append(m.group())
            continue
        # Lv. / 数字のみ / UI ワード / バトル結果テキスト / 英数字のみ はスキップ
        if (text.startswith("Lv") or re.match(r'^[\d\s/]+$', text)
                or text in _UI_WORDS or text in _BATTLE_RESULT_WORDS
                or any(kw in text for kw in _BATTLE_RESULT_WORDS)
                or re.match(r'^[A-Za-z0-9\s]+$', text)):
            continue
        # bbox の中心 y 座標で振り分け
        # y > _COMMAND_Y_MIN はコマンドメニュー（技名UIエリア）なので除外
        bbox = r.get("bbox", [])
        if bbox:
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            if center_y > _COMMAND_Y_MIN:
                continue  # コマンドメニュー内のテキストは除外
            elif center_y < _PLAYER_Y_THRESHOLD:
                name_candidates_opponent.append(text)
            else:
                name_candidates_player.append(text)
        else:
            name_candidates_player.append(text)

    return {
        "hp_values": hp_values,
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

    structured = _extract_structured_info(ocr_results)

    return {
        "pokemon_player":   "（OCR参照）",
        "hp_player":        "?",
        "pokemon_opponent": "（OCR参照）",
        "hp_opponent":      "?",
        "last_move":        "（OCR参照）",
        "status":           status_text,
        "balls_remaining":  [p_balls, o_balls] if (p_balls or o_balls) else [],
        "event_type":       event_type,
        "ocr_text":         _ocr_results_to_text(ocr_results),
        "hp_values":        structured["hp_values"],
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

    _COMMAND_KW    = {"たたかう", "どうする"}
    _SWITCH_KW     = {"こうたい", "ポケモンをえらんで"}
    _ANIM_KW       = {"バツグンだ", "いまひとつ", "こうかなし", "きゅうしょ", "急所", "ひんし"}
    _END_KW        = {"勝負に勝", "勝負に負", "降参が選ばれ", "降参"}
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
        if self._COMMAND_KW & texts:
            return "command_select"
        if any(kw in t for kw in self._ANIM_KW for t in texts):
            return "animation"
        return "unknown"

    def detect(self, ocr_results: list[dict]) -> str | None:
        """フェーズ遷移からイベントを返す。イベントなし or 処理中 は None。"""
        if self._is_processing:
            return None

        curr = self.classify(ocr_results)
        prev = self._prev_phase
        self._prev_phase = curr

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
class PokemonSlot:
    """1匹のポケモンの状態を保持するスロット。"""
    name: str
    confidence: int = 0
    hp_history: list[str] = field(default_factory=list)  # 直近3件の HP 値
    status: str | None = None   # まひ / やけど / どく など
    fainted: bool = False        # 気絶済みフラグ


class BattleStateTracker:
    """
    試合全体の戦況を蓄積するクラス。

    自分・相手それぞれ最大 4 スロットでポケモンを管理する。
    同名ポケモンが両サイドに存在しても混在しない。
    """

    MAX_SLOTS  = 4
    MAX_EVENTS = 8
    _HP_RE = re.compile(r'(\d{1,3})/(\d{1,3})')

    def __init__(self):
        self.turn = 0
        self._player:   list[PokemonSlot] = []  # 自分の4匹
        self._opponent: list[PokemonSlot] = []  # 相手の4匹
        self._event_log: list[str] = []

    # ── 内部ヘルパー ─────────────────────────────────────────────────────────

    def _get_or_create(self, slots: list[PokemonSlot], name: str) -> PokemonSlot | None:
        """名前でスロットを検索。なければ新規作成（MAX_SLOTS を超えたら None）。"""
        for s in slots:
            if s.name == name:
                return s
        if len(slots) < self.MAX_SLOTS:
            slot = PokemonSlot(name=name)
            slots.append(slot)
            return slot
        return None  # 4匹超過は無視

    def _top_slot(self, slots: list[PokemonSlot]) -> PokemonSlot | None:
        """信頼度が最も高いスロットを返す。"""
        return max(slots, key=lambda s: s.confidence) if slots else None

    def _update_hp(self, slot: PokemonSlot, hp: str) -> None:
        if not slot.hp_history or slot.hp_history[-1] != hp:
            slot.hp_history.append(hp)
            if len(slot.hp_history) > 3:
                slot.hp_history.pop(0)

    # ── メイン更新 ───────────────────────────────────────────────────────────

    def update(self, game_state: dict, event_type: str) -> None:
        """1 イベントごとに呼び出して戦況を更新する。"""
        self.turn += 1

        # ポケモン名の蓄積（信頼度加算）
        for name in game_state.get("name_candidates_player", []):
            slot = self._get_or_create(self._player, name)
            if slot:
                slot.confidence += 1

        for name in game_state.get("name_candidates_opponent", []):
            slot = self._get_or_create(self._opponent, name)
            if slot:
                slot.confidence += 1

        # HP 値の蓄積（自分側・相手側それぞれ先頭1件を代表ポケモンへ）
        hp_values = game_state.get("hp_values", [])
        p_top = self._top_slot(self._player)
        o_top = self._top_slot(self._opponent)
        if hp_values and p_top:
            self._update_hp(p_top, hp_values[0])
        if len(hp_values) > 1 and o_top:
            self._update_hp(o_top, hp_values[1])

        # 状態異常の更新（YOLO 由来）
        status_raw = game_state.get("status", "")
        if status_raw and status_raw != "なし":
            parts = status_raw.split(" / 相手: ")
            p_status = parts[0] if parts[0] != "なし" else None
            o_status = parts[1] if len(parts) > 1 else None
            if p_status and p_top:
                p_top.status = p_status
            if o_status and o_top:
                o_top.status = o_status

        # 気絶検知: HP 履歴の末尾が "0/XXX" のスロットを気絶マーク
        if event_type == "faint":
            for side in (self._player, self._opponent):
                for slot in side:
                    if slot.hp_history and slot.hp_history[-1].startswith("0/"):
                        slot.fainted = True

        # イベントログ追記
        ocr_snip = game_state.get("ocr_text", "")[:25]
        self._event_log.append(f"T{self.turn}:{event_type}[{ocr_snip}]")
        if len(self._event_log) > self.MAX_EVENTS:
            self._event_log.pop(0)

    # ── コンテキスト生成 ─────────────────────────────────────────────────────

    def _format_slot(self, slot: PokemonSlot) -> str:
        if slot.fainted:
            return f"{slot.name}(気絶・場に出ていない)"
        s = slot.name
        if slot.status:
            s += f"({slot.status})"
        if slot.hp_history:
            hp = slot.hp_history[-1]
            s += f" HP:{hp}"
            m = self._HP_RE.match(hp)
            if m:
                pct = int(m.group(1)) / int(m.group(2)) * 100
                if pct <= 25:
                    s += "★ピンチ"
        return s

    def _format_side(self, slots: list[PokemonSlot]) -> str:
        if not slots:
            return "情報収集中"
        # 信頼度順にソートして表示
        sorted_slots = sorted(slots, key=lambda s: -s.confidence)
        return " / ".join(self._format_slot(s) for s in sorted_slots)

    def to_context(self) -> dict:
        """Bedrock に渡す戦況サマリーを返す。"""
        return {
            "turn":             self.turn,
            "player_pokemon":   self._format_side(self._player),
            "opponent_pokemon": self._format_side(self._opponent),
            "event_log":        " | ".join(self._event_log[-5:]),
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
              or any(kw in text for kw in _BATTLE_RESULT_WORDS)
              or re.match(r'^[A-Za-z0-9\s]+$', text)):
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
            },
            "history": commentary_history[-3:],
            "battle_state": battle_context or {},
        }
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
        self._battle_tracker = BattleStateTracker()       # 戦況累積
        self._battle_active = False  # battle_start〜battle_end の間のみ True
        self._prev_yolo: BattleState | None = None
        self._commentary_history: list[str] = []

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

                if diff_changed:
                    # ── OCR（差分あり時のみ）────────────────────────────────
                    t_ocr = time.perf_counter()
                    ocr_results = run_ocr(self._reader, frame)
                    log.debug(f"OCR: {len(ocr_results)} 件 ({time.perf_counter()-t_ocr:.2f}s)")

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
        game_state = _build_game_state(ocr_results, yolo_state, event_type, self._prev_yolo)
        log.info(f"[状態] {yolo_state.summary()} | OCR: {game_state['ocr_text']}")
        log.info(f"[構造化] HP={game_state['hp_values']} | 自分={game_state['name_candidates_player']} | 相手={game_state['name_candidates_opponent']}")
        _save_ocr_debug_image(frame, ocr_results, turn)

        # ── 戦況トラッカー更新 ────────────────────────────────────────────────
        if event_type == "battle_start":
            # バトル開始: トラッカーをリセットしてアクティブ化
            self._battle_tracker = BattleStateTracker()
            self._battle_active = True
            self._commentary_history = []
            log.info("[戦況] バトル開始 → トラッカーリセット")

        if self._battle_active:
            self._battle_tracker.update(game_state, event_type)

        battle_context = self._battle_tracker.to_context()
        log.info(f"[戦況] T{battle_context['turn']} 自={battle_context['player_pokemon']} 相={battle_context['opponent_pokemon']}")

        # ── Bedrock Vision（バトル中のみ・対象イベントのみ・EC2 URL が設定されている場合）──
        # _battle_active = False の間（選出画面等）は Bedrock を呼ばない
        bedrock_commentary: str | None = None
        bedrock_analysis: str | None = None
        if self._ec2_url and event_type in BEDROCK_EVENTS and self._battle_active:
            log.info("Bedrock Vision 呼び出し中...")
            t0 = time.perf_counter()
            bedrock_commentary, bedrock_analysis = _call_bedrock_vision(
                self._ec2_url, frame, game_state, event_type,
                self._commentary_history, battle_context
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
