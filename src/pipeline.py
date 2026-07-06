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
import threading
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

from src.capture.hpbar_analyzer import HpBarAnalyzer
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

# 終了画面確認用 OCR キーワード（YOLO検出との AND 条件で誤発火を防ぐ）
# 勝ち: 「勝った」/ 負け: 「負けた」/ 降参: 「選ばれました」
_END_SCREEN_OCR_KEYWORDS = ("勝った", "負けた", "選ばれました")

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
    "通信待機中",  # Champions: 対戦相手との接続状態テキスト
}
# 部分一致で弾くシステムテキストのキーワード（ポケモン名に含まれないもの）
_UI_OVERLAY_SUBSTRINGS = {"通信待機", "待機中", "通信中"}


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


def _proximity_pair(
    names: list[tuple[str, float, float]],
    hp_values: list[tuple[str, float, float]],
) -> list[tuple[str, str]]:
    """名前とHP値を近傍マッチング（Euclidean距離）でペアリングする。
    zip による y 座標順マッチングの代替。
    ダブルバトルで横並びの2匹はy座標が近くx座標が大きく異なるため、
    Euclidean距離でペアリングすると正しく対応付けられる。
    """
    if not names or not hp_values:
        return []
    pairs: list[tuple[str, str]] = []
    used: set[int] = set()
    for name, nx, ny in names:
        best_idx = -1
        best_dist = float("inf")
        for i, (hp, hx, hy) in enumerate(hp_values):
            if i in used:
                continue
            dist = ((nx - hx) ** 2 + (ny - hy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            pairs.append((name, hp_values[best_idx][0]))
            used.add(best_idx)
    return pairs


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
    # チャンピオンズ: 相手HP%パターン（XX%形式）
    hp_pct_pattern = re.compile(r'(\d{1,3})%')
    # "/" を "1" または "7" と誤読するケース補正: "1551155"→"155/155", "1557155"→"155/155"
    _hp_slash_re = re.compile(r'^(\d{1,3})[17](\d{2,3})$')

    def _normalize_hp_text(t: str) -> str:
        """HP文字列のOCR誤読を補正する。O→0、{→8、スラッシュ誤読(1/7)→/"""
        t = t.replace('O', '0').replace('o', '0').replace('{', '8')
        m = _hp_slash_re.match(t)
        if m:
            cur, mx = int(m.group(1)), int(m.group(2))
            if cur <= mx:  # 分子 > 分母は誤補正として弾く
                t = f"{cur}/{mx}"
        return t

    hp_values: list[str] = []
    hp_player_with_xy: list[tuple[str, float, float]] = []   # (hp_str, center_x, center_y) 自分側
    hp_opponent_with_xy: list[tuple[str, float, float]] = []  # (hp_str, center_x, center_y) 相手側
    name_player_with_xy: list[tuple[str, float, float]] = []   # (name, center_x, center_y) 自分側
    name_opponent_with_xy: list[tuple[str, float, float]] = [] # (name, center_x, center_y) 相手側

    # 「相手を見る」状態確認パネル検出
    # "戦闘中" はこのパネル専用テキスト。ここではポケモンが画面上に並ぶため
    # y 座標ベースの自分/相手分類が信頼できない → 名前候補の収集をスキップする
    all_texts = {r["text"] for r in ocr_results if r["confidence"] >= 0.4}
    is_status_panel = any(
        "戦闘中" in t or "たたかえない" in t or "たたかえる" in t
        for t in all_texts
    )
    if is_status_panel:
        log.debug("「相手を見る」パネル検出 → ポケモン名候補収集をスキップ")

    for r in ocr_results:
        if r["confidence"] < 0.4:
            continue
        text = r["text"].strip()
        text_norm = _normalize_hp_text(text)

        # HP 値を抽出（分母 < 50 は PP 値のため除外し continue で名前候補にも入れない）
        m = hp_pattern.search(text_norm)
        if m:
            cur_val = int(m.group(1))
            denom = int(m.group(2))
            if denom >= _HP_MIN_DENOM and cur_val <= denom:  # 現在値 > 最大値は誤読として弾く
                hp_str = f"{m.group(1)}/{m.group(2)}"
                hp_values.append(hp_str)
                # y座標で自分/相手側に分類（HP値を場のポケモンに紐付けるため）
                bbox_hp = r.get("bbox", [])
                if bbox_hp:
                    cx = (bbox_hp[0][0] + bbox_hp[2][0]) / 2
                    cy = (bbox_hp[0][1] + bbox_hp[2][1]) / 2
                    if cy < _PLAYER_Y_THRESHOLD:
                        hp_opponent_with_xy.append((hp_str, cx, cy))
                    else:
                        hp_player_with_xy.append((hp_str, cx, cy))
                else:
                    hp_player_with_xy.append((hp_str, 960.0, 999.0))  # bbox なし → 自分側に追加
            continue

        # チャンピオンズ: 相手HP%パターン（XX%形式）
        # 状態確認パネル中は複数ポケモンのHP%が混在するためスキップ
        if not is_status_panel:
            m_pct = hp_pct_pattern.search(text_norm)
            if m_pct:
                pct_val = int(m_pct.group(1))
                if pct_val <= 100:
                    hp_str = f"{pct_val}%"
                    bbox_hp = r.get("bbox", [])
                    if bbox_hp:
                        cx = (bbox_hp[0][0] + bbox_hp[2][0]) / 2
                        cy = (bbox_hp[0][1] + bbox_hp[2][1]) / 2
                        if cy < _PLAYER_Y_THRESHOLD:
                            hp_opponent_with_xy.append((hp_str, cx, cy))
                    continue

        # 状態確認パネル中は名前候補収集しない（HP値は上で収集済み）
        if is_status_panel:
            continue

        # 「ポケモン名」形式（ゆけ！アニメ中）の山括弧を除去してポケモン名として取り出す
        # 例: 「ガブリアス」→ ガブリアス / 「ガブリアス！」→ ガブリアス！
        if text.startswith("「") or text.endswith("」"):
            text = text.strip("「」").strip()
            if not text:
                continue

        # 共通の軽量フィルター（DB 照合前に除外）
        # _UI_OVERLAY_WORDS は PokeClassifier 使用時も必ず適用（通信中・待機中等のシステムテキスト除外）
        if (text.startswith("Lv") or re.match(r'^[\d\s/]+$', text)
                or text in _UI_WORDS or text in _UI_OVERLAY_WORDS
                or any(kw in text for kw in _UI_OVERLAY_SUBSTRINGS)
                or text in _BATTLE_RESULT_WORDS
                or any(kw in text for kw in _BATTLE_RESULT_WORDS)
                or (re.match(r'^[A-Za-z0-9\s]+$', text) and len(text) < 4)
                or text.endswith("の") or text.endswith("「") or text.endswith("!")):
            continue

        # タイプ表示テキスト除外（「タイプ」→「タイプ：ヌル」への誤 fuzzy マッチ防止）
        if "タイプ" in text or "テラスタイプ" in text:
            continue

        # bbox の中心 x/y 座標で自分/相手エリアを判定
        # PokeClassifier がある場合はコマンドエリアフィルターをスキップ:
        #   技名・特性名は classifier が "move"/"ability" として除外できるため、
        #   フィルターで自分ポケモン名（画面下部 y>700）まで除外してしまう問題を回避する
        bbox = r.get("bbox", [])
        if bbox:
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            if classifier is None and center_y > _COMMAND_Y_MIN:
                continue  # PokeClassifier なし時のみコマンドメニュー内を除外
        else:
            center_x = 960  # bbox なし → 中央扱い
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
                        name_opponent_with_xy.append((text, center_x, center_y))
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

        # 自分/相手エリアに振り分け（x/y座標も記録）
        if center_y < _PLAYER_Y_THRESHOLD:
            name_opponent_with_xy.append((canonical, center_x, center_y))
        else:
            name_player_with_xy.append((canonical, center_x, center_y))

    # y座標（同y時はx座標）でソート（画面の上から下の順）
    hp_player_with_xy.sort(key=lambda t: (t[2], t[1]))
    hp_opponent_with_xy.sort(key=lambda t: (t[2], t[1]))
    name_player_with_xy.sort(key=lambda t: (t[2], t[1]))
    name_opponent_with_xy.sort(key=lambda t: (t[2], t[1]))

    hp_values_player   = [hp for hp, _, _ in hp_player_with_xy[:2]]
    hp_values_opponent = [hp for hp, _, _ in hp_opponent_with_xy[:2]]
    name_candidates_player   = [n for n, _, _ in name_player_with_xy[:5]]
    name_candidates_opponent = [n for n, _, _ in name_opponent_with_xy[:5]]

    # 名前+HPペア: 近傍マッチング（Euclidean距離）で対応付け
    # ダブルバトルでは横並びの2匹のyが近く zip だと順序が逆転するケースがあるため
    player_pokemon_hp   = _proximity_pair(name_player_with_xy[:5],   hp_player_with_xy[:2])
    opponent_pokemon_hp = _proximity_pair(name_opponent_with_xy[:5], hp_opponent_with_xy[:2])

    # x座標でソートしてスロット別HP（0=左スロット, 1=右スロット）を作成
    # 近傍マッチングに依存せず画面左右の固定位置でHPを割り当てるため
    hp_player_by_slot   = [hp for hp, _, _ in sorted(hp_player_with_xy,   key=lambda t: t[1])[:2]]
    hp_opponent_by_slot = [hp for hp, _, _ in sorted(hp_opponent_with_xy, key=lambda t: t[1])[:2]]

    # COMMAND テキストの cy を検出（画面種別の判定に使用）
    # cy ≈ 377: 行動選択・技選択画面（HP表示あり） → HP セット対象
    # cy ≈ 129: 対象選択・交換選択画面（HP混在）   → HP セット除外
    _command_re = re.compile(r'[Cc][O0oQ][Mm][Mm][Aa@][Nn][Dd]')
    command_cy: float | None = None
    for r in ocr_results:
        if r["confidence"] >= 0.4 and _command_re.search(r["text"]):
            bbox = r.get("bbox", [])
            if bbox:
                command_cy = (bbox[0][1] + bbox[2][1]) / 2
                break

    return {
        "hp_values":              hp_values,
        "hp_values_player":       hp_values_player,
        "hp_values_opponent":     hp_values_opponent,
        "name_candidates_player":   name_candidates_player,
        "name_candidates_opponent": name_candidates_opponent,
        "player_pokemon_hp":    player_pokemon_hp,    # [(name, hp), ...] y座標ソート済み
        "opponent_pokemon_hp":  opponent_pokemon_hp,
        "hp_player_by_slot":    hp_player_by_slot,    # [左HP, 右HP] x座標ソート
        "hp_opponent_by_slot":  hp_opponent_by_slot,
        "name_player_with_cx":   [(n, cx) for n, cx, _ in name_player_with_xy[:5]],
        "name_opponent_with_cx": [(n, cx) for n, cx, _ in name_opponent_with_xy[:5]],
        "command_cy":           command_cy,           # COMMAND テキストの y 座標（None=未検出）
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
    ability_msg: dict[str, str] | None = None,
) -> dict:
    """
    OCR + YOLO 結果から Phi-3 に渡す game_state を組み立てる。
    HP・ポケモン名の精密なパースは難しいため、OCR 生テキストを
    実況文生成の追加コンテキストとして渡す。
    """
    # YOLO から状態異常・ボール数を取得
    # 状態異常は現在フレームから、ボール数はボールが見えていた最新フレーム（prev_yolo）から補完
    p_s = [s for s in [yolo_state.player_status_0, yolo_state.player_status_1] if s]
    o_s = [s for s in [yolo_state.opponent_status_0, yolo_state.opponent_status_1] if s]
    status_text = "/".join(p_s) if p_s else "なし"
    if o_s:
        status_text += f" / 相手: {'/'.join(o_s)}"

    ball_src = prev_yolo if prev_yolo else yolo_state
    p_balls = ball_src.player_balls.alive
    o_balls = ball_src.opponent_balls.alive

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
        "player_pokemon_hp":        structured["player_pokemon_hp"],
        "opponent_pokemon_hp":      structured["opponent_pokemon_hp"],
        "hp_player_by_slot":        structured["hp_player_by_slot"],
        "hp_opponent_by_slot":      structured["hp_opponent_by_slot"],
        "name_player_with_cx":      structured["name_player_with_cx"],
        "name_opponent_with_cx":    structured["name_opponent_with_cx"],
        "ability_msg_player":       (ability_msg or {}).get("player", ""),
        "ability_msg_opp":          (ability_msg or {}).get("opp", ""),
        "command_cy":               structured["command_cy"],
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

    # コマンド選択画面の判定キーワード
    # 「ゆけつ」「いけつ」（ゆけっ！繰り出し演出）は除外する:
    # これらがあると繰り出しアニメーション中にも command_select と誤判定され、
    # 余分な move_used → turn_start サイクルが発生してターン数が多重カウントされる。
    # battle_start は「たたかう」初回出現で確実に検知できるため早期検知は不要。
    _COMMAND_KW    = {"たたかう", "どうする"}
    # 「こうたい」「ポケモンをえらんで」はSV版UIのラベル。Championsでは「交代する」
    # （conf0.99〜1.00で安定検出・診断ログ実測）を使うため追加。
    _SWITCH_KW     = {"こうたい", "ポケモンをえらんで", "交代する"}
    # 通信待機中: Champions特有の「全コマンド確定待ち」画面
    # ダブルバトルで双方の全コマンドが揃うまで表示される。
    # この画面の終了が「実際の行動開始（move_used）」の信頼できる唯一のシグナル。
    # 完全一致だと「通信待様中」（機→様）等のOCR誤読と低confで95%以上のフレームを
    # 取りこぼす（診断ログ6本で実測: 完全一致6〜52fに対し誤読・低confが数百〜数千f）。
    # ファジー正規表現＋専用conf閾値で判定する。「マッチング待機中」は対象外。
    _COMM_RE       = re.compile(r'通.?[待侍].中|^待機中$|通信中')
    # conf0.2でも実際は8割超のフレームを取りこぼす（診断ログ6本実測: conf中央値0.117）。
    # conf<0.05帯を確認しても中身はほぼ全て「通信待ば中」「通ル待機中」等の同一テキストの
    # 誤読バリエーションでノイズ混入はごく僅か。_COMM_RE自体が十分specificなため
    # conf閾値はほぼ無効化し、正規表現の一致自体を信頼する。
    _COMM_CONF_MIN = 0.0
    _COMM_ENTRY_FRAMES   = 2    # 通信フェーズ入場に必要な連続検出フレーム数（単発誤検出の排除）
    _COMM_EXIT_GRACE_SEC = 3.0  # 通信フェーズ退出の猶予秒数（単発の取りこぼしを吸収）
    # 技選択画面の型相性ラベル（これが見えている = 技選択UI が開いている = command_select 継続）
    # 「こうかあり」はバトルメッセージには出ず技選択UIにのみ出現するため安全な指標
    # 「いまひとつ」「こうかなし」は技選択UIにも出るため _ANIM_KW から除外済み
    _TECH_SELECT_KW = {"こうかあり"}
    # バツグンだ・きゅうしょ・ひんし は技選択UIに出ないためアニメーション指標として残す
    _ANIM_KW       = {"バツグンだ", "きゅうしょ", "急所", "ひんし"}
    _END_KW        = {"勝負に勝", "勝負に負", "降参が選ばれ", "通信エラー", "切断されました",
                     "更新されるまで"}  # 成績更新待ち画面（正常決着後のフォールバック）
    # 選出画面キーワード（この画面中はバトルイベントを発火させない）
    _SELECTION_KW  = {"ポケモンを選んで", "選出", "きめる", "リーダー", "選出順"}
    # L50競技ポケモンの最低HPは約50以上なので、分母が50未満は除外
    _HP_ZERO_RE    = re.compile(r'(?:\b0/(?:[5-9]\d|\d{3})\b|\b0%\b)')

    # イベント別デバウンス秒数（_debounce はデフォルト値）
    _DEBOUNCE_OVERRIDES: dict[str, float] = {
        "turn_start": 15.0,  # ダブルバトル2匹目コマンド選択での余分な turn_start 発火を防ぐ
        "move_used":  10.0,  # 相手を見るパネル等で通信待機中バッジが十数秒隠れて再出現する
                              # ケースの再発火を抑える。faint後の繰り出し選択で再び通信待機中に
                              # なる正当なケース（20秒以上先）は許容される。
        "faint":      25.0,  # 相手を見るパネルが0/211を表示し続けることで多重発火する問題の防止
    }

    # turn_start脱出弁: move_usedを取りこぼすと_allow_turn_start=Falseのまま固着し、
    # turn_startが恒久的にブロックされてターン番号がズレ続ける（診断ログで実測）。
    # ターン内でも相手を見るパネル等でコマンド画面が最大80秒程度消えることがあるため、
    # セッション区切りは実測のターン間ギャップ最小値（192秒）との間を取って120秒。
    _CMD_SESSION_GAP_SEC   = 120.0  # コマンド画面がこの秒数以上消えていたら新セッション
    _TURN_START_ESCAPE_SEC = 45.0   # 前回turn_start確定からこの秒数経過で脱出弁が開く
    # faint再アーム: 「0/211」等は相手を見るパネルで最大210秒表示され続ける（実測）。
    # 表示がこの秒数以上途切れてから再出現した場合のみ新しいfaintイベントとして扱う。
    _FAINT_REARM_SEC = 20.0

    def __init__(self, debounce_seconds: float = 10.0):
        self._debounce = debounce_seconds
        self._last_event_time: dict[str, float] = {}
        self._prev_phase = "unknown"
        self._battle_started = False
        self._is_processing = False
        # move_usedまたはbattle_startの後にのみturn_startを許可するフラグ。
        # ダブルバトルで1匹目→2匹目コマンド選択中の余分なturn_startを
        # debounceに依存せず完全にブロックする。
        self._allow_turn_start = False
        # 通信フェーズ平滑化（入場確認・退出猶予）
        self._comm_streak = 0        # communication 連続検出フレーム数
        self._comm_active = False    # 確定済み通信フェーズ中か
        self._last_comm_seen = 0.0   # 最後に communication を検出した時刻
        # turn_start脱出弁・faint再アーム用の追跡
        self._last_cmd_seen = 0.0          # 最後に command_select を見た時刻
        self._cmd_session_start = 0.0      # 現在のコマンドセッションの開始時刻
        self._last_turn_start_fired = 0.0  # 最後に turn_start / battle_start が確定した時刻
        self._last_faint_seen = 0.0        # 最後に faint フェーズを見た時刻

    def set_processing(self, v: bool) -> None:
        self._is_processing = v

    def reset_after_processing(self, event_type: str | None = None) -> None:
        """処理完了後にフェーズ履歴をリセットし、直後の誤発火を防ぐ。
        処理中に _prev_phase が command_select で止まっていると、
        処理完了直後のフレームで command_select → unknown 遷移として
        move_used が即再発火する問題を防ぐ。
        """
        self._prev_phase = "unknown"
        now = time.time()
        # move_used デバウンスを現在時刻に更新（処理完了直後の再発火を抑止）
        self._last_event_time["move_used"] = now
        # faint後・move_used後は command_select 誤分類による turn_start 多重発火を抑制
        # （move_used: アニメーション中に一瞬 command_select が映り turn_start が早期発火する問題）
        if event_type in ("faint", "move_used"):
            self._last_event_time["turn_start"] = now

    def _is_communication(self, ocr_results: list[dict]) -> bool:
        """通信待機中/通信中の表示があるかをファジー判定する。"""
        for r in ocr_results:
            if r["confidence"] < self._COMM_CONF_MIN:
                continue
            txt = r["text"].replace(" ", "")
            if "マッチング" in txt:  # マッチメイキング画面の「マッチング待機中」は対象外
                continue
            if self._COMM_RE.search(txt):
                return True
        return False

    def _smooth_communication(self, raw: str) -> str:
        """communication フェーズの入退場を平滑化する。
        入場は連続 _COMM_ENTRY_FRAMES フレームで確定（単発誤検出の排除）、
        退場は _COMM_EXIT_GRACE_SEC の猶予付き（取りこぼしによる move_used 多重発火の防止）。
        退場確定フレームでは実際の画面種別に関わらず "unknown" を返す:
        communication→command_select と直接遷移すると turn_start の prev 条件に
        引っかかって move_used 後の turn_start が永久に発火できなくなるため、
        必ず communication→unknown→(次フレームで実フェーズ) の順に遷移させる。
        """
        if raw == "battle_end":
            self._comm_streak = 0
            self._comm_active = False
            return raw
        now = time.time()
        if raw == "communication":
            self._comm_streak += 1
            self._last_comm_seen = now
            if not self._comm_active and self._comm_streak >= self._COMM_ENTRY_FRAMES:
                self._comm_active = True
            return "communication" if self._comm_active else "unknown"
        self._comm_streak = 0
        if self._comm_active:
            if now - self._last_comm_seen < self._COMM_EXIT_GRACE_SEC:
                return "communication"
            self._comm_active = False
            return "unknown"  # 退場確定の合成フレーム（communication→unknown 遷移で move_used 発火）
        return raw

    def classify(self, ocr_results: list[dict]) -> str:
        """OCR 結果から現在のフェーズを判定する（優先度順）。"""
        texts = {r["text"] for r in ocr_results if r["confidence"] >= 0.4}
        # OCR が1テキストを複数に分割するケースに対応（例:「勝負に」+「勝った！」）
        # ocr_results の順序（上→下・左→右）を維持して結合し、キーワードの分断を吸収する
        joined = "".join(r["text"].replace(" ", "") for r in ocr_results if r["confidence"] >= 0.4)

        if any(kw in t for kw in self._END_KW for t in texts) or any(kw in joined for kw in self._END_KW):
            return "battle_end"
        # 通信待機中: Champions特有の「全コマンド確定待ち」画面
        # この画面の終了を move_used のトリガーとして使う。
        # faint/switch_select より先にチェックして誤分類を防ぐ。
        # OCR誤読（通信待様中等）・低conf対策のためファジー判定（_COMM_RE参照）
        if self._is_communication(ocr_results):
            return "communication"
        # 選出画面: バトル前のポケモン選択画面（ここでのイベント発火を防ぐ）
        if any(kw in t for kw in self._SELECTION_KW for t in texts):
            return "selection_screen"
        if any(self._HP_ZERO_RE.search(r["text"]) for r in ocr_results if r["confidence"] >= 0.4):
            return "faint"
        if any(kw in t for kw in self._SWITCH_KW for t in texts):
            return "switch_select"
        if any(kw in t for kw in self._COMMAND_KW for t in texts):
            return "command_select"
        # 技選択画面中（「こうかあり」が表示）は command_select 継続とみなす
        # → 技選択UIの「いまひとつ」「こうかなし」が animation と誤判定されるのを防ぐ
        if any(kw in t for kw in self._TECH_SELECT_KW for t in texts):
            return "command_select"
        if any(kw in t for kw in self._ANIM_KW for t in texts):
            return "animation"
        return "unknown"

    def detect(self, ocr_results: list[dict]) -> str | None:
        """フェーズ遷移からイベントを返す。イベントなし or 処理中 は None。
        ただし battle_end は処理中でも割り込み検知する（実況中の試合終了を見逃さないため）。
        """
        raw = self.classify(ocr_results)
        curr = self._smooth_communication(raw)
        prev = self._prev_phase
        self._prev_phase = curr
        now = time.time()

        # コマンド画面の出現を追跡（脱出弁用: 長い空白の後の出現 = 新しいコマンドセッション）
        if curr == "command_select":
            if now - self._last_cmd_seen >= self._CMD_SESSION_GAP_SEC:
                self._cmd_session_start = now
            self._last_cmd_seen = now
        # faint表示の継続を追跡（_FAINT_REARM_SEC 以上途切れてからの再出現のみ新イベント扱い）
        faint_rearmed = False
        if curr == "faint":
            faint_rearmed = now - self._last_faint_seen >= self._FAINT_REARM_SEC
            self._last_faint_seen = now

        # 処理中でも battle_end だけは割り込み検知
        if self._is_processing:
            if curr == "battle_end" and prev != "battle_end":
                self._battle_started = False
                # battle_end後の結果画面が command_select と誤分類されて
                # turn_start が発火するのを抑制する
                self._last_event_time["turn_start"] = time.time()
                log.info(f"[フェーズ] {prev} → {curr} | イベント: battle_end (実況中割り込み)")
                return "battle_end"
            return None

        event: str | None = None

        # 選出画面中は battle_started / allow_turn_start をリセット（誤発火対策）
        if curr == "selection_screen":
            self._battle_started = False
            self._allow_turn_start = False

        if curr == "command_select" and not self._battle_started:
            self._battle_started = True
            self._allow_turn_start = True  # battle_start後の最初のturn_startを許可
            event = "battle_start"
        elif (curr == "command_select" and self._battle_started
              and prev not in ("command_select", "selection_screen", "communication")
              and self._allow_turn_start):
            # move_usedまたはbattle_start後の command_select = ターン開始。
            # _allow_turn_start=False（1匹目→2匹目コマンド選択中）はここでスキップされる。
            # debounceに依存しないため、操作が何秒かかっても余分なturn_startは発火しない。
            event = "turn_start"
        elif prev == "communication" and curr not in ("communication", "battle_end"):
            # 通信待機中終了 = 全コマンド確定後にアニメーション開始
            # Champions特有: ダブルバトルで双方の全コマンドが揃ったことを示す唯一の信頼できるシグナル。
            event = "move_used"
        elif curr == "faint" and prev != "faint" and faint_rearmed:
            # エッジトリガ: 相手を見るパネル等で「0/211」が長時間残り、OCRのチラつきで
            # faint⇄他フェーズを往復するたびにデバウンス(25s)を貫通して再発火する問題の防止
            event = "faint"
        elif curr == "switch_select" and prev not in ("switch_select", "communication"):
            event = "switch"
        elif curr == "battle_end" and prev != "battle_end":
            event = "battle_end"
            self._battle_started = False
            self._allow_turn_start = False
            # battle_end後の結果画面が command_select と誤分類されて
            # turn_start が発火するのを抑制する
            self._last_event_time["turn_start"] = time.time()

        # 脱出弁: move_used取りこぼしで_allow_turn_start=Falseのまま固着した場合の回復措置。
        # 新しいコマンドセッションの2フレーム目以降（開始10秒以内）かつ前回turn_start確定から
        # _TURN_START_ESCAPE_SEC 以上経過していれば、遷移条件・デバウンスを無視して発火する。
        # （通常経路がデバウンスで握りつぶされた場合も次フレームでここが拾う）
        escape = False
        if (event is None and curr == "command_select" and self._battle_started
                and self._cmd_session_start > 0
                and 0 < now - self._cmd_session_start <= 10.0
                and now - self._last_turn_start_fired >= self._TURN_START_ESCAPE_SEC):
            event = "turn_start"
            escape = True
            log.info("[脱出弁] turn_start を強制発火（前回確定から %.0f 秒・move_used 取りこぼしの疑い）",
                     now - self._last_turn_start_fired)

        if curr != prev:
            if event:
                log.info(f"[フェーズ] {prev} → {curr} | イベント: {event}")
            else:
                log.debug(f"[フェーズ] {prev} → {curr}")

        if event:
            no_debounce = {"battle_start", "battle_end"}
            debounce = self._DEBOUNCE_OVERRIDES.get(event, self._debounce)
            last = self._last_event_time.get(event, 0.0)
            if event not in no_debounce and not escape and now - last < debounce:
                log.debug(f"デバウンス中のためスキップ: {event} (残り {debounce-(now-last):.1f}s)")
                return None
            self._last_event_time[event] = now
            if event == "move_used":
                # move_used確定 → 次のコマンド選択でturn_startを許可
                self._allow_turn_start = True
                self._last_event_time["turn_start"] = now
            elif event == "turn_start":
                # turn_start確定 → 次のmove_usedまでturn_startを禁止
                # （1匹目→2匹目コマンド選択中の余分なturn_startをdebounce非依存でブロック）
                self._allow_turn_start = False
                self._last_turn_start_fired = now
            elif event == "battle_start":
                # T1のturn_start相当として記録（直後に脱出弁が誤発火するのを防ぐ）
                self._last_turn_start_fired = now

        return event


# ─── バトルメッセージ解析 ──────────────────────────────────────────────────────

# メッセージボックスROIから「[ポケモン名]の[技名]」を抜き出す正規表現
# 例: "バドレックスのブリザードランス！" → group(1)=バドレックス, group(2)=ブリザードランス
_MOVE_IN_MSG_RE = re.compile(r'(.{2,12})の(.{3,15}?)(?:[！!」]|$)')

# OCR変形表記 → 正規技名 のマッピング
# PokeClassifier が技として認識できない OCR 誤読・変形表記を事前に正規化する。
# 例: ゲーム内テキスト「手助けする」は技名「てだすけ」に対応する表現で、
#     OCR がそのまま読んだ場合 PokeClassifier では「move」判定されない。
# 新しい変形表記が見つかったらここに追加するだけで対応できる。
_MOVE_ALIAS_MAP: dict[str, str] = {
    "手助けする":          "てだすけ",
    "手助けした":          "てだすけ",
    "手助け":              "てだすけ",
    "攻撃から身を守った":  "まもる",
}

# OCR 大文字かな→小文字かな正規化テーブル
# 例: 「サイドチエンジ」→「サイドチェンジ」、「きよじゆうだん」→「きょじゅうだん」
_OCR_KANA_NORM_RE = re.compile(
    r'(?<=[きしちにひみりぎじびぴ])[よゆ]|'  # ひらがな: きよ→きょ, じゆ→じゅ 等
    r'(?<=[チシジ])[エユ]|'                   # カタカナ: チエ→チェ, シユ→シュ 等
    r'(?<=[ファフヴウ])[オアイエ]|'            # フォ/ファ/ヴォ/ウィ 等
    r'デイ|テイ'                              # ディ/ティ
)
_OCR_KANA_NORM_MAP = {
    'よ': 'ょ', 'ゆ': 'ゅ',
    'エ': 'ェ', 'ユ': 'ュ',
    'オ': 'ォ', 'ア': 'ァ', 'イ': 'ィ', 'エ': 'ェ',
    'デイ': 'ディ', 'テイ': 'ティ',
}


def _normalize_ocr_kana(text: str) -> str:
    """OCR が大文字かなを小文字かなとして誤読する頻出パターンを補正する。

    例: サイドチエンジ → サイドチェンジ
        きよじゆうだん → きょじゅうだん
        ボデイプレス   → ボディプレス
        ワイドフオース → ワイドフォース
        ダブルウイング → ダブルウィング
    """
    def _replace(m: re.Match) -> str:
        s = m.group(0)
        return _OCR_KANA_NORM_MAP.get(s, _OCR_KANA_NORM_MAP.get(s[-1], s))

    return _OCR_KANA_NORM_RE.sub(_replace, text)


# メッセージボックスROI特有の既知OCR誤読（促音ッ/清音ツの混同など、_normalize_ocr_kana の
# 大文字→小文字かな正規化とは別種のため単純な文字列置換テーブルで対応する）
_MSG_OCR_FIX_MAP: dict[str, str] = {
    "バッグンだ": "バツグンだ",
}


def _normalize_ocr_message(text: str) -> str:
    """メッセージボックスROIのテキストに対する既知OCR誤読の補正。"""
    for wrong, correct in _MSG_OCR_FIX_MAP.items():
        text = text.replace(wrong, correct)
    return text


class BattleMessageParser:
    """
    バトル中に左下のメッセージボックスに表示されるテキストを解析し、
    フェーズ分類に頼らない詳細なバトルイベントを検出する。

    SVのメッセージボックスROI: x < 520, 740 < center_y < 930 (1920x1080)

    検出イベント:
      faint      ─ ○○は/が たおれた
      switch_in  ─ ○○、ゆけ / ○○が とびだした
      switch_out ─ もどれ、○○ / ○○と こうたいした
    """

    MSG_X_MIN  = 120   # メッセージボックス左端マージン
    MSG_X_MAX  = 900   # チャンピオンズ対応: メッセージが画面中央まで広がるため 520→900 に拡張
    MSG_Y_MIN  = 740
    MSG_Y_MAX  = 930
    DEDUP_TTL  = 8.0   # 同一イベントの重複発火を防ぐ秒数

    # 「あいて」「あい」= 「相手の」のOCR誤読プレフィックス
    # 「たおれたり/ゆ」= 「たおれた」のOCR誤読サフィックス
    _FAINT_RE      = re.compile(
        r'((?:あい(?:て)?\s*)?(?:相手の?\s*)?)'  # group1: 相手プレフィックス（空なら自分側）
        r'([^\s]{2,12})'                          # group2: ポケモン名（スペースなし 2〜12文字）
        r'(?:は|が)\s*たおれた[りゆ]?'           # 「たおれた」（OCR誤読サフィックスも許容）
    )
    # 自分がポケモンを繰り出すメッセージのパターン:
    #   旧: 「〇〇、ゆけ！」（名前が前）
    #   新: 「ゆけつ！ (げんきいっぱいの) 〇〇！」（名前が後ろ・SVの実際の表示形式）
    _SWITCH_IN_RE  = re.compile(
        r'(.{2,12})、?\s*ゆけ'                      # 「〇〇、ゆけ！」（名前が前・旧パターン保持）
        r'|ゆけ\S*\s+(?:\S+の\s+)?(\S{2,12})'      # 「ゆけつ！ (げんきいっぱいの) 〇〇！」
        r'|(.{2,12})が\s*とびだした'                 # 「〇〇が とびだした」
    )
    # 相手がポケモンを繰り出すメッセージ: 「〇〇をくりだした！」
    # OCR誤読バリアント: くりだした → くゆだした（り→ゆ誤読）等に対応
    # トークン間スペース対応: 「ウルガモスを くゆだした」のようにスペース区切りでも捕捉
    # プレイヤー名の「は」を読み飛ばし: 「たかひとはウルガモスを」→ ウルガモスのみ捕捉
    # Champions対応: 漢字形式「繰り出した」/「繰ゆ出した」（り→ゆ誤読）も捕捉
    _OPPONENT_SWITCH_IN_RE = re.compile(
        r'(?:[^\s]{2,12}と\s*)?(?:[^\s]*は\s*)?([^\s]{2,12}?)を\s*(?:く[りゆ]だした|繰[りゆ]出した)'
    )
    # ダブルバトル「AとBをくりだした」形式でAとBを両方捕捉する専用RE
    _DUAL_OPPONENT_SWITCH_IN_RE = re.compile(
        r'([^\s]{2,12}?)と\s*([^\s]{2,12}?)を\s*(?:く[りゆ]だした|繰[りゆ]出した)'
    )
    _SWITCH_OUT_RE = re.compile(
        r'もどれ[、,]\s*(.{2,12})'          # SV: もどれ、〇〇（ひらがな）
        r'|(.{2,12})と\s*こうたいした'       # 交代技: 〇〇とこうたいした
        r'|(\S{2,12})\s*戻れ'               # Champions: 〇〇\n戻れ！（漢字）
    )
    # 状態異常メッセージ: 「〇〇は まひじょうたいになった」等
    _STATUS_RE = re.compile(
        r'(.{2,12})(?:は|が)\s*(まひ|やけど|どく|もうどく|こおり|ねむり)\s*(?:じょうたい|状態)?'
    )

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], float] = {}

    def _extract_msg_text(self, ocr_results: list[dict]) -> str:
        """メッセージボックスROI内のテキストをy→x順に結合して返す。"""
        items: list[tuple[float, float, str]] = []
        for r in ocr_results:
            if r["confidence"] < 0.35:
                continue
            bbox = r.get("bbox", [])
            if not bbox:
                continue
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            if self.MSG_X_MIN <= cx < self.MSG_X_MAX and self.MSG_Y_MIN < cy < self.MSG_Y_MAX:
                items.append((cy, cx, r["text"]))
        items.sort(key=lambda t: (round(t[0] / 40), t[1]))
        return _normalize_ocr_message(" ".join(t[2] for t in items))

    def parse(self, ocr_results: list[dict]) -> list[dict]:
        """OCR結果からメッセージイベントのリストを返す（重複はデバウンスでスキップ）。"""
        text = self._extract_msg_text(ocr_results)
        now = time.time()
        events: list[dict] = []

        def _emit(event_type: str, pokemon: str, raw: str = "") -> None:
            # 末尾の感嘆符・括弧・読点はOCRノイズなので除去
            pokemon = pokemon.strip().rstrip('！!」、')
            if not pokemon:
                return
            key = (event_type, pokemon)
            if now - self._seen.get(key, 0.0) < self.DEDUP_TTL:
                return
            self._seen[key] = now
            events.append({"type": event_type, "pokemon": pokemon, "raw": raw or text})

        if text:
            m = self._FAINT_RE.search(text)
            if m:
                prefix, name = m.group(1), m.group(2)
                # 「相手の」プレフィックスがあれば相手側、なければ自分側
                _emit("opponent_faint" if prefix.strip() else "faint", name)

            m = self._SWITCH_IN_RE.search(text)
            if m:
                _emit("switch_in", (m.group(1) or m.group(2) or m.group(3) or ""))

            # 「AとBをくりだした」: 先にDual REで1匹目・2匹目の両方を emit
            m2 = self._DUAL_OPPONENT_SWITCH_IN_RE.search(text)
            if m2:
                _emit("opponent_switch_in", m2.group(1))
                _emit("opponent_switch_in", m2.group(2))
            else:
                m = self._OPPONENT_SWITCH_IN_RE.search(text)
                if m:
                    _emit("opponent_switch_in", m.group(1))

            m = self._SWITCH_OUT_RE.search(text)
            if m:
                _emit("switch_out", (m.group(1) or m.group(2) or m.group(3) or ""))

        # 状態異常はROI外も含めた全OCRから検索（メッセージ表示フレームを取りこぼす場合に備える）
        full_text = " ".join(r["text"] for r in ocr_results if r["confidence"] >= 0.35)
        status_text = f"{text} {full_text}" if text else full_text
        m = self._STATUS_RE.search(status_text)
        if m:
            pokemon_name = m.group(1).strip().rstrip('！!」、')
            status = m.group(2)
            if pokemon_name:
                key = ("status", pokemon_name)
                if now - self._seen.get(key, 0.0) >= self.DEDUP_TTL:
                    self._seen[key] = now
                    events.append({"type": "status", "pokemon": pokemon_name,
                                   "status": status, "raw": status_text})

        return events


# ─── 戦況トラッカー ────────────────────────────────────────────────────────────

@dataclass
class FieldPokemon:
    """1匹のポケモンの戦況スロット（ダブルバトル対応）。"""
    name: str
    hp: str | None = None                        # "176/176" 形式（最新HP）
    hp_pct_pixel: float | None = None            # ピクセル解析によるHP% (0.0-1.0)
    status: str | None = None                    # まひ / やけど / どく / ひんし
    moves_used: list[str] = field(default_factory=list)  # このポケモンが使った技リスト
    on_field: bool = False                        # 現在場にいるか
    fainted: bool = False                         # 気絶済みフラグ
    confidence: int = 0                           # 検出回数（信頼度）
    last_seen_turn: int = 0                       # 最後に検出されたターン番号
    slot_index: int | None = None                 # 画面スロット番号: 0=左(x<960), 1=右(x>=960)


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
    _ON_FIELD_MISS_THRESHOLD = 3
    _HP_RE = re.compile(r'(\d{1,3})/(\d{1,3})')
    # 画面中央x座標: これより左がスロット0（左）、右がスロット1（右）
    _SLOT_X_CENTER = 960

    def __init__(self):
        self.turn = 0       # 内部イベントカウンター（_ON_FIELD_MISS_THRESHOLD 用）
        self.game_turn = 0  # 実際のゲームターン数（command_select 出現ごとに +1）
        self._player:   list[FieldPokemon] = []  # 自分の最大4匹
        self._opponent: list[FieldPokemon] = []  # 相手の最大4匹
        self._event_log: list[str] = []
        # ボール数トラッキング（気絶推定・控え不明表示用）
        self._prev_opponent_alive: int | None = None  # 前ターンの相手生存数
        self._player_alive_count:  int | None = None  # 最新の自分生存数
        self._opponent_alive_count: int | None = None # 最新の相手生存数

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

    def _assign_hp_to_on_field_smart(
        self,
        slots: list[FieldPokemon],
        hp_list: list[str],
        pokemon_hp_pairs: list[tuple[str, str]],
        allow_zero_hp: bool = False,
        has_name_candidates: bool = True,
    ) -> None:
        """HPを割り当てる。ペア情報があれば名前マッチ、なければインデックスベース（y座標ソート済み）。
        allow_zero_hp=False（デフォルト）の場合、HP=0/X は割り当てない（faintイベント以外での誤気絶防止）。
        has_name_candidates=False の場合、faintイベントで0/Xのインデックスベース割り当てをスキップ
        （is_status_panel時に名前なしで0/Xが誤ったポケモンに割り当てられる問題を防ぐ）。
        """
        # faintイベント以外では HP=0/X を除外（バドレックスのHP=0がミライドンに誤割り当てされる問題を防ぐ）
        if not allow_zero_hp:
            hp_list = [hp for hp in hp_list if not hp.startswith("0/")]
            pokemon_hp_pairs = [(n, hp) for n, hp in pokemon_hp_pairs if not hp.startswith("0/")]

        on_field = [s for s in slots if s.on_field]
        if pokemon_hp_pairs:
            # 名前ベースで割り当て（y座標ソート済みのペアを使用）
            matched: set[int] = set()
            for name, hp in pokemon_hp_pairs:
                for slot in on_field:
                    if id(slot) not in matched:
                        if slot.name == name or name in slot.name or slot.name in name:
                            slot.hp = hp
                            matched.add(id(slot))
                            break
        else:
            # フォールバック: インデックスベース（hp_list は y座標ソート済み）
            # has_name_candidates=False（is_status_panel等で名前未検出）かつfaintイベント中に
            # 0/X HP が含まれる場合はスキップ:
            #   相手を見るパネル表示中にHP値が誤った位置に分類され、
            #   インデックスベースで0/Xが別のポケモンに割り当てられる問題を防ぐ
            if not has_name_candidates and allow_zero_hp and any(hp.startswith("0/") for hp in hp_list):
                log.debug("[戦況] faintイベントで名前未検出・HP=0/X → インデックスベース割り当てをスキップ")
                return
            for i, hp in enumerate(hp_list):
                if i < len(on_field):
                    on_field[i].hp = hp

    def _assign_slot_indices(
        self,
        slots: list[FieldPokemon],
        name_with_cx: list[tuple[str, float]],
    ) -> None:
        """初登場時に OCR x座標からスロット番号（0=左, 1=右）を割り当てる。
        固定閾値ではなく、同フレームで見えた未割り当てポケモン同士の相対x順で決定する。
        （SV のプレイヤー側2匹のHPバーは両方とも画面左半分に表示されるため
        cx=960 の固定閾値は使えない）
        既に slot_index が設定済みのポケモンはスキップする。
        """
        # 名前マッチングで未割り当て on_field スロットと cx を収集
        candidates: list[tuple[FieldPokemon, float]] = []
        for name, cx in name_with_cx:
            for slot in slots:
                if slot.name == name or name in slot.name or slot.name in name:
                    if slot.on_field and slot.slot_index is None:
                        candidates.append((slot, cx))
                    break

        if not candidates:
            return

        # cx 昇順（画面左→右）でソートし、小さい方をスロット0、大きい方をスロット1
        candidates.sort(key=lambda t: t[1])

        # 既に割り当て済みのスロット番号を把握し、残り候補に空きスロットを割り当てる
        # （例: slot_0 が早期割済みで candidates に1匹だけ残った場合、slot_1 を割り当てる）
        used = {s.slot_index for s in slots if s.on_field and s.slot_index is not None}
        available = [i for i in range(2) if i not in used]

        if len(candidates) == 1 and len(available) == 1:
            candidates[0][0].slot_index = available[0]
            log.info(f"[スロット] {candidates[0][0].name} → スロット{available[0]} (cx={candidates[0][1]:.0f}, 空きスロット割当)")
        else:
            for i, (slot, cx) in enumerate(candidates[:2]):
                slot.slot_index = i
                log.info(f"[スロット] {slot.name} → スロット{slot.slot_index} (cx={cx:.0f})")

    def _assign_hp_by_slot(
        self,
        slots: list[FieldPokemon],
        hp_by_slot: list[str],
        allow_zero_hp: bool = False,
    ) -> None:
        """スロット番号（0=左, 1=右）でHPをポケモンに割り当てる。
        slot_index が設定済みのポケモンはスロット番号で割り当て。
        slot_index 未設定のポケモンはインデックス順フォールバックで割り当てる
        （テスト環境・初登場フレームで座標が未取得の場合の互換性維持）。
        allow_zero_hp=False の場合、HP=0/X は割り当てない（faintイベント以外での誤気絶防止）。
        """
        if not hp_by_slot:
            return
        effective = [
            hp if (allow_zero_hp or (not hp.startswith("0/") and hp != "0%")) else None
            for hp in hp_by_slot
        ]
        on_field = [s for s in slots if s.on_field]
        assigned_slots: set[int] = set()  # スロット番号ベースで割り当て済みのスロット

        # パス1: slot_index 設定済みのポケモンはスロット番号で割り当て
        for slot in on_field:
            if slot.slot_index is not None:
                i = slot.slot_index
                if i < len(effective) and effective[i]:
                    slot.hp = effective[i]
                    assigned_slots.add(i)

        # パス2: slot_index 未設定のポケモンは残りのHP値をインデックス順で割り当て
        remaining_hp = [hp for i, hp in enumerate(effective) if i not in assigned_slots and hp]
        idx = 0
        for slot in on_field:
            if slot.slot_index is None and idx < len(remaining_hp):
                slot.hp = remaining_hp[idx]
                idx += 1

    # ── メイン更新 ───────────────────────────────────────────────────────────

    def update(self, game_state: dict, event_type: str) -> None:
        """1 イベントごとに呼び出して戦況を更新する。"""
        self.turn += 1

        current_player_names   = set(game_state.get("name_candidates_player", []))
        current_opponent_names = set(game_state.get("name_candidates_opponent", []))

        # ── ポケモン名の蓄積・on_field 更新（自分側） ──────────────────────
        for name in current_player_names:
            # 相手側に登録済みかつ今フレームの相手エリアにも見えている場合 → 同名ポケモンの可能性あり → 両側に登録
            # 相手側に登録済みだが今フレームで自分エリアにしか見えない場合 → y座標誤分類の可能性 → スキップ
            # ただし自分側にも既に登録済みなら同名ポケモン確定 → スキップしない
            already_in_opponent = any(s.name == name for s in self._opponent)
            already_in_player_slots = any(s.name == name for s in self._player)
            if already_in_opponent and not already_in_player_slots and name not in current_opponent_names:
                continue  # 相手側にのみ登録済みで相手エリアにも見えていない → 誤分類として除外
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
            already_in_player = any(s.name == name for s in self._player)
            if already_in_player and name not in current_player_names:
                continue  # 自分側に登録済みで自分エリアにも見えていない → 誤分類として除外
            slot = self._get_or_create(self._opponent, name)
            if slot:
                slot.confidence += 1
                slot.last_seen_turn = self.turn
                if slot.fainted:
                    # OCR で再検出されたにもかかわらず fainted=True → 誤ひんし判定を解除
                    # （ボール数ロジックの誤判定でひんし扱いされたポケモンが復帰できるようにする）
                    slot.fainted = False
                    slot.on_field = True
                    log.warning("[戦況] %s が fainted=True だが OCR 再検出 → 誤ひんし解除", slot.name)
                else:
                    slot.on_field = True

        newly_removed_opponent: list[FieldPokemon] = []
        for slot in self._opponent:
            if slot.on_field and not slot.fainted:
                # OCR で相手名が検出されている場合: そのフレームで見えないなら即座に降ろす
                # （交代直後に旧ポケモンが残り続けるのを防ぐ）
                if current_opponent_names and slot.name not in current_opponent_names:
                    quick_threshold = 1
                    if self.turn - slot.last_seen_turn >= quick_threshold:
                        slot.on_field = False
                        newly_removed_opponent.append(slot)
                        log.info(f"[戦況] {slot.name} がOCR不検出（{slot.name} not in {current_opponent_names}）→ 場から降ろす")
                elif self.turn - slot.last_seen_turn > self._ON_FIELD_MISS_THRESHOLD:
                    slot.on_field = False
                    newly_removed_opponent.append(slot)
                    log.info(f"[戦況] {slot.name} が{self._ON_FIELD_MISS_THRESHOLD}ターン不検出 → 場から降ろす")

        # ダブルバトル制約: 場のポケモンは最大 2 匹
        self._cap_on_field(self._player)
        self._cap_on_field(self._opponent)

        # ── スロット番号の割り当て（初登場時・x座標ベース） ─────────────────
        # battle_start はゆけ！アニメーション中のフレームで、両ポケモン名が
        # 画面左側に集まって表示されるためスロット判定が不正確になる。
        # turn_start/move_used 以降の安定したフレームでのみスロットを確定する。
        if event_type != "battle_start":
            player_name_cx   = game_state.get("name_player_with_cx", [])
            opponent_name_cx = game_state.get("name_opponent_with_cx", [])
            self._assign_slot_indices(self._player,   player_name_cx)
            self._assign_slot_indices(self._opponent, opponent_name_cx)

        # ── OCR HP値をスロット番号（x座標）で割り当て ──────────────────────
        # COMMAND が行動選択・技選択画面（cy 300〜450）にある時のみセット。
        # 対象選択・交換選択画面（cy ≈ 129）やアニメーション中（COMMAND なし）は
        # OCR HP をセットせず HPpx（ピクセル解析）に任せる。
        _CMD_CY_MIN, _CMD_CY_MAX = 300, 450
        command_cy = game_state.get("command_cy")
        ocr_hp_valid = (command_cy is not None and _CMD_CY_MIN <= command_cy <= _CMD_CY_MAX)
        if ocr_hp_valid:
            hp_player_by_slot   = game_state.get("hp_player_by_slot", [])
            hp_opponent_by_slot = game_state.get("hp_opponent_by_slot", [])
            # フォールバック: y座標分類が完全に失敗して両方空の場合、hp_values を均等分配
            if not hp_player_by_slot and not hp_opponent_by_slot:
                all_hp = game_state.get("hp_values", [])
                hp_player_by_slot   = all_hp[:2]
                hp_opponent_by_slot = all_hp[2:4]
            allow_zero = (event_type == "faint")
            self._assign_hp_by_slot(self._player,   hp_player_by_slot,   allow_zero)
            self._assign_hp_by_slot(self._opponent, hp_opponent_by_slot, allow_zero)

        # ── ボール数トラッキング＆相手気絶推定 ──────────────────────────────
        balls = game_state.get("balls_remaining", [])
        cur_p_alive = balls[0] if len(balls) > 0 else None
        cur_o_alive = balls[1] if len(balls) > 1 else None
        # 相手のボール生存数が減少 + 今ターン降ろしたポケモンがいる → 気絶確定
        if (cur_o_alive is not None and self._prev_opponent_alive is not None
                and cur_o_alive < self._prev_opponent_alive
                and newly_removed_opponent):
            diff = self._prev_opponent_alive - cur_o_alive
            for slot in newly_removed_opponent[:diff]:
                if not slot.fainted:
                    slot.fainted = True
                    log.info(f"[戦況] {slot.name} が気絶（ボール数減少で確定: {self._prev_opponent_alive}→{cur_o_alive}）")
        # ボール数を更新（非None・非ゼロのみ採用してノイズを無視）
        if cur_p_alive:
            self._player_alive_count = cur_p_alive
        if cur_o_alive:
            self._opponent_alive_count = cur_o_alive
            self._prev_opponent_alive = cur_o_alive

        # 把握済み相手生存数 > ボール数 → 超過分を気絶確定
        # ボール数変化なし（ex. 2匹同時倒れでボールが一段階しか減らない）でも
        # 場から外れたまま再登場しないポケモンを気絶扱いにして控えを正確にする
        # newly_removed_opponent が存在する場合は check #1 で処理済み or
        # 交代アニメーション中のボール数一時ズレの可能性があるためスキップ。
        # current_opponent_names が空の場合（faintイベント・アニメーション中等）は
        # 相手の場の状態が不明なためボール数比較を信頼せずスキップ。
        if (not newly_removed_opponent
                and current_opponent_names
                and cur_o_alive is not None
                and cur_o_alive > 0):
            known_opponent_alive = len([s for s in self._opponent if not s.fainted])
            if known_opponent_alive > cur_o_alive:
                suspects = [s for s in self._opponent if not s.fainted and not s.on_field]
                excess = known_opponent_alive - cur_o_alive
                for s in suspects[:excess]:
                    s.fainted = True
                    log.info(
                        f"[戦況] {s.name} が気絶（把握数{known_opponent_alive}＞ボール数{cur_o_alive}で確定）"
                    )

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
            newly_fainted_player = 0
            newly_fainted_opponent = 0
            for side in (self._player, self._opponent):
                for slot in side:
                    if slot.hp and (slot.hp.startswith("0/") or slot.hp == "0%") and not slot.fainted:
                        slot.fainted = True
                        slot.on_field = False
                        log.info(f"[戦況] {slot.name} が気絶（faintイベント）")
                        if side is self._player:
                            newly_fainted_player += 1
                        else:
                            newly_fainted_opponent += 1
            # ボール数が遅延している場合（アニメーション中）は fainted 数分デクリメント
            if newly_fainted_player and self._player_alive_count:
                self._player_alive_count = max(0, self._player_alive_count - newly_fainted_player)
            if newly_fainted_opponent and self._opponent_alive_count:
                self._opponent_alive_count = max(0, self._opponent_alive_count - newly_fainted_opponent)
                self._prev_opponent_alive = self._opponent_alive_count

        # ── イベントログ追記 ─────────────────────────────────────────────────
        ocr_snip = game_state.get("ocr_text", "")[:25]
        self._event_log.append(f"T{self.turn}:{event_type}[{ocr_snip}]")
        if len(self._event_log) > self.MAX_EVENTS:
            self._event_log.pop(0)

    def update_status_by_name(self, name: str, status: str) -> bool:
        """メッセージ由来の状態異常を名前で検索して記録する。"""
        slot = self._find_slot(name)
        if slot:
            slot.status = status
            log.info(f"[戦況] {slot.name} 状態異常: {status}（メッセージ由来）")
            return True
        return False

    def update_status_from_yolo(self, side: str, status: str, slot_idx: int | None) -> None:
        """YOLOアイコン/OCR bbox 検出による状態異常を場のポケモンに反映する。
        slot_idx が指定されている場合はそのスロットのポケモンにのみ付与する。
        slot_idx が None の場合、状態異常なしのポケモンが1匹だけなら付与する。
        いずれの場合も、対象ポケモンが既に同じ状態異常を持っていればスキップ。
        """
        target_side = self._opponent if side == "opponent" else self._player
        candidates = [p for p in target_side if p.on_field and not p.fainted]

        if slot_idx is not None:
            # スロット指定あり: そのスロットのポケモンにのみ付与（他スロットの状態は無関係）
            for p in candidates:
                if p.slot_index == slot_idx:
                    if not p.status:
                        p.status = status
                        log.info(f"[戦況] {p.name} 状態異常: {status}（OCRアイコン slot{slot_idx}）")
                    return  # 既設定済みでもスキップして終了
            # slot_indexが一致しない場合: status未設定の候補が1匹のみなら付与
            # （slot_index競合や未割り当ての両方を吸収する）
            no_status = [p for p in candidates if not p.status]
            if len(no_status) == 1:
                no_status[0].status = status
                log.info(f"[戦況] {no_status[0].name} 状態異常: {status}（OCRアイコン slot{slot_idx}）")
        else:
            no_status = [p for p in candidates if not p.status]
            if len(no_status) == 1:
                no_status[0].status = status
                log.info(f"[戦況] {no_status[0].name} 状態異常: {status}（OCRアイコン スロット不明）")

    # 確定状態異常技テーブル (技名 → (状態異常, 全体技かどうか))
    # 技名はpokedb.sqlite の moves.name_ja と一致させること
    # ※ でんじは（単体技）はYOLOアイコン検出で正確なターゲットを特定するため除外
    _STATUS_MOVE_TABLE: dict[str, tuple[str, bool]] = {
        "おにび":         ("やけど",  True),    # 全体技
        "どくどく":       ("もうどく", False),
        "どくのこな":     ("どく",    False),
        "キノコのほうし": ("ねむり",  False),
        "さいみんじゅつ": ("ねむり",  False),
        "うたう":         ("ねむり",  False),
    }

    def apply_status_from_move(self, user_name: str, move_name: str) -> None:
        """確定状態異常技の効果から相手チームへ状態異常を推定付与する。
        OCRでメッセージが取れなかった場合の補完用。
        既に状態異常があるポケモン・気絶済みポケモンはスキップ。
        単体技でも場に複数いる場合は全員に付与する（ダブルバトルでターゲット特定不能のため）。
        """
        entry = self._STATUS_MOVE_TABLE.get(move_name)
        if not entry:
            return
        status, _is_spread = entry

        # 使用者がどちらのチームか判定（部分一致で対応）
        in_player = any(
            s.name == user_name or user_name in s.name or s.name in user_name
            for s in self._player
        )
        target_side = self._opponent if in_player else self._player
        targets = [s for s in target_side if s.on_field and not s.fainted and not s.status]

        for target in targets:
            target.status = status
            log.info(f"[戦況] {target.name} 状態異常: {status}（{user_name}の{move_name}効果推定）")

    def update_pixel_hp(self, pixel_hp: dict[str, float | None]) -> None:
        """ピクセル解析HP%でFieldPokemonのhp_pct_pixelを更新する。
        slot_index 設定済みポケモンに優先割り当て。未設定が1匹だけなら
        物理スロット位置から早期割り当てを行う（T2等のOCR取得前フレームに対応）。
        """
        side_map: dict[str, tuple[list[FieldPokemon], int]] = {
            "player_0":   (self._player,   0),
            "player_1":   (self._player,   1),
            "opponent_0": (self._opponent, 0),
            "opponent_1": (self._opponent, 1),
        }
        for key, pct in pixel_hp.items():
            if pct is None:
                continue
            slots, slot_idx = side_map[key]

            # パス1: slot_index 設定済みのポケモンに割り当て
            matched = False
            for p in slots:
                if p.on_field and p.slot_index == slot_idx:
                    old = p.hp_pct_pixel
                    p.hp_pct_pixel = pct
                    if old is None or abs(pct - old) >= 0.05:
                        log.info("[HPpx] %s %s → %.1f%%", key, p.name, pct * 100)
                    matched = True
                    break

            # パス2: 未割り当てが1匹だけなら位置ベース早期割り当て
            if not matched:
                unassigned = [p for p in slots if p.on_field and p.slot_index is None]
                if len(unassigned) == 1:
                    p = unassigned[0]
                    p.slot_index = slot_idx
                    log.info("[スロット早期割] %s → スロット%d (HPpxフォールバック)", p.name, slot_idx)
                    p.hp_pct_pixel = pct
                    log.info("[HPpx] %s %s → %.1f%%", key, p.name, pct * 100)

    def update_move(self, pokemon_name: str, move_name: str, is_opponent: bool = False) -> None:
        """ポケモンが技を使ったことを記録する（per-pokemon 技リスト更新）。
        is_opponent=True の場合は相手チームのみを検索する（同名ポケモンの誤登録防止）。
        """
        sides = [self._opponent] if is_opponent else [self._player, self._opponent]
        for side in sides:
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

    # ── メッセージ由来イベント ────────────────────────────────────────────────

    def _find_slot(self, name: str) -> FieldPokemon | None:
        """名前で両チームを検索してスロットを返す（部分一致OK）。"""
        for slot in self._player + self._opponent:
            if slot.name == name or name in slot.name or slot.name in name:
                return slot
        return None

    def _confirm_faint_on_side(self, slots: list, name: str) -> bool:
        """指定した側のスロットのみを検索してfaintedフラグを立てる。"""
        for slot in slots:
            if slot.name == name or name in slot.name or slot.name in name:
                if not slot.fainted:
                    slot.fainted = True
                    slot.on_field = False
                    if slot.hp and "/" in slot.hp:
                        slot.hp = f"0/{slot.hp.split('/')[1]}"
                    elif slot.hp and slot.hp.endswith("%"):
                        slot.hp = "0%"
                    log.info(f"[戦況] {slot.name} 気絶確認（メッセージ由来）")
                return True
        return False

    def confirm_faint_by_name(self, name: str) -> bool:
        """メッセージ由来の気絶確認: 両陣営を検索（サイド不明の場合のフォールバック用）。"""
        return self._confirm_faint_on_side(self._player + self._opponent, name)

    def confirm_player_faint_by_name(self, name: str) -> bool:
        """自分側のポケモン気絶確認（「たおれた」メッセージに「相手の」なし）。"""
        return self._confirm_faint_on_side(self._player, name)

    def confirm_opponent_faint_by_name(self, name: str) -> bool:
        """相手側のポケモン気絶確認（「相手の〇〇はたおれた」メッセージ）。"""
        return self._confirm_faint_on_side(self._opponent, name)

    def accumulate_player_name(self, name: str) -> None:
        """定期OCRで検出されたプレイヤーポケモン名を蓄積する（イベント以外の補完用）。
        相手側に同名ポケモンがいる場合はスキップ（y座標誤分類対策）。
        未登録なら新規スロットを作成して on_field=True にする。
        ダブルバトル上限（場2匹）を超える場合は新規追加しない。
        """
        already_in_opponent = any(s.name == name for s in self._opponent)
        if already_in_opponent:
            return
        on_field_count = sum(1 for s in self._player if s.on_field and not s.fainted)
        already_in_player = any(s.name == name for s in self._player)
        if already_in_player:
            for s in self._player:
                if s.name == name and not s.fainted and not s.on_field:
                    if on_field_count < 2:
                        s.on_field = True
                        s.last_seen_turn = self.turn
                        log.info(f"[戦況] {s.name} 定期OCR検出 → 場に追加")
            return
        if on_field_count >= 2:
            return  # ダブルバトル上限: 新規追加しない
        slot = self._get_or_create(self._player, name)
        if slot and not slot.fainted:
            slot.on_field = True
            slot.last_seen_turn = self.turn
            log.info(f"[戦況] {slot.name} 定期OCR検出 → 新規登録して場に追加")

    def mark_on_field_by_name(self, name: str) -> bool:
        """メッセージ由来の繰り出し確認: プレイヤースロットを検索してon_field=Trueにする。
        相手に同名ポケモンがいる場合でも正しく自分側に登録するため、プレイヤー側のみ検索する。
        スロット未登録なら新規作成する。
        """
        slot = None
        for s in self._player:
            if s.name == name or name in s.name or s.name in name:
                slot = s
                break
        if slot is None:
            slot = self._get_or_create(self._player, name)
        if slot and not slot.fainted:
            slot.on_field = True
            slot.last_seen_turn = self.turn
            log.info(f"[戦況] {slot.name} 繰り出し確認（メッセージ由来）")
            return True
        return False

    def register_opponent_on_field(self, name: str) -> bool:
        """相手の繰り出し確認: スロット未登録なら新規登録してon_fieldにする。"""
        slot = self._get_or_create(self._opponent, name)
        if slot is None:
            log.warning(f"[戦況] 相手スロットが満杯のため {name} を登録できません")
            return False
        if not slot.fainted:
            slot.on_field = True
            slot.confidence += 1
            slot.last_seen_turn = self.turn
            log.info(f"[戦況] 相手 {slot.name} 繰り出し確認（メッセージ由来）")
        return True

    def mark_bench_by_name(self, name: str) -> bool:
        """メッセージ由来の引っ込め確認: 名前でスロットを検索してon_field=Falseにする。"""
        slot = self._find_slot(name)
        if slot:
            slot.on_field = False
            log.info(f"[戦況] {slot.name} 引っ込め確認（メッセージ由来）")
            return True
        return False

    # ── コンテキスト生成 ─────────────────────────────────────────────────────

    def _format_pokemon(self, p: FieldPokemon) -> str:
        """場にいるポケモンの詳細フォーマット（HP・状態異常・使用技を含む）。"""
        s = p.name
        if p.status:
            s += f"({p.status})"
        if p.hp_pct_pixel is not None:
            # HPpx優先（座標固定で2匹を独立計測）
            pct_px = p.hp_pct_pixel * 100
            s += f" HP:{pct_px:.0f}%(px)"
            if pct_px <= 25:
                s += "★ピンチ"
        elif p.hp:
            # HPpx未取得時はOCRで補強
            s += f" HP:{p.hp}"
            m = self._HP_RE.match(p.hp)
            if m and int(m.group(2)) > 0:
                pct = int(m.group(1)) / int(m.group(2)) * 100
                if pct <= 25:
                    s += "★ピンチ"
            elif p.hp.endswith("%"):
                try:
                    if int(p.hp[:-1]) <= 25:
                        s += "★ピンチ"
                except ValueError:
                    pass
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

        # ボール数から未把握の控えポケモンを「不明×N」で補完
        # 把握済み生存数 = fainted でないポケモン数
        known_p_alive = len([p for p in self._player   if not p.fainted])
        known_o_alive = len([p for p in self._opponent if not p.fainted])
        if self._player_alive_count and self._player_alive_count > known_p_alive:
            unk = self._player_alive_count - known_p_alive
            suffix = f"不明×{unk}"
            player_bench_str = suffix if player_bench_str == "なし" else f"{player_bench_str} / {suffix}"
        if self._opponent_alive_count and self._opponent_alive_count > known_o_alive:
            unk = self._opponent_alive_count - known_o_alive
            suffix = f"不明×{unk}"
            opponent_bench_str = suffix if opponent_bench_str == "なし" else f"{opponent_bench_str} / {suffix}"

        # RAG 用: 蓄積済み全ポケモン名リスト（信頼度順）
        player_names   = [p.name for p in sorted(self._player,   key=lambda p: -p.confidence)]
        opponent_names = [p.name for p in sorted(self._opponent, key=lambda p: -p.confidence)]

        return {
            "turn":             self.game_turn,
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

def _save_ocr_debug_image(frame: np.ndarray | None, ocr_results: list[dict], turn: int) -> None:
    """
    OCR 結果を frame 上に描画して debug/ に保存する。
    色分け:
      緑  = 自分側ポケモン名候補（y >= _PLAYER_Y_THRESHOLD）
      赤  = 相手側ポケモン名候補（y < _PLAYER_Y_THRESHOLD）
      青  = HP 値
      灰  = フィルター済み / 低信頼度
    """
    if frame is None:
        return
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

    # 先頭の「【...】」ラベルを除去（例: 「【バトル開始！】テキスト」→「テキスト」）
    text = re.sub(r'^(【[^】]*】\s*)+', '', text)
    # 中間に残った「【」以降を除去（Phi-3 の「【画面分析】...」が漏れてくる場合）
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
        # frame=None の場合（動画末尾フォールバック等）は黒ダミー画像を使用
        if frame is None:
            small = np.zeros((450, 800, 3), dtype=np.uint8)
        else:
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
                        f"{info['name_ja']}: タイプ={info['type']} / 特性={abilities_str}"
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
                "faint_context":            game_state.get("faint_context", ""),  # 直前のfaint情報（統合時のみ）
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
        ball_model_path: str | None,
        end_model_path: str | None,
        interval: float,
        speaker: int,
        gpu: bool,
        conf: float,
        ec2_url: str | None,
        audio_device: int | None,
        video_path: str | None = None,
        video_sample_fps: float = 2.0,
    ):
        log.info("=== パイプライン初期化 ===")

        log.info("EasyOCR 初期化中...")
        self._reader = init_reader(gpu=gpu)

        log.info("YoloDetector 初期化中...")
        self._yolo = YoloDetector(model_path=model_path, ball_model_path=ball_model_path, end_model_path=end_model_path, conf=conf)

        log.info("Phi-3 クライアント初期化...")
        self._phi3 = Phi3Client()

        log.info("VOICEVOX クライアント初期化...")
        self._voicevox = VoicevoxClient(speaker=speaker)

        log.info("AudioPlayer 初期化...")
        self._player = AudioPlayer(device=audio_device)

        self._camera_index = camera_index
        self._video_path = video_path
        self._video_sample_fps = video_sample_fps
        self._interval = interval
        self._ec2_url = ec2_url
        self._diff_detector = DiffDetector()             # 静止フレームのスキップ用
        self._phase_classifier = BattlePhaseClassifier() # フェーズ分類 + イベント検知
        self._last_ocr_time: float = 0.0                  # 定期OCR用タイマー
        self._PERIODIC_OCR_INTERVAL_BATTLE = 1.5         # バトル中: 終了画面を取りこぼさないよう短め
        self._PERIODIC_OCR_INTERVAL_IDLE   = 3.0         # バトル外: 重くならないよう長め
        self._battle_tracker = BattleStateTracker()       # 戦況累積
        self._hpbar_analyzer = HpBarAnalyzer()             # HPバーピクセル解析
        self._msg_parser = BattleMessageParser()           # バトルメッセージ解析
        self._battle_active = False  # battle_start〜battle_end の間のみ True
        self._last_battle_end_time: float = 0.0  # battle_end 後のクールダウン用
        self._BATTLE_START_COOLDOWN = 10.0  # battle_end 後この秒数は battle_start をブロック
        self._end_screen_count: int = 0  # 終了画面連続検出カウント
        self._END_SCREEN_CONFIRM = 3      # この回数連続で検出したら battle_end 確定
        self._battle_active_since: float = 0.0  # battle_start の時刻
        self._MIN_BATTLE_DURATION = 25.0  # バトル開始からこの秒数は終了画面チェックをスキップ
        self._prev_yolo: BattleState | None = None
        self._last_ball_yolo: BattleState | None = None  # ボールが見えたフレームの最新 YOLO 結果
        self._last_ability_msg: dict[str, str] = {}     # 最後に検出した特性・道具発動メッセージ
        self._pre_battle_opponent: list[str] = []  # battle_start前に検出した相手ポケモン名キャッシュ
        self._pre_battle_player: list[str] = []    # battle_start前に検出した自分ポケモン名キャッシュ（ゆけっ！検出）
        self._commentary_history: list[str] = []
        self._dense_scan_remaining: int = 0  # move_used後の高密度メッセージROIスキャン残りフレーム数
        self._dense_scan_start_turn: int | None = None  # dense scan起点ターン（技ログのターン番号固定用）
        self._last_full_ocr_results: list[dict] = []  # メインOCR最新結果（dense scan時の使い手特定に使用）
        self._move_log: list[str] = []   # OCRから検出した「使われた技」のリングバッファ
        self._tentative_opponent_moves: list[dict] = []  # dense scan フォールバックで仮確定した相手技（後付け修正用）
        self._MAX_MOVE_LOG = 8
        self._speech_thread: threading.Thread | None = None  # 音声再生スレッド
        # faint保留送信: faintイベントのBedrockを即送信せず次のmove_usedで統合する
        self._pending_faint_state: dict | None = None
        self._pending_faint_battle_context: dict | None = None
        self._pending_faint_frame: "np.ndarray | None" = None
        self._pending_faint_time: float = 0.0
        self._pending_faint_game_turn: int = 0   # faint保留時点の game_turn（統合時に繰り上げ要否を判断）
        self._FAINT_PENDING_TIMEOUT: float = 75.0  # この秒数内にmove_usedが来なければ単独送信
        self._skip_next_turn_start: bool = False  # faint統合でgame_turnを繰り上げた後、直後のturn_startをスキップするフラグ

        # PokeDB 分類器（DB がなければ None でフォールバック動作）
        log.info("PokeClassifier 初期化中...")
        try:
            self._classifier: PokeClassifier | None = PokeClassifier()
        except FileNotFoundError as e:
            log.warning("PokeDB が見つからないため手動フィルターで動作: %s", e)
            self._classifier = None

        log.info("=== 初期化完了 ===")

    def run(self) -> None:
        _is_video = self._video_path is not None
        cap = cv2.VideoCapture(self._video_path if _is_video else self._camera_index)
        if not cap.isOpened():
            if _is_video:
                log.error(f"動画ファイルを開けませんでした: {self._video_path}")
            else:
                log.error(f"カメラ {self._camera_index} を開けませんでした（OBS仮想カメラが起動中か確認）")
            sys.exit(1)

        if not _is_video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            for _ in range(10):
                cap.read()

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if _is_video:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # デフォルト2fps（技名などの短時間表示を取りこぼしにくくする）
            _video_frame_skip = max(1, int(fps // self._video_sample_fps))
            _video_frame_pos = 0
            log.info(f"動画ファイル: {self._video_path} ({w}x{h}, {fps:.1f}fps, {total}フレーム, skip={_video_frame_skip})")
        else:
            log.info(f"カメラ {self._camera_index} オープン: {w}x{h}")
        log.info("パイプライン開始（Ctrl+C で終了）")

        turn = 0
        try:
            while True:
                loop_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    if _is_video:
                        log.info("動画ファイルの末尾に達しました。終了します。")
                        if self._battle_active:
                            log.info("[battle_end] 動画末尾到達 → 試合終了イベントを合成発行")
                            turn += 1
                            self._phase_classifier.set_processing(True)
                            try:
                                self._process_event(None, None, [], "battle_end", turn)
                            finally:
                                self._phase_classifier.set_processing(False)
                                self._phase_classifier._battle_started = False
                        break
                    log.warning("フレーム取得失敗。再試行します...")
                    time.sleep(0.5)
                    continue

                # ── YOLO 検出（毎フレーム）──────────────────────────────────
                yolo_state = self._yolo.detect(frame)
                if yolo_state.detections:
                    log.debug(f"[YOLO] {yolo_state.summary()}")
                # YOLOアイコン → BattleStateTracker に状態異常を同期（per-slot）
                if self._battle_active:
                    for slot, status in enumerate([yolo_state.opponent_status_0, yolo_state.opponent_status_1]):
                        if status:
                            self._battle_tracker.update_status_from_yolo("opponent", status, slot)
                    for slot, status in enumerate([yolo_state.player_status_0, yolo_state.player_status_1]):
                        if status:
                            self._battle_tracker.update_status_from_yolo("player", status, slot)
                # ボールが見えているフレームを記憶（イベント時は animation 中でボールが映らないため）
                if yolo_state.player_balls.total > 0 or yolo_state.opponent_balls.total > 0:
                    self._last_ball_yolo = yolo_state
                elif self._phase_classifier._prev_phase == "command_select":
                    # コマンド選択中はボールアイコンが必ず表示されるため低閾値で再検出
                    ball_state = self._yolo.detect_balls(frame, conf=0.05)
                    if ball_state.player_balls.total > 0 or ball_state.opponent_balls.total > 0:
                        self._last_ball_yolo = ball_state
                        log.debug(f"[YOLO/balls] コマンド選択時に検出: {ball_state.summary()}")

                # ── 終了画面 YOLO 検知（毎フレーム・連続確認あり）──────────
                _battle_elapsed = time.time() - self._battle_active_since
                if (self._battle_active
                        and not self._phase_classifier._is_processing
                        and _battle_elapsed >= self._MIN_BATTLE_DURATION):
                    if self._yolo.detect_end_screen(frame):
                        self._end_screen_count += 1
                        log.debug(f"[YOLO] 終了画面検出 {self._end_screen_count}/{self._END_SCREEN_CONFIRM}")
                        if self._end_screen_count >= self._END_SCREEN_CONFIRM:
                            # OCR で勝敗テキストを確認（誤発火防止の AND 条件）
                            _end_ocr = run_ocr(self._reader, frame)
                            _end_joined = "".join(r["text"] for r in _end_ocr)
                            if not any(kw in _end_joined for kw in _END_SCREEN_OCR_KEYWORDS):
                                log.info(f"[YOLO] 終了画面{self._end_screen_count}回検出 → OCRキーワード不一致のため誤発火と判定 (OCR: {_end_joined[:60]})")
                                self._end_screen_count = 0
                            else:
                                log.info(f"[YOLO] 終了画面を{self._end_screen_count}回連続検出 + OCR確認済 → battle_end")
                                self._end_screen_count = 0
                                turn += 1
                                self._phase_classifier.set_processing(True)
                                try:
                                    self._process_event(frame, yolo_state, [], "battle_end", turn)
                                finally:
                                    self._phase_classifier.set_processing(False)
                                    self._phase_classifier._battle_started = False
                                    self._phase_classifier._last_event_time["turn_start"] = time.time()
                                continue
                    else:
                        self._end_screen_count = 0

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
                    self._last_full_ocr_results = ocr_results  # dense scan時の使い手特定用にキャッシュ
                    ocr_texts = [r["text"] for r in ocr_results if r["confidence"] >= 0.4]
                    if self._battle_active and ocr_texts:
                        log.info("[OCR] %s", " / ".join(ocr_texts[:15]))
                    else:
                        log.debug(f"OCR({reason}): {len(ocr_results)} 件 | {' / '.join(ocr_texts[:10])}")

                    # 定期OCR時はデバッグ画像を保存（終了画面など未検知フェーズの診断用）
                    if periodic_ocr and ocr_results:
                        _save_ocr_debug_image(frame, ocr_results, turn * 1000 + int(now) % 1000)

                    # ── HPバーピクセル解析（バトル中の毎OCRサイクル）──────────
                    if self._battle_active:
                        pixel_hp = self._hpbar_analyzer.analyze(frame)
                        self._battle_tracker.update_pixel_hp(pixel_hp)
                        # 定期OCRでも自分側ポケモン名を蓄積（battle_start時に映らなかったポケモンの補完）
                        # イベントOCRに比べてUIノイズが多いため、y座標分類済みのみを対象とする
                        if ocr_results:
                            _periodic_gs = _extract_structured_info(ocr_results, self._classifier)
                            _periodic_player = _periodic_gs.get("name_candidates_player", [])
                            for _pname in _periodic_player:
                                self._battle_tracker.accumulate_player_name(_pname)

                    # ── 技使用・交代メッセージの検出（バトル中は常時監視）──────
                    if self._battle_active:
                        self._update_move_log(ocr_results, is_main_ocr=True)
                        self._update_switch_out(ocr_results)
                        # OCR bbox 位置から状態異常アイコンを検出してトラッカーに反映
                        fh, fw = frame.shape[:2]
                        self._sync_status_from_ocr_bbox(ocr_results, fh, fw)
                        # 特性・道具発動メッセージを収集
                        ability = self._scan_ability_msg(ocr_results, fh, fw)
                        if ability:
                            self._last_ability_msg = ability
                            log.info(f"[特性/道具] {ability}")
                        # メッセージボックス解析（左下ROI・気絶/交代の補完）
                        for ev in self._msg_parser.parse(ocr_results):
                            log.info(
                                "[メッセージ] %s: %s / 「%s」",
                                ev["type"], ev["pokemon"], ev["raw"],
                            )
                            self._handle_message_event(ev)
                    else:
                        # バトル開始前: 全OCRテキストから「をくりだした」を検出してキャッシュ
                        # ROIフィルタなし（開始演出では通常のメッセージボックス外に表示される可能性あり）
                        pre_ocr_texts = [r["text"] for r in ocr_results if r["confidence"] >= 0.35]
                        log.debug("[事前OCR] %s", " / ".join(pre_ocr_texts[:15]))
                        # OCR誤読（くり→くゆ等）・漢字形式（Champions）に対応
                        _KURI_RE = re.compile(r'く[りゆ]だした|繰[りゆ]出した')
                        # 「AとBを繰り出した」の1匹目（A）も捕捉する専用RE
                        _DUAL_KURI_RE = re.compile(
                            r'([^\s]{2,12}?)と\s*([^\s]{2,12}?)を\s*(?:く[りゆ]だした|繰[りゆ]出した)'
                        )
                        full_pre_text = " ".join(pre_ocr_texts)
                        if _KURI_RE.search(full_pre_text) and self._classifier:
                            # Dual RE で「AとBを繰り出した」の1匹目・2匹目を直接捕捉
                            dm = _DUAL_KURI_RE.search(full_pre_text)
                            if dm:
                                for cand in [dm.group(1).rstrip('とをは！!」、'),
                                             dm.group(2).rstrip('とをは！!」、')]:
                                    if len(cand) < 2:
                                        continue
                                    r2 = self._classifier.classify(cand)
                                    if r2 and r2.category == CATEGORY_POKEMON and r2.score >= 80:
                                        if r2.canonical_ja not in self._pre_battle_opponent:
                                            self._pre_battle_opponent.append(r2.canonical_ja)
                                            log.info("[事前検出] 相手 %s（dual RE）", r2.canonical_ja)
                            # 全トークンもPokeClassifierでスキャン（単体繰り出し・OCR分割ケース対応）
                            for token in pre_ocr_texts:
                                clean = token.strip().rstrip('とをは！!」、')
                                if len(clean) < 2:
                                    continue
                                result = self._classifier.classify(clean)
                                if result and result.category == CATEGORY_POKEMON and result.score >= 80:
                                    if result.canonical_ja not in self._pre_battle_opponent:
                                        self._pre_battle_opponent.append(result.canonical_ja)
                                        log.info("[事前検出] 相手 %s をくりだした（battle_start待ち）", result.canonical_ja)

                        # バトル開始前: 自分ポケモンの「ゆけっ！ 〇〇！」パターンを検出してキャッシュ
                        # （battle_start OCR では技選択画面のため自分ポケモン名が映らないケースに対応）
                        if self._classifier:
                            _YUKE_RE = re.compile(r'ゆけ\S*\s+(?:\S+の\s+)?(\S{2,12})')
                            _yuke_full = " ".join(pre_ocr_texts)
                            for m in _YUKE_RE.finditer(_yuke_full):
                                cand = m.group(1).rstrip('！!」、')
                                if len(cand) < 2:
                                    continue
                                r3 = self._classifier.classify(cand)
                                if r3 and r3.category == CATEGORY_POKEMON and r3.score >= 80:
                                    if r3.canonical_ja not in self._pre_battle_player:
                                        self._pre_battle_player.append(r3.canonical_ja)
                                        log.info("[事前検出] 自分 %s（ゆけっ！検出）", r3.canonical_ja)

                    # ── フェーズ分類 + イベント検知 ─────────────────────────
                    event_type = self._phase_classifier.detect(ocr_results)

                    # battle_end はバトル中のみ有効（バトル未開始時の成績更新画面等を誤検知しない）
                    if event_type == "battle_end" and not self._battle_active:
                        log.debug("battle_end を検知したがバトル未開始のためスキップ")
                        event_type = None

                    # battle_end 後クールダウン中は battle_start をブロック
                    # （リザルト画面・ロビー画面の command_select 誤検知対策）
                    if (event_type == "battle_start"
                            and self._last_battle_end_time > 0
                            and time.time() - self._last_battle_end_time < self._BATTLE_START_COOLDOWN):
                        remaining = self._BATTLE_START_COOLDOWN - (time.time() - self._last_battle_end_time)
                        log.debug(f"battle_end 後クールダウン中のため battle_start をスキップ (残り {remaining:.1f}s)")
                        event_type = None

                    # フェーズ確定後にボール検出を補完（YOLO検出時点では_prev_phaseが未更新のため）
                    if self._phase_classifier._prev_phase == "command_select":
                        if not self._last_ball_yolo or self._last_ball_yolo.player_balls.total == 0:
                            ball_state = self._yolo.detect_balls(frame, conf=0.05)
                            if ball_state.player_balls.total > 0 or ball_state.opponent_balls.total > 0:
                                self._last_ball_yolo = ball_state
                                log.debug(f"[YOLO/balls] フェーズ確定後に検出: {ball_state.summary()}")

                    if event_type:
                        # move_used 検知時は品質チェック結果に関わらず dense scan を即起動する。
                        # オンライン対戦では move_used 直後に「通信中」（OCR 1件）となり品質チェックを
                        # 通過しないことが多いが、通信完了後のバトルメッセージ（てだすけ等）を
                        # 取りこぼさないために dense scan は必ず開始する。
                        if event_type == "move_used" and self._battle_active:
                            # 通信待機中終了 = 全コマンド確定。この時点の game_turn を起点として記録する。
                            # communication→unknown 遷移は1ターンに1回のみなので、常に上書き更新する。
                            self._dense_scan_start_turn = self._battle_tracker.game_turn
                            log.debug("[密集OCR] move_used 検知 → 起点ターン T%s を記録", self._dense_scan_start_turn)
                            # turn_start が来るまでdense scanを継続する（フレーム数で打ち切らない）。
                            # ダブルバトルではバトルアニメーションが2分を超えることがあり、
                            # 90フレーム（約54秒）では技テキストを取りこぼす。
                            self._dense_scan_remaining = 9999
                            log.debug("[密集OCR] move_used 検知 → dense scan 開始（turn_startまで継続）")

                        # turn_start / battle_start / battle_end: dense scan を停止する。
                        # これらのイベントはコマンド入力画面への遷移を示すため、
                        # バトルメッセージが消えており dense scan を継続しても意味がない。
                        if event_type in ("turn_start", "battle_start", "battle_end") and self._dense_scan_remaining > 0:
                            self._dense_scan_remaining = 0
                            self._dense_scan_start_turn = None
                            log.debug("[密集OCR] %s 検知 → dense scan 停止", event_type)

                        # ターンカウント前の品質チェック
                        # OCR 件数不足 or バトル外画面の場合はカウントも実況もスキップ
                        if event_type != "battle_end" and (
                            len(ocr_results) < 2 or not _is_battle_screen(ocr_results)
                        ):
                            log.info(
                                f"OCR 品質不足（{len(ocr_results)} 件）またはバトル外 → "
                                f"イベント '{event_type}' をスキップ（ターン未カウント）"
                            )
                            # ── faint早期フラッシュ: OCR品質不足でもmove_usedが来たら即送信 ──
                            # 「ゆけつ！」アニメーション中（OCR 0件）や通信中（1件）で
                            # move_used がスキップされると、15秒デバウンスで次の機会も
                            # 通らず 75秒タイムアウトまで待ってしまう問題を防ぐ。
                            if (event_type == "move_used"
                                    and self._pending_faint_state is not None):
                                elapsed = time.time() - self._pending_faint_time
                                log.info(
                                    "[faint早期フラッシュ] OCR品質不足だがmove_usedが来たので"
                                    "保留faintを単独送信 (%.1f秒後)", elapsed
                                )
                                self._flush_pending_faint()
                                self._pending_faint_state = None
                                self._pending_faint_battle_context = None
                                self._pending_faint_frame = None
                        else:
                            turn += 1
                            log.info(f"[ターン {turn}] イベント検知 (diff={diff_score:.1f}, type={event_type}, phase={self._phase_classifier._prev_phase})")
                            self._phase_classifier.set_processing(True)
                            try:
                                self._process_event(frame, yolo_state, ocr_results, event_type, turn)
                            finally:
                                self._phase_classifier.set_processing(False)
                                # turn_start はリセットしない: _prev_phase=command_select を保持することで
                                # 次のOCRで再び unknown→command_select が起きて多重発火するのを防ぐ
                                # また command_select→other で move_used が正しく検知されるようになる
                                if event_type != "turn_start":
                                    self._phase_classifier.reset_after_processing(event_type)

                # ── 密集メッセージROIスキャン（move_used後の高速テキスト取りこぼし対策）──
                # move_used 後の技名テキストは < 1秒で消えることがある（まもるで防がれた場合など）。
                # diff_changed に依存せず毎フレーム実行し、メッセージボックスROIのみをスキャンする。
                if self._battle_active and self._dense_scan_remaining > 0:
                    # 通信中フレームは dense scan を一時停止（フレームカウントを消費しない）
                    if any("通信中" in r["text"] for r in ocr_results):
                        pass  # 通信中はスキップ
                    else:
                        self._dense_scan_remaining -= 1
                        if self._dense_scan_remaining <= 0:
                            self._dense_scan_start_turn = None
                            log.debug("[密集OCR] dense scan 終了 → 起点ターンをリセット")
                    _D_Y1 = BattleMessageParser.MSG_Y_MIN   # 740
                    _D_Y2 = BattleMessageParser.MSG_Y_MAX   # 930
                    _D_X2 = BattleMessageParser.MSG_X_MAX   # 520
                    msg_roi = frame[_D_Y1:_D_Y2, 0:_D_X2]
                    dense_raw = run_ocr(self._reader, msg_roi, preprocess_dense=True)
                    # bbox y座標をオリジナルフレーム座標系にオフセット補正
                    # （preprocess_dense=True 時のbboxはすでにオリジナルスケールに戻されている）
                    dense_results = [
                        {
                            "text": r["text"],
                            "confidence": r["confidence"],
                            "bbox": [[pt[0], pt[1] + _D_Y1] for pt in r["bbox"]],
                        }
                        for r in dense_raw
                    ]
                    dense_texts = [r["text"] for r in dense_results if r["confidence"] >= 0.4]
                    if dense_texts:
                        log.info("[密集OCR] %s", " / ".join(dense_texts[:10]))
                        self._update_move_log(dense_results)

                self._prev_yolo = yolo_state

                if _is_video:
                    # 動画モード: 密集スキャン中はフレームスキップを 1/5 に縮小して技名テキストを取りこぼさない
                    if self._dense_scan_remaining > 0:
                        _video_frame_pos += max(1, _video_frame_skip // 5)
                    else:
                        _video_frame_pos += _video_frame_skip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, _video_frame_pos)
                else:
                    # ライブモード: 密集スキャン中はsleepを短縮（0.1s = 10fps相当）
                    elapsed = time.perf_counter() - loop_start
                    if self._dense_scan_remaining > 0:
                        time.sleep(max(0.0, 0.1 - elapsed))
                    else:
                        time.sleep(max(0.0, self._interval - elapsed))

        except KeyboardInterrupt:
            log.info(f"終了します（総ターン数: {turn}）")
        finally:
            cap.release()
            # self._generate_battle_template()  # 手動で scripts/generate_battle_template.py を使うため無効化

    # def _generate_battle_template(self) -> None:
    #     """終了時にログから対戦記録ひな型を自動生成する。"""
    #     try:
    #         import sys as _sys
    #         scripts_dir = Path(__file__).parent.parent / "scripts"
    #         if str(scripts_dir) not in _sys.path:
    #             _sys.path.insert(0, str(scripts_dir))
    #         from generate_battle_template import parse_log, format_template  # type: ignore
    #
    #         result = parse_log(log_file_path)
    #         output = format_template(result)
    #
    #         records_dir = Path("records")
    #         records_dir.mkdir(exist_ok=True)
    #         out_path = records_dir / f"{log_file_path.stem}.txt"
    #         out_path.write_text(output, encoding="utf-8")
    #         log.info(f"対戦記録ひな型を出力しました: {out_path}")
    #     except Exception as e:
    #         log.warning(f"対戦記録ひな型の生成に失敗しました: {e}")

    def _process_event(
        self,
        frame: np.ndarray | None,
        yolo_state: BattleState | None,
        ocr_results: list[dict],
        event_type: str,
        turn: int,
    ) -> None:
        """イベント発生時の一連の処理（game_state 構築 → Phi-3 / Bedrock → VOICEVOX → 再生）。"""
        log.debug(f"OCR: {len(ocr_results)} 件")

        # ── ターン開始イベント: ゲームターンカウントのみ（実況は不要）────────
        if event_type == "turn_start":
            if self._skip_next_turn_start:
                # faint統合でgame_turnを繰り上げ済み。直後のturn_startはスキップして二重加算を防ぐ。
                log.info(f"[ターン] turn_start スキップ（faint統合繰り上げ済み・T{self._battle_tracker.game_turn} 維持）")
                self._skip_next_turn_start = False
            else:
                self._battle_tracker.game_turn += 1
                log.info(f"[ターン] T{self._battle_tracker.game_turn} 開始")
            # 保留中のfaintがタイムアウトしていれば単独Bedrock送信でフラッシュ
            if (self._pending_faint_state is not None
                    and time.time() - self._pending_faint_time >= self._FAINT_PENDING_TIMEOUT):
                log.info("[faintフラッシュ] タイムアウト(%gs超過) → 保留faintを単独送信",
                         self._FAINT_PENDING_TIMEOUT)
                self._flush_pending_faint()
            return

        # ── バトル外画面はスキップ（battle_end は終了画面なので除外しない）────
        if event_type != "battle_end" and not _is_battle_screen(ocr_results):
            log.debug("バトル外の画面を検知 → スキップ")
            return

        # OCR 件数が少なすぎる場合はスキップ（battle_end は例外）
        if event_type != "battle_end" and len(ocr_results) < 2:
            log.info(f"OCR 件数が少なすぎる（{len(ocr_results)} 件）→ スキップ")
            return

        # ── game_state 構築 ───────────────────────────────────────────────────
        # イベント時は animation 中でボールが映らないため、最後にボールが見えたフレームの結果を優先する
        # frame/yolo_state が None の場合（動画末尾フォールバック）は空の BattleState を使う
        if yolo_state is None:
            yolo_state = BattleState()
        ball_yolo = self._last_ball_yolo if self._last_ball_yolo else yolo_state
        # ボール数は最新の確認済み値で yolo_state を上書き（ログと実況に反映）
        yolo_state.player_balls   = ball_yolo.player_balls
        yolo_state.opponent_balls = ball_yolo.opponent_balls
        game_state = _build_game_state(ocr_results, yolo_state, event_type, ball_yolo, self._classifier,
                                       ability_msg=self._last_ability_msg)
        log.info(f"[状態] {yolo_state.summary()} | OCR: {game_state['ocr_text']}")
        log.info(f"[構造化] HP={game_state['hp_values']} | 自分={game_state['name_candidates_player']} | 相手={game_state['name_candidates_opponent']}")
        _save_ocr_debug_image(frame, ocr_results, turn)

        # ── 戦況トラッカー更新 ────────────────────────────────────────────────
        if event_type == "battle_start":
            # バトル開始: トラッカーをリセットしてアクティブ化
            self._battle_tracker = BattleStateTracker()
            self._battle_active = True
            self._battle_active_since = time.time()
            self._end_screen_count = 0
            self._commentary_history = []
            self._move_log = []
            self._last_ball_yolo = None   # バトル開始時にボール情報をリセット
            self._last_ability_msg = {}   # バトル開始時に特性・道具メッセージをリセット
            log.info("[戦況] バトル開始 → トラッカーリセット")
            # バトル開始前にキャッシュした相手ポケモンを登録
            for name in self._pre_battle_opponent:
                self._battle_tracker.register_opponent_on_field(name)
            self._pre_battle_opponent.clear()
            # バトル開始前にキャッシュした自分ポケモンを登録（ゆけっ！検出分）
            for name in self._pre_battle_player:
                self._battle_tracker.mark_on_field_by_name(name)
            self._pre_battle_player.clear()

        # battle_start が OCR品質不足でスキップされた場合のフォールバック
        # バトルイベントが来た時点でアクティブ化（トラッカーはリセットしない・既存情報を保持）
        if not self._battle_active and event_type in {"move_used", "faint", "switch"}:
            log.warning("[戦況] battle_start 未検知 → バトルをアクティブ化（遅延起動）")
            self._battle_active = True
            self._battle_active_since = time.time()
            # 事前キャッシュが残っていれば登録（battle_start スキップで未登録のケース）
            if self._pre_battle_opponent:
                for name in self._pre_battle_opponent:
                    self._battle_tracker.register_opponent_on_field(name)
                    log.info("[戦況] 遅延登録: 相手 %s（battle_start フォールバック）", name)
                self._pre_battle_opponent.clear()
            if self._pre_battle_player:
                for name in self._pre_battle_player:
                    self._battle_tracker.mark_on_field_by_name(name)
                    log.info("[戦況] 遅延登録: 自分 %s（battle_start フォールバック）", name)
                self._pre_battle_player.clear()

        if self._battle_active:
            self._battle_tracker.update(game_state, event_type)

        battle_context = self._battle_tracker.to_context()
        log.info(
            "[戦況] T%s(G%s) 場(自)=%s | 場(相)=%s",
            self._battle_tracker.turn,
            self._battle_tracker.game_turn,
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
            # ── faint保留: 即送信せず次のmove_usedと統合するため保留する ──
            if event_type == "faint":
                log.info("[faint保留] Bedrock送信を保留（次のmove_usedで統合予定）")
                self._pending_faint_state = game_state
                self._pending_faint_battle_context = battle_context
                self._pending_faint_frame = frame
                self._pending_faint_time = time.time()
                self._pending_faint_game_turn = self._battle_tracker.game_turn
                # 実況・VOICEVOX もスキップして終了（戦況更新は済み）
                return

            # ── move_usedで保留中のfaint情報があれば統合 ──
            if event_type == "move_used" and self._pending_faint_state is not None:
                elapsed = time.time() - self._pending_faint_time
                if elapsed < self._FAINT_PENDING_TIMEOUT:
                    log.info("[faint統合] 保留faint(%.1f秒前)をmove_usedに統合", elapsed)
                    # turn_start がデバウンスで飛ばされた場合のみ繰り上げが必要。
                    # 保留時点から game_turn が変わっていなければ turn_start 未発火 → 繰り上げる。
                    # 既に turn_start が来て game_turn が進んでいれば繰り上げ不要。
                    if self._battle_tracker.game_turn == self._pending_faint_game_turn:
                        self._battle_tracker.game_turn += 1
                        log.info(f"[ターン] T{self._battle_tracker.game_turn} 開始（faint統合による繰り上げ）")
                        self._skip_next_turn_start = True  # 直後のturn_startによる二重加算を防ぐ
                    else:
                        log.info(
                            f"[faint統合] turn_start 済み (T{self._battle_tracker.game_turn}) → "
                            "ターン繰り上げスキップ"
                        )
                    # 保留faintの戦況を「直前の気絶情報」としてgame_stateに追加
                    pending_ctx = self._pending_faint_battle_context or {}
                    game_state = dict(game_state)
                    game_state["faint_context"] = (
                        f"場(自)={pending_ctx.get('player_field','')} | "
                        f"場(相)={pending_ctx.get('opponent_field','')}"
                    )
                else:
                    # タイムアウト: 先に単独送信してからmove_usedを処理
                    log.info("[faint統合] タイムアウト(%.1f秒) → 先にフラッシュ", elapsed)
                    self._flush_pending_faint()
                self._pending_faint_state = None
                self._pending_faint_battle_context = None
                self._pending_faint_frame = None

            log.debug("Bedrock Vision 呼び出し中...")
            if self._move_log:
                log.debug(f"[技ログ] {' / '.join(self._move_log[-5:])}")
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
            log.debug("Phi-3 実況文生成中（フォールバック）...")
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

        # ── VOICEVOX 音声合成 + 再生（非同期）──────────────────────────────────
        self._speak_async(commentary)

        # バトル終了後にアクティブフラグをリセット（Bedrock呼び出し後）
        if event_type == "battle_end":
            self._battle_active = False
            self._last_battle_end_time = time.time()
            self._pre_battle_opponent.clear()
            self._pre_battle_player.clear()
            log.info("[戦況] バトル終了 → トラッカー非アクティブ化")

        # デバッグ用スクリーンショット保存
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"pipeline_turn_{turn:03d}.png"), frame)

    def _flush_pending_faint(self) -> None:
        """タイムアウトした保留faintを単独でBedrock送信して実況する。"""
        if self._pending_faint_state is None:
            return
        game_state      = self._pending_faint_state
        battle_context  = self._pending_faint_battle_context
        frame           = self._pending_faint_frame
        self._pending_faint_state          = None
        self._pending_faint_battle_context = None
        self._pending_faint_frame          = None

        if not (self._ec2_url and self._battle_active):
            return

        t0 = time.perf_counter()
        bedrock_commentary, _ = _call_bedrock_vision(
            self._ec2_url, frame, game_state, "faint",
            self._commentary_history, battle_context, self._classifier,
            self._move_log[-5:],
        )
        if bedrock_commentary:
            commentary = _clean_commentary(bedrock_commentary)
            log.info("[faintフラッシュ] Bedrock完了 (%.2fs): 「%s」",
                     time.perf_counter() - t0, commentary)
            self._commentary_history.append(commentary)
            if len(self._commentary_history) > 5:
                self._commentary_history.pop(0)
            self._speak_async(commentary)

    def _speak_async(self, commentary: str) -> None:
        """VOICEVOX 音声合成・再生を別スレッドで実行する（メインループをブロックしない）。
        前の再生が残っていれば停止してから新しい音声を流す。
        """
        def _run() -> None:
            try:
                t0 = time.perf_counter()
                wav_bytes = self._voicevox.generate_wav(commentary)
                log.debug(f"音声合成完了 ({time.perf_counter()-t0:.2f}s): {len(wav_bytes)} bytes")
            except requests.exceptions.ConnectionError:
                log.error("VOICEVOX が起動していません。VOICEVOX を起動してください。")
                return
            except Exception as e:
                log.error(f"VOICEVOX エラー: {e}")
                return
            # 前の再生が残っていれば停止
            self._player.stop()
            try:
                t0 = time.perf_counter()
                self._player.play(wav_bytes)
                log.debug(f"再生完了 ({time.perf_counter()-t0:.2f}s)")
            except Exception as e:
                log.error(f"音声再生エラー: {e}")

        self._speech_thread = threading.Thread(target=_run, daemon=True)
        self._speech_thread.start()

    # 状態異常アイコンとして画面に表示される単語（OCRで単体トークンとして検出される）
    _STATUS_ICON_WORDS: frozenset[str] = frozenset({"まひ", "やけど", "どく", "もうどく", "ねむり", "こおり", "こんらん"})
    # per-pokemon 状態異常アイコンエリア（絶対座標 1920x1080 基準 / visualize_coords.py と同値）
    _STATUS_ICON_AREAS: dict[tuple[str, int], dict] = {
        ("opponent", 0): dict(x1=1135, x2=1215, y1=20,  y2=80),
        ("opponent", 1): dict(x1=1535, x2=1615, y1=20,  y2=80),
        ("player",   0): dict(x1=105,  x2=170,  y1=900, y2=960),
        ("player",   1): dict(x1=505,  x2=570,  y1=900, y2=960),
    }
    # 特性・道具発動メッセージエリア（絶対座標 1920x1080 基準 / visualize_coords.py と同値）
    _ABILITY_MSG_AREAS: dict[str, dict] = {
        "player": dict(x1=0,    x2=555,  y1=450, y2=570),
        "opp":    dict(x1=1365, x2=1920, y1=450, y2=570),
    }

    def _sync_status_from_ocr_bbox(self, ocr_results: list[dict], frame_h: int, frame_w: int) -> None:
        """OCRテキストの bbox 位置から状態異常アイコンを検出してトラッカーに反映する。
        YOLOモデルが状態異常アイコンを取りこぼした場合の補完として機能する。
        per-pokemon エリア（visualize_coords.py の STATUS_ICON_CHAMP と同値）でスロットを判定する。
        """
        scale_x = 1920 / frame_w
        scale_y = 1080 / frame_h
        for r in ocr_results:
            text = r["text"].strip()
            if text not in self._STATUS_ICON_WORDS:
                continue
            bbox = r.get("bbox")
            if not bbox:
                continue
            # bbox は [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 形式 → 1920x1080 スケールに正規化
            cx = sum(p[0] for p in bbox) / 4 * scale_x
            cy = sum(p[1] for p in bbox) / 4 * scale_y
            for (side, slot), area in self._STATUS_ICON_AREAS.items():
                if area["x1"] <= cx <= area["x2"] and area["y1"] <= cy <= area["y2"]:
                    log.debug(f"[状態異常OCR] {text} → {side} slot{slot} (cx={cx:.0f}, cy={cy:.0f})")
                    self._battle_tracker.update_status_from_yolo(side, text, slot)
                    break

    def _scan_ability_msg(self, ocr_results: list[dict], frame_h: int, frame_w: int) -> dict[str, str]:
        """特性・道具発動メッセージエリアのOCRテキストを収集して返す。
        Returns:
            {"player": "テキスト", "opp": "テキスト"}  存在するsideのみ含む
        """
        scale_x = 1920 / frame_w
        scale_y = 1080 / frame_h
        buckets: dict[str, list[str]] = {"player": [], "opp": []}
        for r in ocr_results:
            if r.get("confidence", 0) < 0.3:
                continue
            bbox = r.get("bbox")
            if not bbox:
                continue
            cx = sum(p[0] for p in bbox) / 4 * scale_x
            cy = sum(p[1] for p in bbox) / 4 * scale_y
            for side, area in self._ABILITY_MSG_AREAS.items():
                if area["x1"] <= cx <= area["x2"] and area["y1"] <= cy <= area["y2"]:
                    buckets[side].append(r["text"].strip())
                    break
        return {side: " ".join(tokens) for side, tokens in buckets.items() if tokens}

    def _update_switch_out(self, ocr_results: list[dict]) -> None:
        """「〜は戻っていく」「〇〇\n戻れ！」テキストを検出してポケモンを場から降ろす。

        とんぼがえり・Uターン等の交代技ではフェーズが switch_select を経由しないため、
        テキストパターンで交代を検出して on_field=False を即時反映する。
        例: "ゴリランダーは / ともの元へ / 戻っていく"
        Champions交代: "オオニューラ / (もと) / 戻れ！" → PokeClassifierで名前を特定
        """
        texts = [r["text"].strip() for r in ocr_results if r["confidence"] >= 0.3]
        has_return_text = any("戻" in t or "もど" in t for t in texts)
        if not has_return_text:
            return

        # パターン1: 「〜は戻っていく」（とんぼがえり等の交代技）
        # 例: "ゴリランダーは" → "ゴリランダー"
        for text in texts:
            if text.endswith("は") and len(text) >= 3:
                pokemon_name_candidate = text[:-1]
                found = self._battle_tracker.set_not_on_field(pokemon_name_candidate)
                if found:
                    log.info(f"[交代検知] {pokemon_name_candidate} が場から退いた（「戻っていく」テキスト検出）")

        # パターン2: 「〇〇\n戻れ！」（Champions式コマンド交代）
        # OCR結合テキストに "戻れ" が含まれる場合、MSG ROI内のトークンを縦順にスキャンして
        # PokeClassifierで前方のポケモン名を特定する（「もと」等のノイズトークンをスキップ）
        if not any("戻れ" in t for t in texts) or not self._classifier:
            return
        roi_tokens: list[tuple[float, str]] = []
        for r in ocr_results:
            if r["confidence"] < 0.3 or not r.get("bbox"):
                continue
            cx = (r["bbox"][0][0] + r["bbox"][2][0]) / 2
            cy = (r["bbox"][0][1] + r["bbox"][2][1]) / 2
            if cx < BattleMessageParser.MSG_X_MAX and BattleMessageParser.MSG_Y_MIN < cy < BattleMessageParser.MSG_Y_MAX:
                roi_tokens.append((cy, r["text"].strip()))
        roi_tokens.sort()
        for i, (_, tok_text) in enumerate(roi_tokens):
            if "戻れ" not in tok_text:
                continue
            # "戻れ" トークンの前方（最大4つ）からPokeClassifierで名前を特定
            for j in range(max(0, i - 4), i):
                candidate = roi_tokens[j][1].rstrip("！!」、")
                result = self._classifier.classify(candidate)
                if result and result.canonical_ja and result.score >= 80:
                    found = self._battle_tracker.mark_bench_by_name(result.canonical_ja)
                    if found:
                        log.info(f"[交代検知] {result.canonical_ja} が場から退いた（戻れ検出）")
                    break

    def _handle_message_event(self, ev: dict) -> None:
        """BattleMessageParser から受け取ったメッセージイベントで戦況を補完する。"""
        event_type = ev["type"]
        pokemon = ev["pokemon"]
        if event_type == "faint":
            self._battle_tracker.confirm_player_faint_by_name(pokemon)
        elif event_type == "opponent_faint":
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            self._battle_tracker.confirm_opponent_faint_by_name(canonical)
        elif event_type == "switch_in":
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            self._battle_tracker.mark_on_field_by_name(canonical)
        elif event_type == "opponent_switch_in":
            # PokeClassifierで正規化してから相手スロットに登録
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            self._battle_tracker.register_opponent_on_field(canonical)
        elif event_type == "switch_out":
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            self._battle_tracker.mark_bench_by_name(canonical)
        elif event_type == "status":
            self._battle_tracker.update_status_by_name(pokemon, ev.get("status", ""))

    def _update_move_log(self, ocr_results: list[dict], is_main_ocr: bool = False) -> None:
        """OCR 結果から「〜の → 技名」パターンを検出して _move_log に追記する。

        スキャン対象:
          1. 全 OCR トークンの隣接ペア「[X]の」→「技名」（グローバルスキャン）
          2. メッセージボックスROI の結合テキスト内の「[ポケモン名]の[技名]」（ROIスキャン）
             ROI: x < 520, 740 < cy < 930 (BattleMessageParser と同じ領域)
        is_main_ocr=True の場合: dense scan フォールバックで仮確定した相手技の後付け修正も行う。
        """
        _INVALID_POKEMON_KEYWORDS = {"相手", "あいて", "とも", "自分", "じぶん"}

        # ── 後付け修正: メインOCR時に仮確定エントリを正しい使い手で更新 ──────────
        # dense scan 時は msg ROI のみのため、_find_attacker_from_full_ocr() が空振りして
        # _get_active_opponent_name()（場の1匹目）で仮登録することがある。
        # その後のメインOCR で「ゴリランダーの」等のトークンが見つかれば上書き修正する。
        if is_main_ocr and self._tentative_opponent_moves and self._classifier:
            known_opps = {s.name for s in self._battle_tracker._opponent}
            all_toks = [(r.get("text", "").strip(), i)
                        for i, r in enumerate(ocr_results)
                        if r.get("confidence", 0) >= 0.25]
            for tidx, (tok, _) in enumerate(all_toks):
                if not tok.endswith("の"):
                    continue
                cand = tok[:-1].strip()
                if not cand or any(kw in cand for kw in _INVALID_POKEMON_KEYWORDS):
                    continue
                p_res = self._classifier.classify(_normalize_ocr_kana(cand))
                if not (p_res and p_res.category == CATEGORY_POKEMON and p_res.score >= 80):
                    continue
                canonical = p_res.canonical_ja or cand
                if canonical not in known_opps:
                    continue
                # 次トークンが技名か確認
                if tidx + 1 >= len(all_toks):
                    continue
                next_tok = all_toks[tidx + 1][0].rstrip("！!」、")
                mv_res = self._classifier.classify(_normalize_ocr_kana(next_tok))
                if not (mv_res and mv_res.category == "move" and mv_res.score >= 80):
                    continue
                move_name = mv_res.canonical_ja or next_tok
                # 仮確定エントリと突合（同ターン・同技名・異なるポケモンの場合のみ修正）
                cur_turn = str(self._battle_tracker.game_turn if self._battle_active else "?")
                for tent in self._tentative_opponent_moves[:]:
                    if (tent["move_name"] == move_name
                            and tent["turn_label"] == cur_turn
                            and tent["fallback_pokemon"] != canonical):
                        old_entry = tent["old_entry"]
                        new_entry = f"T{tent['turn_label']}:{canonical}の{move_name}"
                        for idx, e in enumerate(self._move_log):
                            if e == old_entry:
                                self._move_log[idx] = new_entry
                                log.info("[技ログ] 後付け修正: %s → %s", old_entry, new_entry)
                                break
                        self._tentative_opponent_moves.remove(tent)

        def _is_invalid_pokemon(name: str) -> bool:
            """「相手」「あいて」等を含む名前は無効（ROI結合テキストの部分マッチ誤登録を防ぐ）"""
            return any(kw in name for kw in _INVALID_POKEMON_KEYWORDS)

        def _try_register(pokemon_name: str, move_candidate: str, is_opponent: bool = False, tentative: bool = False) -> bool:
            """ポケモン名+技名候補を検証して _move_log に登録。登録したら True を返す。
            is_opponent=True の場合は update_move で相手チームのみを検索する。
            tentative=True の場合は後付け修正の対象として _tentative_opponent_moves に記録する。
            """
            pokemon_name = pokemon_name.strip().rstrip("！!」、")
            move_candidate = move_candidate.strip().rstrip("！!」、")
            if not pokemon_name or _is_invalid_pokemon(pokemon_name):
                return False
            if not move_candidate or len(move_candidate) < 3:
                return False
            if self._classifier:
                # ポケモン名を正規化（OCR揺らぎ補正: 例 イエツサン → イエッサン）
                # entry の重複チェックを確実に機能させるため、技名分類より先に実施する
                p_result = self._classifier.classify(_normalize_ocr_kana(pokemon_name))
                if p_result and p_result.category == CATEGORY_POKEMON and p_result.score >= 80:
                    pokemon_name = p_result.canonical_ja or pokemon_name
                # is_opponent=True の場合、登録済みポケモン優先だが
                # Champions ダブルバトルではスロット割当前のため known_opp に入っていないことがある。
                # 未登録でも弾かず、debug ログだけ残して続行する。
                if is_opponent:
                    known_opp = {s.name for s in self._battle_tracker._opponent}
                    if known_opp and pokemon_name not in known_opp:
                        log.debug("[技ログ] is_opponent=True 未登録ポケモン（スロット割当前の可能性）: %s", pokemon_name)
                # OCR 大文字かな誤読を補正してから分類（例: チエ→チェ, きよじゆ→きょじゅ）
                normalized = _normalize_ocr_kana(move_candidate)
                result = self._classifier.classify(normalized)
                # score < 80 は誤検出リスクが高いため除外
                # confident=False(80-90点台) でも category=move なら採用
                if result.category != "move":
                    return False
                if result.score < 80:
                    return False
                if len(result.canonical_ja) > len(move_candidate) * 1.5:
                    return False
                move_name = result.canonical_ja or move_candidate
            else:
                move_name = move_candidate
            # dense scan 起点ターンが設定されている場合はそれを優先する。
            # 交代演出中の COMMAND 誤検知で game_turn が先行して増えても、
            # 技は dense scan を起動した時点のターン番号に記録される。
            if self._dense_scan_start_turn is not None:
                turn_label = self._dense_scan_start_turn
            else:
                turn_label = self._battle_tracker.game_turn if self._battle_active else "?"
            entry = f"T{turn_label}:{pokemon_name}の{move_name}"
            # 同ターン・同技の重複を除外（OCR誤読による使用者名違いの重複登録を防ぐ）
            # 例: T7:プテラのいわなだれ が登録済みの場合、T7:プーラのいわなだれ は登録しない
            # ⚠️ ダブルバトルで同一ターンに2匹が同じ技を使う場合も除外される
            if any(
                e.startswith(f"T{turn_label}:") and e.endswith(f"の{move_name}")
                for e in self._move_log
            ):
                log.debug("[技ログ] 同ターン同技の重複スキップ: %s", entry)
                return False
            if entry not in self._move_log[-3:]:
                self._move_log.append(entry)
                if len(self._move_log) > self._MAX_MOVE_LOG:
                    self._move_log.pop(0)
                log.info("[技ログ] 検出: %s%s", entry, "（仮確定）" if tentative else "")
                if tentative:
                    self._tentative_opponent_moves.append({
                        "old_entry": entry,
                        "move_name": move_name,
                        "turn_label": str(turn_label),
                        "fallback_pokemon": pokemon_name,
                    })
                if self._battle_active:
                    self._battle_tracker.update_move(pokemon_name, move_name, is_opponent=is_opponent)
                    self._battle_tracker.apply_status_from_move(pokemon_name, move_name)
                    # 技検出成功時: 次の技メッセージを取りこぼさないよう dense scan を最低 90 フレーム維持。
                    # ただし move_used で 9999 にセット済みの場合は縮小しない（max で保護）。
                    if self._dense_scan_remaining < 90:
                        self._dense_scan_remaining = max(self._dense_scan_remaining, 90)
                        log.debug("[密集OCR] 技検出 → dense scan 最低 90 フレーム維持")
                return True
            return False

        def _get_active_opponent_name() -> str | None:
            """場に出ている相手ポケモン名を返す（複数いる場合は1匹目）。
            「相手の[技]」のように相手名がROI外でトークン未取得の場合の代替用。
            """
            on_field_o = [p for p in self._battle_tracker._opponent
                          if p.on_field and not p.fainted]
            return on_field_o[0].name if on_field_o else None

        def _find_attacker_from_full_ocr() -> str | None:
            """全OCR結果（ROI外含む）から「Xの」形式の相手ポケモン名トークンを探す。
            「相手のザマゼンタのきょじゅうだん」のように相手名がROI外に出るケースに対応。
            dense scan 時は msg ROI のみのため、キャッシュしたメインOCR結果も検索する。
            known_opponents に一致すれば即採用。未一致でも有効なポケモン名なら投機的候補として返す。
            上部HPバー等の誤マッチを防ぐためメッセージボックス付近（cy > 600）のみ対象。
            """
            if not self._classifier:
                return None
            known_opponents = {s.name for s in self._battle_tracker._opponent}
            # メッセージボックス直上エリアのみ対象（cy > 600）。
            # 上部HPバー（y≈50-200）やUI中段（y≈200-600）の誤マッチを防ぐ。
            # メッセージ付近（y≈600-740）に出るポケモン名トークンのみを拾う。
            _MSG_AREA_Y_MIN = BattleMessageParser.MSG_Y_MIN - 140  # 600
            # dense scan 時は ocr_results が msg ROI のみ → メインOCRキャッシュも合わせて検索する
            search_sources = ocr_results if ocr_results is not self._last_full_ocr_results else ocr_results
            if self._last_full_ocr_results and self._last_full_ocr_results is not ocr_results:
                search_sources = list(ocr_results) + self._last_full_ocr_results
            speculative: str | None = None  # known_opponents 未一致でも有効なポケモン名
            for r in search_sources:
                if r["confidence"] < 0.25:
                    continue
                bbox = r.get("bbox", [])
                if not bbox:
                    continue
                cy_r = (bbox[0][1] + bbox[2][1]) / 2
                if cy_r < _MSG_AREA_Y_MIN:
                    continue  # 上部HPバー・ポケモン名表示エリアは除外
                text = r.get("text", "").strip()
                # 「Xの」（所有格）または「Xは」（主語助詞）で終わるトークンを使い手候補とする。
                # 例: 「バドレックスの」→ バドレックス / 「イエツサンは」→ イエッサン
                if text.endswith("の") or text.endswith("は"):
                    candidate = text[:-1].strip()
                else:
                    continue
                if not candidate or _is_invalid_pokemon(candidate):
                    continue
                result = self._classifier.classify(_normalize_ocr_kana(candidate))
                if (result
                        and result.category == CATEGORY_POKEMON
                        and result.score >= 80):
                    canonical = result.canonical_ja or candidate
                    if canonical in known_opponents:
                        return canonical  # 確定: known_opponents に一致
                    if speculative is None:
                        speculative = canonical  # 投機的候補（スロット割当前の可能性）
                        log.debug("[技ログ] _find_attacker 投機的候補: %s（未known）", canonical)
            if speculative:
                log.debug("[技ログ] _find_attacker 投機的候補を採用: %s", speculative)
            return speculative

        def _try_register_opponent_attack(move_cand: str) -> bool:
            """「相手の[move_cand]」パターンの技登録。
            move_cand が「ポケモン略称+技名」の連結OCR誤読の場合は分割して正しい使い手で登録。
            例: 「イエてだすけ」→ イエッサンのてだすけ として登録。
            通常ケースは _find_attacker_from_full_ocr → _get_active_opponent_name の順で使い手を特定。
            """
            cleaned_cand = move_cand.rstrip("！!」、")
            if not cleaned_cand:
                return False
            # ── 先頭2文字がポケモン略称 + 残りが有効な技 → 分割登録 ────────────
            # 「イエてだすけ」= イエッサン略称 + てだすけ の連結誤読を修正する。
            # remainder が有効な技かを事前確認し、有効なら split パスで完結する。
            # （重複登録で _try_register が False を返した場合も通常ルートには進まない）
            # ブリザードランスのように「ブリ」=ブリムオン略称で誤判定されても
            # 「ザードランス」が技として無効なら通常ルートへフォールスルーする。
            if self._classifier and len(cleaned_cand) >= 4:
                prefix2 = cleaned_cand[:2]
                # OCR 清音→濁音誤読を考慮した候補を生成。
                # 例: 「バト」→「バド」（バドレックス の OCR 誤読）
                # ト→ド, テ→デ はポケモン名頭文字の混同で最頻出。
                _DAKUTEN = {"ト": "ド", "テ": "デ", "ツ": "ズ"}
                prefix_variants = [prefix2]
                if len(prefix2) == 2 and prefix2[1] in _DAKUTEN:
                    prefix_variants.append(prefix2[0] + _DAKUTEN[prefix2[1]])
                known_opp_set = {s.name for s in self._battle_tracker._opponent}
                for pfx in prefix_variants:
                    p_pfx = self._classifier.classify(_normalize_ocr_kana(pfx))
                    if not (p_pfx
                            and p_pfx.category == CATEGORY_POKEMON
                            and p_pfx.score >= 80):
                        continue
                    canonical = p_pfx.canonical_ja or pfx
                    # Classifierが同スコアで別ポケモン（例: バド→バンバドロ）を返した場合、
                    # known_opponents に pfx で始まる実際の対戦ポケモンがいればそちらを優先する。
                    if canonical not in known_opp_set:
                        alt = next((n for n in known_opp_set if n.startswith(pfx)), None)
                        if not alt:
                            continue
                        canonical = alt
                    remainder = cleaned_cand[2:].lstrip()
                    if len(remainder) >= 3:
                        rem_result = self._classifier.classify(_normalize_ocr_kana(remainder))
                        if rem_result and rem_result.category == "move" and rem_result.score >= 80:
                            # 分割パスが有効 → 登録（重複でも通常ルートには進まない）
                            _try_register(canonical, remainder, is_opponent=True)
                            return True
                    # remainder が技でない → 通常ルートへフォールスルー
                    break
            # ── 通常ルート: 全OCR → active opponent の順で使い手を特定 ────────
            # 同ターン・同技がすでに別ポケモンで登録済みなら重複スキップ。
            # 例: 「イエッサンのてだすけ」登録済み → 「相手のてだすけ」でザマゼンタ重複を防ぐ。
            if self._classifier:
                mv_check = self._classifier.classify(_normalize_ocr_kana(cleaned_cand))
                if mv_check and mv_check.category == "move" and mv_check.score >= 80:
                    canonical_move = mv_check.canonical_ja or cleaned_cand
                    turn_label = self._battle_tracker.game_turn if self._battle_active else "?"
                    for recent in self._move_log[-6:]:
                        if (recent.startswith(f"T{turn_label}:")
                                and recent.endswith(f"の{canonical_move}")):
                            log.debug("[技ログ] 同ターン同技が登録済みのためスキップ: %s", canonical_move)
                            return False
            attacker = _find_attacker_from_full_ocr()
            if attacker:
                return _try_register(attacker, cleaned_cand, is_opponent=True)
            # フォールバック: 場の1匹目を使い手とする（仮確定・後付け修正の対象）
            fallback = _get_active_opponent_name()
            if fallback:
                return _try_register(fallback, cleaned_cand, is_opponent=True, tentative=True)
            return False

        # ── メッセージボックスROI トークン収集 ──────────────────────────────────
        # スキャン1・2・2フォールバック共通: MSG ROI 内のトークンのみを対象とする。
        # ROI外（HPバー・コマンドUI等）のOCR誤読による誤検出を防ぐ。
        msg_items: list[tuple[float, float, str]] = []
        for r in ocr_results:
            if r["confidence"] < 0.35:
                continue
            bbox = r.get("bbox", [])
            if not bbox:
                continue
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            if (cx < BattleMessageParser.MSG_X_MAX
                    and BattleMessageParser.MSG_Y_MIN < cy < BattleMessageParser.MSG_Y_MAX):
                msg_items.append((cy, cx, r["text"].strip()))
        if msg_items:
            msg_items.sort(key=lambda t: (round(t[0] / 40), t[1]))

        # ── スキャン1: メッセージROIトークンのペアスキャン「[X]の/は」→「技名」 ──────
        # 「の」エンディング: 直後トークンを技名候補として試みる
        #   例: 「バドレックスの」→「ブリザードランス」
        # 「は」エンディング: 後続 _HA_SCAN_WINDOW 個のトークンを技名候補として試みる
        #   例: 「イエツサンは」→「ザマゼンタを」「てだす」… 「手助けする」
        #       → _MOVE_ALIAS_MAP で「てだすけ」に変換 → PokeClassifier が move と判定して登録
        #   例: 「バドレックスは」→「まもるを」→ 末尾助詞除去で「まもる」→ move 判定して登録
        # どちらも PokeClassifier (score >= 80, category == move) が誤登録を防ぐ。
        # ※ ROI外のUI要素（HPバー・コマンド選択等）の誤読によるノイズをROI絞り込みで排除。
        _HA_SCAN_WINDOW = 4
        texts = [t[2] for t in msg_items if t[2]]  # msg_items から confidence>=0.35 のトークン
        for i, text in enumerate(texts):
            is_opp = i > 0 and any(kw in texts[i - 1] for kw in {"相手", "あいて"})
            if text.endswith("の"):
                pokemon_name = text[:-1]
                if _is_invalid_pokemon(pokemon_name):
                    # 「相手の」→ _try_register_opponent_attack で使い手特定+分割誤読修正
                    if ("相手" in pokemon_name or "あいて" in pokemon_name) and i + 1 < len(texts):
                        _try_register_opponent_attack(texts[i + 1])
                elif i + 1 < len(texts):
                    _try_register(pokemon_name, texts[i + 1], is_opponent=is_opp)
            elif text.endswith("は"):
                pokemon_name = text[:-1]
                if _is_invalid_pokemon(pokemon_name):
                    continue
                for j in range(i + 1, min(i + 1 + _HA_SCAN_WINDOW, len(texts))):
                    # _MOVE_ALIAS_MAP で変形表記を先に正規化（例: 「手助けする」→「てだすけ」）
                    candidate = _MOVE_ALIAS_MAP.get(texts[j], texts[j])
                    # 末尾の助詞・句読点を除去（例: 「まもるを」→「まもる」）
                    candidate = candidate.rstrip("をにはがもでて！!」、")
                    _try_register(pokemon_name, candidate, is_opponent=is_opp)

        # ── スキャン2: メッセージボックスROI の結合テキスト ─────────────────
        # BattleMessageParser と同じ ROI からトークンを収集・結合し、
        # 「[ポケモン名]の[技名]」パターンを正規表現で抽出する。
        # スキャン1で拾えなかった「の」入り単一トークンや前後に文字が混じるケースを補完。
        if msg_items:
            msg_text = "".join(t[2] for t in msg_items)
            if msg_text.strip():
                log.info("[OCR/メッセージ] %s", msg_text[:120])
            for m in _MOVE_IN_MSG_RE.finditer(msg_text):
                poke_name = m.group(1)
                if _is_invalid_pokemon(poke_name):
                    # 「相手の[技]」→ _try_register_opponent_attack で使い手特定+分割誤読修正
                    if "相手" in poke_name or "あいて" in poke_name:
                        _try_register_opponent_attack(m.group(2))
                else:
                    _try_register(poke_name, m.group(2))

            # ── スキャン2フォールバック: 「の」欠落対応・技名直接検索 ──────────
            # OCRが「ミライドンの」を「ミライドン」と読んだ場合に対応。
            # メッセージROIのトークンを個別に技名判定し、直前トークンをポケモン名として登録する。
            # メッセージROI内のトークンのみを対象とするため誤検出リスクが低く、
            # dense scanの状態に依存しないため技出現タイミングを問わず動作する。
            if self._classifier:
                for i, (_cy, _cx, token) in enumerate(msg_items):
                    cleaned = token.rstrip("！!」、")
                    if len(cleaned) < 3:
                        continue
                    move_result = self._classifier.classify(_normalize_ocr_kana(cleaned))
                    if not (move_result
                            and move_result.category == "move"
                            and move_result.score >= 80):
                        continue
                    if i == 0:
                        continue
                    orig_prev = msg_items[i - 1][2]
                    if orig_prev.endswith("を"):
                        # 「ザマゼンタを」= 技の対象（受け手）であり使い手ではない → スキップ
                        # 例: 「イエッサンはザマゼンタをてだすけ」の「ザマゼンタを」は対象
                        continue
                    prev_text = orig_prev.rstrip("のは！!」、")
                    if len(prev_text) < 2:
                        continue
                    p_result = self._classifier.classify(_normalize_ocr_kana(prev_text))
                    if not (p_result
                            and p_result.category == CATEGORY_POKEMON
                            and p_result.score >= 80):
                        # 「相手」等の無効トークン → _try_register_opponent_attack で使い手特定
                        if "相手" in prev_text or "あいて" in prev_text:
                            _try_register_opponent_attack(cleaned)
                        continue
                    if len(prev_text) < 3:
                        # 2文字トークン（例: 「イエ」）は canonical_ja の先頭一致も必須。
                        # 「ザー」→「リザード」のような後方一致の誤マッチを防ぐ。
                        canonical = p_result.canonical_ja or ""
                        if not canonical.startswith(prev_text):
                            continue
                    is_opp = i > 1 and any(
                        kw in msg_items[i - 2][2] for kw in {"相手", "あいて"}
                    )
                    _try_register(prev_text, cleaned, is_opponent=is_opp)


# ─── エントリポイント ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ポケモン対戦実況AI パイプライン（Sprint 5）")
    parser.add_argument("--camera",  type=int,   default=3,
                        help="OBS仮想カメラのデバイス番号（デフォルト: 3）")
    parser.add_argument("--input",   default=None,
                        help="動画ファイルのパス（指定時はカメラの代わりに動画を使用）")
    parser.add_argument("--model",   default=None,
                        help="YOLOv8 カスタムモデルのパス（状態異常検出用・例: runs/detect/train4/weights/best.pt）")
    parser.add_argument("--ball-model", default=None,
                        help="ボール検出専用モデルのパス（例: runs/detect/train7/weights/best.pt）")
    parser.add_argument("--end-model", default=None,
                        help="終了画面検出モデルのパス（例: runs/detect/train_end_screen2/weights/best.pt）")
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
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="動画解析時のサンプリングレート（fps、デフォルト: 2.0）"
                             " ─ 高いほど技名取りこぼしが減るがCPU負荷増")

    args = parser.parse_args()

    pipeline = Pipeline(
        camera_index=args.camera,
        model_path=args.model,
        ball_model_path=args.ball_model,
        end_model_path=args.end_model,
        interval=args.interval,
        speaker=args.speaker,
        gpu=not args.cpu,
        conf=args.conf,
        ec2_url=args.ec2_url,
        audio_device=args.audio_device,
        video_path=args.input,
        video_sample_fps=args.video_fps,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
