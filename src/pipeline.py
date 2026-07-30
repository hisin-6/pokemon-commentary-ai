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
import random
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

from src.capture.hpbar_analyzer import HpBarAnalyzer, slot_bar_centers
from src.capture.screen_capture import DiffDetector, init_reader, run_ocr
from src.capture.yolo_detector import BattleState, YoloDetector
from src.commentary.phi3_client import Phi3Client
from src.output.audio_player import AudioPlayer
from src.output.render_sink import RenderSink
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

# 自分ネームプレート帯（y > この値のみ自分側ポケモン名として採用・1080p 基準）。
# 診断JSONL実測（2026-07-08・2動画）: 自分ネームプレートは cy 950-999 に集中、
# cy 800-899 はメッセージボックス、cy 500-699 は選出画面/相手を見るパネル、
# cy 900-949 は0件（自然な境界）。メッセージ内の相手名（「ガブリアスを 繰り出した」
# 等が最大10秒表示）が自分側名前候補に毎サイクル蓄積され、新規登録ヒステリシス
# （2サイクル連続目撃）を貫通して相手ポケモンが自分側に混入する主因だった
_PLAYER_NAME_Y_MIN = 930

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

# 漢字のみのUIラベル（中国語ポケモン名フォールバックの除外リスト）
# 相手エリアに出る「かなを含まない漢字テキスト」は中国語名として素通しされるため、
# 日本語UIの漢字ラベルを明示的に除外する（実機で「能力」が幽霊登録された）
_CJK_UI_WORDS = {
    "能力", "特性", "持ち物", "状態", "急所", "選出", "交代", "通信", "待機",
    "説明", "戻", "勝負", "降参", "接続", "対戦", "観戦", "設定", "確認",
    "決定", "中止", "終了", "順番", "変更", "相手", "自分", "味方", "技",
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
                    # 日本語UIの漢字ラベルがこのフォールバックを通過して幽霊登録される
                    # （実機で「能力」が相手ポケモンとして登録され場スロットまで占有した）ため除外
                    if has_cjk and not has_kana and text not in _CJK_UI_WORDS:
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
        elif center_y > _PLAYER_NAME_Y_MIN:
            name_player_with_xy.append((canonical, center_x, center_y))
        # 500 <= cy <= 930 の中間帯（メッセージ/選出/パネル）はどちらの候補にもしない

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
        "hp_player_with_xy":    hp_player_with_xy[:4],    # (hp, cx, cy) 位置ゲート用
        "hp_opponent_with_xy":  hp_opponent_with_xy[:4],
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
    # 通信フェーズ入場に必要な連続検出秒数（単発誤検出・0.1〜0.3秒の微小ブリップの排除）。
    # フレーム数ベース（旧_COMM_ENTRY_FRAMES=2）だとサンプリングレート依存になり、
    # 30fpsリプレイでは0.07秒相当・本番1Hzでは2秒相当と挙動が変わってしまうため秒ベースに統一。
    # 本番1Hzでは連続2フレーム目（1.0秒）で確定し、旧実装と同じ入場遅延になる。
    _COMM_ENTRY_SEC      = 0.7
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
    # 閾値は動画内時間（=本番実時間）で実測して決定（診断JSONL 7本・30fps全数OCR）:
    #   セッション内のコマンド表示途切れ最大 5.7秒 < 10 < ターン間ギャップ最小 18.1秒
    #   turn_start間隔の最小 19.5秒 > 15
    # （旧値120/45秒はOCR処理時間で約11倍間延びした軸での実測に基づいており、
    #   真の時間軸ではターン間ギャップより大きく、脱出弁がほぼ開かなかった）
    _CMD_SESSION_GAP_SEC   = 10.0  # コマンド画面がこの秒数以上消えていたら新セッション
    _TURN_START_ESCAPE_SEC = 15.0  # 前回turn_start確定からこの秒数経過で脱出弁が開く
    # faint再アーム: 「0/211」等は相手を見るパネルで最大210秒表示され続ける（実測）。
    # 表示がこの秒数以上途切れてから再出現した場合のみ新しいfaintイベントとして扱う。
    _FAINT_REARM_SEC = 20.0

    def __init__(self, debounce_seconds: float = 10.0, clock=None):
        self._debounce = debounce_seconds
        # 時計注入: 動画検証では clock=（動画内時間を返す関数）を渡すことで、
        # デバウンス・脱出弁45s・セッション区切り120s等の全時間閾値が
        # 動画内時間で動き、本番（実時間=カメラ入力）と挙動が一致する。
        # デフォルトは実時間（ライブカメラ運用）。
        self._clock = clock if clock is not None else time.time
        self._last_event_time: dict[str, float] = {}
        self._prev_phase = "unknown"
        self._battle_started = False
        self._is_processing = False
        # move_usedまたはbattle_startの後にのみturn_startを許可するフラグ。
        # ダブルバトルで1匹目→2匹目コマンド選択中の余分なturn_startを
        # debounceに依存せず完全にブロックする。
        self._allow_turn_start = False
        # 通信フェーズ平滑化（入場確認・退出猶予）
        # 時刻の初期値は -inf: 動画内時間は0付近から始まるため、初期値0.0だと
        # 「最後に見てから十分経過した」系の判定が動画冒頭で誤って偽になる。
        self._comm_streak_start: float | None = None  # communication 連続検出の開始時刻
        self._comm_active = False                     # 確定済み通信フェーズ中か
        self._last_comm_seen = float("-inf")          # 最後に communication を検出した時刻
        # turn_start脱出弁・faint再アーム用の追跡
        self._last_cmd_seen = float("-inf")           # 最後に command_select を見た時刻
        self._cmd_session_start: float | None = None  # 現在のコマンドセッションの開始時刻
        self._last_turn_start_fired = float("-inf")   # 最後に turn_start / battle_start が確定した時刻
        self._last_faint_seen = float("-inf")         # 最後に faint フェーズを見た時刻

    def set_processing(self, v: bool) -> None:
        self._is_processing = v

    def reset_after_processing(self, event_type: str | None = None) -> None:
        """処理完了後にフェーズ履歴をリセットし、直後の誤発火を防ぐ。
        処理中に _prev_phase が command_select で止まっていると、
        処理完了直後のフレームで command_select → unknown 遷移として
        move_used が即再発火する問題を防ぐ。
        """
        self._prev_phase = "unknown"
        now = self._clock()
        # move_used デバウンスを現在時刻に更新（処理完了直後の再発火を抑止）
        self._last_event_time["move_used"] = now
        # 注: 以前はfaint/move_used後にturn_startデバウンスも押し込んでいたが、
        # 真の時間軸ではfaint直後の本物のコマンド画面（最短6秒後・実測）を握りつぶして
        # ターン欠落の原因になっていたため撤去。turn_startの多重発火は
        # _allow_turn_start フラグと15秒デバウンスで引き続き防がれる
        # （診断JSONL 7本のリプレイで過剰カウントなしを確認済み）。

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
        入場は連続 _COMM_ENTRY_SEC 秒の検出で確定（単発誤検出・微小ブリップの排除）、
        退場は _COMM_EXIT_GRACE_SEC の猶予付き（取りこぼしによる move_used 多重発火の防止）。
        退場確定フレームでは実際の画面種別に関わらず "unknown" を返す:
        communication→command_select と直接遷移すると turn_start の prev 条件に
        引っかかって move_used 後の turn_start が永久に発火できなくなるため、
        必ず communication→unknown→(次フレームで実フェーズ) の順に遷移させる。
        """
        if raw == "battle_end":
            self._comm_streak_start = None
            self._comm_active = False
            return raw
        now = self._clock()
        if raw == "communication":
            if self._comm_streak_start is None:
                self._comm_streak_start = now
            self._last_comm_seen = now
            if not self._comm_active and now - self._comm_streak_start >= self._COMM_ENTRY_SEC:
                self._comm_active = True
            return "communication" if self._comm_active else "unknown"
        self._comm_streak_start = None
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
        now = self._clock()

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
                self._last_event_time["turn_start"] = now
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
            self._last_event_time["turn_start"] = now

        # 脱出弁: move_used取りこぼしで_allow_turn_start=Falseのまま固着した場合の回復措置。
        # 新しいコマンドセッションの2フレーム目以降（開始10秒以内）かつ前回turn_start確定から
        # _TURN_START_ESCAPE_SEC 以上経過していれば、遷移条件・デバウンスを無視して発火する。
        # （通常経路がデバウンスで握りつぶされた場合も次フレームでここが拾う）
        escape = False
        if (event is None and curr == "command_select" and self._battle_started
                and self._cmd_session_start is not None
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
            # デフォルト -inf: 時計の原点が0付近（動画内時間）でも初回イベントが誤デバウンスされない
            last = self._last_event_time.get(event, float("-inf"))
            if event not in no_debounce and not escape and now - last < debounce:
                log.debug(f"デバウンス中のためスキップ: {event} (残り {debounce-(now-last):.1f}s)")
                return None
            self._last_event_time[event] = now
            if event == "move_used":
                # move_used確定 → 次のコマンド選択でturn_startを許可。
                # 注: 以前はここでturn_startデバウンスも押し込んでいたが、本物のコマンド画面は
                # 通信フェーズ終了の数秒後（最短0.1秒・実測）に来るため、15秒デバウンスが
                # 直後の正当なturn_startを握りつぶしてターン欠落の主因になっていた（撤去済み）。
                self._allow_turn_start = True
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
    # チャンピオンズ対応: メッセージが画面中央まで広がるため 520→900 に拡張（2026-04-18）。
    # 900→1150（2026-07-30）: 「(トレーナー名)は (称号) (ポケモン名)を 繰り出した!」形式
    # （称号はゲーム内の正規表示・OCR誤読ではない）で称号の分だけ末尾が右へ伸び、
    # 「繰り出した!」がROI外（cx>900）に切り捨てられて交代検出が丸ごと失敗していた
    # （実機: 07-00-19のガブリアス消失バグの真因）。診断JSONL6本の実データで
    # 「繰り出した!」の最大出現cxは1129だったため、余裕を見て1150に拡張。
    # 6本全数リプレイで新規イベント9件（すべて称号付き交代の正しい検出）・
    # 誤検出0件・既存イベントの欠落0件を確認済み（コマンド選択画面の技名・
    # 効果テキストは同y帯に存在するが、メッセージ表示中のフレームとは
    # 排他的なため実害なし）。
    MSG_X_MAX  = 1150
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
    # 末尾見切れ対応: メッセージROIの下端で「〜を 繰」までしか読めないフレームが
    # 継続するケースがある（実機: 「rixohは ランクマスター ガブリアスを 繰」が
    # 10秒間一度も完全形にならず相手ガブリアスが未登録のままfaintも帰属失敗）。
    # 「繰」+末尾ノイズ最大3文字で文末なら繰り出しとみなす
    _KURIDASHITA_SFX = r'(?:く[りゆ]だした|繰[りゆ]出した|繰[\s\S]{0,3}$)'
    _OPPONENT_SWITCH_IN_RE = re.compile(
        r'(?:[^\s]{2,12}と\s*)?(?:[^\s]*は\s*)?([^\s]{2,12}?)を\s*' + _KURIDASHITA_SFX
    )
    # ダブルバトル「AとBをくりだした」形式でAとBを両方捕捉する専用RE
    _DUAL_OPPONENT_SWITCH_IN_RE = re.compile(
        r'([^\s]{2,12}?)と\s*([^\s]{2,12}?)を\s*' + _KURIDASHITA_SFX
    )
    _SWITCH_OUT_RE = re.compile(
        r'もどれ[、,]\s*(.{2,12})'          # SV: もどれ、〇〇（ひらがな）
        r'|(.{2,12})と\s*こうたいした'       # 交代技: 〇〇とこうたいした
        r'|(\S{2,12})\s*戻れ'               # Champions: 〇〇\n戻れ！（漢字）
        r'|([^\s]{2,12})を\s*引っ?[こ込]めた'  # 相手: 「(名前)は 〇〇を 引っこめた！」
        # 相手の引っ込めを取りこぼすと、そのポケモンが on_field のまま古い
        # slot_index を保持し続け、交代で入った別ポケモンと互いのHPバーを
        # 読み合う完全反転が起きる（実機で確認: リザードン⇔ガブリアス）
    )
    # 状態異常メッセージ: 「〇〇は まひじょうたいになった」等。
    # Champions は漢字形式（「眠ってしまった」「凍りついた」等）でも表示される。
    # 漢字状態語の前の [ぁ-んー\s]{0,4}? は OCR 断片ノイズの挟まり対策
    # （実機: 「ペリッパーは ねむ 眠ってしまった!」の「ねむ」で不成立だった）。
    # ⚠️ひらがな状態語側にはノイズスキップを入れないこと:
    # 「効果は いまひとつだ」の「(い)まひ」を拾う大量誤爆になる（実ログ53本で確認）
    _STATUS_RE = re.compile(
        r'(.{2,12})(?:は|が)\s*'
        r'(?:(まひ|やけど|もうどく|どく|こおり|ねむり)'
        r'|[ぁ-んー\s]{0,4}?(眠って|眠り|凍りつ|凍って|麻痺|猛毒|毒))'
        r'\s*(?:じょうたい|状態)?'
    )
    # 漢字形式 → トラッカー正規形（ひらがな）への変換
    _STATUS_KANJI_MAP = {
        "眠って": "ねむり", "眠り": "ねむり",
        "凍りつ": "こおり", "凍って": "こおり",
        "麻痺": "まひ", "猛毒": "もうどく", "毒": "どく",
    }

    def __init__(self, clock=None) -> None:
        self._seen: dict[tuple[str, str], float] = {}
        # 動画モードでは time.time() が実処理時間(OCR時間)を返し、動画内時間から
        # 大きく乖離するため（BattlePhaseClassifierと同じ問題）、clock注入に対応する。
        self._clock = clock or time.time

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
        now = self._clock()
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
                is_opponent = bool(prefix.strip())
                if not is_opponent:
                    # 「相手の」の崩れ読み対策: 「あい 手の イトウは たおれたー」のように
                    # プレフィックスが分断されると自分側と誤判定され、同名ミラー戦で
                    # 生存中の自分ポケモンを誤ひんし化する（実機: 自分イダイトウ139/201）。
                    # 名前の直前に相手プレフィックスの痕跡があれば相手側として扱う
                    head = text[:m.start(2)][-10:]
                    if re.search(r'(?:あい\s*て?|相?手\s*の?)\s*$', head):
                        is_opponent = True
                if is_opponent:
                    _emit("opponent_faint", name)
                else:
                    # 同名ミラー対策: 直近で相手側として発火済みの名前（OCR欠けの
                    # 部分一致含む）はプレフィックス取りこぼしの再読とみなしスキップ
                    stripped = name.strip().rstrip('！!」、')
                    dup = any(et == "opponent_faint"
                              and now - t < self.DEDUP_TTL
                              and (nm in stripped or stripped in nm)
                              for (et, nm), t in self._seen.items())
                    if not dup:
                        _emit("faint", name)

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
                if m.group(4):
                    # 「(トレーナー名)は 〇〇を 引っこめた」は相手の交代のみ。
                    # 同名ミラー戦で自分側を誤ベンチ化しないようサイドを分けて発行する
                    # （実機: 「rixohは オオニューラを 引っこめた」で自分のオオニューラが
                    #   ベンチ化し、相手のオオニューラは場に残留した）
                    _emit("opponent_switch_out", m.group(4))
                else:
                    # もどれ、〇〇 / 〇〇 戻れ！ / 〇〇と こうたいした = 自分側の交代
                    _emit("switch_out",
                          (m.group(1) or m.group(2) or m.group(3) or ""))

        # 状態異常はROI外も含めた全OCRから検索（メッセージ表示フレームを取りこぼす場合に備える）
        full_text = " ".join(r["text"] for r in ocr_results if r["confidence"] >= 0.35)
        status_text = f"{text} {full_text}" if text else full_text
        m = self._STATUS_RE.search(status_text)
        if m:
            pokemon_name = m.group(1).strip().rstrip('！!」、')
            status = m.group(2) or self._STATUS_KANJI_MAP.get(m.group(3), m.group(3))
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
    hp_turn: int = -1                             # hp を最後に更新した内部ターン（鮮度比較用）
    hp_pct_pixel: float | None = None            # ピクセル解析によるHP% (0.0-1.0)
    hp_px_turn: int = -1                          # hp_pct_pixel を最後に更新した内部ターン
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
    MAX_TURN_HISTORY = 8   # ターン毎スナップショットの保持数（turn_history送信用）
    # on_field=True でこの「ゲームターン数」以上不検出なら場にいないと判断。
    # ⚠️ last_seen_turn は self.game_turn（実ターン数）で管理すること。
    # 内部イベントカウンター self.turn を使うと、メガシンカ・道具発動・毒ダメージ等
    # メッセージが立て込む区間で同一ターン内に update() が何度も呼ばれ、実際は
    # 1ターンも経っていないのに閾値を超えて誤って場から降ろしてしまう（実機:
    # 07-00-19でオオニューラ・イダイトウが同時に誤って場から降ろされ、片方は
    # 試合終了まで復帰しなかった）。
    _ON_FIELD_MISS_THRESHOLD = 3
    _HP_RE = re.compile(r'(\d{1,3})/(\d{1,3})')
    # 画面中央x座標: これより左がスロット0（左）、右がスロット1（右）
    _SLOT_X_CENTER = 960
    # HPバー中心x座標。hpbar_analyzer.slot_bar_centers() で _DEFAULT_SLOTS から算出する
    # （ハードコードだとROI再キャリブレーション時に手計算し忘れて静かにズレるため）。
    # ネームプレートは各バーの直上に表示され、cx はバー中心から±140px以内に収まる
    # （診断JSONL実測: 自分側 cx 200-299/600-699・相手側 cx 1200-1299/1600-1699 の
    #   2クラスタがバー中心 292/688・1336/1732 に対応）
    _SLOT_BAR_CENTERS = slot_bar_centers()
    # 最寄りバー中心からこの距離を超える cx はネームプレート由来ではない
    # （選出リスト cx≈250 が相手側スロットを誤取得するのを防ぐ）
    _SLOT_CX_TOLERANCE = 200

    @staticmethod
    def _fuzzy_name_match(a: str, b: str) -> bool:
        """OCR揺らぎ（前方一致・末尾見切れ）を許容したポケモン名の同一性判定。
        完全一致 or どちらかがもう一方の部分文字列なら同一個体とみなす。
        """
        return a == b or a in b or b in a

    def __init__(self):
        self.turn = 0       # 内部イベントカウンター（_ON_FIELD_MISS_THRESHOLD 用）
        self.game_turn = 0  # 実際のゲームターン数（command_select 出現ごとに +1）
        self._player:   list[FieldPokemon] = []  # 自分の最大4匹
        self._opponent: list[FieldPokemon] = []  # 相手の最大4匹
        self._event_log: list[str] = []
        self._turn_history: list[str] = []  # ターン毎の場の状態スナップショット（turn_history送信用）
        # 低信頼経路（定期OCR）の新規登録ヒステリシス: (側, 名前) → (目撃数, 最終目撃turn)
        self._pending_new: dict[tuple[str, str], tuple[int, int]] = {}
        # スロット占有者の追跡（key例: "player_0"）と交代時コールバック
        # （HpBarAnalyzerの安定化状態リセット用。PipelineRunnerが接続する）
        self._slot_occupant: dict[str, str] = {}
        self.slot_reset_cb = None  # Callable[[str], None] | None
        # ボール数トラッキング（気絶推定・控え不明表示用）
        self._prev_opponent_alive: int | None = None  # 前ターンの相手生存数
        self._player_alive_count:  int | None = None  # 最新の自分生存数
        self._opponent_alive_count: int | None = None # 最新の相手生存数
        # 定期OCR数値HPの確定ヒステリシス: (側, スロット番号) → (読み値, 連続観測数)
        self._pending_ocr_hp: dict[tuple[str, int], tuple[str, int]] = {}

    # ── 内部ヘルパー ─────────────────────────────────────────────────────────

    # 前方一致吸収の最小文字数: 2文字名（ピィ等）同士の偶発一致を避ける
    _ABSORB_MIN_LEN = 3
    # 低信頼経路（定期OCR）の新規登録ヒステリシス:
    # 1フレームのOCRノイズがPokeClassifierで実在名に化けて登録される幽霊
    # （ラン→トランセル等・実機で頻発）を防ぐため、複数サイクルの連続目撃を要求する
    _NEW_NAME_CONFIRM_COUNT = 2   # 新規登録に必要な目撃サイクル数
    _NEW_NAME_PENDING_TTL   = 8   # この内部ターン数以上空いたら目撃カウントを失効

    def _confirm_new_name(self, side: str, name: str) -> bool:
        """低信頼経路の新規名を _NEW_NAME_CONFIRM_COUNT 回目撃するまで保留する。"""
        cnt, last = self._pending_new.get((side, name), (0, -999))
        if self.turn - last > self._NEW_NAME_PENDING_TTL:
            cnt = 0
        cnt += 1
        self._pending_new[(side, name)] = (cnt, self.turn)
        if cnt < self._NEW_NAME_CONFIRM_COUNT:
            log.info(f"[戦況] 新規名 {name}（{side}）を保留（目撃{cnt}回・{self._NEW_NAME_CONFIRM_COUNT}回で登録）")
            return False
        return True

    def _get_or_create(self, slots: list[FieldPokemon], name: str,
                       low_trust: bool = False) -> FieldPokemon | None:
        """名前でスロットを検索。なければ新規作成（MAX_SLOTS を超えたら None）。

        low_trust=True は定期OCR由来の経路（update の名前蓄積・accumulate_player_name）。
        メッセージ確認済みの高信頼経路（繰り出し/ゆけっ！検出）は False（デフォルト）。

        幽霊ポケモン対策:
        ①前方一致吸収: OCRの末尾欠け誤読（リザードン→リザード）で同一ポケモンが
          別スロットに二重登録されるのを防ぐ。短い方が長い方の前方部分文字列なら
          同一個体とみなし、スロット名は長い方に揃える。
          注: ゴース/ゴーストのような実在の前方一致ペアは誤吸収するが、
          同一チームに揃う確率よりOCR末尾欠け誤読の頻度の方が圧倒的に高い。
        ②新規登録ヒステリシス（low_trust時のみ）: 新規名は複数サイクルの連続目撃で
          確定するまでスロットを作らない（1フレームのノイズ由来の幽霊登録防止）。
        ③満杯時eviction（高信頼経路のみ）: 満杯で新規登録できない場合、場におらず
          未気絶で目撃回数（confidence）最少のスロットを幽霊とみなして削除する。
          低信頼経路に許すと相手繰り出しメッセージの誤分類などで本物のスロットが
          道連れ削除される（実機で確認）。
        """
        for s in slots:
            if s.name == name:
                return s
        # ①前方一致吸収
        for s in slots:
            if min(len(s.name), len(name)) < self._ABSORB_MIN_LEN:
                continue
            if s.name.startswith(name):
                log.info(f"[戦況] {name} は既存 {s.name} の前方一致 → 同一個体として吸収")
                return s
            if name.startswith(s.name):
                log.info(f"[戦況] ロスターの {s.name} を {name} に更新（前方一致吸収）")
                s.name = name
                return s
        # ②新規登録ヒステリシス（低信頼経路のみ）
        side = "自分側" if slots is self._player else "相手側"
        if low_trust and not self._confirm_new_name(side, name):
            return None
        if len(slots) < self.MAX_SLOTS:
            slot = FieldPokemon(name=name)
            slots.append(slot)
            return slot
        # ③満杯時eviction（場にいる・気絶済みスロットは本物確定なので対象外）
        candidates = [s for s in slots if not s.on_field and not s.fainted] if not low_trust else []
        if candidates:
            victim = min(candidates, key=lambda s: s.confidence)
            slots.remove(victim)
            log.info(f"[戦況] ロスター満杯 → 幽霊疑い {victim.name}（目撃{victim.confidence}回）を削除して {name} を登録")
            slot = FieldPokemon(name=name)
            slots.append(slot)
            return slot
        return None  # 全スロットが場or気絶済み: 登録不可

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
                        if self._fuzzy_name_match(slot.name, name):
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

    def assign_slots_from_ocr(
        self,
        player_name_cx: list[tuple[str, float]],
        opponent_name_cx: list[tuple[str, float]],
        command_cy: float | None = None,
    ) -> None:
        """定期OCRの名前+x座標からスロット番号を割り当てる（イベント外の補完）。
        イベント発火時のフレームはメッセージ画面等で相手ネームプレートが読めない
        ことが多く、update() 内の割当だけでは相手側の slot_index がほぼ付かない
        （実機で確認: 相手側のHPpxが1件も取れなかった）。

        command_cy が行動選択画面の範囲（300〜450）にある時のみ割り当てる:
        「相手を見る」パネル等では名前が本来と違う位置に表示され、そのcxで
        割り当てると別スロットのバーを読む誤割当が起きる（実機で確認:
        リザードン再登場後にガブリアスのバー位置へ誤割当）。
        """
        if command_cy is None or not (300 <= command_cy <= 450):
            return
        self._release_stale_slot_indices()
        self._assign_slot_indices(self._player,   player_name_cx,   "player")
        self._assign_slot_indices(self._opponent, opponent_name_cx, "opponent")

    # 数値HP表示の実在帯（1080p実測・診断JSONL 07-00-19全数OCRより）:
    # 自分側 X/Y 数値は cy 1000-1049 のみ（交換選択パネル等の危険帯は cy 750-799）
    # 相手側 HP% は cy 100-151 のみ（状態パネル等の危険帯は cy 350-399）
    _NUM_HP_PLAYER_Y_MIN = 950
    _NUM_HP_OPP_Y_MAX = 200

    def assign_hp_from_ocr(
        self,
        hp_player_with_xy: list[tuple[str, float, float]],
        hp_opponent_with_xy: list[tuple[str, float, float]],
    ) -> None:
        """定期OCRの数値HPを位置ゲートで検証してスロットへ割り当てる（イベント外の補完）。
        数値HPの割当は従来 update()（イベント経路）でしか行われなかったが、
        イベント発火時のフレームはメッセージ/アニメ画面でHP数値が映らないことが
        多く、画面に出ている正読値が毎回捨てられていた（実機で確認: 07-00-19 終盤の
        イダイトウ 7/201 が何度も正読されていたのに、HPpx の物理限界（バー15px未満=
        HP<6.6%は読めない）で残った古い 47%(px) が鮮度比較で勝ち続けた）。

        ゲートは command_cy（トークンOCRがフレーム単位で欠落し不安定）ではなく
        表示位置で行う: HP数値の実在帯（y）＋バー中心とのcx近接でスロットを直接
        決定する。数値がバー位置とセットで表示される以上、画面種別に依存しない。

        定期OCRはイベント経路よりサンプル数が多く、単発の数値誤読
        （例: 「0/205」→「1/205」）が刺さるリスクが上がるため、
        2回同じ読み値が観測された場合のみ割り当てる（幽霊登録と同じ考え方）。
        """
        for side, with_xy, slots in (
                ("player",   hp_player_with_xy,   self._player),
                ("opponent", hp_opponent_with_xy, self._opponent)):
            centers = self._SLOT_BAR_CENTERS[side]
            by_slot = ["", ""]
            for hp, cx, cy in with_xy:
                if side == "player" and cy < self._NUM_HP_PLAYER_Y_MIN:
                    continue
                if side == "opponent" and cy > self._NUM_HP_OPP_Y_MAX:
                    continue
                idx = 0 if abs(cx - centers[0]) <= abs(cx - centers[1]) else 1
                if abs(cx - centers[idx]) > self._SLOT_CX_TOLERANCE:
                    continue
                if not by_slot[idx]:
                    by_slot[idx] = hp
            confirmed = self._confirm_ocr_hp(side, by_slot)
            for s in slots:
                i = s.slot_index
                if not (s.on_field and i is not None
                        and i < len(confirmed) and confirmed[i]):
                    continue
                # 1桁%（1-9%）はターゲット選択カーソル等のUI遮蔽で先頭桁が
                # 隠れた誤読（72%→「2%」・conf1.0で14秒継続を実測）と区別が
                # つかないため、既知HPがすでに低い場合のみ受け付ける
                # （誤情報より欠落マシ。本物の低HP進行 13%→2% 等は通る）
                m_pct = re.fullmatch(r'([1-9])%', confirmed[i])
                if m_pct and not self._known_hp_is_low(s):
                    log.info("[数値HP] %s の1桁%%読み %s を保留（既知HPが高く先頭桁欠け疑い）",
                             s.name, confirmed[i])
                    confirmed[i] = ""
                    continue
                # 0/X・0% は _assign_hp_by_slot が誤気絶防止のため
                # 割り当てない → 実際に代入される値のみログする
                if (confirmed[i] != s.hp
                        and not confirmed[i].startswith("0/")
                        and confirmed[i] != "0%"):
                    log.info("[数値HP] %s → %s（定期OCR確定）",
                             s.name, confirmed[i])
            self._assign_hp_by_slot(slots, confirmed)

    @staticmethod
    def _known_hp_is_low(s: FieldPokemon, threshold: float = 0.25) -> bool:
        """既知のHP（数値・pxのいずれか）が threshold 以下なら True。
        どちらも不明なら False（=1桁%は棄却される側に倒す）。"""
        if s.hp_pct_pixel is not None and s.hp_pct_pixel <= threshold:
            return True
        if s.hp:
            m = re.match(r'^(\d+)/(\d+)$', s.hp)
            if m and int(m.group(2)) > 0:
                return int(m.group(1)) / int(m.group(2)) <= threshold
            if s.hp.endswith("%"):
                try:
                    return int(s.hp[:-1]) / 100 <= threshold
                except ValueError:
                    return False
        return False

    # %形式は1トークン内の桁欠け誤読（72%→2%）が同一画面で連続しやすく、
    # 2回一致では貫通した実例があるため3回一致を要求する。
    # X/Y形式はスラッシュ＋分母の構造で頑健なため2回のまま
    _OCR_HP_CONFIRM_XY = 2
    _OCR_HP_CONFIRM_PCT = 3

    def _confirm_ocr_hp(self, side: str, hp_by_slot: list[str]) -> list[str]:
        """同じ読み値が規定回数連続で観測されたスロットのみ通す（誤読の除去）。
        未読（空）のサイクルでは保留値を消さない: コマンド画面中は
        HP数値が読めないフレーム（技の説明パネル等）が頻繁に挟まり、
        消してしまうと連続一致が実機で事実上成立しない
        （実機 07-00-19 で確認: 7/201→空→7/201 の交互列で一度も確定しなかった）。"""
        confirmed: list[str] = []
        for i, hp in enumerate(hp_by_slot):
            key = (side, i)
            if not hp:
                confirmed.append("")
                continue  # 未読は矛盾情報ではない → 保留を保持
            prev_val, prev_cnt = self._pending_ocr_hp.get(key, (None, 0))
            cnt = prev_cnt + 1 if prev_val == hp else 1
            self._pending_ocr_hp[key] = (hp, cnt)
            need = (self._OCR_HP_CONFIRM_PCT if hp.endswith("%")
                    else self._OCR_HP_CONFIRM_XY)
            confirmed.append(hp if cnt >= need else "")
        return confirmed

    def _release_stale_slot_indices(self) -> None:
        """場を離れたポケモン（交代・気絶・不検出降ろし）の slot_index を解放する。
        物理スロット位置は交代のたびに入れ替わるため、古い slot_index を持ったまま
        再登場すると別のポケモンのHPバーを読み続ける（実機で確認: リザードンが
        T1の位置のままガブリアスのバー42%を「リザードン43%」として表示し続けた）。
        """
        for s in self._player + self._opponent:
            if not s.on_field and s.slot_index is not None:
                s.slot_index = None

    def _assign_slot_indices(
        self,
        slots: list[FieldPokemon],
        name_with_cx: list[tuple[str, float]],
        side: str,
    ) -> None:
        """初登場時に OCR x座標からスロット番号（0=左, 1=右）を割り当てる。
        固定閾値ではなく、同フレームで見えた未割り当てポケモン同士の相対x順で決定する。
        （SV のプレイヤー側2匹のHPバーは両方とも画面左半分に表示されるため
        cx=960 の固定閾値は使えない）
        候補が1匹しか見えず両スロット空きの場合は相対順が使えないため、
        _SLOT_BAR_CENTERS（side="player"/"opponent"）との近接で決定する。
        既に slot_index が設定済みのポケモンはスキップする。
        """
        # 名前マッチングで未割り当て on_field スロットと cx を収集
        candidates: list[tuple[FieldPokemon, float]] = []
        for name, cx in name_with_cx:
            for slot in slots:
                if self._fuzzy_name_match(slot.name, name):
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
        elif len(candidates) < len(available):
            # 両スロット空きで候補1匹: 相対x順では左右を決められない。
            # zip で機械的にスロット0を振ると画面右の候補が先取りして完全反転する
            # （実機で確認: フシギバナ cx=1672 がスロット0を先取り→後続リザードンが
            #   空きスロット割当で1に入り、HP表示が2匹丸ごと入れ替わった）。
            # 最寄りバー中心とのx近接で決定し、どのバーにも近くない候補
            # （選出リスト・パネル等の座標）は次フレーム以降に保留する。
            centers = self._SLOT_BAR_CENTERS[side]
            for slot, cx in candidates:
                if not available:
                    break
                idx = min(available, key=lambda i: abs(cx - centers[i]))
                if abs(cx - centers[idx]) > self._SLOT_CX_TOLERANCE:
                    log.info(f"[スロット] {slot.name} 保留 (cx={cx:.0f} がバー位置と不一致)")
                    continue
                slot.slot_index = idx
                available.remove(idx)
                log.info(f"[スロット] {slot.name} → スロット{idx} (cx={cx:.0f}, バー近接判定)")
        else:
            # 空きスロット番号のみを cx 昇順の候補に割り当てる
            # （固定で0,1を振ると使用中スロットと重複し、同じバーを2匹が読む）
            for (slot, cx), idx in zip(candidates, available):
                slot.slot_index = idx
                log.info(f"[スロット] {slot.name} → スロット{idx} (cx={cx:.0f})")

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
                    slot.hp_turn = self.turn
                    assigned_slots.add(i)

        # パス2: slot_index 未設定のポケモンは「1対1で曖昧さがない場合のみ」割り当てる。
        # 2匹以上を登録順×x座標順で機械的にzipすると左右スワップが起きる
        # （実機で確認: T1でプテラのHP157がオオニューラに付いた）
        remaining_hp = [hp for i, hp in enumerate(effective) if i not in assigned_slots and hp]
        unassigned = [s for s in on_field if s.slot_index is None]
        if len(unassigned) == 1 and len(remaining_hp) == 1:
            unassigned[0].hp = remaining_hp[0]
            unassigned[0].hp_turn = self.turn

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
            slot = self._get_or_create(self._player, name, low_trust=True)
            if slot:
                slot.confidence += 1
                slot.last_seen_turn = self.game_turn
                if not slot.fainted:
                    slot.on_field = True  # 現フレームで見えた → 場にいる

        # 長期間不検出のポケモンを場から降ろす（OCRノイズで一時的に消える場合は維持）
        for slot in self._player:
            if slot.on_field and not slot.fainted:
                if self.game_turn - slot.last_seen_turn > self._ON_FIELD_MISS_THRESHOLD:
                    slot.on_field = False
                    log.info(f"[戦況] {slot.name} が{self._ON_FIELD_MISS_THRESHOLD}ターン不検出 → 場から降ろす")

        # ── ポケモン名の蓄積・on_field 更新（相手側） ──────────────────────
        for name in current_opponent_names:
            already_in_player = any(s.name == name for s in self._player)
            if already_in_player and name not in current_player_names:
                continue  # 自分側に登録済みで自分エリアにも見えていない → 誤分類として除外
            slot = self._get_or_create(self._opponent, name, low_trust=True)
            if slot:
                slot.confidence += 1
                slot.last_seen_turn = self.game_turn
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
                    # こちらは「今フレームに他の相手名が見えているのにこの名前だけ無い」
                    # という直接証拠に基づく即時判定のため、意図的に self.turn
                    # （イベント単位）のままにする。self.game_turn 化すると次のターン
                    # 境界まで除去が遅延し、交代直後に旧ポケモンが残り続けてしまう。
                    quick_threshold = 1
                    if self.turn - slot.last_seen_turn >= quick_threshold:
                        slot.on_field = False
                        newly_removed_opponent.append(slot)
                        log.info(f"[戦況] {slot.name} がOCR不検出（{slot.name} not in {current_opponent_names}）→ 場から降ろす")
                elif self.game_turn - slot.last_seen_turn > self._ON_FIELD_MISS_THRESHOLD:
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
            # 場を離れたポケモンの古い slot_index を解放してから割り当てる
            self._release_stale_slot_indices()
            player_name_cx   = game_state.get("name_player_with_cx", [])
            opponent_name_cx = game_state.get("name_opponent_with_cx", [])
            self._assign_slot_indices(self._player,   player_name_cx,   "player")
            self._assign_slot_indices(self._opponent, opponent_name_cx, "opponent")

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

    # 確定状態異常技テーブル (技名 → 状態異常)
    # 技名はpokedb.sqlite の moves.name_ja と一致させること
    # ※ でんじは（単体技）はYOLOアイコン検出で正確なターゲットを特定するため除外
    # ※ 全体技/単体技を問わず、場に複数いる場合は全員に付与する（下記docstring参照）ため
    #   技ごとの範囲情報は持たない
    _STATUS_MOVE_TABLE: dict[str, str] = {
        "おにび":         "やけど",   # 全体技
        "どくどく":       "もうどく",
        "どくのこな":     "どく",
        "キノコのほうし": "ねむり",
        "さいみんじゅつ": "ねむり",
        "うたう":         "ねむり",
    }

    def apply_status_from_move(self, user_name: str, move_name: str) -> None:
        """確定状態異常技の効果から相手チームへ状態異常を推定付与する。
        OCRでメッセージが取れなかった場合の補完用。
        既に状態異常があるポケモン・気絶済みポケモンはスキップ。
        単体技でも場に複数いる場合は全員に付与する（ダブルバトルでターゲット特定不能のため）。
        """
        status = self._STATUS_MOVE_TABLE.get(move_name)
        if not status:
            return

        # 使用者がどちらのチームか判定（部分一致で対応）
        in_player = any(
            self._fuzzy_name_match(s.name, user_name)
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
        # 場を離れたポケモンの古い slot_index を解放（再登場時の別バー誤読防止）
        self._release_stale_slot_indices()
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
                    if self._claim_slot(key, p.name):
                        break  # 占有者交代を検知 → 今サイクルの値は前任者のものなのでスキップ
                    old = p.hp_pct_pixel
                    p.hp_pct_pixel = pct
                    # 鮮度スタンプは値が変わった時のみ更新する。
                    # アナライザーは読めないフレームで最後の確定値を返し続けるため
                    # （_apply_stabilizer の raw=None 分岐）、無条件に再スタンプすると
                    # 古い保持値が「常に最新」となり、HPpx物理限界（バー15px未満）で
                    # px が止まった後の数値OCR正読（イダイトウ 7/201）を鮮度比較で
                    # 覆い隠し続ける（実機 07-00-19 で確認）。
                    if old is None or pct != old:
                        p.hp_px_turn = self.turn
                    # HPバーが物理スロットで読めている＝そのポケモンが確実に場にいる証拠。
                    # 名前OCRでの独立再検出頻度が低い個体（実機: 07-00-19のオオニューラ）が
                    # _ON_FIELD_MISS_THRESHOLD で誤って場から降ろされるのを防ぐ。
                    p.last_seen_turn = self.game_turn
                    if old is None or abs(pct - old) >= 0.05:
                        log.info("[HPpx] %s %s → %.1f%%", key, p.name, pct * 100)
                    matched = True
                    break

            # パス2: 未割り当てが1匹だけなら位置ベース早期割り当て。
            # ただし同サイドの両スロットに読み値があり、もう一方のスロットの主が
            # 不明な場合はどちらのバーか判別できないため保留する
            # （実機で確認: 相手2匹中1匹しか把握していない時に左バーへ誤割当され、
            #   リザードンがガブリアスのバー42%を読み続けた）
            if not matched:
                unassigned = [p for p in slots if p.on_field and p.slot_index is None]
                other_idx = 1 - slot_idx
                other_key = key.rsplit("_", 1)[0] + f"_{other_idx}"
                other_held = any(q.on_field and q.slot_index == other_idx for q in slots)
                unambiguous = other_held or pixel_hp.get(other_key) is None
                if len(unassigned) == 1 and unambiguous:
                    p = unassigned[0]
                    p.slot_index = slot_idx
                    log.info("[スロット早期割] %s → スロット%d (HPpxフォールバック)", p.name, slot_idx)
                    if self._claim_slot(key, p.name):
                        continue  # 占有者交代 → 前任者の値はスキップ（次サイクル以降の新確定値を待つ）
                    old = p.hp_pct_pixel
                    p.hp_pct_pixel = pct
                    if old is None or pct != old:  # 保持値の再読では鮮度を偽装しない（パス1と同様）
                        p.hp_px_turn = self.turn
                    p.last_seen_turn = self.game_turn  # HPバー実測＝場にいる証拠（上のパス1と同様）
                    log.info("[HPpx] %s %s → %.1f%%", key, p.name, pct * 100)

    def _claim_slot(self, key: str, name: str) -> bool:
        """物理スロットの占有者を更新する。占有者が交代した場合 True を返し、
        コールバック（HpBarAnalyzerの安定化状態リセット）を呼ぶ。
        アナライザーの確定値はスロット（画面位置）に紐づくため、交代直後は
        前任ポケモンの値が返り続ける。リセットして新しい確定値を待つ。"""
        prev = self._slot_occupant.get(key)
        if prev == name:
            return False
        self._slot_occupant[key] = name
        if prev is not None and self.slot_reset_cb is not None:
            log.info("[HPpx] %s 占有者交代 %s → %s（安定化リセット）", key, prev, name)
            self.slot_reset_cb(key)
            return True
        return False

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

    def move_user_side(self, pokemon_name: str, is_opponent: bool = False) -> str | None:
        """技の使い手がどちらの陣営かを返す（"自分"/"相手"・判定不能はNone）。

        瞬間ログ（timeline.jsonl）の陣営タグ用。同名ミラー（両陣営に同名がいる）
        の場合は場に出ている側を優先し、両方場に出ている・どちらも出ていない
        場合は判定不能として None を返す（誤タグよりタグ無しの方が安全）。
        """
        if is_opponent:
            return "相手"
        p_slots = [s for s in self._player if s.name == pokemon_name]
        o_slots = [s for s in self._opponent if s.name == pokemon_name]
        if p_slots and not o_slots:
            return "自分"
        if o_slots and not p_slots:
            return "相手"
        if p_slots and o_slots:
            p_on = any(s.on_field for s in p_slots)
            o_on = any(s.on_field for s in o_slots)
            if p_on != o_on:
                return "自分" if p_on else "相手"
        return None

    def set_not_on_field(self, pokemon_name: str) -> bool:
        """指定ポケモンを場から降ろす（交代・とんぼがえり検出時に呼ぶ）。
        見つかった場合は True を返す。

        「〜は戻っていく」メッセージには陣営を示すプレフィックスが無く（faintの
        「相手の」やswitch_outの「引っこめた」形式のような手がかりが無い）、
        同名ミラー戦では両陣営に一致する場に出ているスロットが存在しうる。
        従来は自分側→相手側の走査順で先に見つかった方を無条件に降ろしていたため、
        実際は相手の交代なのに自分側を誤ベンチ化する（逆も同様）曖昧さがあった。
        一致する「場に出ている」スロットが2つ以上ある場合は判定不能とみなし、
        誤ベンチ化を避けるため何もしない（move_user_side と同じ「誤タグより
        タグ無しの方が安全」の方針）。
        """
        # fuzzy マッチ（OCR誤読でポケモン名が少し違う場合も対応）
        def _match(slot: FieldPokemon) -> bool:
            return slot.name == pokemon_name or slot.name in pokemon_name or pokemon_name in slot.name

        on_field_matches = [slot for side in (self._player, self._opponent)
                            for slot in side if _match(slot) and slot.on_field]
        if len(on_field_matches) == 1:
            on_field_matches[0].on_field = False
            return True
        return False

    # ── メッセージ由来イベント ────────────────────────────────────────────────

    def _find_slot(self, name: str) -> FieldPokemon | None:
        """名前で両チームを検索してスロットを返す（部分一致OK）。"""
        for slot in self._player + self._opponent:
            if self._fuzzy_name_match(slot.name, name):
                return slot
        return None

    def _confirm_faint_on_side(self, slots: list, name: str) -> bool:
        """指定した側のスロットのみを検索してfaintedフラグを立てる。"""
        for slot in slots:
            if self._fuzzy_name_match(slot.name, name):
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

    def confirm_player_faint_by_name(self, name: str) -> bool:
        """自分側のポケモン気絶確認（「たおれた」メッセージに「相手の」なし）。"""
        return self._confirm_faint_on_side(self._player, name)

    def confirm_opponent_faint_by_name(self, name: str) -> bool:
        """相手側のポケモン気絶確認（「相手の〇〇はたおれた」メッセージ）。"""
        return self._confirm_faint_on_side(self._opponent, name)

    def accumulate_player_name(self, name: str,
                               opponent_candidates: list[str] | None = None) -> None:
        """定期OCRで検出されたプレイヤーポケモン名を蓄積する（イベント以外の補完用）。
        相手側に同名ポケモンがいる場合はスキップ（y座標誤分類対策）。
        opponent_candidates: 同一フレームで相手エリアに見えている名前（渡されると
        その名前は相手側の誤分類とみなしてスキップする）。
        未登録なら新規スロットを作成して on_field=True にする。
        ダブルバトル上限（場2匹）を超える場合は新規追加しない。
        """
        # 相手を見るパネル等では相手ポケモン名が画面中央（y>500=自分エリア扱い）に
        # 表示され、自分側に誤蓄積される（実機でユキメノコ・ロトムが自分側に混入）。
        # 相手ロスターとの照合は前方一致（OCR末尾欠け）も含めて行う
        already_in_opponent = any(
            s.name == name
            or (min(len(s.name), len(name)) >= self._ABSORB_MIN_LEN
                and (s.name.startswith(name) or name.startswith(s.name)))
            for s in self._opponent)
        if already_in_opponent:
            return
        if opponent_candidates and name in opponent_candidates:
            return  # 同一フレームで相手エリアにも見えている → 相手側の誤分類
        on_field_count = sum(1 for s in self._player if s.on_field and not s.fainted)
        already_in_player = any(s.name == name for s in self._player)
        if already_in_player:
            for s in self._player:
                if s.name == name:
                    # 目撃カウント: eviction判定（目撃最少=幽霊疑い）が本物を誤爆しないよう、
                    # 定期OCRで見えている本物のスロットには目撃回数を積む
                    s.confidence += 1
                    if not s.fainted and not s.on_field and on_field_count < 2:
                        s.on_field = True
                        s.last_seen_turn = self.game_turn
                        log.info(f"[戦況] {s.name} 定期OCR検出 → 場に追加")
            return
        if on_field_count >= 2:
            return  # ダブルバトル上限: 新規追加しない
        # 定期OCRは相手繰り出しメッセージ等の誤分類が混入しうる低信頼経路
        # （eviction禁止＋新規登録ヒステリシス）
        slot = self._get_or_create(self._player, name, low_trust=True)
        if slot and not slot.fainted:
            slot.on_field = True
            slot.last_seen_turn = self.game_turn
            log.info(f"[戦況] {slot.name} 定期OCR検出 → 新規登録して場に追加")

    def mark_on_field_by_name(self, name: str) -> bool:
        """メッセージ由来の繰り出し確認: プレイヤースロットを検索してon_field=Trueにする。
        相手に同名ポケモンがいる場合でも正しく自分側に登録するため、プレイヤー側のみ検索する。
        スロット未登録なら新規作成する。
        """
        slot = None
        for s in self._player:
            if self._fuzzy_name_match(s.name, name):
                slot = s
                break
        if slot is None:
            slot = self._get_or_create(self._player, name)
        if slot and not slot.fainted:
            slot.on_field = True
            slot.last_seen_turn = self.game_turn
            log.info(f"[戦況] {slot.name} 繰り出し確認（メッセージ由来）")
            return True
        if slot is None:
            # 実機でフラエッテがこの経路で無言のまま登録失敗していた（幽霊が場を占有）
            log.warning(f"[戦況] 自分スロットが満杯のため {name} を登録できません")
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
            slot.last_seen_turn = self.game_turn
            log.info(f"[戦況] 相手 {slot.name} 繰り出し確認（メッセージ由来）")
        return True

    def register_opponent_fainted(self, name: str) -> bool:
        """気絶メッセージ由来の遅延登録: 未登録の相手が「たおれた」場合に登録して気絶確定する。
        繰り出しメッセージの取りこぼし（battle_startリセットで消滅・OCR末尾見切れ等）で
        ロスターにいないまま倒れた相手の救済（実機: ロトム・ガブリアスの気絶が無言消滅）。
        """
        slot = self._get_or_create(self._opponent, name)
        if slot is None:
            log.warning(f"[戦況] 相手スロットが満杯のため {name} の気絶登録に失敗")
            return False
        slot.fainted = True
        slot.on_field = False
        log.info(f"[戦況] 相手 {slot.name} 気絶確認（メッセージ由来・遅延登録）")
        return True

    def mark_bench_by_name(self, name: str, side: str = "both") -> bool:
        """メッセージ由来の引っ込め確認: 名前でスロットを検索してon_field=Falseにする。
        side="player"/"opponent" で検索範囲を限定する。同名ミラー戦では両側検索だと
        相手の引っ込めメッセージが自分側スロットを誤ベンチ化する（実機で確認）。
        """
        pools = {"player": self._player, "opponent": self._opponent}
        slots = pools.get(side) if side in pools else (self._player + self._opponent)
        for slot in slots:
            if self._fuzzy_name_match(slot.name, name):
                slot.on_field = False
                log.info(f"[戦況] {slot.name} 引っ込め確認（メッセージ由来）")
                return True
        return False

    # ── コンテキスト生成 ─────────────────────────────────────────────────────

    def _format_pokemon(self, p: FieldPokemon) -> str:
        """場にいるポケモンの詳細フォーマット（HP・状態異常・使用技を含む）。

        HP表示は「より新しく更新された方」を採用する:
        OCR数値HP（62/135等・正確）が同ターン以降に取れていればそれを優先し、
        古い場合のみHPpx（ピクセル解析・近似値）を表示する。
        従来のHPpx無条件優先は、アニメーション中に更新が止まった古いpx値が
        正確な数値HPを隠す問題があった（実機で確認）。
        """
        s = p.name
        if p.status:
            s += f"({p.status})"
        px_is_fresher = (p.hp_pct_pixel is not None
                         and (p.hp is None or p.hp_px_turn > p.hp_turn))
        if px_is_fresher:
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

    def _display_hp_pct(self, p: FieldPokemon) -> tuple[int | None, str | None]:
        """パネル表示用のHP（％整数と表示テキスト）を返す。

        鮮度比較（数値HPとHPpxの新しい方を採用）は ``_format_pokemon`` と同一。
        """
        px_is_fresher = (p.hp_pct_pixel is not None
                         and (p.hp is None or p.hp_px_turn > p.hp_turn))
        if px_is_fresher:
            pct = round(p.hp_pct_pixel * 100)
            return pct, f"{pct}%"
        if p.hp:
            m = self._HP_RE.match(p.hp)
            if m and int(m.group(2)) > 0:
                return round(int(m.group(1)) / int(m.group(2)) * 100), p.hp
            if p.hp.endswith("%"):
                try:
                    return int(p.hp[:-1]), p.hp
                except ValueError:
                    pass
        return None, None

    def record_turn_snapshot(self) -> None:
        """ターン開始時点の場の状態を turn_history に記録する（Bedrockへの
        turn_history送信用）。MAX_TURN_HISTORY 件を超えたら古い方から捨てる。
        """
        def side(slots: list[FieldPokemon]) -> str:
            on_field = [q for q in slots if q.on_field and not q.fainted]
            parts = []
            for p in on_field:
                pct, _ = self._display_hp_pct(p)
                parts.append(f"{p.name}{pct}%" if pct is not None else p.name)
            return "/".join(parts) if parts else "不明"

        snapshot = f"T{self.game_turn}: 自分={side(self._player)} / 相手={side(self._opponent)}"
        self._turn_history.append(snapshot)
        if len(self._turn_history) > self.MAX_TURN_HISTORY:
            self._turn_history.pop(0)

    def to_panel_state(self) -> dict:
        """実況動画の戦況パネル（v2b）用スナップショットを返す。

        場のポケモン（スロット順・最大2匹/側）の名前・HP・状態異常と、
        ターン数・残り頭数。レンダーモードで states.jsonl に記録され、
        パス2がASS描画で右サイドパネルに時刻同期表示する。
        """
        def side(slots: list[FieldPokemon]) -> list[dict]:
            on_field = [q for q in slots if q.on_field and not q.fainted]
            on_field.sort(key=lambda q: q.slot_index if q.slot_index is not None else 9)
            out = []
            for p in on_field[:self.MAX_ON_FIELD]:
                pct, text = self._display_hp_pct(p)
                out.append({"name": p.name, "hp_pct": pct,
                            "hp_text": text, "status": p.status})
            return out

        known_p = len([p for p in self._player if not p.fainted])
        known_o = len([p for p in self._opponent if not p.fainted])
        return {
            "turn": self.game_turn,
            "player": side(self._player),
            "opponent": side(self._opponent),
            "alive_player": self._player_alive_count or known_p,
            "alive_opponent": self._opponent_alive_count or known_o,
        }

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
            "turn_history":     " | ".join(self._turn_history) or "なし",
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


def _detect_battle_result(text: str) -> str | None:
    """OCRテキストから勝敗を判定する（battle_end画面の「勝負に勝った/負けた」）。

    OCRの分割・スペース混入（「勝負に 勝った!」等）を吸収するため
    スペース除去後に部分一致で判定する。判定不能（降参・通信エラー等）は None。
    """
    joined = text.replace(" ", "")
    if "勝負に勝" in joined:
        return "勝ち"
    if "勝負に負" in joined:
        return "負け"
    return None


# ─── AIグリッチ差し替え（Bedrock保留・困惑応答対策） ─────────────────────────

# Bedrockがデータ矛盾等で実況を保留・困惑する応答（「データが矛盾していて
# 実況できません」等）を返した場合の検出キーワード。先頭グループから順に照合し、
# 最初にマッチしたグループの原因文言を定型文に差し込む。
# 言い回しの変化に応じてキーワードは今後拡張していく前提。
_GLITCH_CAUSE_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("矛盾", "ちぐはぐ"), "データがちぐはぐさん"),
    (("見えにく", "読み取れ"), "画面がチカチカしてた"),
    (("確定できて", "お待ち"), "情報がまだ揃ってない"),
    (("モヤモヤ", "教えてほし", "教えてもらえ", "実況できな"), "ナゾのノイズ"),
]

# くれぴ口調の「AIグリッチ」定型文（{cause}に原因文言が入る）。
# テンプレートはランダム選択、原因はキーワードマッチで決定する
# （原因までランダムにすると実態と合わない文になるため）。
# ⚠️絵文字（U+1F300以降）はMeiryoに無く字幕が豆腐化するので使わないこと（♪♡は可）。
_GLITCH_TEMPLATES = [
    "あれれ？{cause}で、くれぴの目がちょっとバグっちゃったかも…！次いくよ次〜♪",
    "ちょっと待って、{cause}でデータがぐるぐるしてる…！ま、いっか♪試合に戻ろ〜！",
    "エラー発生〜！原因は{cause}だって♪ くれぴだってたまには混乱するもん！",
]


def _detect_glitch_cause(text: str) -> str | None:
    """保留・困惑応答なら差し込む原因文言を、通常の実況文なら None を返す。"""
    for keywords, cause in _GLITCH_CAUSE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return cause
    return None


def _replace_glitch_commentary(text: str) -> str:
    """Bedrockの保留・困惑応答をくれぴ口調の「AIグリッチ」定型文に差し替える。

    通常の実況文はそのまま返す。差し替えはVOICEVOX合成前に呼ぶこと
    （合成後に字幕テキストだけ差し替えると音声と不一致になるため）。
    """
    cause = _detect_glitch_cause(text)
    if cause is None:
        return text
    replaced = random.choice(_GLITCH_TEMPLATES).format(cause=cause)
    log.info("[AIグリッチ] 保留・困惑応答を検出→定型文に差し替え: 「%s」→「%s」",
             text, replaced)
    return replaced


# ─── Bedrock Vision 呼び出し（EC2 API 経由・オプション） ─────────────────────

def _build_bedrock_context(
    game_state: dict,
    event_type: str,
    battle_context: dict | None,
    classifier,
    move_log: list[str] | None,
) -> dict:
    """Bedrock に送る context 辞書を組み立てる（画像に依存しない部分）。

    ``_call_bedrock_vision``（ライブ・画像あり）と ``_call_bedrock_text``
    （動画モードの後付け生成・画像なし）で共通利用する。
    """
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

    return {
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
        "battle_result":            game_state.get("battle_result", ""),  # 勝敗（battle_endのみ・"勝ち"/"負け"）
    }


def _log_bedrock_send(context: dict, battle_state: dict) -> None:
    log.info(
        "[Bedrock送信] event=%s | 自分=%s | 相手=%s | HP=%s | 技ログ=%s | RAG=%s",
        context["event_type"],
        battle_state.get("player_pokemon", "不明"),
        battle_state.get("opponent_pokemon", "不明"),
        context["hp_values"],
        context["detected_moves"],
        " / ".join(context["rag_pokemon_info"]) if context["rag_pokemon_info"] else "なし",
    )


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
    EC2 API に画像と状況を送り、Bedrock Vision 分析結果を受け取る（ライブ経路）。
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

        context = _build_bedrock_context(game_state, event_type, battle_context, classifier, move_log)
        payload = {
            "image_base64": image_b64,
            "context": context,
            "history": commentary_history[-3:],
            "battle_state": battle_context or {},
        }
        _log_bedrock_send(context, payload["battle_state"])
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


def _call_bedrock_text(
    ec2_url: str,
    game_state: dict,
    event_type: str,
    commentary_history: list[str],
    battle_context: dict | None = None,
    classifier=None,
    move_log: list[str] | None = None,
) -> str | None:
    """
    EC2 API に画像なし・構造化データのみを送り、Bedrockの実況文を受け取る
    （動画モードの後付け生成専用・ADR-009追記）。
    蓄積済みの戦況追跡（OCR・HPバー解析）を正確な事実として扱わせるため、
    単フレーム画像からの再判定はさせない。失敗してもパイプラインを止めない（None を返す）。
    """
    try:
        context = _build_bedrock_context(game_state, event_type, battle_context, classifier, move_log)
        payload = {
            "context": context,
            "history": commentary_history[-3:],
            "battle_state": battle_context or {},
        }
        _log_bedrock_send(context, payload["battle_state"])
        resp = requests.post(f"{ec2_url}/api/vision", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            log.debug(f"Bedrock tokens: in={data.get('usage',{}).get('input_tokens')} out={data.get('usage',{}).get('output_tokens')} latency={data.get('latency_ms')}ms")
        return data.get("commentary"), data.get("analysis")
    except Exception as e:
        log.warning(f"Bedrock Text 呼び出しスキップ: {e}")
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
        render_out: str | None = None,
    ):
        log.info("=== パイプライン初期化 ===")

        log.info("EasyOCR 初期化中...")
        self._reader = init_reader(gpu=gpu)

        log.info("YoloDetector 初期化中...")
        # 状態異常検出はテキストOCR（_sync_status_from_ocr_bbox）で代替済みのため、
        # model未指定時はCOCO事前学習フォールバックを使わず無効化する（無駄な推論を防ぐ）。
        self._yolo = YoloDetector(model_path=model_path, ball_model_path=ball_model_path, end_model_path=end_model_path, conf=conf,
                                   enable_pretrained_fallback=False)

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
        # 動画モードでは run() が毎フレーム _video_now に動画内時間をセットする。
        # ライブモード（None のまま）では実時間にフォールバック。
        self._video_now: float | None = None
        self._phase_classifier = BattlePhaseClassifier(clock=self._now)  # フェーズ分類 + イベント検知
        self._last_ocr_time: float = 0.0                  # 定期OCR用タイマー
        self._PERIODIC_OCR_INTERVAL_BATTLE = 1.5         # バトル中: 終了画面を取りこぼさないよう短め
        self._PERIODIC_OCR_INTERVAL_IDLE   = 3.0         # バトル外: 重くならないよう長め
        self._battle_tracker = BattleStateTracker()       # 戦況累積
        self._hpbar_analyzer = HpBarAnalyzer()             # HPバーピクセル解析
        # スロット占有者交代時にアナライザーの安定化状態をリセットする配線
        self._battle_tracker.slot_reset_cb = self._hpbar_analyzer.reset_slot
        self._msg_parser = BattleMessageParser(clock=self._now)  # バトルメッセージ解析
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
        self._battle_result: str | None = None  # 「勝負に勝った/負けた」検出結果（"勝ち"/"負け"）
        self._pre_battle_opponent: list[str] = []  # battle_start前に検出した相手ポケモン名キャッシュ
        self._pre_battle_player: list[str] = []    # battle_start前に検出した自分ポケモン名キャッシュ（ゆけっ！検出）
        # メッセージ由来の繰り出し履歴 (時刻, side, 名前)。繰り出し演出は
        # battle_start（コマンド画面）より先に流れるため、遅延起動中のトラッカーに
        # 登録済みでもbattle_startのリセットで消える → リセット直後に引き継ぐ
        self._recent_sendouts: list[tuple[float, str, str]] = []
        self._SENDOUT_CARRYOVER_SEC = 25.0  # battle_startからこの秒数以内の繰り出しを引き継ぐ
        self._commentary_history: list[str] = []
        self._dense_scan_remaining: int = 0  # move_used後の高密度メッセージROIスキャン残りフレーム数
        self._dense_scan_start_turn: int | None = None  # dense scan起点ターン（技ログのターン番号固定用）
        self._last_full_ocr_results: list[dict] = []  # メインOCR最新結果（dense scan時の使い手特定に使用）
        self._move_log: list[str] = []   # OCRから検出した「使われた技」のリングバッファ
        self._tentative_opponent_moves: list[dict] = []  # dense scan フォールバックで仮確定した相手技（後付け修正用）
        self._MAX_MOVE_LOG = 8
        self._speech_thread: threading.Thread | None = None  # 音声再生スレッド
        # レンダリング素材出力（ADR-009 パス1）: 指定時は音声を再生せずWAV＋マニフェスト保存
        self._render_sink: RenderSink | None = None
        if render_out:
            if not video_path:
                log.warning("--render-out はライブモードでは非推奨: event_time が実時間になる")
            self._render_sink = RenderSink(render_out)
            self._render_sink.write_info({
                "video": video_path,
                "sample_fps": video_sample_fps,
                "speaker": speaker,
            })
            log.info("[レンダ] 素材出力モード: %s（実況音声は再生せず保存）",
                     self._render_sink.out_dir)
        # 動画モード＋素材出力時のみ: 実況生成をスキャン完了後に後付けで行う（ADR-009追記）。
        # ライブモードは即時Bedrock Vision経路を維持するため対象外。
        self._posthoc_mode: bool = self._render_sink is not None and video_path is not None
        self._pending_render_events: list[dict] = []
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

    def _now(self) -> float:
        """BattlePhaseClassifier 用の時計。
        動画モードでは動画内時間（frame_pos / fps）、ライブでは実時間を返す。
        動画のフレームはOCR処理時間と無関係にフレーム数で進むため、実時間を使うと
        デバウンス・脱出弁等の時間閾値が約10倍間延びした軸で動いてしまう（実測）。
        """
        return self._video_now if self._video_now is not None else time.time()

    def _reset_battle_state(self) -> None:
        """バトル開始時の状態リセット＆アクティブ化。
        battle_start と遅延起動（battle_start 取り逃しフォールバック）の共通処理。
        トラッカー・アナライザー・実況履歴・技ログ等、前試合の残骸を全て捨てる。"""
        self._battle_tracker = BattleStateTracker()
        self._battle_tracker.slot_reset_cb = self._hpbar_analyzer.reset_slot
        # アナライザーの確定値も前試合・試合前画面の値が残るためリセット
        self._hpbar_analyzer.reset()
        self._battle_active = True
        self._battle_active_since = self._now()
        self._end_screen_count = 0
        self._commentary_history = []
        self._move_log = []
        self._last_ball_yolo = None   # バトル開始時にボール情報をリセット
        self._last_ability_msg = {}   # バトル開始時に特性・道具メッセージをリセット
        self._battle_result = None    # 前試合の勝敗をリセット（連戦動画対策）

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
                if ret and _is_video:
                    # 今読んだフレームの動画内時間を classifier の時計に反映
                    self._video_now = _video_frame_pos / fps
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
                _battle_elapsed = self._now() - self._battle_active_since
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
                                # YOLO経路のbattle_endはocr_results=[]で発行するため
                                # 確認用OCRテキストからここで勝敗を拾っておく
                                if self._battle_result is None:
                                    self._battle_result = _detect_battle_result(_end_joined)
                                self._end_screen_count = 0
                                turn += 1
                                self._phase_classifier.set_processing(True)
                                try:
                                    self._process_event(frame, yolo_state, [], "battle_end", turn)
                                finally:
                                    self._phase_classifier.set_processing(False)
                                    self._phase_classifier._battle_started = False
                                    # classifier の時計（動画モードでは動画内時間）で書き込む。
                                    # time.time() だと動画モードで時計の原点が異なり debounce 判定が壊れる
                                    self._phase_classifier._last_event_time["turn_start"] = self._now()
                                continue
                    else:
                        self._end_screen_count = 0

                # ── HPバーピクセル解析（バトル中は毎フレーム）────────────────
                # 軽量なピクセル処理のためOCRサイクルに依存させない。
                # コマンド画面は静止していてdiff検出が発火せず、OCRゲート内（旧実装）だと
                # 定期1.5秒間隔のみ→安定化フィルタ（連続6サンプル）が埋まらず
                # HP%がほぼ更新されなかった（実機で確認）
                if self._battle_active:
                    _pixel_hp = self._hpbar_analyzer.analyze(frame)
                    self._battle_tracker.update_pixel_hp(_pixel_hp)

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

                    if self._battle_active:
                        # 定期OCRでも自分側ポケモン名を蓄積（battle_start時に映らなかったポケモンの補完）
                        # イベントOCRに比べてUIノイズが多いため、y座標分類済みのみを対象とする
                        if ocr_results:
                            _periodic_gs = _extract_structured_info(ocr_results, self._classifier)
                            _periodic_player = _periodic_gs.get("name_candidates_player", [])
                            _periodic_opp = _periodic_gs.get("name_candidates_opponent", [])
                            for _pname in _periodic_player:
                                self._battle_tracker.accumulate_player_name(_pname, _periodic_opp)
                            # スロット番号の補完割当（イベント時フレームでは相手名が
                            # 読めないことが多く、ここでの割当が相手側HPpxの生命線。
                            # command_cyゲート付き=行動選択画面のみ）
                            self._battle_tracker.assign_slots_from_ocr(
                                _periodic_gs.get("name_player_with_cx", []),
                                _periodic_gs.get("name_opponent_with_cx", []),
                                _periodic_gs.get("command_cy"))
                            # 数値HPの補完割当（イベント時フレームはHP数値が
                            # 映らないことが多く、画面の正読値はここでしか拾えない。
                            # HPpxが物理限界で読めない低HP帯の生命線）。
                            # ゲートは位置ベース（y帯＋バー中心cx近接）で
                            # assign_hp_from_ocr 内で行う
                            self._battle_tracker.assign_hp_from_ocr(
                                _periodic_gs.get("hp_player_with_xy", []),
                                _periodic_gs.get("hp_opponent_with_xy", []))
                            # レンダーモード: 戦況パネル用スナップショット（v2b）
                            self._record_panel_state()

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
                            and self._now() - self._last_battle_end_time < self._BATTLE_START_COOLDOWN):
                        remaining = self._BATTLE_START_COOLDOWN - (self._now() - self._last_battle_end_time)
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
                                elapsed = self._now() - self._pending_faint_time
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
                    _D_X2 = BattleMessageParser.MSG_X_MAX   # 1150
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
            # 動画モードの後付け実況生成（ADR-009追記）: スキャン完了後にまとめて
            # 実況文を生成する。画像は使わず、スキャン中に蓄積した構造化データのみを根拠にする。
            if self._posthoc_mode and self._pending_render_events:
                self._generate_posthoc_commentary()
            # レンダリング素材出力のサマリー（音声合成スレッドの完了を待ってから集計）
            if self._render_sink is not None:
                if self._speech_thread is not None:
                    self._speech_thread.join(timeout=30)
                saved = self._render_sink.count
                if saved > 0:
                    log.info("[レンダ] 素材出力完了: %d 件 → %s", saved, self._render_sink.out_dir)
                else:
                    log.warning("[レンダ] ⚠ 実況素材が 1 件も保存されていません。"
                                "VOICEVOX の起動と Bedrock（--ec2-url）の設定を確認してください。")
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
                self._battle_tracker.record_turn_snapshot()
                log.info(f"[ターン] T{self._battle_tracker.game_turn} 開始")
            # 保留中のfaintがタイムアウトしていれば単独Bedrock送信でフラッシュ
            if (self._pending_faint_state is not None
                    and self._now() - self._pending_faint_time >= self._FAINT_PENDING_TIMEOUT):
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
            self._reset_battle_state()
            log.info("[戦況] バトル開始 → トラッカーリセット")
            # バトル開始前にキャッシュした相手ポケモンを登録
            for name in self._pre_battle_opponent:
                self._battle_tracker.register_opponent_on_field(name)
            self._pre_battle_opponent.clear()
            # バトル開始前にキャッシュした自分ポケモンを登録（ゆけっ！検出分）
            for name in self._pre_battle_player:
                self._battle_tracker.mark_on_field_by_name(name)
            self._pre_battle_player.clear()
            # battle_start直前のメッセージ由来繰り出しをリセット後のトラッカーに引き継ぐ
            # （実機: 相手ロトム/オオニューラが登録7秒後のリセットで消滅し、以降
            #   保留どまり→opponent_faintが帰属先なしで消えていた）
            now = self._now()
            for ts, side, name in self._recent_sendouts:
                if now - ts > self._SENDOUT_CARRYOVER_SEC:
                    continue
                if side == "opponent":
                    self._battle_tracker.register_opponent_on_field(name)
                else:
                    self._battle_tracker.mark_on_field_by_name(name)
            self._recent_sendouts.clear()

        # battle_start が OCR品質不足でスキップされた場合のフォールバック
        # バトルイベントが来た時点でアクティブ化（遅延起動）。
        # トラッカーは battle_start と同様にリセットする: 以前は既存情報保持のまま
        # 起動していたため、battle_end→battle_start 間の遅延起動が前試合ロスターの
        # まま走り、新試合の繰り出しで eviction 連発していた（実機 08-15-22 で
        # 目撃53回のフラエッテまで削除・自己修復まで約21秒の表示汚染）
        if not self._battle_active and event_type in {"move_used", "faint", "switch"}:
            log.warning("[戦況] battle_start 未検知 → バトルをアクティブ化（遅延起動・前試合ロスターをクリア）")
            self._reset_battle_state()
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
        # レンダーモード: イベント処理後の戦況をパネル用に記録（v2b）
        self._record_panel_state()
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
        attempt_bedrock = bool(self._ec2_url and event_type in BEDROCK_EVENTS and self._battle_active)
        if attempt_bedrock:
            # ── faint保留: 即送信せず次のmove_usedと統合するため保留する ──
            if event_type == "faint":
                log.info("[faint保留] Bedrock送信を保留（次のmove_usedで統合予定）")
                self._pending_faint_state = game_state
                self._pending_faint_battle_context = battle_context
                self._pending_faint_frame = frame
                self._pending_faint_time = self._now()
                self._pending_faint_game_turn = self._battle_tracker.game_turn
                # 実況・VOICEVOX もスキップして終了（戦況更新は済み）
                return

            # ── move_usedで保留中のfaint情報があれば統合 ──
            if event_type == "move_used" and self._pending_faint_state is not None:
                elapsed = self._now() - self._pending_faint_time
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

        # ── 勝敗検出（battle_endのみ）─────────────────────────────────────────
        # OCRテキストから「勝負に勝った/負けた」を拾い、battle_endイベントの
        # game_stateにのみ注入する（それ以前のイベントに混ぜるとネタバレになる）。
        # 後付け生成はこのgame_stateをそのままバッファするため、ライブ・動画
        # モード共通でこの1箇所で済む。
        if event_type == "battle_end":
            if self._battle_result is None:
                self._battle_result = _detect_battle_result(game_state.get("ocr_text", ""))
            if self._battle_result:
                game_state = dict(game_state)
                game_state["battle_result"] = self._battle_result
                log.info("[戦況] 勝敗検出: %s", self._battle_result)

        # ── 実況文の生成・再生（ライブ）／後付け生成用バッファへの追加（動画モード）──
        # event_time はこのハンドラが処理中のフレームの動画内時刻（同期実行なので
        # _now() はイベント検知時点と同値）
        self._dispatch_commentary(event_type, frame, game_state, battle_context,
                                   self._move_log_display(5), attempt_bedrock)

        # バトル終了後にアクティブフラグをリセット（Bedrock呼び出し後）
        if event_type == "battle_end":
            self._battle_active = False
            self._last_battle_end_time = self._now()
            self._pre_battle_opponent.clear()
            self._pre_battle_player.clear()
            self._recent_sendouts.clear()  # 前試合の繰り出しを次試合に引き継がない
            log.info("[戦況] バトル終了 → トラッカー非アクティブ化")

        # デバッグ用スクリーンショット保存
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"pipeline_turn_{turn:03d}.png"), frame)

    def _dispatch_commentary(
        self,
        event_type: str,
        frame: "np.ndarray | None",
        game_state: dict,
        battle_context: dict | None,
        move_log: list[str],
        attempt_bedrock: bool,
        event_time: float | None = None,
    ) -> None:
        """実況文を決定して再生する（ライブ）か、後付け生成用にバッファする（動画モード）。

        動画モード＋素材出力時（``self._posthoc_mode``）は、Bedrock/Phi-3の呼び出しを
        一切ここでは行わず、``run()``完了後にまとめて生成する（ADR-009追記）。
        ライブモードでは従来どおりこの場でBedrock Visionを呼び、即座に再生する。
        """
        if self._posthoc_mode:
            self._pending_render_events.append({
                "event_time": event_time if event_time is not None else self._now(),
                "event_type": event_type,
                "game_state": game_state,
                "battle_context": battle_context,
                "move_log": move_log,
                "render_context": self._render_context(battle_context),
            })
            return

        bedrock_commentary: str | None = None
        bedrock_analysis: str | None = None
        if attempt_bedrock:
            log.debug("Bedrock Vision 呼び出し中...")
            if self._move_log:
                log.debug(f"[技ログ] {' / '.join(self._move_log[-5:])}")
            t0 = time.perf_counter()
            bedrock_commentary, bedrock_analysis = _call_bedrock_vision(
                self._ec2_url, frame, game_state, event_type,
                self._commentary_history, battle_context, self._classifier, move_log,
            )
            if bedrock_commentary:
                log.info(f"Bedrock 完了 ({time.perf_counter()-t0:.2f}s): 「{bedrock_commentary}」")

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

        # 保留・困惑応答は「AIグリッチ」定型文に差し替える（VOICEVOX合成前）
        commentary = _replace_glitch_commentary(commentary)

        self._commentary_history.append(commentary)
        if len(self._commentary_history) > 5:
            self._commentary_history.pop(0)

        self._speak_async(commentary, event_type=event_type, event_time=event_time,
                          context=self._render_context(battle_context))

    def _flush_pending_faint(self) -> None:
        """タイムアウトした保留faintを単独でBedrock送信して実況する。"""
        if self._pending_faint_state is None:
            return
        game_state      = self._pending_faint_state
        battle_context  = self._pending_faint_battle_context
        frame           = self._pending_faint_frame
        pending_time    = self._pending_faint_time
        self._pending_faint_state          = None
        self._pending_faint_battle_context = None
        self._pending_faint_frame          = None

        if not (self._ec2_url and self._battle_active):
            return

        # event_time は faint 検知時点の動画内時刻（保留中に動画が進んでいるため
        # 現在時刻ではなく保留開始時刻を使う）
        self._dispatch_commentary(
            "faint", frame, game_state, battle_context,
            self._move_log_display(5), attempt_bedrock=True,
            event_time=pending_time,
        )

    def _generate_posthoc_commentary(self) -> None:
        """動画モードの後付け実況生成（ADR-009追記）。

        `run()` の動画スキャンが完了した後に呼ばれる。バッファ済みの各イベントに
        ついて、画像なし・構造化データのみで Bedrock（失敗時は Phi-3）に実況文を
        生成させ、VOICEVOXで音声合成して `render_sink` に追記する。

        イベントは検知順（＝動画内時刻順）にバッファされているため、順番に処理
        することで `history`（直前の実況の繰り返し防止）がライブ経路と同じように
        機能する。各イベントのcontext/battle_stateはスキャン中に捕捉した時点の
        値のままなので、未来の情報が混ざることはない（スポイラー安全性の担保）。
        """
        history: list[str] = []
        for ev in self._pending_render_events:
            bedrock_commentary, bedrock_analysis = _call_bedrock_text(
                self._ec2_url, ev["game_state"], ev["event_type"], history,
                ev["battle_context"], self._classifier, ev["move_log"],
            )
            if bedrock_commentary:
                commentary = _clean_commentary(bedrock_commentary)
            else:
                phi3_context = bedrock_analysis or ev["game_state"]["ocr_text"]
                try:
                    commentary = _clean_commentary(
                        self._phi3.generate_commentary(ev["game_state"], bedrock_analysis=phi3_context))
                except Exception as e:
                    log.error(f"Phi-3 エラー（後付け）: {e}")
                    continue
            if not commentary:
                continue

            # 保留・困惑応答は「AIグリッチ」定型文に差し替える（VOICEVOX合成前・
            # manifest.jsonlに問題テキストを一度も書き込ませない）
            commentary = _replace_glitch_commentary(commentary)

            history.append(commentary)
            if len(history) > 5:
                history.pop(0)

            try:
                wav_bytes = self._voicevox.generate_wav(commentary)
            except Exception as e:
                log.error(f"VOICEVOX エラー（後付け）: {e}")
                continue

            entry = self._render_sink.add(ev["event_time"], ev["event_type"], commentary,
                                          wav_bytes, context=ev["render_context"])
            log.info("[レンダ][後付け] #%d %s t=%.1fs (%.1f秒) → %s",
                     entry["seq"], ev["event_type"], entry["event_time"],
                     entry["duration"], entry["wav"])

    def _record_panel_state(self) -> None:
        """レンダーモード時、戦況パネル用スナップショットを states.jsonl に記録する。

        イベント処理後と定期OCR後に呼ばれる。同一状態はRenderSink側で
        デデュープされるため高頻度で呼んでも肥大しない。
        """
        render_sink = getattr(self, "_render_sink", None)
        if render_sink is None or not self._battle_active:
            return
        try:
            render_sink.add_state(self._now(), self._battle_tracker.to_panel_state())
        except Exception as e:
            log.error(f"パネル状態記録エラー: {e}")

    def _render_context(self, battle_context: dict | None) -> dict | None:
        """レンダリング素材のマニフェストに記録する戦況サマリーを組み立てる。

        台本パス（ADR-009・ギャップフィラー生成）がイベント間の戦況を
        把握できるようにするための情報。レンダーモード以外では None。
        """
        if self._render_sink is None:
            return None
        ctx: dict = {"move_log": self._move_log_display(5)}
        if battle_context:
            ctx["turn"] = battle_context.get("turn")
            ctx["player"] = battle_context.get("player_pokemon")
            ctx["opponent"] = battle_context.get("opponent_pokemon")
        return ctx

    def _speak_async(self, commentary: str, event_type: str = "unknown",
                     event_time: float | None = None,
                     context: dict | None = None) -> None:
        """VOICEVOX 音声合成・再生を別スレッドで実行する（メインループをブロックしない）。
        前の再生が残っていれば停止してから新しい音声を流す。

        レンダリング素材出力モード（--render-out）では再生せず、WAV保存＋
        マニフェスト追記に切り替わる。event_time はイベント検知時点の動画内時刻。
        呼び出し元スレッドで確定させる（合成スレッド内では動画が先に進んでいるため）。
        context はマニフェストに記録する戦況サマリー（``_render_context()``）。
        """
        if event_time is None:
            event_time = self._now()

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
            # レンダリング素材出力モード: 再生せず保存して終了
            if self._render_sink is not None:
                try:
                    entry = self._render_sink.add(event_time, event_type, commentary,
                                                  wav_bytes, context=context)
                    log.info("[レンダ] #%d %s t=%.1fs (%.1f秒) → %s",
                             entry["seq"], event_type, entry["event_time"],
                             entry["duration"], entry["wav"])
                except Exception as e:
                    log.error(f"レンダ素材保存エラー: {e}")
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

        # パターン1: 「〜は戻っていく」（とんぼがえり等の交代技）
        # 例: "ゴリランダーは" → "ゴリランダー"
        # ⚠️ ゲートは「戻って」（連用形）に限定する。旧実装は「戻」1文字含有で
        # ゲートしていたため、「しろいハーブで ステータスを 元に戻した」（アイテム回復
        # メッセージ）や「攻撃から 身を守った」のOCR誤読断片「戻る」にも誤反応し、
        # set_not_on_field の無条件両側検索で無関係なポケモンを誤ベンチ化していた
        # （実機: 20-14-17でオンバーン/オオニューラが一時誤ベンチ化・自己修復。
        #   07-00-19では自分のオオニューラが試合終了まで誤ベンチのまま残留）。
        if any("戻って" in t for t in texts):
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
                    # 「〇〇 戻れ！」は自分のコマンド交代のみ（相手は「引っこめた」形式）
                    found = self._battle_tracker.mark_bench_by_name(result.canonical_ja, side="player")
                    if found:
                        log.info(f"[交代検知] {result.canonical_ja} が場から退いた（戻れ検出）")
                    break

    def _note_sendout(self, side: str, name: str) -> None:
        """メッセージ由来の繰り出しを履歴に記録する（battle_startリセット後の引き継ぎ用）。"""
        now = self._now()
        self._recent_sendouts = [(t, s, n) for (t, s, n) in self._recent_sendouts
                                 if now - t <= 60.0]
        self._recent_sendouts.append((now, side, name))

    def _handle_message_event(self, ev: dict) -> None:
        """BattleMessageParser から受け取ったメッセージイベントで戦況を補完する。"""
        event_type = ev["type"]
        pokemon = ev["pokemon"]
        if event_type == "faint":
            if not self._battle_tracker.confirm_player_faint_by_name(pokemon):
                log.warning(f"[戦況] 気絶メッセージの帰属先が見つかりません: 自分 {pokemon}")
        elif event_type == "opponent_faint":
            canonical = pokemon
            confident = False
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
                    confident = (result.category == CATEGORY_POKEMON
                                 and result.score >= 90)
            if not self._battle_tracker.confirm_opponent_faint_by_name(canonical):
                # 繰り出し取りこぼしで未登録のまま倒れた相手の救済。
                # 気絶メッセージ＋DB高確信マッチの二重根拠がある場合のみ登録する
                if confident:
                    self._battle_tracker.register_opponent_fainted(canonical)
                else:
                    log.warning(f"[戦況] 気絶メッセージの帰属先が見つかりません: 相手 {canonical}")
        elif event_type == "switch_in":
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            self._battle_tracker.mark_on_field_by_name(canonical)
            self._note_sendout("player", canonical)
        elif event_type == "opponent_switch_in":
            # PokeClassifierで正規化してから相手スロットに登録
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            self._battle_tracker.register_opponent_on_field(canonical)
            self._note_sendout("opponent", canonical)
        elif event_type == "switch_out":
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            # もどれ/戻れ/こうたいした = 自分の交代（同名ミラーで相手側を誤ベンチ化しない）
            self._battle_tracker.mark_bench_by_name(canonical, side="player")
        elif event_type == "opponent_switch_out":
            canonical = pokemon
            if self._classifier:
                result = self._classifier.classify(pokemon)
                if result and result.canonical_ja:
                    canonical = result.canonical_ja
            # 「(トレーナー名)は 〇〇を 引っこめた」= 相手の交代
            self._battle_tracker.mark_bench_by_name(canonical, side="opponent")
        elif event_type == "status":
            self._battle_tracker.update_status_by_name(pokemon, ev.get("status", ""))

    def _move_log_display(self, n: int = 5) -> list[str]:
        """Bedrock 送信用に技ログ末尾 n 件を整形する。

        使い手を特定できず「場の1匹目」フォールバックで仮登録し、まだ後付け修正
        （_update_move_log 冒頭の仮確定エントリ突合）で確定していないエントリには
        「（推定）」を付け、LLM に使い手の確度が低いことを伝える。
        """
        tentative_entries = {t["old_entry"] for t in self._tentative_opponent_moves}
        return [
            f"{e}（推定）" if e in tentative_entries else e
            for e in self._move_log[-n:]
        ]

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
                else:
                    # 図鑑全体のファジー分類に失敗した場合でも、既にロスターにいる
                    # ポケモンの使用者名見切れ（例: 「ドドゲザンの」→「ドゲザンの」）の
                    # 可能性があるため、ロスター名との前方一致吸収（_get_or_create と同じ
                    # 考え方）で救済を試みる。それも失敗するなら生のOCR断片を技ログへ
                    # 流すリスクが高いため登録を拒否する。
                    # （実機で「トムの」「キの」等の見切れ断片が無検証のまま
                    #   「信頼度高」技として実況に渡っていたバグの再発防止）
                    roster_names = {s.name for s in self._battle_tracker._player} | \
                                   {s.name for s in self._battle_tracker._opponent}
                    matched = next(
                        (rn for rn in roster_names
                         if min(len(rn), len(pokemon_name)) >= self._battle_tracker._ABSORB_MIN_LEN
                         and self._battle_tracker._fuzzy_name_match(rn, pokemon_name)),
                        None,
                    )
                    if matched:
                        log.info("[技ログ] 使用者名候補 %s はロスターの %s の見切れ断片と判定 → 補正",
                                 pokemon_name, matched)
                        pokemon_name = matched
                    else:
                        log.debug("[技ログ] 使用者名候補 %s を分類・ロスター一致とも失敗のため棄却（技候補: %s）",
                                  pokemon_name, move_candidate)
                        return False
                # is_opponent=True で技ログにだけ記録されロスター未登録のケースを
                # 即座に校正登録する（実機: 07-00-19でガブリアスの繰り出しメッセージが
                # OCR取りこぼしで検知されないまま技ログにだけ記録され、ロスター未登録
                # 状態がBedrockへの矛盾したcontextとして露呈し「保留」応答を誘発した）。
                # 技名はこの時点で既にPokeClassifierスコア80以上（またはロスター名
                # 前方一致）で確定済みの高信頼情報のため、低信頼OCR経路の幽霊登録
                # ヒステリシス（_NEW_NAME_CONFIRM_COUNT）を経由せず register_opponent_on_field
                # （_get_or_create low_trust=False）で即時登録してよいと判断。
                if is_opponent:
                    known_opp = {s.name for s in self._battle_tracker._opponent}
                    if pokemon_name not in known_opp:
                        self._battle_tracker.register_opponent_on_field(pokemon_name)
                        log.info("[戦況] 技ログから相手 %s をロスターへ校正登録（繰り出し検知漏れ救済）",
                                 pokemon_name)
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
                # レンダーモード: 技が画面に映った瞬間の動画内時刻を記録
                # （台本パスがライブ実況風の時刻アンカーとして使う）
                # getattr: テストが部分構築のPipelineで_update_move_logを呼ぶため
                render_sink = getattr(self, "_render_sink", None)
                if render_sink is not None:
                    render_sink.add_moment(
                        self._now(), "move", entry,
                        side=self._battle_tracker.move_user_side(
                            pokemon_name, is_opponent=is_opponent))
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
            search_sources = ocr_results
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
                        help="YOLOv8 カスタムモデルのパス（状態異常検出用・例: runs/detect/train4/weights/best.pt）。"
                             "現在はテキストOCRで代替済みのためデフォルト無効。将来再利用する場合のみ指定")
    parser.add_argument("--ball-model", default=None,
                        help="ボール検出専用モデルのパス（例: runs/detect/train7/weights/best.pt）。"
                             "現在はパイプラインで未使用（デフォルト無効）。将来再利用する場合のみ指定")
    parser.add_argument("--end-model", default=None,
                        help="終了画面検出モデルのパス（例: runs/detect/train_end_screen2/weights/best.pt）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="キャプチャ間隔（秒、デフォルト: 1.0）")
    parser.add_argument("--speaker", type=int,   default=2,
                        help="VOICEVOX 話者 ID（デフォルト: 2 = 四国めたん）")
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
    parser.add_argument("--render-out", default=None,
                        help="実況動画レンダリング素材の出力ディレクトリ（ADR-009 パス1）。"
                             "指定時は実況音声を再生せず WAV + manifest.jsonl を保存する。"
                             "--input（動画モード）との併用を想定。")

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
        render_out=args.render_out,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
