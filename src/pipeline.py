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
import sys
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Bedrock を呼ぶイベント種別（ADR-007）
BEDROCK_EVENTS = {"turn_end", "switch", "faint"}


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


def _ocr_results_to_text(ocr_results: list[dict]) -> str:
    """OCR 結果を読みやすいテキスト文字列にまとめる（最大 OCR_MAX_CHARS 文字）。"""
    lines = [r["text"] for r in ocr_results if r["confidence"] >= 0.4]
    text = " / ".join(lines) if lines else "（テキスト未検出）"
    return text[:OCR_MAX_CHARS]


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
    }


# ─── イベント種別の判定 ────────────────────────────────────────────────────────

def _detect_event_type(
    yolo_state: BattleState,
    prev_yolo: BattleState | None,
) -> str:
    """YOLO の前後フレーム差分からイベント種別を判定する。"""
    if prev_yolo is None:
        return "turn_end"

    # ボール数が減った → 気絶
    if (prev_yolo.player_balls.alive > yolo_state.player_balls.alive or
            prev_yolo.opponent_balls.alive > yolo_state.opponent_balls.alive):
        return "faint"

    # 状態異常が新たに付与された → 状態異常付与
    if (yolo_state.player_status and yolo_state.player_status != prev_yolo.player_status or
            yolo_state.opponent_status and yolo_state.opponent_status != prev_yolo.opponent_status):
        return "status_change"

    # それ以外の差分検知 → ターン切替とみなす
    return "turn_end"


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
    import re

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
) -> str | None:
    """
    EC2 API に画像と状況を送り、Bedrock Vision 分析結果を受け取る。
    失敗してもパイプラインを止めない（None を返す）。
    """
    try:
        # 縮小してから PNG エンコード（nginx の 5MB 制限対策・640x360 で確実に収まる）
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".png", small)
        image_b64 = base64.b64encode(buf.tobytes()).decode()

        # server.py の /api/vision は context.event_type でバリデーションする
        status_parts = (game_state.get("status", "") or "").split(" / 相手: ")
        status_player   = status_parts[0] if status_parts[0] != "なし" else "なし"
        status_opponent = status_parts[1] if len(status_parts) > 1 else "なし"

        balls = game_state.get("balls_remaining", [])
        payload = {
            "image_base64": image_b64,
            "context": {
                "pokemon_player":          "不明",
                "hp_player":               "?",
                "pokemon_opponent":        "不明",
                "hp_opponent":             "?",
                "last_move":               "不明",
                "status_player":           status_player,
                "status_opponent":         status_opponent,
                "balls_remaining_player":  balls[0] if len(balls) > 0 else "?",
                "balls_remaining_opponent": balls[1] if len(balls) > 1 else "?",
                "event_type":              event_type,  # context 内に入れる（server.py の要件）
                "ocr_text":                game_state.get("ocr_text", ""),
            },
            "history": commentary_history[-3:],
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
        self._diff_detector = DiffDetector()
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

                # ── 差分検出 ────────────────────────────────────────────────
                event_detected, diff_score = self._diff_detector.detect(frame)

                if event_detected:
                    turn += 1
                    event_type = _detect_event_type(yolo_state, self._prev_yolo)
                    log.info(f"[ターン {turn}] イベント検知 (diff={diff_score:.1f}, type={event_type})")

                    self._process_event(frame, yolo_state, event_type, turn)

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
        event_type: str,
        turn: int,
    ) -> None:
        """イベント発生時の一連の処理（OCR → Phi-3 → VOICEVOX → 再生）。"""

        # ── OCR ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        ocr_results = run_ocr(self._reader, frame)
        log.info(f"OCR: {len(ocr_results)} 件検出 ({time.perf_counter()-t0:.2f}s)")

        # ── バトル外画面はスキップ ─────────────────────────────────────────────
        if not _is_battle_screen(ocr_results):
            log.info("バトル外の画面を検知 → スキップ")
            return

        # OCR 件数が少なすぎる場合はハルシネーションが起きるためスキップ
        if len(ocr_results) < 2:
            log.info(f"OCR 件数が少なすぎる（{len(ocr_results)} 件）→ スキップ")
            return

        # ── game_state 構築 ───────────────────────────────────────────────────
        game_state = _build_game_state(ocr_results, yolo_state, event_type, self._prev_yolo)
        log.info(f"[状態] {yolo_state.summary()} | OCR: {game_state['ocr_text']}")

        # ── Bedrock Vision（対象イベントのみ・EC2 URL が設定されている場合）──
        bedrock_commentary: str | None = None
        bedrock_analysis: str | None = None
        if self._ec2_url and event_type in BEDROCK_EVENTS:
            log.info("Bedrock Vision 呼び出し中...")
            t0 = time.perf_counter()
            bedrock_commentary, bedrock_analysis = _call_bedrock_vision(
                self._ec2_url, frame, game_state, event_type, self._commentary_history
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
