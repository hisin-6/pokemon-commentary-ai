"""v2c（VMC録画ワイプ）アバター収録時に、`commentary_track.wav` の再生と同時に
VMCへOSCで表情ブレンドシェイプ・腕の姿勢を送り、実況イベントに合わせて表情を
切り替える（改善ロードマップ③・表情・モーション連動）。

`scripts/play_commentary_track.bat` の代わりにこちらを使う。動作:
  0. 再生開始前に一度、腕を下ろした姿勢をOSCで送る（トラッキングデバイス
     無しだとT-poseのまま表示される問題への対策・`scripts/test_vmc_pose.py`で
     角度を実機検証済み）
  1. renders/<動画名>/schedule.json（パス2の出力）を読み、イベント時刻と
     event_type/context（faint_side・battle_result）から表情を決定
  2. WAVをPython側（sounddevice）で自前に再生し、再生開始を基準時刻にする
  3. 各イベント時刻が来るたびVMCへ /VMC/Ext/Blend/Val + /VMC/Ext/Blend/Apply を送信
  4. 最後のイベントの少し後にNeutralへ戻す

事前準備（Windows側・初回のみ）:
    venv\\Scripts\\pip.exe install python-osc

VMC側の設定（`scripts/test_vmc_expression.py`/`test_vmc_pose.py`で確認済みの手順）:
  - VMCの設定画面でReceiver（39540 or 39541）の「有効化」チェックボックスをON
  - WAV再生の出力先はCABLE Input（口パクの自動連動はVMC側の音声入力設定のまま）

使い方:
    venv\\Scripts\\python.exe scripts\\play_and_animate_avatar.py 16-14-39 --osc-port 39540
    （OBS録画を先に開始してから実行すること。schedule.jsonが無ければ先に
     render_commentary_video.py --dry-run を実行して生成する）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path

import soundfile as sf
import sounddevice as sd
from pythonosc.udp_client import SimpleUDPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# VRM 0.x 標準ブレンドシェイププリセット名
NEUTRAL = "Neutral"

# T-pose回避: 腕を下ろした姿勢（Z軸まわり回転・`test_vmc_pose.py`で実機確認済みの角度）。
# LeftUpperArmはこの角度そのまま、RightUpperArmは符号反転（鏡像）で送る。
# 現状は上半身のみ表示（--avatar-crop）のため、指先が服にわずかに埋まるのは実害なしと判断。
_IDLE_POSE_DEG = 80.0
_IDLE_POSE_AXIS = "z"
_IDLE_POSE_BONES = ["LeftUpperArm", "RightUpperArm"]

# event_type → ブレンドシェイプ名（context依存が無いもの）
_EVENT_EXPRESSION = {
    "battle_start": "Fun",
    "move_used": None,   # 無表情のまま（技1つ1つには反応させない・過剰演出防止）
    "filler": None,      # 中立のまま
}

# faint_side → 表情（自分が倒れたら哀しい／相手を倒したら嬉しい）
_FAINT_EXPRESSION = {
    "player": "Sorrow",
    "opponent": "Joy",
}

# battle_result（"勝ち"/"負け"）→ 表情
_BATTLE_RESULT_EXPRESSION = {
    "勝ち": "Joy",
    "負け": "Sorrow",
}

# 表情を維持する秒数（この時間が経つとNeutralに自動で戻る。次のイベントが
# それより早く来ればそちらが上書きするので実質「短命な表情」として機能する）
_EXPRESSION_HOLD_SEC = 4.0


def expression_for(event_type: str, context: dict) -> str | None:
    """イベント種別とcontextから送るべきブレンドシェイプ名を決める。
    None の場合は表情を変えない（現状維持）。
    """
    if event_type == "faint":
        side = context.get("faint_side")
        return _FAINT_EXPRESSION.get(side)  # 判定不能(None)なら表情変更なし
    if event_type == "battle_end":
        result = context.get("battle_result")
        return _BATTLE_RESULT_EXPRESSION.get(result)
    return _EVENT_EXPRESSION.get(event_type)


def load_schedule(render_dir: Path) -> list[dict]:
    schedule_path = render_dir / "schedule.json"
    if not schedule_path.exists():
        raise FileNotFoundError(
            f"schedule.json が見つかりません: {schedule_path}\n"
            "先に render_commentary_video.py --dry-run（またはパス2本実行）を実行してください")
    data = json.loads(schedule_path.read_text(encoding="utf-8"))
    return data["scheduled"]


def build_expression_timeline(scheduled: list[dict]) -> list[tuple[float, str]]:
    """(発火時刻, ブレンドシェイプ名) のリストを開始時刻順で返す（表情変更なしイベントは除く）。"""
    timeline = []
    for entry in scheduled:
        expr = expression_for(entry.get("event_type", ""), entry.get("context") or {})
        if expr:
            timeline.append((float(entry["start"]), expr))
    timeline.sort(key=lambda t: t[0])
    return timeline


def _axis_angle_quat(axis: str, deg: float) -> tuple[float, float, float, float]:
    """指定軸まわり deg 度の回転をクォータニオン(x,y,z,w)で返す。"""
    rad = math.radians(deg)
    s, c = math.sin(rad / 2), math.cos(rad / 2)
    if axis == "x":
        return (s, 0.0, 0.0, c)
    if axis == "y":
        return (0.0, s, 0.0, c)
    return (0.0, 0.0, s, c)


def send_idle_pose(client: SimpleUDPClient) -> None:
    """トラッキングデバイス無しでT-poseのまま表示される問題への対策として、
    腕を下ろした姿勢をOSCで一度送る（`_IDLE_POSE_DEG`等は`test_vmc_pose.py`で
    実機確認済みの角度）。"""
    for bone in _IDLE_POSE_BONES:
        deg = -_IDLE_POSE_DEG if "Right" in bone else _IDLE_POSE_DEG
        qx, qy, qz, qw = _axis_angle_quat(_IDLE_POSE_AXIS, deg)
        client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
    log.info("[姿勢] 腕を下ろした初期姿勢を送信")


def send_expression(client: SimpleUDPClient, name: str) -> None:
    """指定のブレンドシェイプだけを1.0にし、他はNeutral含め全部0にして送る。"""
    all_known = {NEUTRAL, "Fun", "Joy", "Sorrow", "Angry"}
    for preset in all_known:
        client.send_message("/VMC/Ext/Blend/Val", [preset, 1.0 if preset == name else 0.0])
    client.send_message("/VMC/Ext/Blend/Apply", [])
    log.info("[表情] %s", name)


def run_expression_scheduler(client: SimpleUDPClient, timeline: list[tuple[float, str]],
                             start_clock: float, total_duration: float) -> None:
    """再生開始時刻(start_clock)を基準に、timelineの時刻が来るたびOSCを送る。
    表情は_EXPRESSION_HOLD_SEC秒後（次のイベントより先に来れば）Neutralへ戻す。
    """
    for i, (t, expr) in enumerate(timeline):
        wait = start_clock + t - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        send_expression(client, expr)

        next_t = timeline[i + 1][0] if i + 1 < len(timeline) else total_duration
        hold_until = min(t + _EXPRESSION_HOLD_SEC, next_t)
        wait = start_clock + hold_until - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if hold_until < next_t:
            send_expression(client, NEUTRAL)


def main() -> int:
    parser = argparse.ArgumentParser(description="実況WAV再生＋VMC表情連動（v2cアバター収録用）")
    parser.add_argument("name", help="renders配下の動画名フォルダ")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, required=True, help="VMCのReceiverポート番号")
    parser.add_argument("--device", default=None, help="WAV再生の出力デバイス（既定: システムデフォルト。"
                        "CABLE Inputに流す場合はデバイス名/番号を指定）")
    parser.add_argument("--no-idle-pose", action="store_true",
                        help="腕を下ろす初期姿勢の送信をスキップ（別途トラッキングデバイスを"
                             "使う場合などT-pose対策が不要な時に指定）")
    args = parser.parse_args()

    render_dir = Path(__file__).parent.parent / "renders" / args.name
    wav_path = render_dir / "commentary_track.wav"
    if not wav_path.exists():
        log.error("commentary_track.wav が見つかりません: %s", wav_path)
        log.error("パス2（render_commentary_video.py）を先に実行してください")
        return 1

    scheduled = load_schedule(render_dir)
    timeline = build_expression_timeline(scheduled)
    log.info("表情イベント %d件を検出（全%d件中）", len(timeline), len(scheduled))

    data, samplerate = sf.read(wav_path)
    duration = len(data) / samplerate

    client = SimpleUDPClient(args.host, args.osc_port)
    send_expression(client, NEUTRAL)
    if not args.no_idle_pose:
        send_idle_pose(client)

    log.info("※OBS録画を先に開始してから実行してください")
    log.info("再生開始: %s（%.1f秒）", wav_path, duration)
    sd.play(data, samplerate, device=args.device)
    start_clock = time.monotonic()

    scheduler = threading.Thread(
        target=run_expression_scheduler, args=(client, timeline, start_clock, duration),
        daemon=True)
    scheduler.start()

    sd.wait()
    scheduler.join(timeout=_EXPRESSION_HOLD_SEC + 1.0)
    send_expression(client, NEUTRAL)
    log.info("完了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
