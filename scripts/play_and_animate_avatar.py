"""v2c（VMC録画ワイプ）アバター収録時に、`commentary_track.wav` の再生と同時に
VMCへOSCで表情ブレンドシェイプ・腕の姿勢を送り、実況イベントに合わせて表情を
切り替える（改善ロードマップ③・表情・モーション連動）。

`scripts/play_commentary_track.bat` の代わりにこちらを使う。動作:
  0. 起動直後、腕を下ろした姿勢をOSCで送る（トラッキングデバイス無しだと
     T-poseのまま表示される問題への対策・`scripts/test_vmc_pose.py`で
     角度を実機検証済み）→ ここでEnter入力待ちになる（下記「使い方」参照）
  1. renders/<動画名>/schedule.json（パス2の出力）を読み、イベント時刻と
     event_type/context（faint_side・battle_result）・実況テキストから表情を決定
     （気絶/勝敗/試合開始は専用ルール。技ごとの実況move_single/move_used/switchは
     実況テキストのキーワードからJoy寄り/Sorrow寄りを推定=2026-08-01追加）
  2. WAVをPython側（sounddevice）で自前に再生し、再生開始を基準時刻にする
  3. 各イベント時刻が来るたびVMCへ /VMC/Ext/Blend/Val + /VMC/Ext/Blend/Apply を送信
     （表情が変わった瞬間は/VMC/Ext/Bone/PosでNeckのうなずきジェスチャーも1回送る）
  4. 最後のイベントの少し後にNeutralへ戻す
  5. 再生中はSpineへゆっくりスウェイ（待機モーション）を送り続ける（`--no-sway`で無効化可）

事前準備（Windows側・初回のみ）:
    venv\\Scripts\\pip.exe install python-osc

VMC側の設定（`scripts/test_vmc_expression.py`/`test_vmc_pose.py`で確認済みの手順）:
  - VMCの設定画面でReceiver（39540 or 39541）の「有効化」チェックボックスをON
  - WAV再生の出力先はCABLE Input（口パクの自動連動はVMC側の音声入力設定のまま）

使い方（2026-07-30〜: スクリプト起動→アイドル姿勢送信→OBS録画→Enterの順に変更。
録画開始前にTポーズが解消されるので、録画にTポーズが映り込まない）:
    venv\\Scripts\\python.exe scripts\\play_and_animate_avatar.py 16-14-39 --osc-port 39540
    1. スクリプトを起動する（この時点でVMCの腕が下がる）
    2. 表示された「録画が始まったらEnterを押してください」を待って、ここでOBS録画を開始
    3. 録画が始まったらEnterを押す（この瞬間からWAV再生・表情連動が始まる）
    → 録画開始がWAV再生開始より先になるので --avatar-offset は 0（または
      Enterを押す一瞬のラグ分の1秒未満）でよい。schedule.jsonが無ければ先に
      render_commentary_video.py --dry-run を実行して生成する
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
    "filler": None,      # 中立のまま（雑談なので実況内容と紐付いた反応はさせない）
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

# 技ごとの実況（move_single/move_used/switch）にコンテンツ連動の表情を付ける
# （改善ロードマップ③続き・2026-08-01）。実況テキストのキーワードから感情を
# 推定するシンプルな方式（_GLITCH_CAUSE_KEYWORDS等、このプロジェクトで既に
# 実績のあるキーワード分類パターンを踏襲）。くれぴのキャラ性質上テキストは
# 常にポジティブ寄りなので、"心配・苦戦"系の言葉だけを別枠で拾う設計。
_POSITIVE_KEYWORDS = (
    "バツグン", "急所", "決まった", "炸裂", "やったぁ", "素晴らしい",
    "ナイス", "無傷", "きっちり", "頑張れ", "いいぞ",
)
_NEGATIVE_KEYWORDS = (
    "ピンチ", "削られ", "いまひとつ", "外れ", "厳しい", "危ない",
    "食らって", "くらって", "苦戦",
)
# 感情反応を付けるイベント種別（move_used=無印/faint統合先を含む・move_single=技単独反応）
_SENTIMENT_EVENT_TYPES = {"move_used", "move_single", "switch"}


def _sentiment_expression(commentary: str) -> str | None:
    """実況テキストのキーワードから表情を推定する。両方/どちらも無ければNone
    （Noneは「表情を変えない」= 直前の表情を維持、を意味する。呼び出し側の
    expression_for/run_expression_schedulerを参照）。"""
    if any(kw in commentary for kw in _NEGATIVE_KEYWORDS):
        return "Sorrow"
    if any(kw in commentary for kw in _POSITIVE_KEYWORDS):
        return "Fun"
    return None


# 表情を維持する秒数（この時間が経つとNeutralに自動で戻る。次のイベントが
# それより早く来ればそちらが上書きするので実質「短命な表情」として機能する）
_EXPRESSION_HOLD_SEC = 4.0


def expression_for(event_type: str, context: dict, commentary: str = "") -> str | None:
    """イベント種別とcontextから送るべきブレンドシェイプ名を決める。
    None の場合は表情を変えない（現状維持）。

    faint_side は event_type=="faint"（保留タイムアウト時の単独dispatch）だけでなく
    event_type=="move_used"（Bedrock節約のため次のmove_usedに統合された通常経路）
    にも乗ってくるため、event_typeを問わずcontextを先に見る（pipeline.py側の
    faint統合バグ修正とセット・詳細はpipeline.pyの該当コメント参照）。

    上記どちらにも該当しない move_used/move_single/switch は、実況テキストの
    キーワードから_sentiment_expressionで表情を推定する（技1つ1つに反応させる・
    2026-08-01改善ロードマップ③続き）。battle_start/filler等は_EVENT_EXPRESSION
    の明示的な値（Noneも含む）を優先し、テキスト連動の対象にしない。
    """
    if "faint_side" in context:
        side = context.get("faint_side")
        return _FAINT_EXPRESSION.get(side)  # 判定不能(None)なら表情変更なし
    if event_type == "battle_end":
        result = context.get("battle_result")
        return _BATTLE_RESULT_EXPRESSION.get(result)
    if event_type in _EVENT_EXPRESSION:
        return _EVENT_EXPRESSION[event_type]
    if event_type in _SENTIMENT_EVENT_TYPES:
        return _sentiment_expression(commentary)
    return None


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
        expr = expression_for(entry.get("event_type", ""), entry.get("context") or {},
                              entry.get("commentary", ""))
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


# ── 常時の揺れ（モーション② 2026-08-01）────────────────────────────────────
# トラッキングデバイスが無く静止したままだと不自然なため、Spineをゆっくり
# 左右にスウェイさせて「呼吸・待機モーション」らしさを出す。腕下げ姿勢
# （_IDLE_POSE_BONES=UpperArm）とは別のボーンを使うので競合しない。
_SWAY_BONE = "Spine"
_SWAY_AXIS = "y"
_SWAY_AMPLITUDE_DEG = 4.0     # 振れ幅（小さめ・上半身のみ表示なので目立ちすぎないように）
_SWAY_PERIOD_SEC = 5.0        # 1往復にかかる秒数
_SWAY_UPDATE_INTERVAL_SEC = 0.15

# ── 表情変化時のうなずきジェスチャー（モーション② 2026-08-01）──────────────
# Neckを一瞬前後に傾けて「相槌」らしい動きを付ける。表情が変わった瞬間にだけ
# 発火し（Neutralへの自動復帰時は発火しない）、_NOD_SEQUENCE_DEGを順に送って
# 最後は0°（=Neck自体はニュートラル。基本姿勢はSwayが担当）に戻す。
_NOD_BONE = "Neck"
_NOD_AXIS = "x"
_NOD_SEQUENCE_DEG = (10.0, 14.0, 6.0, 0.0)
_NOD_STEP_SEC = 0.12


def _sway_quat(elapsed: float) -> tuple[float, float, float, float]:
    """待機モーション（スウェイ）の経過時間elapsed（秒）における回転を返す。
    正弦波なので周期_SWAY_PERIOD_SECで滑らかに往復する。"""
    deg = _SWAY_AMPLITUDE_DEG * math.sin(2 * math.pi * elapsed / _SWAY_PERIOD_SEC)
    return _axis_angle_quat(_SWAY_AXIS, deg)


def run_idle_sway(client: SimpleUDPClient, stop_event: threading.Event,
                  start_clock: float) -> None:
    """再生中ずっとSpineへ揺れを送り続ける（stop_eventがセットされたら終了）。"""
    while not stop_event.is_set():
        elapsed = time.monotonic() - start_clock
        qx, qy, qz, qw = _sway_quat(elapsed)
        client.send_message("/VMC/Ext/Bone/Pos", [_SWAY_BONE, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        stop_event.wait(_SWAY_UPDATE_INTERVAL_SEC)
    # 停止時はSpineをニュートラルに戻す
    client.send_message("/VMC/Ext/Bone/Pos", [_SWAY_BONE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def send_nod(client: SimpleUDPClient) -> None:
    """表情が変わった瞬間に一度だけ呼ぶ「うなずき」ジェスチャー（短時間ブロッキング）。"""
    for deg in _NOD_SEQUENCE_DEG:
        qx, qy, qz, qw = _axis_angle_quat(_NOD_AXIS, deg)
        client.send_message("/VMC/Ext/Bone/Pos", [_NOD_BONE, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        time.sleep(_NOD_STEP_SEC)


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
        send_nod(client)  # 表情が変わった瞬間だけ「うなずき」を1回入れる

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
    parser.add_argument("--no-sway", action="store_true",
                        help="待機モーション（Spineのスウェイ）を無効化")
    args = parser.parse_args()
    device = int(args.device) if args.device is not None and args.device.isdigit() else args.device

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

    log.info("初期姿勢を送信済み（Tポーズ解消）。ここでOBS録画を開始してください。")
    input("録画が始まったらEnterを押してください（そのまま再生開始します）...")
    log.info("再生開始: %s（%.1f秒）", wav_path, duration)
    sd.play(data, samplerate, device=device)
    start_clock = time.monotonic()

    scheduler = threading.Thread(
        target=run_expression_scheduler, args=(client, timeline, start_clock, duration),
        daemon=True)
    scheduler.start()

    sway_stop = threading.Event()
    sway_thread = None
    if not args.no_sway:
        sway_thread = threading.Thread(
            target=run_idle_sway, args=(client, sway_stop, start_clock), daemon=True)
        sway_thread.start()

    try:
        while sd.get_stream().active:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("中断されました。再生を停止します。")
        sd.stop()

    scheduler.join(timeout=_EXPRESSION_HOLD_SEC + 1.0)
    if sway_thread is not None:
        sway_stop.set()
        sway_thread.join(timeout=1.0)
    send_expression(client, NEUTRAL)
    log.info("完了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
