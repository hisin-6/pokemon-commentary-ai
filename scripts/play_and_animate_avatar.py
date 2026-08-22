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
     （表情が変わった瞬間はリアクション動作も送る。Joy/Sorrow/Funは
     `explore_avatar_poses.py`で実機検証済みのポーズ
     （victory_arms_up/bow_apologetic/head_tilt_curious）へSLERP遷移→保持→
     idle_downへ戻す一連の動作、それ以外は標準のうなずきのみ＝2026-08-03統合）
  4. 最後のイベントの少し後にNeutralへ戻す
  5. 再生中はSpine+Chestへレイヤードサイン（主周期＋副周期を重ねた呼吸っぽい
     揺れ）を送り続ける待機モーション（`--no-sway`で無効化可）
  6. 加えてHeadへ数秒〜十数秒おきにランダムな軽い仕草（前傾のみ・首かしげ等）を挟み、
     長い無反応区間の単調さを崩す（`--no-idle-gestures`で無効化可）

ポーズ再生中、そのポーズが使うボーンは常時スウェイ（Spine/Chest）・ランダム仕草
（Head）から一時的に除外する（`_suspend_bones`/`_resume_bones`）。同じボーンへ
複数スレッドが同時に書き込むと最後に送られた方が勝ってアニメーションが競合する
（VMCは相対加算ではなく絶対値上書きのため）。

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
import random
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import soundfile as sf
import sounddevice as sd
from pythonosc.udp_client import SimpleUDPClient

sys.path.insert(0, str(Path(__file__).parent))
import explore_avatar_poses as eap  # noqa: E402  POSES/クォータニオン補間ヘルパーを再利用

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
# 2026-08-22: `--persona neutral`運用（2026-08-15〜）でテンション高めの言い回し
# （「キマってほしいっ！」等）が「決まってほしいところです」のような控えめな
# 言い回しに変わり、過去形固定の"決まった"等がヒットしなくなっていたため、
# neutral実例（kurepi_persona.NEUTRAL_TONE_EXAMPLES）に合わせて語幹マッチに
# 緩めたキーワード（"決まっ"）とneutralで出やすい言葉（チャンス・有利・順調／
# 劣勢・手痛い・苦しい）を追加。
_POSITIVE_KEYWORDS = (
    "バツグン", "急所", "決まっ", "炸裂", "やったぁ", "素晴らしい",
    "ナイス", "無傷", "きっちり", "頑張れ", "いいぞ", "チャンス", "有利", "順調",
)
_NEGATIVE_KEYWORDS = (
    "ピンチ", "削られ", "いまひとつ", "外れ", "厳しい", "危ない",
    "食らって", "くらって", "苦戦", "劣勢", "手痛い", "苦しい",
)
# 感情反応を付けるイベント種別（move_used=無印/faint統合先を含む・move_single=技単独反応）。
# 2026-08-22: fillerも対象に追加（従来は「雑談なので実況内容と紐付いた反応はさせない」
# 方針で明示的に無反応=Noneだったが、2026-08-21の対象閾値引き下げでフィラーが
# 発話時間の相当割合を占めるようになり、そこが無反応のままだとポーズの体感頻度が
# 大きく下がる問題が発生したため方針変更。フィラーの内容も他のイベントと同じ
# キーワード判定に委ねる）
_SENTIMENT_EVENT_TYPES = {"move_used", "move_single", "switch", "filler"}


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

    上記どちらにも該当しない move_used/move_single/switch/filler は、実況テキストの
    キーワードから_sentiment_expressionで表情を推定する（技1つ1つに反応させる・
    2026-08-01改善ロードマップ③続き。fillerは2026-08-22追加＝雑談扱いで無反応
    だった従来方針から、キーワード判定に委ねる方針へ変更）。battle_start等
    _EVENT_EXPRESSIONに明示的な値がある種別はそちらを優先し、テキスト連動の
    対象にしない。
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


def build_expression_timeline(scheduled: list[dict]) -> list[tuple[float, str, str]]:
    """(発火時刻, ブレンドシェイプ名, ポーズバリエーションキー) のリストを
    開始時刻順で返す（表情変更なしイベントは除く）。"""
    timeline = []
    for entry in scheduled:
        event_type = entry.get("event_type", "")
        context = entry.get("context") or {}
        expr = expression_for(event_type, context, entry.get("commentary", ""))
        if expr:
            variant = _pose_variant_key(event_type, context)
            timeline.append((float(entry["start"]), expr, variant))
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


# ── 常時の揺れ（モーション② 2026-08-01 → 多ボーン化 2026-08-01）───────────
# Spine単発の正弦波1本だと「ゆらゆらしてるだけ」で機械的に見えるため、
# Spine+Chestの2ボーンに、それぞれ主周期＋副周期（振幅小さめ・周期をずらす）を
# 重ねたレイヤードサインを送る。ボーンごとに位相もずらすので、全身が
# 同じタイミングでピタッと揺れる不自然さを避けられる。
# Headは使わない（ランダム待機仕草＝run_idle_gesturesが専用で使うため競合を避ける）。
# 腕下げ姿勢（_IDLE_POSE_BONES=UpperArm）とも別ボーンなので競合しない。
_IdleBoneConfig = namedtuple(
    "_IdleBoneConfig",
    ["bone", "axis", "amplitude_deg", "period_sec",
     "sub_amplitude_deg", "sub_period_sec", "phase_rad"])

_IDLE_BONES: tuple[_IdleBoneConfig, ...] = (
    _IdleBoneConfig("Spine", "y", 4.0, 5.0, 1.0, 3.3, 0.0),
    _IdleBoneConfig("Chest", "x", 1.5, 6.5, 0.6, 2.7, 0.6),
)
_SWAY_UPDATE_INTERVAL_SEC = 0.15

# ── 表情変化時のリアクション動作（モーション② 2026-08-01 → ポーズ統合 2026-08-03
# → 場面別ポーズ分岐 2026-08-15）
# Joy/Sorrow/Funは`explore_avatar_poses.py`で実機検証済みのポーズへ滑らかに
# 遷移する（_POSE_VARIANTS・play_pose_reaction）。それ以外（Angryなど現状未使用の
# 表情）はNeckの標準うなずきにフォールバックする。
_NOD_BONE = "Neck"
_NOD_AXIS = "x"
_NOD_SEQUENCE_DEG = (10.0, 14.0, 6.0, 0.0)
_NOD_STEP_SEC = 0.12

# 表情 → 遷移させるポーズ名（`explore_avatar_poses.POSES`のキー）。
# 2026-08-15: 同じ表情でも発生場面（variant）によってポーズを変え、単調さを崩す
# ＋victory_arms_up（腕を大きく振り上げるV字バンザイ・--avatar-cropからはみ出し
# やすい）の発火頻度を「試合の勝敗が決まった瞬間」だけに絞る（従来は1匹倒す度に
# 毎回発火していた）。fist_pump_right/thinking_chin/lean_back_confidentは
# 実機検証済みだが未使用だった3ポーズ（`explore_avatar_poses.py`参照）。
# "default"キーは_pose_variant_keyが未知のvariantを返した場合のフォールバック。
_POSE_VARIANTS = {
    "Joy": {
        "default": "victory_arms_up",   # battle_end（試合の勝利）用・派手なバンザイ
        "faint": "fist_pump_right",     # 1匹倒す度の軽いガッツポーズ
    },
    "Sorrow": {
        "default": "bow_apologetic",    # 自分が倒された時の会釈
        "sentiment": "thinking_chin",   # ピンチ等の実況キーワード反応・考え込む仕草
    },
    "Fun": {
        "default": "head_tilt_curious",  # 技が決まった時の首かしげ
        "battle_start": "lean_back_confident",  # 対戦開始の意気込み
    },
}


def _pose_variant_key(event_type: str, context: dict) -> str:
    """ポーズのバリエーション選択に使う分類キーを返す。faint_sideがcontextに
    乗っていればevent_type（faint/move_used統合どちらも）を問わず'faint'扱いにする
    （pipeline.py側のfaint統合仕様に合わせる。expression_forと同じ優先順）。"""
    if "faint_side" in context:
        return "faint"
    if event_type == "battle_start":
        return "battle_start"
    if event_type in _SENTIMENT_EVENT_TYPES:
        return "sentiment"
    return "default"


def _pose_for(expression: str, variant: str = "default") -> str | None:
    variants = _POSE_VARIANTS.get(expression)
    if not variants:
        return None
    return variants.get(variant, variants["default"])

# ポーズ遷移1回あたりの秒数（遷移in→保持→遷移outの3段。合計は遷移×2+保持）。
# 2026-08-22: 実機fb「ポーズを取っている時間が短い」を受けて0.6秒→2.0秒に延長
# （合計1.5秒→2.9秒）。次のイベントとの間隔が短いと重なって見えることがあるので、
# 詰まって見える場合は_EXPRESSION_HOLD_SEC（表情自体の保持秒数）とのバランスも見直すこと。
_POSE_TRANSITION_SEC = 0.45
_POSE_HOLD_SEC = 2.0

# ポーズ再生中、そのポーズが使うボーンを常時スウェイ/ランダム仕草から
# 一時的に除外するための共有集合。マルチスレッドで軽く読み書きするが、
# 見た目のアニメーションにしか影響しないため厳密なロックは省略している
# （最悪でも1フレーム分のちらつき程度で実害が無いと判断）。
_suspended_bones: set[str] = set()


def _suspend_bones(bones) -> None:
    _suspended_bones.update(bones)


def _resume_bones(bones) -> None:
    _suspended_bones.difference_update(bones)


def _pose_transition_frames(client: SimpleUDPClient,
                            starts: dict[str, tuple[float, float, float, float]],
                            targets: dict[str, tuple[float, float, float, float]],
                            duration_sec: float, fps: float = 30.0) -> None:
    """startsからtargetsへ、`explore_avatar_poses`のSLERP遷移ロジック（swing-twist
    分解＋加減速）をそのまま再利用して補間送信する（`transition_to_pose`参照）。"""
    frame_count = max(1, round(duration_sec * fps))
    interval = duration_sec / frame_count
    for i in range(1, frame_count + 1):
        global_t = i / frame_count
        for bone, target in targets.items():
            local_t = eap.ease_smoothstep(eap._windowed_t(global_t, eap._bone_time_window(bone)))
            start = starts[bone]
            if any(k in bone for k in eap._TWIST_AXIS_BONE_KEYWORDS):
                qx, qy, qz, qw = eap.quat_slerp_swing_twist(start, target, local_t)
            else:
                qx, qy, qz, qw = eap.quat_slerp(start, target, local_t)
            client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        time.sleep(interval)


def play_pose_reaction(client: SimpleUDPClient, pose_name: str,
                       hold_sec: float = _POSE_HOLD_SEC,
                       transition_sec: float = _POSE_TRANSITION_SEC) -> None:
    """`explore_avatar_poses.POSES[pose_name]`へ滑らかに遷移→保持→idle_downへ
    戻す一連の動作。ポーズが使うボーンは実行中`_suspended_bones`に加えて常時
    スウェイ/ランダム仕草からの書き込みを止め、終わったら必ず解放する。
    idle_down（腕の下げ角度）に含まれないボーン（Neck/Head/Spineなど）は、
    遷移開始前のニュートラル(0,0,0,1)からの遷移とみなす（常時スウェイ等の
    実際の途中角度は数度程度の振れ幅しか無いため、この近似による見た目の
    ずれは軽微）。"""
    idle = eap.POSES["idle_down"]
    pose = eap.POSES[pose_name]
    bones = set(pose) | set(idle)
    _suspend_bones(bones)
    try:
        starts = {b: eap.compose_quat(idle[b]) if b in idle else (0.0, 0.0, 0.0, 1.0)
                 for b in bones}
        targets = {b: eap.compose_quat(pose[b]) if b in pose else starts[b] for b in bones}
        _pose_transition_frames(client, starts, targets, transition_sec)
        time.sleep(hold_sec)
        _pose_transition_frames(client, targets, starts, transition_sec)
    finally:
        _resume_bones(bones)

# ── ランダム待機仕草（モーション③ 2026-08-01）────────────────────────────
# 長い無反応区間が単調にならないよう、Headへ数秒〜十数秒おきにランダムな
# 軽い仕草（首かしげ等）を挟む。Neckの相槌（前後）と軸を分けず同じx軸だが、
# ボーン自体をHeadにして常時スウェイ（Spine/Chest）とも独立させている。
# 2026-08-01実機fb「少し上を向く動作があり不自然」対応: うなずき（+方向=前傾）は
# OKだった一方、-方向（後ろに傾げる=見上げる）の仕草が脈絡なく発生して不自然と
# 判明したため削除。前傾方向（うなずきと同じ+方向）のバリエーションのみに絞る。
_GESTURE_BONE = "Head"
_GESTURE_AXIS = "x"
_GESTURE_SEQUENCES = (
    (6.0, 9.0, 4.0, 0.0),     # 軽く前に傾げる（強め）
    (4.0, 6.0, 2.0, 0.0),     # 軽く前に傾げる（弱め）
    (5.0, 0.0),               # 小さな相槌一往復
)
_GESTURE_STEP_SEC = 0.15
_GESTURE_MIN_INTERVAL_SEC = 9.0
_GESTURE_MAX_INTERVAL_SEC = 20.0


def _idle_bone_deg(cfg: _IdleBoneConfig, elapsed: float) -> float:
    """レイヤードサイン（主周期＋副周期）による経過時間elapsed（秒）時点の角度。"""
    deg = cfg.amplitude_deg * math.sin(2 * math.pi * elapsed / cfg.period_sec + cfg.phase_rad)
    if cfg.sub_period_sec:
        deg += cfg.sub_amplitude_deg * math.sin(
            2 * math.pi * elapsed / cfg.sub_period_sec + cfg.phase_rad * 1.7)
    return deg


def _sway_quat(elapsed: float, cfg: _IdleBoneConfig = _IDLE_BONES[0]
              ) -> tuple[float, float, float, float]:
    """指定ボーン設定cfg（既定はSpine）の経過時間elapsed（秒）時点の回転。"""
    return _axis_angle_quat(cfg.axis, _idle_bone_deg(cfg, elapsed))


def run_idle_sway(client: SimpleUDPClient, stop_event: threading.Event,
                  start_clock: float) -> None:
    """再生中ずっとSpine/Chestへレイヤードサインの揺れを送り続ける
    （stop_eventがセットされたら終了）。"""
    while not stop_event.is_set():
        elapsed = time.monotonic() - start_clock
        for cfg in _IDLE_BONES:
            if cfg.bone in _suspended_bones:
                continue  # ポーズ再生中はそのボーンへの書き込みを譲る
            qx, qy, qz, qw = _axis_angle_quat(cfg.axis, _idle_bone_deg(cfg, elapsed))
            client.send_message("/VMC/Ext/Bone/Pos", [cfg.bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        stop_event.wait(_SWAY_UPDATE_INTERVAL_SEC)
    # 停止時は対象ボーンを全てニュートラルに戻す
    for cfg in _IDLE_BONES:
        client.send_message("/VMC/Ext/Bone/Pos", [cfg.bone, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def run_idle_gestures(client: SimpleUDPClient, stop_event: threading.Event,
                      rng: random.Random | None = None) -> None:
    """ランダムな間隔でHeadへ軽い仕草を送り続ける（stop_eventがセットされたら終了）。"""
    rng = rng or random
    while not stop_event.wait(rng.uniform(_GESTURE_MIN_INTERVAL_SEC, _GESTURE_MAX_INTERVAL_SEC)):
        if _GESTURE_BONE in _suspended_bones:
            continue  # ポーズ再生中はHeadへの書き込みを譲る
        for deg in rng.choice(_GESTURE_SEQUENCES):
            qx, qy, qz, qw = _axis_angle_quat(_GESTURE_AXIS, deg)
            client.send_message("/VMC/Ext/Bone/Pos", [_GESTURE_BONE, 0.0, 0.0, 0.0, qx, qy, qz, qw])
            stop_event.wait(_GESTURE_STEP_SEC)
    client.send_message("/VMC/Ext/Bone/Pos", [_GESTURE_BONE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _send_smooth_keyframes(client: SimpleUDPClient, bone: str, axis: str,
                           degrees_sequence: tuple[float, ...], duration_sec: float,
                           fps: float = 30.0) -> None:
    """boneをaxis軸まわりで、degrees_sequenceの折れ線をキーフレーム間SLERP+
    ease_smoothstepでなぞるように滑らかに送る。単一ボーン・単一軸の簡易
    アニメーション用（Neckの標準うなずき等）。2026-08-03実機fb「標準うなずきだけ
    ロボットっぽい」対応: 従来は各角度を瞬間切り替え→静止の繰り返し（補間なし）
    だったため、ポーズ遷移（SLERP+イージング）と並べると相対的にカクついて
    見えていた。"""
    keyframes = [_axis_angle_quat(axis, deg) for deg in degrees_sequence]
    segment_count = len(keyframes) - 1
    if segment_count <= 0:
        return
    segment_duration = duration_sec / segment_count
    frame_count = max(1, round(segment_duration * fps))
    interval = segment_duration / frame_count
    for start, end in zip(keyframes, keyframes[1:]):
        for i in range(1, frame_count + 1):
            t = eap.ease_smoothstep(i / frame_count)
            qx, qy, qz, qw = eap.quat_slerp(start, end, t)
            client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
            time.sleep(interval)


def send_reaction(client: SimpleUDPClient, expression: str, variant: str = "default") -> None:
    """表情が変わった瞬間に一度だけ呼ぶリアクション動作（短時間ブロッキング）。
    _POSE_VARIANTSにマッピングされた表情はポーズ遷移（play_pose_reaction）、
    それ以外はNeckの標準うなずき（滑らか化済み）のみ。variant省略時は"default"
    （=従来のvictory_arms_up/bow_apologetic/head_tilt_curiousと同じ挙動）。"""
    pose_name = _pose_for(expression, variant)
    if pose_name:
        play_pose_reaction(client, pose_name)
        return
    _send_smooth_keyframes(client, _NOD_BONE, _NOD_AXIS, (0.0,) + _NOD_SEQUENCE_DEG,
                           _NOD_STEP_SEC * len(_NOD_SEQUENCE_DEG))


def send_expression(client: SimpleUDPClient, name: str) -> None:
    """指定のブレンドシェイプだけを1.0にし、他はNeutral含め全部0にして送る。"""
    all_known = {NEUTRAL, "Fun", "Joy", "Sorrow", "Angry"}
    for preset in all_known:
        client.send_message("/VMC/Ext/Blend/Val", [preset, 1.0 if preset == name else 0.0])
    client.send_message("/VMC/Ext/Blend/Apply", [])
    log.info("[表情] %s", name)


def run_expression_scheduler(client: SimpleUDPClient, timeline: list[tuple[float, str, str]],
                             start_clock: float, total_duration: float) -> None:
    """再生開始時刻(start_clock)を基準に、timelineの時刻が来るたびOSCを送る。
    表情は_EXPRESSION_HOLD_SEC秒後（次のイベントより先に来れば）Neutralへ戻す。
    """
    for i, (t, expr, variant) in enumerate(timeline):
        wait = start_clock + t - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        send_expression(client, expr)
        send_reaction(client, expr, variant)  # 表情が変わった瞬間だけ感情別のリアクションを1回入れる

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
                        help="待機モーション（Spine/Chestのスウェイ）を無効化")
    parser.add_argument("--no-idle-gestures", action="store_true",
                        help="ランダム待機仕草（Headの首かしげ等）を無効化")
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

    idle_stop = threading.Event()
    sway_thread = None
    if not args.no_sway:
        sway_thread = threading.Thread(
            target=run_idle_sway, args=(client, idle_stop, start_clock), daemon=True)
        sway_thread.start()

    gesture_thread = None
    if not args.no_idle_gestures:
        gesture_thread = threading.Thread(
            target=run_idle_gestures, args=(client, idle_stop), daemon=True)
        gesture_thread.start()

    try:
        while sd.get_stream().active:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("中断されました。再生を停止します。")
        sd.stop()

    scheduler.join(timeout=_EXPRESSION_HOLD_SEC + 1.0)
    idle_stop.set()
    if sway_thread is not None:
        sway_thread.join(timeout=1.0)
    if gesture_thread is not None:
        gesture_thread.join(timeout=1.0)
    send_expression(client, NEUTRAL)
    log.info("完了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
