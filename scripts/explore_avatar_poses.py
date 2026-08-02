"""VMC ProtocolのOSCで複数ボーンを組み合わせた「ポーズ」を送り、3Dアバターの
表現の幅を実機で探るための検証スクリプト（`play_and_animate_avatar.py`への
本実装まえの下調べ用。`test_vmc_pose.py`の単一ボーン版を拡張したもの）。

position は今回も動かさないので(0,0,0)固定・回転のみ。1ボーンにつき複数の
軸回転（例: Z軸で持ち上げてからX軸で肘を曲げる）をクォータニオン合成で
1つの姿勢にまとめて送る（VMCは同一ボーンへの新しいBone/Posで前回の回転を
「上書き」する＝複数メッセージを送っても加算されないため、合成してから
1回で送る必要がある）。

⚠️角度は全て**未検証の当たりずっぽう**（`_IDLE_POSE_DEG`=80のZ軸だけが
`test_vmc_pose.py`で実機確認済み・他は初見の推測）。座標系・軸の向きは
モデル依存なので、実際に送ってみて「変な方向に曲がる/伸びる」場合は
`--deg-override 骨名:軸:角度`で個別に上書きしながら追い込むこと。

`--pose`実行時・`--cycle`の各ステップ実行時とも、送信前に必ず全ボーンを
リセットしてから適用する（`reset_all_bones`）。ポーズごとに触るボーンが
違うため、リセットせず切り替えると前のポーズの曲げが残ったまま次のポーズが
乗ってしまう（`--cycle`で最後に腕がねじれたまま終わっていたのはこれが原因）。

2026-08-01追加: 「動きがなく急にポーズを取る」というfbを受けて、既定では
現在の姿勢→目標ポーズへクォータニオンSLERPで補間しながら滑らかに遷移する
（`transition_to_pose`・既定0.6秒・`--duration`で変更可）。角度をざっくり
数値確認したいだけの時は`--instant`で従来通りの瞬間切り替えに戻せる。

事前準備（Windows側・初回のみ）:
    venv\\Scripts\\pip.exe install python-osc

使い方（VMC起動中・Receiver有効化チェック済みの状態で）:
    venv\\Scripts\\python.exe scripts\\explore_avatar_poses.py --port 39540 --list
    venv\\Scripts\\python.exe scripts\\explore_avatar_poses.py --port 39540 --pose victory_arms_up
    venv\\Scripts\\python.exe scripts\\explore_avatar_poses.py --port 39540 --pose idle_down   # 基本姿勢に戻す

    骨1本だけ角度を上書きして微調整したい場合（例: 右肘の曲げ角度だけ変えたい）:
    venv\\Scripts\\python.exe scripts\\explore_avatar_poses.py --port 39540 --pose fist_pump_right \\
        --deg-override RightLowerArm:x:-50

    掌の向き（手首の捻り）がY軸で合わない場合、軸を変えて探る例:
    venv\\Scripts\\python.exe scripts\\explore_avatar_poses.py --port 39540 --pose victory_arms_up \\
        --deg-override LeftHand:x:90 --deg-override RightHand:x:-90

    一通り順番に見たい場合（各ポーズを--hold秒ずつ表示、間にidle_downを挟む）:
    venv\\Scripts\\python.exe scripts\\explore_avatar_poses.py --port 39540 --cycle --hold 3
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from pythonosc.udp_client import SimpleUDPClient

# T-pose対策の基本姿勢（`play_and_animate_avatar.py`の_IDLE_POSE_DEGと同じ・実機確認済み）
_IDLE_ARM_DEG = 80.0

# 拳を握る指の曲げについての初期知見（2026-08-01・VRM実測）: 標準的なHumanoidの
# T-poseは掌が下（-Y）を向くため、「+X（指の伸びる方向）を-Y（掌側）へ曲げる」＝
# 肘の上げ下げ（Z軸）とは逆符号のZ軸回転が握り込む方向になる。実際の`fist_pump_right`
# の指の値は`pose_tuner_gui.py`で1関節ずつ実機を見ながら手直しされているため、
# 上記はあくまで最初の当たりを付けるための目安（新しく拳ポーズを作る時の出発点用）。

# ── ポーズ定義 ───────────────────────────────────────────────────────────
# ⚠️1ボーンに複数軸を指定する場合は必ずX→Y→Zの順で書くこと（`pose_tuner_gui.py`の
# スライダーはX→Y→Zの固定順で合成するため、順番が違うと同じ数値でも見た目が変わる）。
# bone名 → [(軸, 角度deg), ...]（この順でクォータニオン合成。実機確認済みなのは
# LeftUpperArm/RightUpperArmのZ軸±80のみ・それ以外は「こう動くはず」という推測）。
# 上半身のみ表示（--avatar-crop）を前提に、体全体の派手な動きより上半身の
# 表情豊かさを狙った構成にしている。
POSES: dict[str, dict[str, list[tuple[str, float]]]] = {
    "idle_down": {
        "LeftUpperArm": [("z", _IDLE_ARM_DEG)],
        "RightUpperArm": [("z", -_IDLE_ARM_DEG)],
    },
    "victory_arms_up": {
        # 両手を高く挙げるバンザイ・勝利の瞬間用。T-pose(0°)を超えてさらに
        # 上へ回転させる想定（idle_downと逆方向）。
        # ⚠️2026-08-01実機fb①「腕が頭を貫通して反対側に行く」→ -140°は垂直(-90°付近)を
        # 50°も超えて振り切っていたと判明→-100°に修正。
        # ⚠️実機fb②「-100°でも頭に干渉する」→ 垂直(-90°)にほぼ近い角度だと腕が頭の
        # 真横〜真上をかすめるため、真上まで振り切らず斜め上に開く「V字バンザイ」になる
        # 角度（垂直より35°ほど手前）まで下げた
        "LeftUpperArm": [("z", -55.0)],
        "RightUpperArm": [("z", 55.0)],
        "LeftLowerArm": [("x", -8.0)],
        "RightLowerArm": [("x", 8.0)],
        # 掌をこちら（カメラ側）に向ける手首の捻り。2026-08-01実機確認済み:
        # X軸±90°で「掌がこちらを向く」を確認（当初の推測Y軸は外れだった）
        "LeftHand": [("x", 90.0)],
        "RightHand": [("x", -90.0)],
    },
    "fist_pump_right": {
        # 右腕を突き上げるガッツポーズ。VRM実測＋シミュレーションでの計算値を
        # 出発点に、`pose_tuner_gui.py`（スライダーでリアルタイム調整できる
        # GUIツール）を使ってユーザーが実機を見ながら最終確定した値（2026-08-01）。
        # 計算だけでは追い込みきれなかった細部（各指の関節ごとの微妙な角度・
        # 手首や肩のY/X軸の微調整等）はこのGUIで実際の見た目を見ながら
        # 手直しされている。調整の経緯（VRMファイル実測手法・見つかった不具合・
        # ボツ案）は git履歴・会話ログ参照。
        # ⚠️既知の課題（2026-08-02実機確認・改善は後日）: idle→このポーズへの
        # transition_to_pose遷移中、腕がくるっと回ってからガッツポーズを取るように
        # 見え、他の6ポーズより不自然。RightUpperArm/RightLowerArmの回転量が大きい
        # （especially LowerArmのx=-180°）ためswing-twist分解でも経路が素直に
        # ならない可能性。一旦許容してこのまま採用。改善するなら中間キーフレームの
        # 追加や_bone_time_windowでのRightUpperArm/RightLowerArm遅延スタートを検討。
        "LeftUpperArm": [("y", 6.5), ("z", 80.0)],
        "RightUpperArm": [("x", 95.4), ("y", -23.9), ("z", -73.7)],
        "RightLowerArm": [("x", -180.0), ("y", 6.5), ("z", 119.3)],
        "RightThumbProximal": [("x", 15.2), ("y", 2.2), ("z", -26.0)],
        "RightThumbIntermediate": [("y", 91.1), ("z", -4.3)],
        "RightThumbDistal": [("y", 54.2), ("z", 8.7)],
        "RightIndexProximal": [("y", 8.7), ("z", -84.6)],
        "RightIndexIntermediate": [("z", -80.2)],
        "RightIndexDistal": [("z", -78.1)],
        "RightMiddleProximal": [("z", -78.1)],
        "RightMiddleIntermediate": [("z", -95.4)],
        "RightMiddleDistal": [("z", -82.4)],
        "RightRingProximal": [("y", -6.5), ("z", -78.1)],
        "RightRingIntermediate": [("z", -106.3)],
        "RightRingDistal": [("z", -73.7)],
        "RightLittleProximal": [("y", -13.0), ("z", -80.2)],
        "RightLittleIntermediate": [("z", -125.8)],
        "RightLittleDistal": [("z", -71.6)],
    },
    "thinking_chin": {
        # 右手を顔の近くに持っていく「考え中」ポーズ。`pose_tuner_gui.py`で
        # ユーザーが実機を見ながら最終確定した値（2026-08-02）。
        "Neck": [("x", -2.0), ("z", -5.0)],
        "LeftUpperArm": [("x", 70.0), ("y", 26.0), ("z", 87.0)],
        "LeftLowerArm": [("y", 86.0), ("z", -10.0)],
        "LeftHand": [("x", -130.0)],
        "RightUpperArm": [("y", -40.0), ("z", -80.0)],
        "RightLowerArm": [("x", -40.0), ("y", -150.0), ("z", 29.0)],
        "RightHand": [("x", -80.0)],
        "LeftThumbProximal": [("x", 10.0), ("y", -30.0)],
        "LeftIndexProximal": [("y", -6.0), ("z", 40.0)],
        "LeftIndexIntermediate": [("z", 10.0)],
        "LeftIndexDistal": [("z", 30.0)],
        "LeftMiddleProximal": [("x", -10.0), ("z", 32.0)],
        "LeftMiddleIntermediate": [("z", 20.0)],
        "LeftMiddleDistal": [("z", 30.0)],
        "LeftRingProximal": [("x", -20.0), ("y", 6.0), ("z", 28.0)],
        "LeftRingIntermediate": [("z", 30.0)],
        "LeftLittleProximal": [("y", 17.0), ("z", 40.0)],
        "RightThumbProximal": [("y", -10.0)],
        "RightThumbIntermediate": [("y", -30.0)],
        "RightMiddleProximal": [("z", -80.0)],
        "RightMiddleIntermediate": [("z", -80.0)],
        "RightMiddleDistal": [("z", -100.0)],
        "RightRingProximal": [("x", -30.0), ("z", -90.0)],
        "RightRingIntermediate": [("z", -80.0)],
        "RightRingDistal": [("z", -100.0)],
        "RightLittleProximal": [("x", -60.0), ("z", -100.0)],
        "RightLittleIntermediate": [("z", -80.0)],
        "RightLittleDistal": [("z", -100.0)],
    },
    "head_tilt_curious": {
        # 腕は動かさず首だけ傾ける「興味津々」ポーズ。`pose_tuner_gui.py`で
        # 実機を見ながら微調整して確定（2026-08-01・Neckも少し加えるとより自然）。
        "Neck": [("z", 4.0)],
        "Head": [("z", 21.0)],
        "LeftUpperArm": [("z", _IDLE_ARM_DEG)],
        "RightUpperArm": [("z", -_IDLE_ARM_DEG)],
    },
    "bow_apologetic": {
        # 少し前かがみに会釈するポーズ。謝罪・恐縮系の実況リアクション候補
        "LeftUpperArm": [("z", _IDLE_ARM_DEG)],
        "RightUpperArm": [("z", -_IDLE_ARM_DEG)],
        "Spine": [("x", 12.0)],
        "Head": [("x", 8.0)],
    },
    "lean_back_confident": {
        # 胸を張って少し反り返る自信満々ポーズ。連勝中・快勝時の実況向け候補
        "LeftUpperArm": [("z", _IDLE_ARM_DEG)],
        "RightUpperArm": [("z", -_IDLE_ARM_DEG)],
        "Spine": [("x", -10.0)],
        "Chest": [("x", -6.0)],
    },
}


def axis_angle_quat(axis: str, deg: float) -> tuple[float, float, float, float]:
    """指定軸まわり deg 度の回転をクォータニオン(x,y,z,w)で返す。"""
    rad = math.radians(deg)
    s, c = math.sin(rad / 2), math.cos(rad / 2)
    if axis == "x":
        return (s, 0.0, 0.0, c)
    if axis == "y":
        return (0.0, s, 0.0, c)
    return (0.0, 0.0, s, c)


def quat_mul(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """クォータニオン積 a∘b（bを先に適用し、その上からaを適用する合成）。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def compose_quat(steps: list[tuple[str, float]]) -> tuple[float, float, float, float]:
    """[(軸,角度), ...] を順番に合成した1つのクォータニオンを返す。"""
    q = (0.0, 0.0, 0.0, 1.0)
    for axis, deg in steps:
        q = quat_mul(axis_angle_quat(axis, deg), q)
    return q


def apply_overrides(pose: dict[str, list[tuple[str, float]]],
                    overrides: list[str]) -> dict[str, list[tuple[str, float]]]:
    """--deg-override 骨名:軸:角度 をposeにマージした新しい辞書を返す（元は変更しない）。
    指定した骨がposeに無ければ新規追加、あればその骨のステップを丸ごと置き換える
    （複数軸を組み合わせたい場合はカンマ区切りで軸:角度を複数指定: 骨名:z:80,x:-10）。"""
    merged = {bone: list(steps) for bone, steps in pose.items()}
    for spec in overrides:
        try:
            bone, rest = spec.split(":", 1)
            steps = []
            for part in rest.split(","):
                axis, deg_str = part.split(":")
                steps.append((axis, float(deg_str)))
        except ValueError:
            raise SystemExit(
                f"--deg-override の書式が不正です: {spec!r}"
                "（正しい例: RightLowerArm:x:-50 や Spine:x:10,y:5）")
        merged[bone] = steps
    return merged


# ポーズを跨いで前回の回転が残らないよう、全ポーズが触るボーンの一覧を洗い出しておく
# （個々のポーズ定義は自分が使うボーンしか書いていないため、切り替え時に「新しいポーズが
# 触らないボーン」は前の状態のまま残ってしまう＝これが--cycleで腕がねじれたまま
# 終わっていた原因）。
_ALL_POSE_BONES: set[str] = {bone for pose in POSES.values() for bone in pose}


def send_pose(client: SimpleUDPClient, pose: dict[str, list[tuple[str, float]]]) -> None:
    for bone, steps in pose.items():
        qx, qy, qz, qw = compose_quat(steps)
        client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        print(f"  [送信] {bone}: {steps} -> quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")


def reset_all_bones(client: SimpleUDPClient) -> None:
    """全ポーズが使うボーンを一旦きれいな状態に戻す（腕はT-pose対策の下げ角度・
    それ以外はニュートラル回転）。ポーズ送信の直前に必ず呼ぶことで、前回どのポーズを
    送っていても毎回同じ状態から始められる。"""
    idle = POSES["idle_down"]
    for bone in sorted(_ALL_POSE_BONES):
        if bone in idle:
            continue  # 後段のsend_poseがidle_downの角度を送るのでここでは触らない
        client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    send_pose(client, idle)


# ── ポーズ間の滑らかな遷移（2026-08-01追加）─────────────────────────────
# 「動きがなく急にポーズを取る」というfb対応。reset_all_bones→send_poseの
# 単発切り替えだと1フレームで一気に飛ぶため、開始姿勢→目標姿勢をクォータニオン
# SLERPで補間し、複数フレームに分けて送ることで滑らかな動きにする。
def quat_slerp(q0: tuple[float, float, float, float], q1: tuple[float, float, float, float],
               t: float) -> tuple[float, float, float, float]:
    """クォータニオンの球面線形補間（t=0でq0・t=1でq1）。"""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1, dot = -x1, -y1, -z1, -w1, -dot
    if dot > 0.9995:
        # ほぼ同じ回転同士は線形補間で近似（sinがほぼ0でゼロ割りになるのを回避）
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        w = w0 + t * (w1 - w0)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        return (x / n, y / n, z / n, w / n)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * x0 + s1 * x1, s0 * y0 + s1 * y1, s0 * z0 + s1 * z1, s0 * w0 + s1 * w1)


def quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def swing_twist_decompose(q: tuple[float, float, float, float],
                          axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
                          ) -> tuple[tuple[float, float, float, float],
                                     tuple[float, float, float, float]]:
    """qを、axis周りの「捻り(twist)」と、それ以外の「振り(swing)」に分解する
    （q = swing ⊗ twist）。2026-08-01実機fb「ガッツポーズで腕が変な軸で捻れながら
    動く」への対処用: 腕・指のような1本軸の骨（このモデルではX軸=長さ方向と実測済み・
    victory_arms_upの掌捻り/fist_pump_rightの肘曲げの実測結果より）は、開始姿勢→
    目標姿勢の単純なSLERPだと「振り」と「捻り」が混ざった最短経路を通ってしまい、
    人間の腕の動きとしては不自然に見える。swingとtwistを別々に補間してから
    合成することで、自然な「振ってから捻る」ような経路に近づける。"""
    qx, qy, qz, qw = q
    ax, ay, az = axis
    d = qx * ax + qy * ay + qz * az
    twist = (ax * d, ay * d, az * d, qw)
    n = math.sqrt(sum(c * c for c in twist))
    if n < 1e-9:
        twist = (0.0, 0.0, 0.0, 1.0)
    else:
        twist = tuple(c / n for c in twist)
    swing = quat_mul(q, quat_conjugate(twist))
    return swing, twist


def quat_slerp_swing_twist(q0: tuple[float, float, float, float],
                           q1: tuple[float, float, float, float], t: float,
                           axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
                           ) -> tuple[float, float, float, float]:
    """swing/twistをそれぞれ独立にSLERPしてから合成する補間（`swing_twist_decompose`参照）。"""
    swing0, twist0 = swing_twist_decompose(q0, axis)
    swing1, twist1 = swing_twist_decompose(q1, axis)
    swing_t = quat_slerp(swing0, swing1, t)
    twist_t = quat_slerp(twist0, twist1, t)
    return quat_mul(swing_t, twist_t)


def ease_smoothstep(t: float) -> float:
    """線形のtを滑らかな加減速カーブに変換する（3t²-2t³）。0/1の端点は不変。"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# 腕・指（1本軸で長さ方向を持つボーン）はswing-twist分解を使う。Spine/Chest/Neck/Head
# はこの実測（X軸=長さ方向）が成り立つ根拠が無いため、通常のSLERPのままにする
_TWIST_AXIS_BONE_KEYWORDS = ("UpperArm", "LowerArm", "Hand", "Proximal", "Intermediate", "Distal")

# 指は腕がある程度動いてから遅れて握り込み始めるようにする（2026-08-01実機fb
# 「全部同時に動いて不自然」対応・「腕を伸ばしてから握る」という自然な順番に近づける）。
# (開始, 終了)は全体の時間に対する割合（0.0〜1.0）。指以外は0.0〜1.0（フル区間）のまま
_FINGER_BONE_SUFFIXES = ("Proximal", "Intermediate", "Distal")
_FINGER_TIME_WINDOW = (0.45, 1.0)


def _is_finger_bone(bone: str) -> bool:
    return any(bone.endswith(suffix) for suffix in _FINGER_BONE_SUFFIXES)


def _bone_time_window(bone: str) -> tuple[float, float]:
    return _FINGER_TIME_WINDOW if _is_finger_bone(bone) else (0.0, 1.0)


def _windowed_t(global_t: float, window: tuple[float, float]) -> float:
    """全体の進行度global_tを、そのボーン専用の(開始,終了)区間に写像したローカルtへ変換する
    （区間の外では0または1でクランプ＝始まる前は動かず、終わったらそこで静止）。"""
    start, end = window
    if global_t <= start:
        return 0.0
    if global_t >= end:
        return 1.0
    return (global_t - start) / (end - start)


def transition_to_pose(client: SimpleUDPClient, pose: dict[str, list[tuple[str, float]]],
                       duration_sec: float = 0.6, fps: float = 30.0) -> None:
    """現在の姿勢（reset_all_bonesが作る基準姿勢）から目標ポーズへ、poseが触る
    ボーンだけを滑らかに遷移させながら送る。それ以外のボーン（このポーズが使わない
    ボーン）はreset_all_bonesで一度だけ即座にニュートラルへ戻す（動かす必要が
    無い＝アニメーションさせる意味がないため）。
    腕・指ボーンはswing-twist分解＋SLERP、それ以外は通常のSLERPを使う。さらに
    指ボーンは`_FINGER_TIME_WINDOW`だけ全体の進行より遅れて動き出す（腕が先に
    振られてから握り込む、という自然な順番に近づけるため）。どちらもease_smoothstep
    で加減速を付ける。"""
    reset_all_bones(client)
    idle = POSES["idle_down"]
    starts: dict[str, tuple[float, float, float, float]] = {}
    targets: dict[str, tuple[float, float, float, float]] = {}
    for bone, steps in pose.items():
        starts[bone] = compose_quat(idle[bone]) if bone in idle else (0.0, 0.0, 0.0, 1.0)
        targets[bone] = compose_quat(steps)

    frame_count = max(1, round(duration_sec * fps))
    interval = duration_sec / frame_count
    for i in range(1, frame_count + 1):
        global_t = i / frame_count
        for bone in pose:
            local_t = ease_smoothstep(_windowed_t(global_t, _bone_time_window(bone)))
            if any(k in bone for k in _TWIST_AXIS_BONE_KEYWORDS):
                qx, qy, qz, qw = quat_slerp_swing_twist(starts[bone], targets[bone], local_t)
            else:
                qx, qy, qz, qw = quat_slerp(starts[bone], targets[bone], local_t)
            client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        time.sleep(interval)
    print(f"  [遷移完了] {duration_sec:.2f}秒かけて{len(pose)}ボーンを補間送信")


def main() -> int:
    parser = argparse.ArgumentParser(description="VMCアバターのポーズ探索スクリプト")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="VMCのReceiverポート番号")
    parser.add_argument("--pose", choices=sorted(POSES), help="送信するポーズ名")
    parser.add_argument("--list", action="store_true", help="ポーズ名の一覧を表示して終了")
    parser.add_argument("--cycle", action="store_true",
                        help="全ポーズを順番に送信（各ポーズの間にidle_downを挟む）")
    parser.add_argument("--hold", type=float, default=3.0,
                        help="--cycle時に1ポーズあたり維持する秒数")
    parser.add_argument("--deg-override", action="append", default=[],
                        metavar="骨名:軸:角度",
                        help="指定ボーンの角度を上書き（複数指定可・例: RightLowerArm:x:-50）")
    parser.add_argument("--repeat", action="store_true",
                        help="Ctrl+Cで止めるまで同じポーズを送り続ける"
                             "（1回だけ送ると時間経過で崩れる場合の切り分け用。"
                             "VMC側の物理演算/アイドル制御に上書きされていないか確認できる）")
    parser.add_argument("--repeat-interval", type=float, default=0.5,
                        help="--repeat時の再送間隔（秒・既定0.5秒）")
    parser.add_argument("--duration", type=float, default=0.6,
                        help="ポーズ遷移にかける秒数（既定0.6秒・0で--instantと同じ扱い）")
    parser.add_argument("--instant", action="store_true",
                        help="遷移アニメーションを使わず即座にポーズを切り替える"
                             "（角度を数値でざっくり確認したい時向け・旧来の挙動）")
    args = parser.parse_args()

    if args.list or (not args.pose and not args.cycle):
        print("利用可能なポーズ:")
        for name, pose in sorted(POSES.items()):
            bones = "・".join(pose.keys())
            print(f"  {name}: {bones}")
        if not args.pose and not args.cycle:
            print("\n--pose <名前> または --cycle を指定してください。")
        return 0

    client = SimpleUDPClient(args.host, args.port)
    animate = not args.instant and args.duration > 0

    def _apply(pose: dict[str, list[tuple[str, float]]]) -> None:
        if animate:
            transition_to_pose(client, pose, duration_sec=args.duration)
        else:
            reset_all_bones(client)  # 前回の呼び出しの状態を引きずらないよう毎回リセットしてから適用
            send_pose(client, pose)

    if args.cycle:
        for name, pose in POSES.items():
            if name == "idle_down":
                continue
            print(f"\n=== {name} ===")
            _apply(apply_overrides(pose, args.deg_override))
            time.sleep(args.hold)
        print("\n--- idle_down に戻す ---")
        _apply(POSES["idle_down"])
        print("一巡完了。")
        return 0

    pose = apply_overrides(POSES[args.pose], args.deg_override)
    print(f"[ポーズ送信] {args.pose}" + ("（遷移アニメーションあり）" if animate else "（即時切り替え）"))
    _apply(pose)
    print("アバターの姿勢を確認してください。")

    if args.repeat:
        print(f"\n[--repeat] {args.repeat_interval}秒間隔で送り続けます。Ctrl+Cで停止...")
        try:
            while True:
                time.sleep(args.repeat_interval)
                for bone, steps in pose.items():
                    qx, qy, qz, qw = compose_quat(steps)
                    client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        except KeyboardInterrupt:
            print("\n停止しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
