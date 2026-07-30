"""VMC ProtocolのOSCで腕のボーン姿勢を上書きし、トラッキングデバイス無しの
T-poseを自然な「腕を下ろしたポーズ」に変えられるか確認するテストスクリプト。

/VMC/Ext/Bone/Pos (string)name (float)x,y,z (float)qx,qy,qz,qw
ボーン名はUnityのHumanBodyBones形式（LeftUpperArm/RightUpperArm等）。
位置は今回動かさないので(0,0,0)固定・回転のみZ軸周りで指定角度を送る。

事前準備（Windows側）:
    venv\\Scripts\\pip.exe install python-osc

使い方（VMC起動中・Receiver有効化チェック済みの状態で）:
    venv\\Scripts\\python.exe scripts\\test_vmc_pose.py --port 39540 --deg -80

--deg の符号・角度は座標系依存で正解が分からないため、値を変えながら
（例: 80, -80, 45, -45, 90, -90）どれが「腕が自然に下がる」か目視で探ること。
軸を変えたい場合は --axis x|y|z を指定（既定はz）。
"""

from __future__ import annotations

import argparse
import math
import time

from pythonosc.udp_client import SimpleUDPClient


def axis_angle_quat(axis: str, deg: float) -> tuple[float, float, float, float]:
    """指定軸まわり deg 度の回転をクォータニオン(x,y,z,w)で返す。"""
    rad = math.radians(deg)
    s = math.sin(rad / 2)
    c = math.cos(rad / 2)
    if axis == "x":
        return (s, 0.0, 0.0, c)
    if axis == "y":
        return (0.0, s, 0.0, c)
    return (0.0, 0.0, s, c)


def send_bone(client: SimpleUDPClient, name: str, quat: tuple[float, float, float, float]) -> None:
    qx, qy, qz, qw = quat
    client.send_message("/VMC/Ext/Bone/Pos", [name, 0.0, 0.0, 0.0, qx, qy, qz, qw])


def main() -> None:
    parser = argparse.ArgumentParser(description="VMCの腕ボーン姿勢テスト（T-pose回避の実験）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deg", type=float, default=-80.0,
                        help="LeftUpperArmに送る回転角度（度）。RightUpperArmには符号反転で送る")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--bones", nargs="*",
                        default=["LeftUpperArm", "RightUpperArm"],
                        help="対象ボーン名（既定: 左右上腕）")
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    for bone in args.bones:
        # 左右で符号を反転（鏡像の腕を同時に自然な向きへ）。
        # "Right" を含むボーン名だけ反転する簡易判定
        deg = -args.deg if "Right" in bone else args.deg
        quat = axis_angle_quat(args.axis, deg)
        print(f"[送信] {bone}: axis={args.axis} deg={deg} quat={quat}")
        send_bone(client, bone, quat)

    print("送信完了。アバターの腕の角度を確認してください。")
    print("違う角度を試す場合は --deg / --axis を変えて再実行（Ctrl+Cで終了不要・毎回上書きされます）")


if __name__ == "__main__":
    main()
