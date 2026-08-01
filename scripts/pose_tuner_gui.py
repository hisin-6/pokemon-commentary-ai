"""VMCへリアルタイムでボーン角度を送りながら、数値入力＋±ボタンで見た目を確認しつつ
ポーズを調整できるGUIツール。

`explore_avatar_poses.py`での「角度を数値で当てずっぽう→実機で見て→また数値を
変えて再送信」というCLI往復に限界があった（座標や角度を目視だけで伝えるのが
難しい）ため追加。値を変えるたびに即座にOSCで送信し、その場でアバターの
見た目を確認しながら追い込める。

追加インストール不要（tkinterはPython標準ライブラリ）。python-oscのみ必要:
    venv\\Scripts\\pip.exe install python-osc

使い方（VMC起動中・Receiver有効化チェック済みの状態で）:
    venv\\Scripts\\python.exe scripts\\pose_tuner_gui.py --port 39540

画面の使い方:
  - ボーンごとにX/Y/Z軸の「数値入力欄＋±1/±10ボタン」が並ぶ（2026-08-01・
    スライダーだと微調整しづらいというfbを受けて置き換え）。数値欄は入力して
    Enterかフォーカスを外すと確定、ボタンはクリックのたびにその場で反映される
  - 体幹・左腕・右腕・左手の指・右手の指の見出しでグループ分けされている
    （2026-08-01・左手の指も追加。従来は右手のみ対応だった）
  - 上部の「読み込み」ドロップダウンで`explore_avatar_poses.py`のPOSESにある
    既存ポーズ（fist_pump_right等）を初期値として読み込める
  - 「idle_downにリセット」で基準姿勢に戻せる
  - 「Pythonコードを出力」で、現在の数値を`explore_avatar_poses.py`の
    POSES辞書にそのまま貼り付けられる形式のテキストを表示する（コピペ用）
  - Handボーン・指ボーンはスプリング物理の影響で送りっぱなしにしないと崩れる
    ことが判明済みのため（2026-08-01調査）、このツールは起動中ずっと
    全ボーンの現在値を一定間隔（既定0.2秒）で送り続ける
  - 「カメラ操作を有効にする」チェックボックスをONにすると、**オービットカメラ**
    （水平角度・垂直角度・距離・注視点の高さ・画角のスライダー）がVMCへ
    `/VMC/Ext/Cam`で送られ続ける。ポーズ確認用に「モデルを中心にぐるっと
    見る角度を変える」ことに特化した操作方法で、位置と向きを別々に合わせる
    必要がない（常に注視点＝モデルの方を自動で向く）。VMC Protocol仕様上、
    このメッセージを受信すると**強制的にフリーカメラモードに切り替わる**
    （元のカメラ設定を上書きする）ため、普段のカメラワークを崩したくない時は
    チェックを入れないこと。チェックを外しても送信が止まるだけで、VMC側の
    フリーカメラモード自体は自動的には元に戻らない（戻す場合はVMCの操作で
    切り替える）
"""

from __future__ import annotations

import argparse
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent))
import explore_avatar_poses as eap  # axis_angle_quat/quat_mul/compose_quat/POSESを再利用

from pythonosc.udp_client import SimpleUDPClient

# 調整対象ボーン。explore_avatar_poses.pyの_ALL_POSE_BONES相当（体幹・両腕）に
# 加えて、拳を握る調整用に左右両方の指ボーンを含める。UIでは見出し付きで
# グループ表示するため、(見出し, ボーン名リスト)のタプルで管理する
TRUNK_BONES = ["Spine", "Chest", "Neck", "Head"]
LEFT_ARM_BONES = ["LeftUpperArm", "LeftLowerArm", "LeftHand"]
RIGHT_ARM_BONES = ["RightUpperArm", "RightLowerArm", "RightHand"]
LEFT_FINGER_BONES = [
    f"Left{finger}{joint}"
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little")
    for joint in ("Proximal", "Intermediate", "Distal")
]
RIGHT_FINGER_BONES = [
    f"Right{finger}{joint}"
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little")
    for joint in ("Proximal", "Intermediate", "Distal")
]
BONE_GROUPS: list[tuple[str, list[str]]] = [
    ("体幹", TRUNK_BONES),
    ("左腕", LEFT_ARM_BONES),
    ("右腕", RIGHT_ARM_BONES),
    ("左手の指", LEFT_FINGER_BONES),
    ("右手の指", RIGHT_FINGER_BONES),
]
ALL_BONES = [bone for _, bones in BONE_GROUPS for bone in bones]

_RESEND_INTERVAL_MS = 200  # Hand/指ボーンの物理演算対策（play_and_animate_avatar.py調査で判明）
_NUDGE_STEPS = (-10.0, -1.0, 1.0, 10.0)  # 数値入力の横に並べる±ボタンの刻み幅

# ── 変更箇所のハイライト（2026-08-01追加）───────────────────────────────
# 120項目（40ボーン×3軸）もあると数値だけではどこを変えたか分かりづらいという
# fbを受けて、値が0（＝ポーズに効いていない）かどうかで背景色を変える
_AXIS_DEFAULT_BG = "white"
_AXIS_CHANGED_BG = "#ffe58a"  # 淡い黄色
_BONE_DEFAULT_BG = "SystemButtonFace"  # ttk.Frameの既定背景に合わせる（Windows既定色）
_BONE_CHANGED_BG = "#ffd166"  # ボーン名側は少し濃いめにして軸側と区別する
_AXES = ("x", "y", "z")

# ── カメラ操作（2026-08-01追加 → オービットカメラ化）───────────────────
# VMC Protocol仕様の"/VMC/Ext/Cam"を使用（sh-akira/VirtualMotionCaptureProtocol
# サンプルのCameraPositionSend.csで確認）。引数は
# (string)"camera", (float)pos.x,y,z, (float)rot.x,y,z,w(クォータニオン), (float)fov
# ⚠️「受信時、強制的にフリーカメラになる」と仕様書に明記されている（未実機検証）ため、
# デフォルトでは送信しない（チェックボックスでの明示的なON時のみ）。
#
# ポーズ確認用途（「モデルを中心にいろんな角度から見たい」）に合わせて、位置・回転を
# 別々に手打ちするのではなく、**注視点を中心に球面上を回る「オービットカメラ」**として
# 実装する: 水平角度(yaw)・垂直角度(pitch)・距離(distance)・注視点の高さ(target_height)
# の4つだけを操作すれば、常にモデルの方を向いたカメラ位置・回転が自動計算される。
_CAM_ADDR = "/VMC/Ext/Cam"
_CAM_NAME = "camera"
_CAM_DEFAULT_FOV = 60.0
# 初期値は完全な当てずっぽう（キャラクターがワールド原点付近・胸の高さ1.1m前後に
# いる想定）。実機で位置がズレていたら注視点の高さ(target_height)をスライダーで
# 調整すること
_ORBIT_DEFAULT = {"yaw": 0.0, "pitch": 10.0, "distance": 2.0, "target_height": 1.1}


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]
          ) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def look_rotation_quat(forward: tuple[float, float, float],
                       up: tuple[float, float, float] = (0.0, 1.0, 0.0)
                       ) -> tuple[float, float, float, float]:
    """forward方向を向くクォータニオンを返す（Unityの`Quaternion.LookRotation`相当）。
    ローカル+Zがforward・ローカル+Yがupになるよう回転行列を作り、クォータニオンへ変換する
    （Shepperdの方法・数値的に安定した4パターン分岐版）。カメラの実際の前方軸がVMC側で
    本当に+Zかどうかは未実機検証（違ったら180°反転などの補正が必要になる可能性あり）。"""
    f = _normalize(forward)
    r = _normalize(_cross(up, f))
    u = _cross(f, r)
    m00, m01, m02 = r[0], u[0], f[0]
    m10, m11, m12 = r[1], u[1], f[1]
    m20, m21, m22 = r[2], u[2], f[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return ((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s)
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


def orbit_to_camera(yaw_deg: float, pitch_deg: float, distance: float, target_height: float
                    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """注視点(0, target_height, 0)を中心に、水平角度yaw・垂直角度pitch・距離distanceの
    球面上に置いたカメラの(位置, 常に注視点を向く回転クォータニオン)を返す。"""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    dx = math.cos(pitch) * math.sin(yaw)
    dy = math.sin(pitch)
    dz = math.cos(pitch) * math.cos(yaw)
    pos = (distance * dx, target_height + distance * dy, distance * dz)
    forward = _normalize((-dx, -dy, -dz))
    return pos, look_rotation_quat(forward)

# ── ボーン名の日本語ラベル（表示専用。VMCへ送るボーン名は英語のまま変更しない） ──
_CORE_BONE_JA = {
    "Spine": "背骨", "Chest": "胸", "Neck": "首", "Head": "頭",
    "LeftUpperArm": "左上腕", "LeftLowerArm": "左前腕", "LeftHand": "左手首",
    "RightUpperArm": "右上腕", "RightLowerArm": "右前腕", "RightHand": "右手首",
}
_FINGER_JA = {"Thumb": "親指", "Index": "人差し指", "Middle": "中指",
              "Ring": "薬指", "Little": "小指"}
_JOINT_JA = {"Proximal": "1(根元)", "Intermediate": "2(中間)", "Distal": "3(先端)"}


def bone_label_ja(bone: str) -> str:
    """ボーンの英語名（HumanBodyBones）に日本語ラベルを添えた表示用文字列を返す。"""
    if bone in _CORE_BONE_JA:
        ja = _CORE_BONE_JA[bone]
    else:
        for finger, finger_ja in _FINGER_JA.items():
            if bone.startswith(f"Right{finger}") or bone.startswith(f"Left{finger}"):
                side_ja = "右" if bone.startswith("Right") else "左"
                joint = bone[len(f"{'Right' if bone.startswith('Right') else 'Left'}{finger}"):]
                ja = f"{side_ja}{finger_ja}{_JOINT_JA.get(joint, joint)}"
                break
        else:
            ja = bone  # 未知のボーンは英語名のままフォールバック
    return f"{ja}（{bone}）"


def pose_to_axis_values(pose: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
    """POSES形式の1ポーズを、ボーンごとの{"x":deg,"y":deg,"z":deg}辞書に変換する
    （ALL_BONESに含まれないボーンは無視・同一ボーン内で同じ軸が複数回出てくる
    ケースは現状のPOSES定義には無いため、後勝ちで上書きする単純な実装で良い）。"""
    result = {bone: {"x": 0.0, "y": 0.0, "z": 0.0} for bone in ALL_BONES}
    for bone, steps in pose.items():
        if bone not in result:
            continue
        for axis, deg in steps:
            if axis in result[bone]:
                result[bone][axis] = deg
    return result


def axis_values_to_steps(values: dict[str, float]) -> list[tuple[str, float]]:
    """{"x":deg,...} からゼロでない軸だけを[(軸,角度),...]として返す（出力用）。"""
    return [(axis, values[axis]) for axis in _AXES if abs(values[axis]) > 1e-6]


def format_pose_dict(bones_values: dict[str, dict[str, float]]) -> str:
    """現在の全ボーン値を、explore_avatar_poses.pyのPOSES辞書にそのまま貼れる
    Pythonコード文字列にする（ゼロ回転のボーンは出力しない）。"""
    lines = ["{"]
    for bone in ALL_BONES:
        steps = axis_values_to_steps(bones_values[bone])
        if not steps:
            continue
        steps_str = ", ".join(f'("{axis}", {deg:.1f})' for axis, deg in steps)
        lines.append(f'    "{bone}": [{steps_str}],')
    lines.append("}")
    return "\n".join(lines)


class PoseTunerApp:
    def __init__(self, root: tk.Tk, client: SimpleUDPClient):
        self.root = root
        self.client = client
        self.vars: dict[str, dict[str, tk.DoubleVar]] = {
            bone: {axis: tk.DoubleVar(value=0.0) for axis in _AXES} for bone in ALL_BONES
        }
        # 数値入力欄の表示専用（self.varsとは別管理。入力途中の不正な文字列
        # （例:"-"だけ打った瞬間）でも送信ループがクラッシュしないようにするため）
        self.entry_vars: dict[str, dict[str, tk.StringVar]] = {
            bone: {axis: tk.StringVar(value="0.0") for axis in _AXES} for bone in ALL_BONES
        }
        self.cam_enabled = tk.BooleanVar(value=False)
        self.orbit = {k: tk.DoubleVar(value=v) for k, v in _ORBIT_DEFAULT.items()}
        self.cam_fov = tk.DoubleVar(value=_CAM_DEFAULT_FOV)
        self._build_ui()
        self.reset_to_idle()
        self._resend_loop()

    # ── UI構築 ──────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.root.title("VMCポーズ調整ツール")

        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="idle_downにリセット", command=self.reset_to_idle).pack(side=tk.LEFT)

        ttk.Label(top, text="  読み込み:").pack(side=tk.LEFT)
        self.pose_choice = tk.StringVar(value="")
        pose_names = sorted(eap.POSES)
        combo = ttk.Combobox(top, textvariable=self.pose_choice, values=pose_names,
                             state="readonly", width=20)
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda e: self.load_pose(self.pose_choice.get()))

        ttk.Button(top, text="Pythonコードを出力", command=self._show_export).pack(side=tk.LEFT, padx=8)

        # カメラ操作（/VMC/Ext/Cam）。既定OFF＝受信すると強制的にフリーカメラに
        # 切り替わる仕様のため、明示的にONにするまでは一切送信しない。
        # 「モデルを中心にいろんな角度から見たい」（ポーズ確認用途）に合わせて、
        # 位置・回転を別々に合わせるのではなく、注視点を中心に回る
        # オービットカメラ（水平角度・垂直角度・距離・注視点の高さ）として操作する
        cam_frame = ttk.LabelFrame(self.root, text="オービットカメラ操作（⚠️ONにするとVMCが強制的にフリーカメラモードになります）", padding=8)
        cam_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        ttk.Checkbutton(cam_frame, text="カメラ操作を有効にする", variable=self.cam_enabled,
                        command=self._on_cam_toggle).pack(side=tk.LEFT)

        cam_sliders = ttk.Frame(cam_frame)
        cam_sliders.pack(side=tk.LEFT, padx=12)
        self.cam_value_labels: dict[str, ttk.Label] = {}

        def _add_cam_slider(parent, label, var, key, frm, to):
            cell = ttk.Frame(parent)
            cell.pack(side=tk.LEFT, padx=4)
            ttk.Label(cell, text=label, width=10).pack(side=tk.LEFT)
            scale = ttk.Scale(cell, from_=frm, to=to, orient=tk.HORIZONTAL, length=120,
                              variable=var, command=lambda _v: self._on_cam_change())
            scale.pack(side=tk.LEFT)
            value_label = ttk.Label(cell, text=f"{var.get():.1f}", width=6)
            value_label.pack(side=tk.LEFT)
            self.cam_value_labels[key] = value_label

        _add_cam_slider(cam_sliders, "水平角度", self.orbit["yaw"], "yaw", -180.0, 180.0)
        _add_cam_slider(cam_sliders, "垂直角度", self.orbit["pitch"], "pitch", -80.0, 80.0)
        _add_cam_slider(cam_sliders, "距離", self.orbit["distance"], "distance", 0.5, 5.0)
        _add_cam_slider(cam_sliders, "注視点の高さ", self.orbit["target_height"], "target_height", 0.0, 2.0)
        _add_cam_slider(cam_sliders, "画角", self.cam_fov, "fov", 10.0, 120.0)

        ttk.Button(cam_frame, text="カメラをリセット", command=self._reset_camera).pack(side=tk.LEFT, padx=8)

        # スクロール可能なボーン一覧
        container = ttk.Frame(self.root)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(container, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        header = ttk.Frame(inner)
        header.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(header, text="ボーン", width=40).pack(side=tk.LEFT)
        for axis in _AXES:
            ttk.Label(header, text=f"{axis.upper()}軸（数値Enter確定／±ボタン）", width=34,
                     anchor="center").pack(side=tk.LEFT)

        self.entries: dict[str, dict[str, tk.Entry]] = {}
        self.bone_labels: dict[str, tk.Label] = {}
        for group_label, bones in BONE_GROUPS:
            group_header = ttk.Label(inner, text=f"── {group_label} ──", font=("", 10, "bold"))
            group_header.pack(anchor=tk.W, pady=(8, 2))
            for bone in bones:
                row = ttk.Frame(inner)
                row.pack(fill=tk.X, pady=1)
                label = tk.Label(row, text=bone_label_ja(bone), width=40, anchor="w")
                label.pack(side=tk.LEFT)
                self.bone_labels[bone] = label
                self.entries[bone] = {}
                for axis in _AXES:
                    self._build_axis_control(row, bone, axis)

        # 出力表示用テキストボックス（下部固定）
        export_frame = ttk.Frame(self.root, padding=8)
        export_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(export_frame, text="出力（コピペ用）:").pack(anchor=tk.W)
        self.export_text = tk.Text(export_frame, height=8)
        self.export_text.pack(fill=tk.X)

    def _build_axis_control(self, parent: tk.Widget, bone: str, axis: str) -> None:
        """1ボーン・1軸ぶんの「数値入力欄＋±ボタン」を作る（スライダーだと微調整しづらい
        というfbを受けて2026-08-01にスライダーから置き換え）。
        値が0（＝ポーズに効いていない）かどうかで背景色を変える（2026-08-01追加・
        120項目もあるとどこを変えたか分かりづらいというfb対応）ため、背景色を
        直接指定できる素のtk.Entryを使う（ttk.Entryはテーマ経由でしか色を変えられない）。"""
        cell = ttk.Frame(parent)
        cell.pack(side=tk.LEFT, padx=2)
        ttk.Label(cell, text=f"{axis.upper()}:", width=2).pack(side=tk.LEFT)
        entry = tk.Entry(cell, textvariable=self.entry_vars[bone][axis], width=7)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", lambda e, b=bone, a=axis: self._on_entry_commit(b, a))
        entry.bind("<FocusOut>", lambda e, b=bone, a=axis: self._on_entry_commit(b, a))
        self.entries[bone][axis] = entry
        for step in _NUDGE_STEPS:
            text = f"{step:+.0f}"
            ttk.Button(cell, text=text, width=3,
                      command=lambda b=bone, a=axis, s=step: self._nudge_axis(b, a, s)
                      ).pack(side=tk.LEFT)

    # ── OSC送信 ──────────────────────────────────────────────────────────
    def _send_bone(self, bone: str) -> None:
        v = self.vars[bone]
        steps = [(axis, v[axis].get()) for axis in _AXES]
        qx, qy, qz, qw = eap.compose_quat(steps)
        self.client.send_message("/VMC/Ext/Bone/Pos", [bone, 0.0, 0.0, 0.0, qx, qy, qz, qw])

    def _sync_entry_display(self, bone: str, axis: str) -> None:
        self.entry_vars[bone][axis].set(f"{self.vars[bone][axis].get():.1f}")
        self._update_highlight(bone, axis)

    def _update_highlight(self, bone: str, axis: str) -> None:
        """値が0（ポーズに効いていない）かどうかで背景色を変える。ボーン名ラベルも
        3軸のうちどれか1つでも0でなければハイライトし、ボーン単位でも一目で
        分かるようにする。"""
        changed = abs(self.vars[bone][axis].get()) > 1e-6
        self.entries[bone][axis].configure(
            bg=_AXIS_CHANGED_BG if changed else _AXIS_DEFAULT_BG)
        bone_changed = any(abs(self.vars[bone][a].get()) > 1e-6 for a in _AXES)
        self.bone_labels[bone].configure(
            bg=_BONE_CHANGED_BG if bone_changed else _BONE_DEFAULT_BG,
            font=("", 9, "bold" if bone_changed else "normal"))

    def _on_entry_commit(self, bone: str, axis: str) -> None:
        """数値入力欄でEnterまたはフォーカスが外れた時に確定させる。不正な数値
        （空欄・記号だけ等）は無視して直前の値の表示に戻す。"""
        try:
            value = float(self.entry_vars[bone][axis].get())
        except ValueError:
            self._sync_entry_display(bone, axis)
            return
        self.vars[bone][axis].set(value)
        self._sync_entry_display(bone, axis)
        self._send_bone(bone)

    def _nudge_axis(self, bone: str, axis: str, step: float) -> None:
        self.vars[bone][axis].set(self.vars[bone][axis].get() + step)
        self._sync_entry_display(bone, axis)
        self._send_bone(bone)

    def _resend_loop(self) -> None:
        for bone in ALL_BONES:
            self._send_bone(bone)
        if self.cam_enabled.get():
            self._send_camera()
        self.root.after(_RESEND_INTERVAL_MS, self._resend_loop)

    # ── カメラ操作（オービット）──────────────────────────────────────────
    def _send_camera(self) -> None:
        pos, quat = orbit_to_camera(
            self.orbit["yaw"].get(), self.orbit["pitch"].get(),
            self.orbit["distance"].get(), self.orbit["target_height"].get())
        px, py, pz = pos
        qx, qy, qz, qw = quat
        fov = self.cam_fov.get()
        self.client.send_message(_CAM_ADDR, [_CAM_NAME, px, py, pz, qx, qy, qz, qw, fov])

    def _on_cam_change(self) -> None:
        for key, var in {**{k: self.orbit[k] for k in self.orbit}, "fov": self.cam_fov}.items():
            self.cam_value_labels[key].configure(text=f"{var.get():.1f}")
        if self.cam_enabled.get():
            self._send_camera()

    def _on_cam_toggle(self) -> None:
        if self.cam_enabled.get():
            self._send_camera()

    def _reset_camera(self) -> None:
        for k, v in _ORBIT_DEFAULT.items():
            self.orbit[k].set(v)
        self.cam_fov.set(_CAM_DEFAULT_FOV)
        self._on_cam_change()

    # ── ポーズ操作 ──────────────────────────────────────────────────────
    def _set_all(self, bones_values: dict[str, dict[str, float]]) -> None:
        for bone in ALL_BONES:
            for axis in _AXES:
                self.vars[bone][axis].set(bones_values[bone][axis])
                self._sync_entry_display(bone, axis)
        for bone in ALL_BONES:
            self._send_bone(bone)

    def reset_to_idle(self) -> None:
        values = {bone: {"x": 0.0, "y": 0.0, "z": 0.0} for bone in ALL_BONES}
        values["LeftUpperArm"]["z"] = eap._IDLE_ARM_DEG
        values["RightUpperArm"]["z"] = -eap._IDLE_ARM_DEG
        self._set_all(values)

    def load_pose(self, name: str) -> None:
        if not name:
            return
        base = {bone: {"x": 0.0, "y": 0.0, "z": 0.0} for bone in ALL_BONES}
        base["LeftUpperArm"]["z"] = eap._IDLE_ARM_DEG
        base["RightUpperArm"]["z"] = -eap._IDLE_ARM_DEG
        pose_values = pose_to_axis_values(eap.POSES.get(name, {}))
        for bone in ALL_BONES:
            for axis in _AXES:
                if abs(pose_values[bone][axis]) > 1e-9:
                    base[bone][axis] = pose_values[bone][axis]
        self._set_all(base)

    def _show_export(self) -> None:
        values = {bone: {axis: self.vars[bone][axis].get() for axis in _AXES} for bone in ALL_BONES}
        text = format_pose_dict(values)
        self.export_text.delete("1.0", tk.END)
        self.export_text.insert(tk.END, text)


def main() -> int:
    parser = argparse.ArgumentParser(description="VMCポーズ調整GUIツール")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="VMCのReceiverポート番号")
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)
    root = tk.Tk()
    PoseTunerApp(root, client)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
