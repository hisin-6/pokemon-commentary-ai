"""
scripts/pose_tuner_gui.py（VMCへボーン角度を送りながらポーズを調整するGUI）の単体テスト

GUI本体（PoseTunerApp）は対象外。モジュールレベルの純粋関数だけを確認する:
  - look_rotation_quat     forward方向を向くクォータニオン（Unity LookRotation相当）
  - orbit_to_camera        オービットカメラの位置・常に注視点を向く回転
  - bone_label_ja          ボーン名の日本語ラベル
  - pose_to_axis_values    POSES形式 → ボーンごとの{x,y,z}辞書
  - axis_values_to_steps   {x,y,z} → ゼロでない軸だけの[(軸,角度)]
  - format_pose_dict       POSES辞書に貼れるPythonコード文字列

tkinter / python-osc が無い環境（WSLのシステムPython等）でもimportできるよう、
見つからない時だけMagicMockで差し替える（ある環境では本物をそのまま使う）。
"""

import ast
import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _stub_if_missing(*names):
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules.setdefault(name, MagicMock())


_stub_if_missing("tkinter", "tkinter.ttk", "pythonosc", "pythonosc.udp_client")


def _load(name):
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))  # pose_tuner_gui が explore_avatar_poses をimportする
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ptg = _load("pose_tuner_gui")
eap = sys.modules["explore_avatar_poses"]


# ── ヘルパー: クォータニオンでベクトルを回す ─────────────────────────────
def _rotate(q, v):
    """クォータニオンq=(x,y,z,w)でベクトルvを回転させる（q v q*）。"""
    qx, qy, qz, qw = q
    vq = (v[0], v[1], v[2], 0.0)
    conj = (-qx, -qy, -qz, qw)
    x, y, z, _ = eap.quat_mul(eap.quat_mul(q, vq), conj)
    return (x, y, z)


def _approx_vec(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


# ── look_rotation_quat ───────────────────────────────────────────────────
class TestLookRotationQuat:
    def test_forward_plus_z_is_identity(self):
        q = ptg.look_rotation_quat((0.0, 0.0, 1.0))
        assert q == pytest.approx((0.0, 0.0, 0.0, 1.0))

    @pytest.mark.parametrize("forward", [
        (0.0, 0.0, 1.0),    # trace>0 の分岐
        (0.0, 0.0, -1.0),   # 真後ろ（trace<0・m11最大の分岐）
        (1.0, 0.0, 0.0),    # 真横
        (-1.0, 0.0, 0.0),
        (0.3, -0.4, -0.8),  # 斜め（見下ろし気味の後方）
        (0.0, 0.0, 5.0),    # 長さ1でなくても正規化される
    ])
    def test_local_z_maps_to_forward_and_y_stays_up(self, forward):
        q = ptg.look_rotation_quat(forward)
        n = math.sqrt(sum(c * c for c in forward))
        f = tuple(c / n for c in forward)
        assert _approx_vec(_rotate(q, (0.0, 0.0, 1.0)), f)
        # ローカル+Yはforwardと直交したまま、上向き成分を保つ（逆さにならない）
        up = _rotate(q, (0.0, 1.0, 0.0))
        assert abs(sum(a * b for a, b in zip(up, f))) < 1e-6
        assert up[1] > 0

    def test_result_is_unit_quaternion(self):
        q = ptg.look_rotation_quat((0.2, 0.5, -0.7))
        assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0)


# ── orbit_to_camera ──────────────────────────────────────────────────────
class TestOrbitToCamera:
    def test_default_front_position(self):
        pos, _ = ptg.orbit_to_camera(0.0, 0.0, 2.0, 1.1)
        assert pos == pytest.approx((0.0, 1.1, 2.0))

    def test_yaw_90_moves_to_plus_x(self):
        pos, _ = ptg.orbit_to_camera(90.0, 0.0, 3.0, 1.0)
        assert pos == pytest.approx((3.0, 1.0, 0.0), abs=1e-9)

    def test_pitch_raises_camera(self):
        pos, _ = ptg.orbit_to_camera(0.0, 30.0, 2.0, 1.0)
        assert pos == pytest.approx((0.0, 1.0 + 2.0 * 0.5, 2.0 * math.cos(math.radians(30))))

    def test_distance_is_kept(self):
        pos, _ = ptg.orbit_to_camera(37.0, 22.0, 2.5, 1.1)
        d = math.sqrt(pos[0] ** 2 + (pos[1] - 1.1) ** 2 + pos[2] ** 2)
        assert d == pytest.approx(2.5)

    @pytest.mark.parametrize("yaw,pitch", [(0, 0), (90, 10), (180, -20), (-135, 45)])
    def test_camera_always_looks_at_target(self, yaw, pitch):
        pos, q = ptg.orbit_to_camera(yaw, pitch, 2.0, 1.1)
        to_target = (-pos[0], 1.1 - pos[1], -pos[2])
        n = math.sqrt(sum(c * c for c in to_target))
        assert _approx_vec(_rotate(q, (0.0, 0.0, 1.0)), tuple(c / n for c in to_target))


# ── bone_label_ja ────────────────────────────────────────────────────────
class TestBoneLabelJa:
    @pytest.mark.parametrize("bone,expected", [
        ("Spine", "背骨（Spine）"),
        ("RightHand", "右手首（RightHand）"),
        ("LeftThumbProximal", "左親指1(根元)（LeftThumbProximal）"),
        ("RightLittleDistal", "右小指3(先端)（RightLittleDistal）"),
        ("LeftIndexIntermediate", "左人差し指2(中間)（LeftIndexIntermediate）"),
    ])
    def test_known_bones(self, bone, expected):
        assert ptg.bone_label_ja(bone) == expected

    def test_unknown_bone_falls_back_to_english(self):
        assert ptg.bone_label_ja("LeftToes") == "LeftToes（LeftToes）"

    def test_every_tuned_bone_gets_japanese_label(self):
        for bone in ptg.ALL_BONES:
            label = ptg.bone_label_ja(bone)
            assert label != f"{bone}（{bone}）", bone


# ── pose_to_axis_values / axis_values_to_steps ──────────────────────────
class TestPoseConversion:
    def test_bone_groups_cover_40_bones_without_duplicates(self):
        assert len(ptg.ALL_BONES) == 40
        assert len(set(ptg.ALL_BONES)) == 40

    def test_fills_zeros_and_ignores_unknown_bones(self):
        values = ptg.pose_to_axis_values({
            "RightUpperArm": [("z", -80.0), ("x", 15.0)],
            "LeftUpperLeg": [("x", 30.0)],   # 調整対象外 → 無視
        })
        assert set(values) == set(ptg.ALL_BONES)
        assert values["RightUpperArm"] == {"x": 15.0, "y": 0.0, "z": -80.0}
        assert values["Spine"] == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert "LeftUpperLeg" not in values

    def test_same_axis_twice_last_wins(self):
        values = ptg.pose_to_axis_values({"Head": [("y", 10.0), ("y", -5.0)]})
        assert values["Head"]["y"] == -5.0

    def test_steps_skip_zero_axes_in_xyz_order(self):
        assert ptg.axis_values_to_steps({"x": 0.0, "y": 0.0, "z": 0.0}) == []
        assert ptg.axis_values_to_steps({"z": 3.0, "x": -1.5, "y": 1e-9}) == [("x", -1.5), ("z", 3.0)]

    @pytest.mark.parametrize("name", list(eap.POSES))
    def test_existing_poses_load_without_error(self, name):
        # explore_avatar_poses のPOSESを「読み込み」ドロップダウンで開けること
        values = ptg.pose_to_axis_values(eap.POSES[name])
        assert set(values) == set(ptg.ALL_BONES)


# ── format_pose_dict ─────────────────────────────────────────────────────
class TestFormatPoseDict:
    def test_all_zero_gives_empty_dict(self):
        values = ptg.pose_to_axis_values({})
        assert ptg.format_pose_dict(values) == "{\n}"

    def test_output_is_valid_python_and_round_trips(self):
        pose = {
            "Spine": [("x", -10.0)],
            "RightUpperArm": [("x", 15.0), ("z", -80.0)],
            "RightIndexProximal": [("z", 72.5)],
        }
        text = ptg.format_pose_dict(ptg.pose_to_axis_values(pose))
        parsed = ast.literal_eval(text)
        assert parsed == {bone: [tuple(s) for s in steps] for bone, steps in pose.items()}
        # 再読み込みしても同じ値に戻る（コピペ→POSES追加→読み込みの往復）
        assert ptg.pose_to_axis_values(parsed) == ptg.pose_to_axis_values(pose)

    def test_bones_follow_all_bones_order_and_one_decimal(self):
        values = ptg.pose_to_axis_values({
            "RightHand": [("y", 1.25)],
            "Spine": [("x", -3.0)],
        })
        assert ptg.format_pose_dict(values) == (
            "{\n"
            '    "Spine": [("x", -3.0)],\n'
            '    "RightHand": [("y", 1.2)],\n'
            "}"
        )
