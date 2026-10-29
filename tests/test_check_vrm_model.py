"""
scripts/check_vrm_model.py（外注VRM差し替え前の表情・ボーン一発チェック）の単体テスト

テスト対象:
  - check_gltf          VRM 0.x / 1.0 のJSONから必須表情・ボーンの欠けを判定
  - unity_to_vrm_bone   VMC(Unity)ボーン名 → VRM humanoidボーン名（1.0の親指ずれ含む）
  - _ANIMATE_BONES      play_and_animate_avatar.py 側の回すボーン定数との同期
  - main                .vrm(glb)ファイルを読んで終了コードを返す
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("pythonosc", MagicMock())
sys.modules.setdefault("pythonosc.udp_client", MagicMock())

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclassの型解決にsys.modules登録が要る
    spec.loader.exec_module(mod)
    return mod


cvm = _load("check_vrm_model")

_BONES = {"Spine", "LeftUpperArm", "RightThumbIntermediate"}


def _vrm0(groups, bones=("spine", "leftUpperArm", "rightThumbIntermediate")):
    return {"extensions": {"VRM": {
        "meta": {"title": "くれぴ", "author": "エンドウ", "licenseName": "Other"},
        "blendShapeMaster": {"blendShapeGroups": groups},
        "humanoid": {"humanBones": [{"bone": b, "node": i} for i, b in enumerate(bones)]},
    }}}


def _group(name, preset=None, binds=1):
    return {"name": name, "presetName": preset or name.lower(),
            "binds": [{"mesh": 0, "index": 0, "weight": 100}] * binds}


_ALL0 = [_group(n) for n in ("Neutral", "Joy", "Sorrow", "Fun", "Angry",
                             "A", "I", "U", "E", "O", "Blink")]


class TestVrm0:
    def test_all_present_is_ok(self):
        r = cvm.check_gltf(_vrm0(_ALL0), _BONES)
        assert r.ok
        assert r.spec_version == "0.x"
        assert not r.missing_recommended_expressions
        assert r.meta["author"] == "エンドウ"

    def test_missing_required_expression_is_ng(self):
        groups = [g for g in _ALL0 if g["name"] != "Sorrow"]
        r = cvm.check_gltf(_vrm0(groups), _BONES)
        assert not r.ok
        assert r.missing_required_expressions == ["Sorrow"]

    def test_missing_recommended_only_is_still_ok(self):
        groups = [g for g in _ALL0 if g["name"] not in ("A", "Blink")]
        r = cvm.check_gltf(_vrm0(groups), _BONES)
        assert r.ok
        assert r.missing_recommended_expressions == ["A", "Blink"]

    def test_preset_name_alone_is_enough(self):
        # 表情名が日本語でもpresetNameが正しければVMCから引ける
        groups = [_group("よろこび", "joy") if g["name"] == "Joy" else g for g in _ALL0]
        r = cvm.check_gltf(_vrm0(groups), _BONES)
        assert "Joy" not in r.missing_required_expressions

    def test_empty_binds_warned_except_neutral(self):
        groups = [_group(g["name"], binds=0) if g["name"] in ("Neutral", "Fun") else g
                  for g in _ALL0]
        r = cvm.check_gltf(_vrm0(groups), _BONES)
        assert r.empty_expressions == ["Fun"]

    def test_missing_bone_is_ng(self):
        r = cvm.check_gltf(_vrm0(_ALL0, bones=("spine",)), _BONES)
        assert not r.ok
        assert r.missing_bones == ["LeftUpperArm", "RightThumbIntermediate"]


class TestVrm1:
    def _gltf(self, presets):
        bind = {"morphTargetBinds": [{"node": 0, "index": 0, "weight": 1.0}]}
        return {"extensions": {"VRMC_vrm": {
            "meta": {"name": "x", "authors": ["a", "b"]},
            "expressions": {"preset": {p: bind for p in presets}},
            # 1.0の親指: Unity Intermediate ＝ 1.0 Proximal
            "humanoid": {"humanBones": {"spine": {"node": 0}, "leftUpperArm": {"node": 1},
                                        "rightThumbProximal": {"node": 2}}},
        }}}

    def test_aliases_and_no_neutral_ok(self):
        r = cvm.check_gltf(self._gltf(["happy", "sad", "relaxed", "angry", "aa", "ih",
                                       "ou", "ee", "oh", "blink"]), _BONES)
        assert r.ok
        assert r.spec_version == "1.0"
        assert not r.missing_recommended_expressions
        assert r.meta["authors"] == "a, b"

    def test_missing_happy_is_ng(self):
        r = cvm.check_gltf(self._gltf(["sad", "relaxed", "angry"]), _BONES)
        assert r.missing_required_expressions == ["Joy"]


def test_unity_to_vrm_bone():
    assert cvm.unity_to_vrm_bone("LeftUpperArm") == "leftUpperArm"
    assert cvm.unity_to_vrm_bone("RightThumbIntermediate") == "rightThumbIntermediate"
    assert cvm.unity_to_vrm_bone("RightThumbIntermediate", "1.0") == "rightThumbProximal"
    assert cvm.unity_to_vrm_bone("LeftThumbProximal", "1.0") == "leftThumbMetacarpal"


def test_not_vrm_is_ng():
    r = cvm.check_gltf({"extensions": {}}, _BONES)
    assert r.spec_version == "unknown"
    assert not r.ok


def test_animate_bones_match_play_and_animate_avatar():
    paa = _load("play_and_animate_avatar")
    expected = set(paa._IDLE_POSE_BONES) | {b.bone for b in paa._IDLE_BONES} | {paa._NOD_BONE}
    assert set(cvm._ANIMATE_BONES) == expected


def test_required_bones_include_pose_bones():
    eap = _load("explore_avatar_poses")
    pose_bones = {b for pose in eap.POSES.values() for b in pose}
    assert pose_bones <= cvm.required_bones()


def _write_glb(path, gltf):
    body = json.dumps(gltf).encode("utf-8")
    body += b" " * (-len(body) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(body))
                     + struct.pack("<I4s", len(body), b"JSON") + body)


def test_main_exit_codes(tmp_path, capsys):
    ok = tmp_path / "ok.vrm"
    _write_glb(ok, _vrm0(_ALL0, bones=[cvm.unity_to_vrm_bone(b) for b in cvm.required_bones()]))
    assert cvm.main([str(ok)]) == 0
    assert "OK ✓" in capsys.readouterr().out

    ng = tmp_path / "ng.vrm"
    _write_glb(ng, _vrm0(_ALL0[:2]))
    assert cvm.main([str(ng), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert "Sorrow" in out["missing_required_expressions"]
