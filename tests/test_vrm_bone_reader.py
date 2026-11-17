"""
scripts/vrm_bone_reader.py（VRMのノード階層・Humanoidボーン対応表の最小パーサー）の単体テスト

テスト対象:
  - _read_glb_json       GLBヘッダ・先頭JSONチャンクの解釈（パディング・BINチャンク付き含む）
  - parse_vrm            VRM 0.x / 1.0 の humanBones → ノード番号対応・ノードの既定値
  - node_world_position  親子ノードの平行移動の積算
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclassの型解決にsys.modules登録が要る
    spec.loader.exec_module(mod)
    return mod


vbr = _load("vrm_bone_reader")


def _glb_bytes(gltf, *, pad=False, bin_chunk=b""):
    body = json.dumps(gltf, ensure_ascii=False).encode("utf-8")
    if pad:  # glTF仕様どおりJSONチャンクを4バイト境界まで空白で埋める
        body += b" " * (-len(body) % 4)
    chunks = struct.pack("<I4s", len(body), b"JSON") + body
    if bin_chunk:
        chunks += struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _write(tmp_path, gltf, **kw):
    p = tmp_path / "model.vrm"
    p.write_bytes(_glb_bytes(gltf, **kw))
    return p


# 3段の親子: root(0) → hips(1) → spine(2)、hipsのもう一人の子に leftUpperLeg(3)
_NODES = [
    {"name": "Root", "children": [1]},
    {"name": "J_Bip_C_Hips", "children": [2, 3], "translation": [0.0, 0.9, 0.0]},
    {"name": "J_Bip_C_Spine", "translation": [0.0, 0.1, 0.02]},
    {"name": "J_Bip_L_UpperLeg", "translation": [0.08, -0.05, 0.0]},
]


# ── _read_glb_json ────────────────────────────────────────


class TestReadGlbJson:
    def test_reads_json_chunk(self, tmp_path):
        p = _write(tmp_path, {"asset": {"version": "2.0"}, "nodes": []})
        assert vbr._read_glb_json(p) == {"asset": {"version": "2.0"}, "nodes": []}

    def test_padding_and_bin_chunk_are_ignored(self, tmp_path):
        p = _write(tmp_path, {"nodes": [{"name": "くれぴ"}]}, pad=True, bin_chunk=b"\x01\x02\x03\x04")
        assert vbr._read_glb_json(p) == {"nodes": [{"name": "くれぴ"}]}

    def test_bad_magic_raises(self, tmp_path):
        p = tmp_path / "x.vrm"
        p.write_bytes(b"NOPE" + _glb_bytes({})[4:])
        with pytest.raises(ValueError, match="シグネチャ"):
            vbr._read_glb_json(p)

    def test_first_chunk_not_json_raises(self, tmp_path):
        p = tmp_path / "x.vrm"
        p.write_bytes(struct.pack("<4sII", b"glTF", 2, 24) + struct.pack("<I4s", 4, b"BIN\x00") + b"\0" * 4)
        with pytest.raises(ValueError, match="JSON"):
            vbr._read_glb_json(p)


# ── parse_vrm ────────────────────────────────────────────


class TestParseVrm:
    def test_vrm0_human_bones(self, tmp_path):
        gltf = {"nodes": _NODES, "extensions": {"VRM": {"humanoid": {"humanBones": [
            {"bone": "hips", "node": 1}, {"bone": "spine", "node": 2}, {"bone": "leftUpperLeg", "node": 3},
        ]}}}}
        m = vbr.parse_vrm(_write(tmp_path, gltf))
        assert m.humanoid_bone_to_node == {"hips": 1, "spine": 2, "leftUpperLeg": 3}
        assert [n.name for n in m.nodes] == ["Root", "J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_L_UpperLeg"]

    def test_vrm1_human_bones(self, tmp_path):
        gltf = {"nodes": _NODES, "extensions": {"VRMC_vrm": {"humanoid": {"humanBones": {
            "hips": {"node": 1}, "spine": {"node": 2},
        }}}}}
        m = vbr.parse_vrm(_write(tmp_path, gltf))
        assert m.humanoid_bone_to_node == {"hips": 1, "spine": 2}

    def test_node_defaults(self, tmp_path):
        m = vbr.parse_vrm(_write(tmp_path, {"nodes": [{}]}))
        node = m.nodes[0]
        assert node.name == ""
        assert node.children == []
        assert node.translation == (0.0, 0.0, 0.0)
        assert isinstance(node.translation, tuple)

    def test_plain_gltf_without_vrm_extension(self, tmp_path):
        m = vbr.parse_vrm(_write(tmp_path, {"nodes": _NODES}))
        assert m.humanoid_bone_to_node == {}
        assert len(m.nodes) == 4

    def test_extension_without_humanoid_is_skipped(self, tmp_path):
        gltf = {"nodes": _NODES, "extensions": {"VRM": {"meta": {"title": "x"}}}}
        assert vbr.parse_vrm(_write(tmp_path, gltf)).humanoid_bone_to_node == {}

    def test_no_nodes_key(self, tmp_path):
        m = vbr.parse_vrm(_write(tmp_path, {"asset": {"version": "2.0"}}))
        assert m.nodes == []


# ── node_world_position ───────────────────────────────────


def _model():
    nodes = [vbr.GltfNode(name=n["name"], children=n.get("children", []),
                          translation=tuple(n.get("translation", [0.0, 0.0, 0.0]))) for n in _NODES]
    return vbr.VrmModel(nodes=nodes)


class TestNodeWorldPosition:
    def test_root_is_its_own_translation(self):
        assert vbr.node_world_position(_model(), 0) == (0.0, 0.0, 0.0)

    def test_accumulates_parent_chain(self):
        assert vbr.node_world_position(_model(), 2) == pytest.approx((0.0, 1.0, 0.02))

    def test_sibling_uses_only_own_chain(self):
        # spine(2) の translation は leftUpperLeg(3) に混ざらない
        assert vbr.node_world_position(_model(), 3) == pytest.approx((0.08, 0.85, 0.0))

    def test_child_listed_before_parent(self):
        # 親ノードが配列の後ろにあっても親子関係はchildrenから引く
        m = vbr.VrmModel(nodes=[
            vbr.GltfNode(name="child", children=[], translation=(1.0, 0.0, 0.0)),
            vbr.GltfNode(name="parent", children=[0], translation=(0.0, 2.0, 0.0)),
        ])
        assert vbr.node_world_position(m, 0) == pytest.approx((1.0, 2.0, 0.0))
