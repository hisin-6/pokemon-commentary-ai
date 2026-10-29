"""VRMモデル差し替え前の一発チェック（表情ブレンドシェイプ・Humanoidボーン・メタ情報）。

外注VRM（ADR-006・VRM 0.x納品予定）が届いたら、VMCに読み込む前にこれを流して
`play_and_animate_avatar.py` / `explore_avatar_poses.py` が送るブレンドシェイプ名と
ボーン名がモデル側に揃っているかを確認する。GPU・VMC・OSC通信は一切使わない
（JSONチャンクを読むだけなのでWSLからでも実行可）。

使い方:
    python scripts/check_vrm_model.py C:\\path\\to\\model.vrm
    python scripts/check_vrm_model.py model.vrm --json   # 機械可読出力

終了コード: 必須項目が全部揃っていれば0、欠けがあれば1。

チェック内容:
  - VRMバージョン（0.x想定。1.0なら警告）
  - 必須表情: 実況で送る Neutral/Joy/Sorrow/Fun/Angry と、口パク・まばたき用の
    A/I/U/E/O/Blink（後者は現状パイプライン未使用なので「推奨」扱い）
  - 表情グループにバインドが1つも無い（名前だけあって顔が動かない）ものを警告
  - 必須ボーン: ポーズ・待機揺れ・うなずき・腕下げで実際に回すボーン
  - メタ情報（タイトル・作者・ライセンス系）の表示
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vrm_bone_reader import _read_glb_json  # noqa: E402

# play_and_animate_avatar.send_expression が /VMC/Ext/Blend/Val で送る名前
REQUIRED_EXPRESSIONS = ("Neutral", "Joy", "Sorrow", "Fun", "Angry")
# 現状パイプラインでは送っていないが、口パク・まばたきを足す時に要るもの
RECOMMENDED_EXPRESSIONS = ("A", "I", "U", "E", "O", "Blink")

# VRM 1.0 のプリセット名（VRM 0.x名 → 1.0名）
_VRM1_PRESET_ALIASES = {
    "Joy": "happy", "Sorrow": "sad", "Fun": "relaxed", "Angry": "angry",
    "Neutral": "neutral", "A": "aa", "I": "ih", "U": "ou", "E": "ee", "O": "oh",
    "Blink": "blink",
}

# play_and_animate_avatar.py 側で回すボーン（_IDLE_POSE_BONES・_IDLE_BONES・_NOD_BONE）。
# test_check_vrm_model.py でこの定数と実装側の一致を検証している。
_ANIMATE_BONES = ("LeftUpperArm", "RightUpperArm", "Spine", "Chest", "Neck")


def _pose_bones() -> set[str]:
    """explore_avatar_poses.POSES が触るボーン名。pythonoscが無い環境
    （WSL等）でも読めるよう、無ければダミーを差し込んでからimportする。"""
    try:
        import pythonosc.udp_client  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("pythonosc")
        stub_udp = types.ModuleType("pythonosc.udp_client")
        stub_udp.SimpleUDPClient = object
        stub.udp_client = stub_udp
        sys.modules.setdefault("pythonosc", stub)
        sys.modules.setdefault("pythonosc.udp_client", stub_udp)
    import explore_avatar_poses as eap
    return {bone for pose in eap.POSES.values() for bone in pose}


def required_bones() -> set[str]:
    return set(_ANIMATE_BONES) | _pose_bones()


# VRM 1.0は親指の名前が1段ずれている（Unity/VRM 0.xのProximal＝1.0のMetacarpal）
_VRM1_THUMB_RENAME = {"ThumbProximal": "ThumbMetacarpal", "ThumbIntermediate": "ThumbProximal"}


def unity_to_vrm_bone(name: str, spec: str = "0.x") -> str:
    """VMC Protocolが使うUnity HumanBodyBones名（LeftUpperArm）を
    VRM humanoidのボーン名（leftUpperArm）へ変換する。"""
    if spec == "1.0":
        for old, new in _VRM1_THUMB_RENAME.items():
            if name.endswith(old):
                name = name[:-len(old)] + new
                break
    return name[:1].lower() + name[1:]


@dataclass
class VrmReport:
    spec_version: str  # "0.x" / "1.0" / "unknown"
    expressions: dict[str, int] = field(default_factory=dict)  # 表情名 → バインド数
    humanoid_bones: set[str] = field(default_factory=set)
    meta: dict[str, str] = field(default_factory=dict)
    missing_required_expressions: list[str] = field(default_factory=list)
    missing_recommended_expressions: list[str] = field(default_factory=list)
    empty_expressions: list[str] = field(default_factory=list)
    missing_bones: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required_expressions and not self.missing_bones


def extract_expressions(gltf: dict) -> tuple[str, dict[str, int]]:
    """(バージョン, {表情名: バインド数}) を返す。VRM 0.xはname・presetNameの
    両方を登録する（VMCはどちらでも引ける）。"""
    ext = gltf.get("extensions", {})
    exprs: dict[str, int] = {}
    if "VRM" in ext:
        for g in ext["VRM"].get("blendShapeMaster", {}).get("blendShapeGroups", []):
            n_binds = len(g.get("binds", [])) + len(g.get("materialValues", []))
            for key in (g.get("name"), g.get("presetName")):
                if key and key != "unknown":
                    exprs[key] = n_binds
        return "0.x", exprs
    if "VRMC_vrm" in ext:
        e = ext["VRMC_vrm"].get("expressions", {})
        for group in ("preset", "custom"):
            for key, val in e.get(group, {}).items():
                exprs[key] = (len(val.get("morphTargetBinds", []))
                              + len(val.get("materialColorBinds", []))
                              + len(val.get("textureTransformBinds", [])))
        return "1.0", exprs
    return "unknown", exprs


def extract_humanoid_bones(gltf: dict) -> set[str]:
    ext = gltf.get("extensions", {})
    if "VRM" in ext:
        return {b["bone"] for b in ext["VRM"].get("humanoid", {}).get("humanBones", [])}
    if "VRMC_vrm" in ext:
        return set(ext["VRMC_vrm"].get("humanoid", {}).get("humanBones", {}))
    return set()


def extract_meta(gltf: dict) -> dict[str, str]:
    ext = gltf.get("extensions", {})
    if "VRM" in ext:
        m = ext["VRM"].get("meta", {})
        keys = ("title", "version", "author", "licenseName", "commercialUssageName",
                "allowedUserName", "otherPermissionUrl")
    elif "VRMC_vrm" in ext:
        m = dict(ext["VRMC_vrm"].get("meta", {}))
        if isinstance(m.get("authors"), list):
            m["authors"] = ", ".join(m["authors"])
        keys = ("name", "version", "authors", "licenseUrl", "commercialUsage",
                "avatarPermission", "otherLicenseUrl")
    else:
        return {}
    return {k: str(m[k]) for k in keys if m.get(k) not in (None, "")}


def _find_expression(name: str, spec: str, exprs: dict[str, int]) -> str | None:
    candidates = [name, name.lower()]
    if spec == "1.0" and name in _VRM1_PRESET_ALIASES:
        candidates.insert(0, _VRM1_PRESET_ALIASES[name])
    lowered = {k.lower(): k for k in exprs}
    for c in candidates:
        if c in exprs:
            return c
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def check_gltf(gltf: dict, bones_needed: set[str] | None = None) -> VrmReport:
    spec, exprs = extract_expressions(gltf)
    report = VrmReport(spec_version=spec, expressions=exprs,
                       humanoid_bones=extract_humanoid_bones(gltf), meta=extract_meta(gltf))
    for name in REQUIRED_EXPRESSIONS + RECOMMENDED_EXPRESSIONS:
        found = _find_expression(name, spec, exprs)
        if found is None and name == "Neutral" and spec == "1.0":
            continue  # VRM 1.0ではneutralプリセット自体が廃止されている
        if found is None:
            target = (report.missing_required_expressions if name in REQUIRED_EXPRESSIONS
                      else report.missing_recommended_expressions)
            target.append(name)
        elif exprs[found] == 0 and name != "Neutral":  # Neutralは空でも正常
            report.empty_expressions.append(name)
    if bones_needed is None:
        bones_needed = required_bones()
    report.missing_bones = sorted(b for b in bones_needed
                                  if unity_to_vrm_bone(b, spec) not in report.humanoid_bones)
    return report


def format_report(path: Path, r: VrmReport) -> str:
    lines = [f"=== {path.name} ===", f"VRMバージョン: {r.spec_version}"]
    if r.spec_version == "1.0":
        lines.append("  ⚠ VRM 1.0です（ADR-006は0.x想定・VMC 0.60で読めるか要実機確認）")
    elif r.spec_version == "unknown":
        lines.append("  ✗ VRM拡張が見つかりません（ただのglTF？）")
    if r.meta:
        lines.append("[メタ情報]")
        lines += [f"  {k}: {v}" for k, v in r.meta.items()]
    # VRM 0.xはname/presetName両方を登録しているので大文字小文字違いをまとめて表示
    names = sorted({k.lower(): k for k in sorted(r.expressions, reverse=True)}.values())
    lines.append(f"[表情] {len(names)}個: {', '.join(names)}")
    lines.append("  必須(Neutral/Joy/Sorrow/Fun/Angry): "
                 + ("✓ 全部あり" if not r.missing_required_expressions
                    else "✗ 不足 " + ", ".join(r.missing_required_expressions)))
    lines.append("  推奨(A/I/U/E/O/Blink): "
                 + ("✓ 全部あり" if not r.missing_recommended_expressions
                    else "⚠ 不足 " + ", ".join(r.missing_recommended_expressions)))
    if r.empty_expressions:
        lines.append("  ⚠ 名前はあるが中身が空（顔が動かない）: " + ", ".join(r.empty_expressions))
    lines.append(f"[Humanoidボーン] {len(r.humanoid_bones)}本")
    lines.append("  アニメーションで回すボーン: "
                 + ("✓ 全部あり" if not r.missing_bones
                    else "✗ 不足 " + ", ".join(r.missing_bones)))
    lines.append("結果: " + ("OK ✓" if r.ok else "NG ✗（必須項目が不足）"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vrm", type=Path)
    ap.add_argument("--json", action="store_true", help="結果をJSONで出力")
    args = ap.parse_args(argv)

    report = check_gltf(_read_glb_json(args.vrm))
    if args.json:
        d = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(report).items()}
        d["ok"] = report.ok
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        print(format_report(args.vrm, report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
