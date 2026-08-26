"""選出前チームプレビュー（自分・相手それぞれ6匹の種族名のみ）の保存・読み込み。

背景（2026-08-24）: 「対戦準備中」画面には両陣営6匹の構築がスプライト（画像）で
表示されるが、テキスト名は一切無いためOCR+PokeClassifierの既存パイプラインでは
自動取得できない（実機2026-08-23_22-15-43のフレーム確認で判明）。スプライト認識の
新規実装はコスト（時間）がかかるとユーザー判断のため、ユーザーが画面を目視して
GUI（`scripts/team_preview_gui.py`）から手入力→ファイル保存→パス1が読み込んで
LLMプロンプトに注入する、という人力入力の運用にした。

持ち物・特性・技構成はこの画面には表示されない（本家VGCの選出画面と同様）ため
不明。種族名のみを扱う。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

TEAM_PREVIEW_FILENAME = "team_preview.json"

# 自分の構築プリセット（同じ構築を複数戦で使い回すことが多いため、GUIで名前を
# 付けて保存・再利用できるようにする）。ユーザー固有データのため.gitignore対象。
OWN_TEAM_PRESETS_PATH = Path("data/own_team_presets.json")


def save_team_preview(render_dir: Path | str, own_team: list[str],
                       opponent_team: list[str]) -> Path:
    """render_dir配下にteam_preview.jsonを書き込む（render_dirが無ければ作成）。

    戻り値: 書き込んだファイルのパス。
    """
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    path = render_dir / TEAM_PREVIEW_FILENAME
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "own_team": [n for n in own_team if n],
        "opponent_team": [n for n in opponent_team if n],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_team_preview(render_dir: Path | str) -> dict | None:
    """render_dir配下のteam_preview.jsonを読み込む。無ければNone。"""
    path = Path(render_dir) / TEAM_PREVIEW_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "own_team": list(data.get("own_team") or []),
        "opponent_team": list(data.get("opponent_team") or []),
    }


def format_team_preview_hint(data: dict) -> str:
    """team_preview辞書からプロンプト注入用の1行ヒントを組み立てる。
    どちらも空なら空文字を返す（呼び出し側で注入をスキップする判定に使う）。
    """
    own = data.get("own_team") or []
    opponent = data.get("opponent_team") or []
    parts = []
    if own:
        parts.append(f"自分の構築（選出前・種族のみ）: {' / '.join(own)}")
    if opponent:
        parts.append(f"相手の構築（選出前・種族のみ）: {' / '.join(opponent)}")
    return " ／ ".join(parts)


def load_own_team_presets(path: Path | str = OWN_TEAM_PRESETS_PATH) -> dict[str, list[str]]:
    """保存済みの自分の構築プリセット一覧を返す（{プリセット名: [種族名x6]}）。
    ファイルが無ければ空辞書。"""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    presets = data.get("presets") if isinstance(data, dict) else None
    if not isinstance(presets, dict):
        return {}
    return {str(name): list(team) for name, team in presets.items()}


def save_own_team_preset(name: str, team: list[str],
                          path: Path | str = OWN_TEAM_PRESETS_PATH) -> None:
    """自分の構築プリセットを1件追加・上書き保存する。"""
    path = Path(path)
    presets = load_own_team_presets(path)
    presets[name] = [n for n in team if n]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"presets": presets}, ensure_ascii=False, indent=2), encoding="utf-8")
