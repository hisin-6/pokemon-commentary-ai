"""選出前チームプレビュー（自分・相手それぞれ6匹の種族名）を手入力してファイル保存するGUI。

背景: `docs`/メモリ参照。「対戦準備中」画面は両陣営6匹の構築をスプライトで表示するが
テキスト名が無く、既存のOCR+PokeClassifierパイプラインでは自動取得できない
（2026-08-24フレーム確認で判明）。スプライト認識の新規実装はコストが高いと判断し、
ユーザーが画面を目視してこのGUIから手入力する運用にした。保存されたJSONは
パス1（`src/pipeline.py`）が`--render-out`の直下から自動で読み込み、実況プロンプトに
「自分の構築（選出前・種族のみ）」「相手の構築（選出前・種族のみ）」として注入する。

追加インストール不要（tkinterはPython標準ライブラリ）。

使い方:
    venv\\Scripts\\python.exe scripts\\team_preview_gui.py
    venv\\Scripts\\python.exe scripts\\team_preview_gui.py --render-dir renders\\2026-08-24_20-00-00

画面の使い方:
  - 保存先フォルダ: パス1で使う`--render-out`と同じフォルダを指定する（「参照」で
    フォルダ選択、または直接入力）。まだ存在しなくてもよい（保存時に自動作成）
  - 自分の構築: 6匹分のコンボボックス。入力するとチャンピオンズ収録ポケモンの中から
    前方一致・部分一致で候補を絞り込む。同じ構築を繰り返し使う場合は「プリセット」欄で
    名前を付けて保存し、次回以降はドロップダウンから選んで「プリセットを読込」で一括反映できる
  - 相手の構築: 6匹分のコンボボックス。試合ごとに「対戦準備中」画面を見ながら入力する
  - 「保存」ボタンで`<保存先フォルダ>/team_preview.json`に書き込む。空欄のまま保存すると
    そのポケモンは構築リストから除外される（6匹揃っていなくても保存できる・見切れて
    確認できなかった場合等を許容）
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pokedb.team_preview import (  # noqa: E402
    load_own_team_presets,
    save_own_team_preset,
    save_team_preview,
)

_DEFAULT_POKEDB_PATH = Path("data/pokedb.sqlite")


def load_champions_roster(db_path: Path = _DEFAULT_POKEDB_PATH) -> list[str]:
    """champions_pokemonに登録済みのポケモン日本語名を五十音順で返す。
    DBが無い/テーブルが無い場合は空リスト（GUI自体は自由入力可能なので動作は継続する）。
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT p.name_ja FROM pokemon p "
            "INNER JOIN champions_pokemon c ON c.pokemon_id = p.id "
            "ORDER BY p.name_ja"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


class _PokemonCombobox(ttk.Combobox):
    """入力に応じて候補を部分一致で絞り込むコンボボックス。"""

    def __init__(self, master, roster: list[str], **kwargs):
        super().__init__(master, **kwargs)
        self._roster = roster
        self["values"] = roster
        self.bind("<KeyRelease>", self._on_key_release)

    def _on_key_release(self, event) -> None:
        if event.keysym in ("Up", "Down", "Left", "Right", "Return", "Tab"):
            return
        text = self.get()
        if not text:
            self["values"] = self._roster
            return
        self["values"] = [n for n in self._roster if text in n]


class TeamPreviewApp:
    def __init__(self, root: tk.Tk, render_dir: str = ""):
        self.root = root
        root.title("選出前チームプレビュー入力")

        self._roster = load_champions_roster()
        if not self._roster:
            messagebox.showwarning(
                "PokeDB未検出",
                f"{_DEFAULT_POKEDB_PATH} が見つからないため候補の絞り込みは効きません"
                "（自由入力は可能）。scripts/build_pokedb.py の実行を確認してください。",
            )

        # ── 保存先フォルダ ──────────────────────────────────────────
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="保存先フォルダ（renders/<動画名>）:").pack(side="left")
        self.render_dir_var = tk.StringVar(value=render_dir)
        ttk.Entry(top, textvariable=self.render_dir_var, width=50).pack(
            side="left", padx=4, fill="x", expand=True)
        ttk.Button(top, text="参照", command=self._browse_render_dir).pack(side="left")

        # ── 自分/相手の構築 ─────────────────────────────────────────
        body = ttk.Frame(root, padding=8)
        body.pack(fill="both", expand=True)

        own_frame = ttk.LabelFrame(body, text="自分の構築（6匹）", padding=8)
        own_frame.grid(row=0, column=0, padx=8, sticky="n")
        self.own_vars = self._build_team_column(own_frame)

        preset_row = ttk.Frame(own_frame)
        preset_row.grid(row=7, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(
            preset_row, textvariable=self.preset_var, width=20,
            values=list(load_own_team_presets().keys()))
        self.preset_combo.pack(side="left", padx=(0, 4))
        ttk.Button(preset_row, text="読込", command=self._load_preset).pack(side="left")
        ttk.Button(preset_row, text="名前を付けて保存", command=self._save_preset).pack(
            side="left", padx=(4, 0))

        opp_frame = ttk.LabelFrame(body, text="相手の構築（6匹）", padding=8)
        opp_frame.grid(row=0, column=1, padx=8, sticky="n")
        self.opponent_vars = self._build_team_column(opp_frame)

        # ── 保存・ステータス ────────────────────────────────────────
        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="保存", command=self._save).pack(side="left")
        self.status_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=8)

    def _build_team_column(self, parent: ttk.LabelFrame) -> list[tk.StringVar]:
        vars_: list[tk.StringVar] = []
        for i in range(6):
            var = tk.StringVar()
            combo = _PokemonCombobox(parent, self._roster, textvariable=var, width=18)
            combo.grid(row=i, column=0, pady=2, sticky="w")
            vars_.append(var)
        return vars_

    def _browse_render_dir(self) -> None:
        path = filedialog.askdirectory(title="保存先フォルダを選択（新規フォルダ名も入力可）")
        if path:
            self.render_dir_var.set(path)

    def _load_preset(self) -> None:
        presets = load_own_team_presets()
        team = presets.get(self.preset_var.get())
        if not team:
            messagebox.showinfo("プリセット", "プリセットが選択されていません")
            return
        for i, var in enumerate(self.own_vars):
            var.set(team[i] if i < len(team) else "")

    def _save_preset(self) -> None:
        name = simpledialog.askstring("プリセット保存", "プリセット名を入力してください")
        if not name:
            return
        team = [v.get() for v in self.own_vars]
        save_own_team_preset(name, team)
        self.preset_combo["values"] = list(load_own_team_presets().keys())
        self.preset_var.set(name)
        self.status_var.set(f"プリセット「{name}」を保存しました")

    def _save(self) -> None:
        render_dir = self.render_dir_var.get().strip()
        if not render_dir:
            messagebox.showerror("エラー", "保存先フォルダを指定してください")
            return
        own = [v.get().strip() for v in self.own_vars]
        opponent = [v.get().strip() for v in self.opponent_vars]
        path = save_team_preview(render_dir, own, opponent)
        self.status_var.set(f"保存しました: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="選出前チームプレビュー入力GUI")
    parser.add_argument("--render-dir", default="",
                        help="保存先フォルダの初期値（renders/<動画名>）。省略可・後から入力欄で変更可")
    args = parser.parse_args()

    root = tk.Tk()
    TeamPreviewApp(root, render_dir=args.render_dir)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
