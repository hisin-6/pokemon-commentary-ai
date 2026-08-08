"""対戦状況の記録倉庫（データウェアハウスの「箱」・改善ロードマップ「戦況推論強化」続き）。

将来的に「似たような状況では有利/不利だった」という経験則判断を行うための土台として、
処理した試合ごとの状況スナップショット（戦況・タイプ相性ヒント・HP等）と最終的な勝敗を
`data/battle_situations.sqlite`に蓄積する。

⚠️ 2026-08-04時点では**記録のみ**で、類似状況検索・判断ロジックは実装しない
（現時点の蓄積試合数ではサンプル数が少なすぎて統計的に意味のある判断ができないため。
十分な件数が溜まってから読み出し側を実装する）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("data/battle_situations.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS situations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id          TEXT NOT NULL,
    event_time        REAL,
    turn              TEXT,
    event_type        TEXT,
    player_pokemon    TEXT,
    opponent_pokemon  TEXT,
    weather           TEXT,
    screens_player    TEXT,
    screens_opponent  TEXT,
    trick_room        INTEGER,
    tailwind_player   INTEGER,
    tailwind_opponent INTEGER,
    type_hint         TEXT,
    hp_player         TEXT,
    hp_opponent       TEXT,
    outcome           TEXT
);
CREATE INDEX IF NOT EXISTS idx_situations_match_id ON situations(match_id);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def record_situation(snapshot: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    """1件の状況スナップショットを追記する。

    snapshot は situations テーブルの列名に対応するキーのみ使う（無い列は NULL）。
    match_id は必須（動画ファイル名（拡張子なし）等、試合を一意に識別できる文字列を
    想定。render_dir名は同じ動画でも出力先フォルダ名を変えて再実行すると別値に
    なってしまうため2026-08-09より非推奨）。
    """
    if "match_id" not in snapshot or not snapshot["match_id"]:
        raise ValueError("snapshot には match_id が必須です")

    columns = [
        "match_id", "event_time", "turn", "event_type", "player_pokemon",
        "opponent_pokemon", "weather", "screens_player", "screens_opponent",
        "trick_room", "tailwind_player", "tailwind_opponent", "type_hint",
        "hp_player", "hp_opponent", "outcome",
    ]
    values = [snapshot.get(c) for c in columns]

    conn = _connect(db_path)
    try:
        placeholders = ", ".join("?" for _ in columns)
        conn.execute(
            f"INSERT INTO situations ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def clear_match(match_id: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """指定match_idの既存行を全て削除する（同じ動画の再実行に備えた事前クリア）。

    record_situationは追記のみのため、同じmatch_idでパス1を再実行すると新旧の
    スナップショットが同じmatch_idの下に混在してしまう（RenderSinkが同種の事故を
    「前回素材の自動クリア」で防いでいるのと同じ問題。2026-08-08発見: バグ修正の
    検証で同じ動画を3回実行したところ、本来5行のところ20行に膨れ、未修正・一部
    修正・全修正の3世代が同一match_idの下に混在していた）。
    呼び出し側（`Pipeline._generate_posthoc_commentary`）が、1試合分の記録を
    開始する前に一度だけ呼ぶ想定。

    ⚠️2026-08-09追記: match_idをrender_dir名にしていた頃は、同じ動画でも
    出力先フォルダ名（--render-out）を変えて再実行すると別match_id扱いとなり
    このクリアが効かず重複が残るケースがあった（動画ファイル名ベースに変更して
    解消。既存DBの過去分の重複は別途手動クリーンアップが必要）。

    Returns: 削除した行数。
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM situations WHERE match_id = ?", (match_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def backfill_outcome(match_id: str, outcome: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """指定match_idの全行にoutcome（"勝ち"/"負け"等）を後付けする。

    battle_end時点で初めて勝敗が確定するため、記録済みの全イベント行にまとめて反映する。
    Returns: 更新した行数。
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE situations SET outcome = ? WHERE match_id = ?", (outcome, match_id))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def count_situations(db_path: Path = DEFAULT_DB_PATH) -> int:
    """蓄積件数を返す（将来「十分溜まったか」を判断する目安用）。"""
    if not Path(db_path).exists():
        return 0
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM situations").fetchone()[0]
    finally:
        conn.close()
