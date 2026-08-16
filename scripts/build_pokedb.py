#!/usr/bin/env python3
"""
PokeAPI から第9世代（SV）対応のポケモン・技・特性・アイテムデータを取得し
SQLite データベース（data/pokedb.sqlite）に保存する。

実行例:
    python scripts/build_pokedb.py

初回実行は API コール数が多いため 30〜60 分程度かかる場合があります。
レスポンスは data/pokeapi_cache/ にキャッシュされるため 2 回目以降は高速です。

依存:
    pip install requests
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import requests

# ─── 設定 ────────────────────────────────────────────────────────────────────

POKEAPI_BASE    = "https://pokeapi.co/api/v2"
DB_PATH         = Path("data/pokedb.sqlite")
CACHE_DIR       = Path("data/pokeapi_cache")
REQUEST_INTERVAL = 0.2   # API レート制限対策（秒）

# ID 1〜1025 = Gen1〜9 の全ポケモン（SV で登場しうる全種）
MAX_POKEMON_ID = 1025

# PokeAPI の version_group 名称（SV）
SV_VERSION_GROUPS = {"scarlet-violet"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── API ヘルパー ─────────────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    key = url.replace(POKEAPI_BASE + "/", "").rstrip("/")
    # Windows で使えない文字（? & = / \ : * " < > |）を _ に置換
    for ch in r'?&=/\:*"<>|':
        key = key.replace(ch, "_")
    return CACHE_DIR / (key + ".json")


def _fetch(url: str, retries: int = 3) -> dict:
    """キャッシュ付き GET。キャッシュがなければ API を叩く。"""
    cp = _cache_path(url)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))

    for attempt in range(retries):
        try:
            time.sleep(REQUEST_INTERVAL)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return {}
            resp.raise_for_status()
            data = resp.json()
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data
        except requests.RequestException as e:
            log.warning("リトライ %d/%d: %s — %s", attempt + 1, retries, url, e)
            time.sleep(2 ** attempt)

    log.error("取得失敗: %s", url)
    return {}


def _name(names: list[dict], *langs: str, fallback: str = "") -> str:
    """names リストから優先言語順に名前を返す。"""
    for lang in langs:
        for n in names:
            if n["language"]["name"] == lang:
                return n["name"]
    return fallback


def _id_from_url(url: str) -> int | None:
    try:
        return int(url.rstrip("/").rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        return None

# ─── 日本語名キャッシュ付きヘルパー ─────────────────────────────────────────

_type_ja_cache: dict[str, str] = {}

def _type_ja(url: str) -> str:
    if url not in _type_ja_cache:
        data = _fetch(url)
        _type_ja_cache[url] = _name(data.get("names", []), "ja", "ja-Hrkt", fallback=data.get("name", ""))
    return _type_ja_cache[url]


_ability_ja_cache: dict[str, str] = {}

def _ability_ja(url: str) -> str:
    if url not in _ability_ja_cache:
        data = _fetch(url)
        _ability_ja_cache[url] = _name(data.get("names", []), "ja", "ja-Hrkt", fallback=data.get("name", ""))
    return _ability_ja_cache[url]

# ─── DB 初期化 ────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS pokemon (
    id             INTEGER PRIMARY KEY,
    name_ja        TEXT NOT NULL,
    name_en        TEXT NOT NULL,
    name_zh        TEXT,
    type1          TEXT,
    type2          TEXT,
    ability1       TEXT,
    ability2       TEXT,
    ability_hidden TEXT,
    name_ko        TEXT
);

CREATE TABLE IF NOT EXISTS moves (
    id       INTEGER PRIMARY KEY,
    name_ja  TEXT NOT NULL,
    name_en  TEXT NOT NULL,
    type     TEXT,
    category TEXT,
    power    INTEGER,
    accuracy INTEGER,
    effect   TEXT,
    target   TEXT
);

CREATE TABLE IF NOT EXISTS abilities (
    id      INTEGER PRIMARY KEY,
    name_ja TEXT NOT NULL,
    name_en TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id      INTEGER PRIMARY KEY,
    name_ja TEXT NOT NULL,
    name_en TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pokemon_moves (
    pokemon_id INTEGER NOT NULL,
    move_id    INTEGER NOT NULL,
    PRIMARY KEY (pokemon_id, move_id)
);

-- ポケモンチャンピオンズで使用可能な pokemon_id の許可リスト（game_mode="champions" 用）。
-- pokemon テーブルは削除・変更せず共用し、このテーブルへの追加だけで対応する
-- （データ投入はローカル管理のスクリプトで行う）。
CREATE TABLE IF NOT EXISTS champions_pokemon (
    pokemon_id     INTEGER PRIMARY KEY,
    added_version  TEXT,   -- 取り込み日（例: "2026-07-30"）。ゲーム内バージョン文字列は非公開のため代用
    note           TEXT,   -- データ取得元・補足
    FOREIGN KEY (pokemon_id) REFERENCES pokemon(id)
);

-- 検索用インデックス（rapidfuzz のスコアリング前にある程度絞り込む用）
CREATE INDEX IF NOT EXISTS idx_pokemon_name_ja ON pokemon(name_ja);
CREATE INDEX IF NOT EXISTS idx_pokemon_name_en ON pokemon(name_en);
CREATE INDEX IF NOT EXISTS idx_moves_name_ja   ON moves(name_ja);
CREATE INDEX IF NOT EXISTS idx_moves_name_en   ON moves(name_en);
CREATE INDEX IF NOT EXISTS idx_abilities_ja    ON abilities(name_ja);
CREATE INDEX IF NOT EXISTS idx_items_name_ja   ON items(name_ja);
"""

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()

    # 既存 DB へのマイグレーション（name_ko カラムがない古い DB に対応）
    # ※ executescript の後に実行することで、インデックス作成前にカラムを確保する
    try:
        conn.execute("ALTER TABLE pokemon ADD COLUMN name_ko TEXT")
        conn.commit()
        log.info("マイグレーション: pokemon.name_ko カラムを追加しました")
    except sqlite3.OperationalError:
        pass  # カラムが既に存在する場合は無視

    # name_ko のインデックスはカラム追加後に作成
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pokemon_name_ko ON pokemon(name_ko)")
    conn.commit()

    # 既存 DB へのマイグレーション（effect カラムがない古い DB に対応・2026-08-14
    # 技効果ヒントRAG新設。技効果誤認パターン対策）
    try:
        conn.execute("ALTER TABLE moves ADD COLUMN effect TEXT")
        conn.commit()
        log.info("マイグレーション: moves.effect カラムを追加しました")
    except sqlite3.OperationalError:
        pass  # カラムが既に存在する場合は無視

    # 既存 DB へのマイグレーション（target カラムがない古い DB に対応・2026-08-16
    # 技の対象範囲ヒント新設。move_target_hintが事後観測ベースのみで、変化技/範囲技の
    # 事前情報が無く対象を誤爆する問題への対策）
    try:
        conn.execute("ALTER TABLE moves ADD COLUMN target TEXT")
        conn.commit()
        log.info("マイグレーション: moves.target カラムを追加しました")
    except sqlite3.OperationalError:
        pass  # カラムが既に存在する場合は無視

    log.info("DB スキーマ初期化完了")

# ─── 技データ取得 ─────────────────────────────────────────────────────────────

DAMAGE_CLASS_JA = {"physical": "物理", "special": "特殊", "status": "変化"}

# 技効果テキスト（flavor_text）のバージョングループ優先度。新しい世代ほど
# 説明文が整理されている傾向があるため先頭を優先し、無ければ後方にフォールバックする
_EFFECT_VERSION_GROUP_PRIORITY = [
    "sword-shield", "ultra-sun-ultra-moon", "sun-moon",
    "omega-ruby-alpha-sapphire", "x-y", "lets-go-pikachu-lets-go-eevee",
]
# SWSHのflavor_textの一部（bide/barrier等、SWSHで技マシン等から削除された技）に
# 「この技は使えません」という説明文自体のダミーテキストが混入する
# （実測: sword-shield 826件中147件が該当）ため除外する
_EFFECT_PLACEHOLDER_MARKERS = ("この技は", "使えません")


def _pick_move_effect_text(flavor_text_entries: list[dict]) -> str | None:
    """flavor_text_entries からバージョングループ優先度順に日本語の効果説明文を選ぶ。"""
    by_vg: dict[str, str] = {}
    for e in flavor_text_entries:
        if e.get("language", {}).get("name") != "ja":
            continue
        txt = e.get("flavor_text", "").replace("　", "").replace("\n", "")
        if not txt or any(marker in txt for marker in _EFFECT_PLACEHOLDER_MARKERS):
            continue
        by_vg[e.get("version_group", {}).get("name", "")] = txt

    for vg in _EFFECT_VERSION_GROUP_PRIORITY:
        if vg in by_vg:
            return by_vg[vg]
    return next(iter(by_vg.values()), None)


# PokeAPI の move.target.name → 日本語の対象範囲ラベル（2026-08-16新設）。
# ダブルバトルでのmove_target_hint（技の対象誤認対策）が事後観測ベースのみで、
# 変化技（自分対象）・範囲技（相手複数対象）の事前情報を一切持っていなかったため、
# 「そもそも対象がどういう種類の技か」をPython側の確定事実として持たせる目的で追加。
_TARGET_JA = {
    "selected-pokemon":          "相手単体",
    "selected-pokemon-me-first": "相手単体",
    "random-opponent":           "相手のランダム1体",
    "all-opponents":             "相手全体",
    "all-other-pokemon":         "自分以外全員",
    "all-pokemon":               "場の全員",
    "user":                      "自分自身",
    "user-and-allies":           "自分と味方",
    "user-or-ally":              "自分か味方",
    "ally":                      "味方1体",
    "all-allies":                "味方全体",
    "users-field":               "自分の場",
    "opponents-field":           "相手の場",
    "entire-field":              "場全体",
    "specific-move":             "直前の技に関連",
    "fainting-pokemon":          "瀕死になった味方",
}


def fetch_moves(conn: sqlite3.Connection) -> None:
    log.info("=== 技データ取得開始 ===")
    data = _fetch(f"{POKEAPI_BASE}/move?limit=2000")
    entries = data.get("results", [])
    count = 0

    for i, entry in enumerate(entries):
        move_id = _id_from_url(entry["url"])
        if not move_id:
            continue
        if conn.execute("SELECT 1 FROM moves WHERE id=?", (move_id,)).fetchone():
            continue

        m = _fetch(entry["url"])
        if not m:
            continue

        name_ja = _name(m.get("names", []), "ja", "ja-Hrkt")
        name_en = _name(m.get("names", []), "en")
        if not name_ja or not name_en:
            continue

        type_ja  = _type_ja(m["type"]["url"]) if m.get("type") else None
        category = DAMAGE_CLASS_JA.get(m.get("damage_class", {}).get("name", ""), None) if m.get("damage_class") else None

        conn.execute(
            "INSERT OR REPLACE INTO moves VALUES (?,?,?,?,?,?,?)",
            (move_id, name_ja, name_en, type_ja, category, m.get("power"), m.get("accuracy")),
        )
        count += 1

        if i % 100 == 0:
            conn.commit()
            log.info("  技: %d / %d 処理済み", i, len(entries))

    conn.commit()
    log.info("技データ完了: %d 件", conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0])


def fetch_move_effects(conn: sqlite3.Connection) -> None:
    """既存 moves 全行に、PokeAPI キャッシュ済みの flavor_text から効果テキストを
    後付け補完する（2026-08-14・技効果ヒントRAG新設）。

    fetch_moves() は `SELECT 1 FROM moves WHERE id=?` で既存行を丸ごとスキップする
    ため、旧DBの919件には新設した effect 列が入らない。このバックフィル専用関数で
    全件を対象に埋める。data/pokeapi_cache/move_*.json が既に取得済みのため、
    通常は新規ネットワークアクセスは発生しない（_fetch() のキャッシュ層に乗る）。
    """
    log.info("=== 技効果データ backfill 開始 ===")
    rows = conn.execute("SELECT id FROM moves").fetchall()
    updated = 0
    for i, (move_id,) in enumerate(rows):
        m = _fetch(f"{POKEAPI_BASE}/move/{move_id}/")
        effect = _pick_move_effect_text(m.get("flavor_text_entries", [])) if m else None
        if effect:
            conn.execute("UPDATE moves SET effect = ? WHERE id = ?", (effect, move_id))
            updated += 1
        if i % 100 == 0:
            conn.commit()
            log.info("  技効果: %d / %d 処理済み", i, len(rows))

    conn.commit()
    log.info("技効果データ backfill 完了: %d / %d 件", updated, len(rows))


def fetch_move_targets(conn: sqlite3.Connection) -> None:
    """既存 moves 全行に、PokeAPI キャッシュ済みの target フィールドから
    対象範囲（自分自身/相手単体/相手全体等）を後付け補完する（2026-08-16新設）。

    move_target_hint（技の対象誤認対策）が「技の直後に画面で観測された結果」
    からの事後推測のみで、変化技（自分対象）・範囲技（相手複数対象）の事前情報を
    一切持っていなかった。つるぎのまい等の自分対象技で対象不明のまま実況したり、
    観測が無い単体対象技の対象をLLMが憶測で決め打ちする事故があったための対策。
    fetch_move_effects() と同じくキャッシュ済みJSONを使うため通常は新規
    ネットワークアクセスは発生しない。
    """
    log.info("=== 技対象範囲データ backfill 開始 ===")
    rows = conn.execute("SELECT id FROM moves").fetchall()
    updated = 0
    for i, (move_id,) in enumerate(rows):
        m = _fetch(f"{POKEAPI_BASE}/move/{move_id}/")
        target_en = (m or {}).get("target", {}).get("name")
        target_ja = _TARGET_JA.get(target_en)
        if target_ja:
            conn.execute("UPDATE moves SET target = ? WHERE id = ?", (target_ja, move_id))
            updated += 1
        if i % 100 == 0:
            conn.commit()
            log.info("  技対象範囲: %d / %d 処理済み", i, len(rows))

    conn.commit()
    log.info("技対象範囲データ backfill 完了: %d / %d 件", updated, len(rows))


# ─── 特性データ取得 ───────────────────────────────────────────────────────────

def fetch_abilities(conn: sqlite3.Connection) -> None:
    log.info("=== 特性データ取得開始 ===")
    data = _fetch(f"{POKEAPI_BASE}/ability?limit=500")
    entries = data.get("results", [])

    for i, entry in enumerate(entries):
        ab_id = _id_from_url(entry["url"])
        if not ab_id:
            continue
        if conn.execute("SELECT 1 FROM abilities WHERE id=?", (ab_id,)).fetchone():
            continue

        a = _fetch(entry["url"])
        if not a:
            continue

        name_ja = _name(a.get("names", []), "ja", "ja-Hrkt")
        name_en = _name(a.get("names", []), "en")
        if not name_ja or not name_en:
            continue

        conn.execute(
            "INSERT OR REPLACE INTO abilities VALUES (?,?,?)",
            (ab_id, name_ja, name_en),
        )

        if i % 50 == 0:
            conn.commit()

    conn.commit()
    log.info("特性データ完了: %d 件", conn.execute("SELECT COUNT(*) FROM abilities").fetchone()[0])


# ─── アイテムデータ取得 ───────────────────────────────────────────────────────

def fetch_items(conn: sqlite3.Connection) -> None:
    log.info("=== アイテムデータ取得開始 ===")
    data = _fetch(f"{POKEAPI_BASE}/item?limit=2000")
    entries = data.get("results", [])

    for i, entry in enumerate(entries):
        item_id = _id_from_url(entry["url"])
        if not item_id:
            continue
        if conn.execute("SELECT 1 FROM items WHERE id=?", (item_id,)).fetchone():
            continue

        it = _fetch(entry["url"])
        if not it:
            continue

        name_ja = _name(it.get("names", []), "ja", "ja-Hrkt")
        name_en = _name(it.get("names", []), "en")
        if not name_ja or not name_en:
            continue

        conn.execute(
            "INSERT OR REPLACE INTO items VALUES (?,?,?)",
            (item_id, name_ja, name_en),
        )

        if i % 100 == 0:
            conn.commit()
            log.info("  アイテム: %d / %d 処理済み", i, len(entries))

    conn.commit()
    log.info("アイテムデータ完了: %d 件", conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])


# ─── ポケモンデータ取得 ────────────────────────────────────────────────────────

def fetch_pokemon(conn: sqlite3.Connection) -> None:
    log.info("=== ポケモンデータ取得開始（ID 1〜%d） ===", MAX_POKEMON_ID)

    for pokemon_id in range(1, MAX_POKEMON_ID + 1):
        if conn.execute("SELECT 1 FROM pokemon WHERE id=?", (pokemon_id,)).fetchone():
            continue

        # species: 多言語名
        species = _fetch(f"{POKEAPI_BASE}/pokemon-species/{pokemon_id}/")
        if not species:
            continue

        name_ja = _name(species.get("names", []), "ja-Hrkt", "ja")
        name_en = _name(species.get("names", []), "en")
        name_zh = _name(species.get("names", []), "zh-Hans", "zh-Hant")
        name_ko = _name(species.get("names", []), "ko")

        if not name_ja or not name_en:
            log.warning("ID %d: 名前取得不能 → スキップ", pokemon_id)
            continue

        # pokemon: タイプ・特性・技
        poke = _fetch(f"{POKEAPI_BASE}/pokemon/{pokemon_id}/")
        if not poke:
            continue

        types = poke.get("types", [])
        type1 = _type_ja(types[0]["type"]["url"]) if len(types) > 0 else None
        type2 = _type_ja(types[1]["type"]["url"]) if len(types) > 1 else None

        ability1 = ability2 = ability_hidden = None
        for ab in poke.get("abilities", []):
            ja_name = _ability_ja(ab["ability"]["url"])
            if ab["is_hidden"]:
                ability_hidden = ja_name
            elif ability1 is None:
                ability1 = ja_name
            else:
                ability2 = ja_name

        conn.execute(
            """INSERT OR REPLACE INTO pokemon
               (id, name_ja, name_en, name_zh, type1, type2, ability1, ability2, ability_hidden, name_ko)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (pokemon_id, name_ja, name_en, name_zh or None,
             type1, type2, ability1, ability2, ability_hidden, name_ko or None),
        )

        # pokemon_moves: 覚えられる技をすべて登録
        for m_entry in poke.get("moves", []):
            move_id = _id_from_url(m_entry["move"]["url"])
            if move_id:
                conn.execute(
                    "INSERT OR IGNORE INTO pokemon_moves VALUES (?,?)",
                    (pokemon_id, move_id),
                )

        if pokemon_id % 50 == 0:
            conn.commit()
            log.info("  ポケモン: %d / %d 完了", pokemon_id, MAX_POKEMON_ID)

    conn.commit()
    log.info("ポケモンデータ完了: %d 件", conn.execute("SELECT COUNT(*) FROM pokemon").fetchone()[0])


# ─── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("DB パス: %s", DB_PATH.resolve())
    log.info("キャッシュ: %s", CACHE_DIR.resolve())

    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)

        # 技・特性・アイテムを先に取得（ポケモン取得時に参照するため）
        fetch_moves(conn)
        fetch_move_effects(conn)
        fetch_move_targets(conn)
        fetch_abilities(conn)
        fetch_items(conn)
        fetch_pokemon(conn)

        log.info("=== 全完了 ===")
        for table in ("pokemon", "moves", "abilities", "items", "pokemon_moves"):
            cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            log.info("  %-20s: %d 件", table, cnt)
        log.info("DB保存先: %s", DB_PATH.resolve())

    finally:
        conn.close()


if __name__ == "__main__":
    main()
