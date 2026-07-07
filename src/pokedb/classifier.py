"""
PokeDB テキスト分類モジュール（Sprint 6）
------------------------------------------
OCR で読み取ったテキストを rapidfuzz で fuzzy マッチングし、
「ポケモン名 / 技名 / 特性名 / アイテム名 / 不明」に分類する。

主な用途:
  - pipeline.py での name_candidates フィルタリング・正規化
  - server.py での RAG（ポケモン情報のプロンプト付与）

使用例:
    clf = PokeClassifier()
    r = clf.classify("エルフー")
    # → ClassifyResult(category="pokemon", canonical_ja="エルフーン", score=90.9, confident=True)

    info = clf.get_pokemon_info("エルフーン")
    # → {"name_ja": "エルフーン", "type": "くさ / フェアリー", "abilities": [...], "moves": [...]}
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

# OCR誤読: 大カナ→小カナ（ユ→ュ, ヨ→ョ等）と濁点混同（ブ→プ等）の正規化テーブル
# ア行は変換しない（独立した母音として正常に使われるため）
_OCR_LARGE_TO_SMALL = str.maketrans("ヤユヨワツ", "ャュョヮッ")
# 濁点除去（ブ→プ, ズ→ス等）- 短い名前でスコアが閾値を下回る場合の補助バリアント
_DAKUTEN_REMOVE = str.maketrans(
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボヴ",
    "カキクケコサシスセソタチツテトパピプペポウ",
)

# ─── 閾値 ────────────────────────────────────────────────────────────────────

CONFIDENT_THRESHOLD = 90   # これ以上: 確定採用
CANDIDATE_THRESHOLD = 75   # これ以上: 低信頼度候補として採用
# 75 未満: 不明扱い（除外）

# カテゴリ定数
CATEGORY_POKEMON = "pokemon"
CATEGORY_MOVE    = "move"
CATEGORY_ABILITY = "ability"
CATEGORY_ITEM    = "item"
CATEGORY_UNKNOWN = "unknown"

# RAG 用: ポケモンごとに取得する代表技の最大数
MAX_MOVES_FOR_RAG = 12


# ─── 結果型 ──────────────────────────────────────────────────────────────────

@dataclass
class ClassifyResult:
    """classify() の返り値。"""
    category:     str    # pokemon / move / ability / item / unknown
    canonical_ja: str    # DB に登録された正規の日本語名
    score:        float  # 類似スコア（0〜100）
    confident:    bool   # score >= CONFIDENT_THRESHOLD


_UNKNOWN = ClassifyResult(
    category=CATEGORY_UNKNOWN,
    canonical_ja="",
    score=0.0,
    confident=False,
)


# ─── 分類器 ──────────────────────────────────────────────────────────────────

class PokeClassifier:
    """
    SQLite PokeDB を読み込み、OCR テキストをカテゴリ分類するクラス。

    起動時に全レコードをメモリに展開するため、初期化は一度だけ行うこと。
    """

    def __init__(self, db_path: str | Path = "data/pokedb.sqlite"):
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"PokeDB が見つかりません: {self._db_path}\n"
                "先に scripts/build_pokedb.py を実行してください。"
            )

        # カテゴリ別に日本語名のリストをメモリに展開
        self._pokemon:    list[str] = []
        self._moves:      list[str] = []
        self._abilities:  list[str] = []
        self._items:      list[str] = []

        # pokemon_name_ja → row のキャッシュ（RAG 用）
        self._pokemon_rows: dict[str, dict] = {}

        self._load()
        log.info(
            "PokeClassifier 初期化完了: pokemon=%d, moves=%d, abilities=%d, items=%d",
            len(self._pokemon),
            len(self._moves), len(self._abilities), len(self._items),
        )

    # ── ロード ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, name_ja, type1, type2, ability1, ability2, ability_hidden FROM pokemon"
            ).fetchall()
            for row in rows:
                self._pokemon.append(row["name_ja"])
                self._pokemon_rows[row["name_ja"]] = dict(row)

            for row in conn.execute("SELECT name_ja FROM moves"):
                self._moves.append(row["name_ja"])

            for row in conn.execute("SELECT name_ja FROM abilities"):
                self._abilities.append(row["name_ja"])

            for row in conn.execute("SELECT name_ja FROM items"):
                self._items.append(row["name_ja"])
        finally:
            conn.close()

    # ── メイン分類 ───────────────────────────────────────────────────────────

    def classify(self, text: str) -> ClassifyResult:
        """
        テキストを最もスコアが高いカテゴリに分類して返す。
        スコアが CANDIDATE_THRESHOLD 未満の場合は UNKNOWN を返す。
        """
        if not text or len(text) < 2:
            return _UNKNOWN

        best = _UNKNOWN

        for category, entries in (
            (CATEGORY_POKEMON, self._pokemon),
            (CATEGORY_MOVE,    self._moves),
            (CATEGORY_ABILITY, self._abilities),
            (CATEGORY_ITEM,    self._items),
        ):
            result = self._best_match(text, entries)
            if result and result.score > best.score:
                best = ClassifyResult(
                    category=category,
                    canonical_ja=result.canonical_ja,
                    score=result.score,
                    confident=result.score >= CONFIDENT_THRESHOLD,
                )

        if best.score < CANDIDATE_THRESHOLD:
            return _UNKNOWN

        # 短断片ゲート: 3文字以下の入力はWRatioのpartial一致が実質無意味で、
        # OCRノイズ断片が実在名に化ける（実測: ラン→トランセル90・オニ→オニスズメ90・
        # スピ→スピアー90・ゲン→ケンタロス90）。正読・正規化バリアント一致なら
        # 100が出るため、95未満は棄却する（プデラ→プテラ100等は影響なし）。
        if len(text) <= 3 and best.score < 95:
            return _UNKNOWN

        return best

    def _best_match(
        self, text: str, entries: list[str]
    ) -> ClassifyResult | None:
        """rapidfuzz で最も近いエントリを返す（内部用）。"""
        if not entries:
            return None

        # OCR正規化バリアント: 大カナ→小カナ / 濁点除去 を試して最高スコアを採用
        variants = {text}
        normalized = text.translate(_OCR_LARGE_TO_SMALL)
        if normalized != text:
            variants.add(normalized)
        dakuten_removed = text.translate(_DAKUTEN_REMOVE)
        if dakuten_removed != text:
            variants.add(dakuten_removed)

        # WRatio: partial_ratio と token_sort_ratio を組み合わせた汎用スコアラー
        match = max(
            (process.extractOne(v, entries, scorer=fuzz.WRatio) for v in variants),
            key=lambda m: m[1] if m else 0,
        )
        if not match:
            return None

        matched_ja, score, _ = match
        return ClassifyResult(
            category=CATEGORY_UNKNOWN,   # 呼び出し元で上書き
            canonical_ja=matched_ja,
            score=float(score),
            confident=score >= CONFIDENT_THRESHOLD,
        )

    # ── バッチ分類 ───────────────────────────────────────────────────────────

    def classify_batch(self, texts: list[str]) -> list[ClassifyResult]:
        """複数テキストをまとめて分類する。"""
        return [self.classify(t) for t in texts]

    def filter_pokemon_names(self, texts: list[str]) -> list[str]:
        """
        テキストリストからポケモン名（確定 or 候補）だけを返す。
        canonical_ja（正規化済み名前）に変換して返す。
        """
        result = []
        for text in texts:
            r = self.classify(text)
            if r.category == CATEGORY_POKEMON:
                result.append(r.canonical_ja)
        return result

    # ── RAG 用: ポケモン詳細情報 ─────────────────────────────────────────────

    def get_pokemon_info(self, name_ja: str) -> dict | None:
        """
        ポケモン名（日本語）からタイプ・特性・代表技を取得して返す。
        server.py の Bedrock プロンプト構築に使用する。

        Returns:
            {
                "name_ja": "エルフーン",
                "type": "くさ / フェアリー",
                "abilities": ["いたずらごころ", "すりぬけ", "ようりょくそ（夢）"],
                "moves": ["ムーンフォース", "まもる", "おいかぜ", ...],
            }
            見つからない場合は None。
        """
        row = self._pokemon_rows.get(name_ja)
        if not row:
            # 完全一致しない場合は fuzzy で補正してから再試行
            r = self.classify(name_ja)
            if r.category == CATEGORY_POKEMON and r.confident:
                row = self._pokemon_rows.get(r.canonical_ja)
            if not row:
                return None

        # タイプ
        types = [t for t in (row["type1"], row["type2"]) if t]
        type_str = " / ".join(types) if types else "不明"

        # 特性
        abilities: list[str] = []
        for ab in (row["ability1"], row["ability2"]):
            if ab:
                abilities.append(ab)
        if row["ability_hidden"]:
            abilities.append(f"{row['ability_hidden']}（夢）")

        # 代表技（DB から取得）
        moves = self._get_moves_for_pokemon(row["id"])

        return {
            "name_ja":   row["name_ja"],
            "type":      type_str,
            "abilities": abilities,
            "moves":     moves,
        }

    def _get_moves_for_pokemon(self, pokemon_id: int) -> list[str]:
        """pokemon_moves テーブルから代表技リストを取得する。"""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT m.name_ja, m.category, m.power
                FROM pokemon_moves pm
                JOIN moves m ON pm.move_id = m.id
                WHERE pm.pokemon_id = ?
                ORDER BY
                    CASE m.category WHEN '物理' THEN 1 WHEN '特殊' THEN 2 ELSE 3 END,
                    COALESCE(m.power, 0) DESC
                LIMIT ?
                """,
                (pokemon_id, MAX_MOVES_FOR_RAG),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    # ── ユーティリティ ────────────────────────────────────────────────────────

    def is_pokemon(self, text: str) -> bool:
        r = self.classify(text)
        return r.category == CATEGORY_POKEMON

    def is_move(self, text: str) -> bool:
        r = self.classify(text)
        return r.category == CATEGORY_MOVE

    def is_ability(self, text: str) -> bool:
        r = self.classify(text)
        return r.category == CATEGORY_ABILITY
