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
    canonical_en: str    # 英語名
    score:        float  # 類似スコア（0〜100）
    confident:    bool   # score >= CONFIDENT_THRESHOLD


_UNKNOWN = ClassifyResult(
    category=CATEGORY_UNKNOWN,
    canonical_ja="",
    canonical_en="",
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

        # カテゴリ別に (canonical_ja, canonical_en) のリストをメモリに展開
        # rapidfuzz は日本語名リストに対して検索する
        self._pokemon:    list[tuple[str, str]] = []
        self._moves:      list[tuple[str, str]] = []
        self._abilities:  list[tuple[str, str]] = []
        self._items:      list[tuple[str, str]] = []

        # 韓国語名 → 日本語名 の対応リスト（ハングルOCR用）
        # (name_ko, name_ja) のタプルリスト
        self._pokemon_ko: list[tuple[str, str]] = []

        # pokemon_id → row のキャッシュ（RAG 用）
        self._pokemon_rows: dict[str, dict] = {}

        self._load()
        log.info(
            "PokeClassifier 初期化完了: pokemon=%d (ko=%d), moves=%d, abilities=%d, items=%d",
            len(self._pokemon), len(self._pokemon_ko),
            len(self._moves), len(self._abilities), len(self._items),
        )

    # ── ロード ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            try:
                rows = conn.execute(
                    "SELECT id, name_ja, name_en, name_ko, type1, type2, ability1, ability2, ability_hidden FROM pokemon"
                ).fetchall()
                has_ko = True
            except sqlite3.OperationalError:
                # 古い DB（name_ko カラムなし）→ 韓国語対応スキップ
                log.warning("name_ko カラムが見つかりません。scripts/build_pokedb.py を再実行してください。")
                rows = conn.execute(
                    "SELECT id, name_ja, name_en, type1, type2, ability1, ability2, ability_hidden FROM pokemon"
                ).fetchall()
                has_ko = False

            for row in rows:
                self._pokemon.append((row["name_ja"], row["name_en"]))
                self._pokemon_rows[row["name_ja"]] = dict(row)
                if has_ko and row["name_ko"]:
                    self._pokemon_ko.append((row["name_ko"], row["name_ja"]))

            for row in conn.execute("SELECT name_ja, name_en FROM moves"):
                self._moves.append((row["name_ja"], row["name_en"]))

            for row in conn.execute("SELECT name_ja, name_en FROM abilities"):
                self._abilities.append((row["name_ja"], row["name_en"]))

            for row in conn.execute("SELECT name_ja, name_en FROM items"):
                self._items.append((row["name_ja"], row["name_en"]))
        finally:
            conn.close()

    # ── ハングル検出 ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_hangul(text: str) -> bool:
        """文字列にハングル（AC00–D7A3）が1文字以上含まれるか判定する。"""
        return any("\uAC00" <= ch <= "\uD7A3" for ch in text)

    @staticmethod
    def _is_pure_hangul(text: str) -> bool:
        """スペースを除いた全文字がハングルかどうかを判定する。
        韓国語ポケモン名は純粋なハングルのみで構成される（例: 뽀록나, 테라파고스）。
        数字・英字・記号が混じるOCRノイズを除外するために使用する。
        """
        stripped = text.replace(" ", "")
        return len(stripped) >= 2 and all("\uAC00" <= ch <= "\uD7A3" for ch in stripped)

    # ── メイン分類 ───────────────────────────────────────────────────────────

    def classify(self, text: str) -> ClassifyResult:
        """
        テキストを最もスコアが高いカテゴリに分類して返す。

        ハングルを含む場合は韓国語ポケモン名リストで優先マッチングする。
        スコアが CANDIDATE_THRESHOLD 未満の場合は UNKNOWN を返す。
        """
        if not text or len(text) < 2:
            return _UNKNOWN

        # 純粋なハングル（数字・記号混じりのOCRノイズを除外）の場合は韓国語ポケモン名で先にマッチング
        if self._is_pure_hangul(text) and self._pokemon_ko:
            ko_result = self._best_match_ko(text)
            if ko_result and ko_result.score >= CANDIDATE_THRESHOLD:
                log.debug("韓国語ポケモン名マッチ: %s → %s (%.1f)", text, ko_result.canonical_ja, ko_result.score)
                return ko_result
            return _UNKNOWN  # ハングル文字がポケモン名以外にマッチしないよう抑制

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
                    canonical_en=result.canonical_en,
                    score=result.score,
                    confident=result.score >= CONFIDENT_THRESHOLD,
                )

        if best.score < CANDIDATE_THRESHOLD:
            return _UNKNOWN

        return best

    def _best_match(
        self, text: str, entries: list[tuple[str, str]]
    ) -> ClassifyResult | None:
        """rapidfuzz で最も近いエントリを返す（内部用）。
        日本語名・英語名の両方を検索し、スコアの高い方を返す。
        """
        if not entries:
            return None

        ja_names = [e[0] for e in entries]
        en_names = [e[1] for e in entries]

        # WRatio: partial_ratio と token_sort_ratio を組み合わせた汎用スコアラー
        # 日本語の短い名前には ratio も有効だが、OCR 短縮（エルフー→エルフーン）に WRatio が有効
        ja_match = process.extractOne(text, ja_names, scorer=fuzz.WRatio)
        en_match = process.extractOne(text, en_names, scorer=fuzz.WRatio) if any(en_names) else None

        # 日本語・英語のうちスコアが高い方を採用
        best_match = None
        best_score = 0.0
        matched_ja = matched_en = ""

        if ja_match and ja_match[1] > best_score:
            matched_ja, best_score, idx = ja_match
            matched_en = entries[idx][1]
            best_match = (matched_ja, best_score, idx, "ja")

        if en_match and en_match[1] > best_score:
            matched_en_name, en_score, en_idx = en_match
            matched_ja = entries[en_idx][0]
            matched_en = matched_en_name
            best_score = en_score
            best_match = (matched_ja, best_score, en_idx, "en")

        if not best_match:
            return None

        return ClassifyResult(
            category=CATEGORY_UNKNOWN,   # 呼び出し元で上書き
            canonical_ja=matched_ja,
            canonical_en=matched_en,
            score=float(best_score),
            confident=best_score >= CONFIDENT_THRESHOLD,
        )

    def _best_match_ko(self, text: str) -> ClassifyResult | None:
        """韓国語ポケモン名リストに対して rapidfuzz マッチングし、日本語名を返す（内部用）。"""
        if not self._pokemon_ko:
            return None

        ko_names = [e[0] for e in self._pokemon_ko]
        match = process.extractOne(text, ko_names, scorer=fuzz.WRatio)
        if not match:
            return None

        matched_ko, score, idx = match
        canonical_ja = self._pokemon_ko[idx][1]
        # canonical_en は pokemon_rows から引く
        row = self._pokemon_rows.get(canonical_ja)
        canonical_en = row["name_en"] if row else ""

        return ClassifyResult(
            category=CATEGORY_POKEMON,
            canonical_ja=canonical_ja,
            canonical_en=canonical_en,
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
                "name_en": "Whimsicott",
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
            "name_en":   row["name_en"],
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
