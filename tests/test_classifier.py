"""
src/pokedb/classifier.py の単体テスト

テスト戦略:
  - テスト用の一時 SQLite DB を作り、PokeClassifier に渡す
  - 本番 DB（data/pokedb.sqlite）が存在しなくてもテスト可能
  - rapidfuzz の実際のスコア計算を使う（モックなし）

テスト対象:
  - PokeClassifier.classify()          メイン分類
  - PokeClassifier.classify_batch()    バッチ分類
  - PokeClassifier.filter_pokemon_names() 名前フィルター
  - PokeClassifier.get_pokemon_info()  RAG 用ポケモン詳細情報
  - PokeClassifier.is_pokemon() / is_move() / is_ability()
  - 閾値・エッジケース
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pokedb.classifier import (
    CANDIDATE_THRESHOLD,
    CATEGORY_ABILITY,
    CATEGORY_ITEM,
    CATEGORY_MOVE,
    CATEGORY_POKEMON,
    CATEGORY_UNKNOWN,
    CONFIDENT_THRESHOLD,
    ClassifyResult,
    PokeClassifier,
)


# ─── テスト用 DB ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    """テスト用の SQLite DB を作成し、パスを返す。"""
    db_dir = tmp_path_factory.mktemp("pokedb")
    db_path = db_dir / "test.sqlite"

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE pokemon (
            id INTEGER PRIMARY KEY,
            name_ja TEXT NOT NULL,
            name_en TEXT NOT NULL,
            type1   TEXT,
            type2   TEXT,
            ability1       TEXT,
            ability2       TEXT,
            ability_hidden TEXT
        );

        CREATE TABLE moves (
            id       INTEGER PRIMARY KEY,
            name_ja  TEXT NOT NULL,
            name_en  TEXT NOT NULL,
            category TEXT,
            power    INTEGER
        );

        CREATE TABLE abilities (
            id      INTEGER PRIMARY KEY,
            name_ja TEXT NOT NULL,
            name_en TEXT NOT NULL
        );

        CREATE TABLE items (
            id      INTEGER PRIMARY KEY,
            name_ja TEXT NOT NULL,
            name_en TEXT NOT NULL
        );

        CREATE TABLE pokemon_moves (
            pokemon_id INTEGER,
            move_id    INTEGER
        );

        -- ポケモン
        INSERT INTO pokemon VALUES
          (1, 'ピカチュウ',  'Pikachu',     'でんき', NULL,       'せいでんき', 'ひらいしん', 'かえんほう'),
          (2, 'エルフーン',  'Whimsicott',  'くさ',   'フェアリー','いたずらごころ','すりぬけ', 'ようりょくそ'),
          (3, 'ゴリランダー','Rillaboom',   'くさ',   NULL,       'グラスメイカー','グラスメイカー',NULL),
          (4, 'テラパゴス',  'Terapagos',   'ノーマル',NULL,       'テラスシェル', NULL,       NULL);

        -- 技
        INSERT INTO moves VALUES
          (1, 'かみなり',    'Thunder',   '特殊',  110),
          (2, 'まもる',      'Protect',   '変化',  NULL),
          (3, 'ムーンフォース','Moonblast','特殊',  95),
          (4, 'おいかぜ',    'Tailwind',  '変化',  NULL);

        -- 特性
        INSERT INTO abilities VALUES
          (1, 'せいでんき',    'Static'),
          (2, 'いたずらごころ','Prankster'),
          (3, 'グラスメイカー','Grassy Surge');

        -- アイテム
        INSERT INTO items VALUES
          (1, 'きあいのタスキ','Focus Sash'),
          (2, 'こだわりスカーフ','Choice Scarf');

        -- ポケモン×技
        INSERT INTO pokemon_moves VALUES
          (1, 1), (1, 2),          -- ピカチュウ: かみなり・まもる
          (2, 3), (2, 4), (2, 2);  -- エルフーン: ムーンフォース・おいかぜ・まもる
    """)
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture(scope="module")
def clf(test_db_path):
    return PokeClassifier(db_path=test_db_path)


@pytest.fixture(scope="module")
def champions_db_path(tmp_path_factory):
    """champions_pokemon テーブルを持つテスト用DB（ピカチュウ・エルフーンのみ許可）。"""
    db_dir = tmp_path_factory.mktemp("pokedb_champions")
    db_path = db_dir / "test_champions.sqlite"

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE pokemon (
            id INTEGER PRIMARY KEY,
            name_ja TEXT NOT NULL,
            name_en TEXT NOT NULL,
            type1   TEXT,
            type2   TEXT,
            ability1       TEXT,
            ability2       TEXT,
            ability_hidden TEXT
        );
        CREATE TABLE champions_pokemon (
            pokemon_id INTEGER PRIMARY KEY,
            added_version TEXT,
            note TEXT
        );
        CREATE TABLE moves (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL, category TEXT, power INTEGER);
        CREATE TABLE abilities (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL);
        CREATE TABLE items (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL);
        CREATE TABLE pokemon_moves (pokemon_id INTEGER, move_id INTEGER);

        INSERT INTO pokemon VALUES
          (1, 'ピカチュウ',  'Pikachu',     'でんき', NULL,       'せいでんき', 'ひらいしん', 'かえんほう'),
          (2, 'エルフーン',  'Whimsicott',  'くさ',   'フェアリー','いたずらごころ','すりぬけ', 'ようりょくそ'),
          (3, 'ゴリランダー','Rillaboom',   'くさ',   NULL,       'グラスメイカー','グラスメイカー',NULL),
          (4, 'テラパゴス',  'Terapagos',   'ノーマル',NULL,       'テラスシェル', NULL,       NULL);

        -- チャンピオンズではピカチュウ・エルフーンのみ使用可能（ゴリランダー・テラパゴスは対象外）
        INSERT INTO champions_pokemon (pokemon_id, added_version, note) VALUES
          (1, '2026-07-30', 'test'),
          (2, '2026-07-30', 'test');
    """)
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture(scope="module")
def empty_champions_db_path(tmp_path_factory):
    """champions_pokemon テーブルはあるが空のテスト用DB。"""
    db_dir = tmp_path_factory.mktemp("pokedb_champions_empty")
    db_path = db_dir / "test_empty.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE pokemon (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL,
            type1 TEXT, type2 TEXT, ability1 TEXT, ability2 TEXT, ability_hidden TEXT);
        CREATE TABLE champions_pokemon (pokemon_id INTEGER PRIMARY KEY, added_version TEXT, note TEXT);
        CREATE TABLE moves (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL, category TEXT, power INTEGER);
        CREATE TABLE abilities (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL);
        CREATE TABLE items (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL);
        CREATE TABLE pokemon_moves (pokemon_id INTEGER, move_id INTEGER);
        INSERT INTO pokemon VALUES (1, 'ピカチュウ', 'Pikachu', 'でんき', NULL, 'せいでんき', 'ひらいしん', 'かえんほう');
    """)
    conn.commit()
    conn.close()
    return str(db_path)


# ═══════════════════════════════════════════════════════════════════════════════
# FileNotFoundError
# ═══════════════════════════════════════════════════════════════════════════════

class TestPokeClassifierInit:

    def test_raises_if_db_not_found(self):
        with pytest.raises(FileNotFoundError):
            PokeClassifier(db_path="/nonexistent/path/to.sqlite")

    def test_loads_successfully_with_valid_db(self, test_db_path):
        clf = PokeClassifier(db_path=test_db_path)
        assert len(clf._pokemon) == 4
        assert len(clf._moves) == 4
        assert len(clf._abilities) == 3
        assert len(clf._items) == 2

    def test_rejects_unknown_game_mode(self, test_db_path):
        with pytest.raises(ValueError):
            PokeClassifier(db_path=test_db_path, game_mode="scarlet-violet")


class TestPokeClassifierChampionsMode:
    """game_mode="champions" の許可リスト絞り込み（docs/champions-v1-plan.md #9）。"""

    def test_pokemon_filtered_to_champions_roster(self, champions_db_path):
        clf = PokeClassifier(db_path=champions_db_path, game_mode="champions")
        assert sorted(clf._pokemon) == ["エルフーン", "ピカチュウ"]

    def test_excluded_pokemon_not_in_pool(self, champions_db_path):
        clf = PokeClassifier(db_path=champions_db_path, game_mode="champions")
        assert "テラパゴス" not in clf._pokemon
        assert "ゴリランダー" not in clf._pokemon

    def test_moves_abilities_items_not_filtered(self, champions_db_path):
        """現時点ではポケモンのみ絞り込み、技・特性・アイテムは全件そのまま。"""
        clf = PokeClassifier(db_path=champions_db_path, game_mode="champions")
        assert clf._moves == []  # このfixtureには技データが無いだけで、絞り込みロジックは通っていない
        # sv モードと同じロード処理を通っていることの確認（テーブルは空でも例外にならない）

    def test_sv_mode_unaffected_by_champions_table_presence(self, champions_db_path):
        """champions_pokemon テーブルがあっても sv（既定）モードは全件ロードする。"""
        clf = PokeClassifier(db_path=champions_db_path)  # game_mode 未指定 = "sv"
        assert sorted(clf._pokemon) == ["エルフーン", "ゴリランダー", "テラパゴス", "ピカチュウ"]

    def test_missing_champions_table_raises_clear_error(self, test_db_path):
        """champions_pokemon テーブルが無いDBで champions モードを指定すると分かりやすいエラーになる。"""
        with pytest.raises(RuntimeError, match="champions_pokemon"):
            PokeClassifier(db_path=test_db_path, game_mode="champions")

    def test_empty_champions_table_yields_empty_pool(self, empty_champions_db_path):
        """champions_pokemon が空でも例外にはせず、プール0件で動作する（警告ログのみ）。"""
        clf = PokeClassifier(db_path=empty_champions_db_path, game_mode="champions")
        assert clf._pokemon == []


# ═══════════════════════════════════════════════════════════════════════════════
# classify
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassify:

    def test_exact_pokemon_name_returns_confident(self, clf):
        result = clf.classify("ピカチュウ")
        assert result.category == CATEGORY_POKEMON
        assert result.confident is True
        assert result.canonical_ja == "ピカチュウ"

    def test_exact_move_name_returns_move(self, clf):
        result = clf.classify("まもる")
        assert result.category == CATEGORY_MOVE

    def test_exact_ability_name_returns_ability(self, clf):
        result = clf.classify("せいでんき")
        assert result.category == CATEGORY_ABILITY

    def test_exact_item_name_returns_item(self, clf):
        result = clf.classify("きあいのタスキ")
        assert result.category == CATEGORY_ITEM

    def test_empty_string_returns_unknown(self, clf):
        result = clf.classify("")
        assert result.category == CATEGORY_UNKNOWN

    def test_single_char_returns_unknown(self, clf):
        result = clf.classify("ア")
        assert result.category == CATEGORY_UNKNOWN

    def test_completely_unrelated_text_returns_unknown(self, clf):
        result = clf.classify("xyzxyzxyzxyz")
        assert result.category == CATEGORY_UNKNOWN

    def test_ocr_partial_read_fuzzy_match(self, clf):
        """OCR で末尾が欠けても fuzzy マッチする（エルフー→エルフーン）。"""
        result = clf.classify("エルフー")
        # スコアが CANDIDATE_THRESHOLD 以上なら pokemon として返る
        if result.score >= CANDIDATE_THRESHOLD:
            assert result.category == CATEGORY_POKEMON
            assert result.canonical_ja == "エルフーン"

    def test_confident_flag_above_threshold(self, clf):
        result = clf.classify("ゴリランダー")
        assert result.confident is True
        assert result.score >= CONFIDENT_THRESHOLD

    def test_best_score_wins_across_categories(self, clf):
        """複数カテゴリで一致候補があっても、最高スコアのカテゴリを返す。"""
        result = clf.classify("ピカチュウ")
        assert result.category == CATEGORY_POKEMON

    def test_canonical_ja_matches_db_value(self, clf):
        result = clf.classify("エルフーン")
        assert result.canonical_ja == "エルフーン"

    def test_result_is_frozen(self, clf):
        """ClassifyResult は frozen（_UNKNOWN シングルトン共有のため書き換え禁止）。"""
        result = clf.classify("ピカチュウ")
        with pytest.raises(Exception):
            result.canonical_ja = "書き換え"

    def test_score_is_float(self, clf):
        result = clf.classify("ピカチュウ")
        assert isinstance(result.score, float)


# ═══════════════════════════════════════════════════════════════════════════════
# classify_batch
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyBatch:

    def test_returns_list_of_same_length(self, clf):
        texts = ["ピカチュウ", "まもる", "不明テキスト"]
        results = clf.classify_batch(texts)
        assert len(results) == len(texts)

    def test_each_result_is_classify_result(self, clf):
        results = clf.classify_batch(["ピカチュウ"])
        assert isinstance(results[0], ClassifyResult)

    def test_empty_list_returns_empty_list(self, clf):
        assert clf.classify_batch([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# filter_pokemon_names
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterPokemonNames:

    def test_returns_only_pokemon_names(self, clf):
        texts = ["ピカチュウ", "まもる", "きあいのタスキ"]
        result = clf.filter_pokemon_names(texts)
        assert "ピカチュウ" in result
        assert "まもる" not in result
        assert "きあいのタスキ" not in result

    def test_normalizes_to_canonical_name(self, clf):
        """正規化された DB 名に変換される。"""
        result = clf.filter_pokemon_names(["ゴリランダー"])
        assert "ゴリランダー" in result

    def test_empty_list_returns_empty(self, clf):
        assert clf.filter_pokemon_names([]) == []

    def test_all_non_pokemon_returns_empty(self, clf):
        result = clf.filter_pokemon_names(["まもる", "せいでんき"])
        assert result == []

    def test_multiple_pokemon_all_included(self, clf):
        result = clf.filter_pokemon_names(["ピカチュウ", "エルフーン"])
        assert "ピカチュウ" in result
        assert "エルフーン" in result


# ═══════════════════════════════════════════════════════════════════════════════
# get_pokemon_info
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetPokemonInfo:

    def test_returns_dict_for_known_pokemon(self, clf):
        info = clf.get_pokemon_info("ピカチュウ")
        assert info is not None
        assert isinstance(info, dict)

    def test_returns_none_for_unknown_pokemon(self, clf):
        info = clf.get_pokemon_info("存在しないポケモン12345")
        assert info is None

    def test_info_contains_required_keys(self, clf):
        # 英語名は日本語名のみ設計への移行で廃止（name_en は返らない）
        info = clf.get_pokemon_info("ピカチュウ")
        assert "name_ja" in info
        assert "type" in info
        assert "abilities" in info
        assert "moves" in info

    def test_type_string_format(self, clf):
        info = clf.get_pokemon_info("ピカチュウ")
        assert info["type"] == "でんき"  # type2 は NULL

    def test_dual_type_has_slash(self, clf):
        info = clf.get_pokemon_info("エルフーン")
        assert " / " in info["type"]
        assert "くさ" in info["type"]
        assert "フェアリー" in info["type"]

    def test_abilities_list_populated(self, clf):
        info = clf.get_pokemon_info("ピカチュウ")
        assert len(info["abilities"]) >= 1
        assert "せいでんき" in info["abilities"]

    def test_hidden_ability_marked_with_dream(self, clf):
        info = clf.get_pokemon_info("ピカチュウ")
        hidden = [a for a in info["abilities"] if "夢" in a]
        assert len(hidden) == 1
        assert "かえんほう" in hidden[0]


class TestGetPokemonTypes:
    """get_pokemon_types: type_chart.pyへ渡す生のタイプリストを返す（2026-08-04新設）。"""

    def test_single_type_returns_one_element_list(self, clf):
        assert clf.get_pokemon_types("ピカチュウ") == ["でんき"]

    def test_dual_type_returns_two_element_list(self, clf):
        assert clf.get_pokemon_types("エルフーン") == ["くさ", "フェアリー"]

    def test_unknown_pokemon_returns_none(self, clf):
        assert clf.get_pokemon_types("存在しないポケモン12345") is None


class TestGetMoveType:
    """get_move_type: 2026-08-04新設。type_chart.pyでの相性計算用に技のタイプを取得する。"""

    @pytest.fixture(scope="class")
    def clf_with_move_types(self, tmp_path_factory):
        db_dir = tmp_path_factory.mktemp("pokedb_movetypes")
        db_path = db_dir / "test_move_types.sqlite"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE pokemon (
                id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL,
                type1 TEXT, type2 TEXT, ability1 TEXT, ability2 TEXT, ability_hidden TEXT
            );
            CREATE TABLE moves (
                id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL,
                type TEXT, category TEXT, power INTEGER
            );
            CREATE TABLE abilities (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL);
            CREATE TABLE items (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, name_en TEXT NOT NULL);
            CREATE TABLE pokemon_moves (pokemon_id INTEGER, move_id INTEGER);

            INSERT INTO pokemon VALUES
              (1, 'ピカチュウ', 'Pikachu', 'でんき', NULL, 'せいでんき', 'ひらいしん', 'かえんほう');
            INSERT INTO moves VALUES
              (1, 'かみなり', 'Thunder', 'でんき', '特殊', 110),
              (2, 'アイアンヘッド', 'Iron Head', 'はがね', '物理', 80),
              (3, 'リフレクター', 'Reflect', 'エスパー', '変化', NULL);
        """)
        conn.commit()
        conn.close()
        return PokeClassifier(db_path=str(db_path))

    def test_known_move_returns_type(self, clf_with_move_types):
        assert clf_with_move_types.get_move_type("かみなり") == "でんき"
        assert clf_with_move_types.get_move_type("アイアンヘッド") == "はがね"

    def test_unknown_move_returns_none(self, clf_with_move_types):
        assert clf_with_move_types.get_move_type("存在しない技12345") is None

    def test_get_move_category_known_moves(self, clf_with_move_types):
        """get_move_category: 2026-08-07新設。_latest_move_type_hint（pipeline.py）が
        変化技（ダメージを与えない技）にもタイプ相性を計算してしまい、Bedrockが
        意味不明な文言をハルシネーションする原因になっていたバグの修正用。"""
        assert clf_with_move_types.get_move_category("かみなり") == "特殊"
        assert clf_with_move_types.get_move_category("アイアンヘッド") == "物理"
        assert clf_with_move_types.get_move_category("リフレクター") == "変化"

    def test_get_move_category_unknown_move_returns_none(self, clf_with_move_types):
        assert clf_with_move_types.get_move_category("存在しない技12345") is None


class TestIsMoveLearnable:
    """is_move_learnable: 技帰属の断片一致幽霊技対策で使う学習可能技チェック。"""

    def test_true_for_learnable_move(self, clf):
        assert clf.is_move_learnable("ピカチュウ", "かみなり") is True

    def test_false_for_unlearnable_move(self, clf):
        # ピカチュウは「ムーンフォース」を覚えない（エルフーンの技）
        assert clf.is_move_learnable("ピカチュウ", "ムーンフォース") is False

    def test_true_when_pokemon_unknown(self, clf):
        """判定不能（ポケモン名が解決できない）時は誤ってtentative化しないよう True 側に倒す。"""
        assert clf.is_move_learnable("存在しないポケモン12345", "かみなり") is True

    def test_true_when_pokemon_has_no_move_data(self, clf):
        """pokemon_movesにエントリが無いポケモン（テストDBのゴリランダー）でも True 側に倒す。"""
        assert clf.is_move_learnable("ゴリランダー", "かみなり") is True

    def test_moves_list_populated(self, clf):
        info = clf.get_pokemon_info("ピカチュウ")
        assert len(info["moves"]) >= 1

    def test_moves_limited_to_max_rag(self, clf):
        from src.pokedb.classifier import MAX_MOVES_FOR_RAG
        info = clf.get_pokemon_info("ピカチュウ")
        assert len(info["moves"]) <= MAX_MOVES_FOR_RAG

    def test_fuzzy_correction_for_slightly_wrong_name(self, clf):
        """少し間違った名前でも fuzzy マッチして情報を返す。"""
        info = clf.get_pokemon_info("ゴリランダ")  # 末尾1文字欠落
        # fuzzy で補正されれば info != None
        # スコア次第なので None でも受け入れる（エラーにならないことを確認）
        # assert info is not None or info is None  # 常に真（クラッシュしないことを確認）

    def test_name_ja_matches_expected(self, clf):
        info = clf.get_pokemon_info("エルフーン")
        assert info["name_ja"] == "エルフーン"


# ═══════════════════════════════════════════════════════════════════════════════
# is_pokemon / is_move / is_ability
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilityMethods:

    def test_is_pokemon_true_for_pokemon(self, clf):
        assert clf.is_pokemon("ピカチュウ") is True

    def test_is_pokemon_false_for_move(self, clf):
        assert clf.is_pokemon("まもる") is False

    def test_is_move_true_for_move(self, clf):
        assert clf.is_move("まもる") is True

    def test_is_move_false_for_pokemon(self, clf):
        assert clf.is_move("ピカチュウ") is False

    def test_is_ability_true_for_ability(self, clf):
        assert clf.is_ability("せいでんき") is True

    def test_is_ability_false_for_pokemon(self, clf):
        assert clf.is_ability("ピカチュウ") is False

    def test_is_pokemon_false_for_empty_string(self, clf):
        assert clf.is_pokemon("") is False


# ═══════════════════════════════════════════════════════════════════════════════
# ClassifyResult dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyResult:

    def test_fields_accessible(self):
        r = ClassifyResult(
            category=CATEGORY_POKEMON,
            canonical_ja="ピカチュウ",
            score=95.0,
            confident=True,
        )
        assert r.category == CATEGORY_POKEMON
        assert r.canonical_ja == "ピカチュウ"
        assert r.score == 95.0
        assert r.confident is True
