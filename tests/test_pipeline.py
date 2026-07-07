"""
pipeline.py の純粋ロジック単体テスト

テスト対象:
  - _clean_commentary()       実況文クリーンアップ
  - _is_battle_screen()       バトル画面判定
  - _ocr_results_to_text()    OCR テキスト変換
  - _extract_structured_info() HP・名前候補抽出
  - BattlePhaseClassifier     フェーズ分類・イベント検知
  - BattleStateTracker        戦況蓄積・コンテキスト生成
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# プロジェクトルートを sys.path に追加（pytest がルートから実行されない場合の保険）
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# conftest.py で重いモジュールがモック済みなのでここで安全にインポートできる
from src.pipeline import (
    _clean_commentary,
    _extract_structured_info,
    _is_battle_screen,
    _ocr_results_to_text,
    BattlePhaseClassifier,
    BattleStateTracker,
    FieldPokemon,
    _PLAYER_Y_THRESHOLD,
    _COMMAND_Y_MIN,
)


# ─── OCR 結果ヘルパー ────────────────────────────────────────────────────────

def _ocr(text: str, confidence: float = 0.9, y_center: float = 300.0) -> dict:
    """テスト用の OCR 結果辞書を作る。bbox は y_center を使う簡略形式。"""
    x = 100.0
    half_h = 15.0
    return {
        "text": text,
        "confidence": confidence,
        "bbox": [
            [x, y_center - half_h],
            [x + 100, y_center - half_h],
            [x + 100, y_center + half_h],
            [x, y_center + half_h],
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# _clean_commentary
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanCommentary:

    def test_removes_after_triple_dash(self):
        text = "ピカチュウがかみなりを放つ！---追加情報"
        assert "---" not in _clean_commentary(text)
        assert "ピカチュウがかみなりを放つ" in _clean_commentary(text)

    def test_removes_after_bracket(self):
        text = "素晴らしい一撃だ！【画面分析】何か"
        result = _clean_commentary(text)
        assert "【" not in result
        assert "素晴らしい" in result

    def test_removes_lines_with_shiji(self):
        text = "激しいバトルが続く。\n指示: 続けなさい\n余分な行"
        result = _clean_commentary(text)
        assert "指示" not in result

    def test_removes_lines_with_shitsumon(self):
        text = "技が命中した。\n質問: これは何？\n別の内容"
        result = _clean_commentary(text)
        assert "質問" not in result

    def test_removes_leading_bullet(self):
        text = "- ピカチュウが技を使った！"
        result = _clean_commentary(text)
        assert not result.startswith("-")
        assert "ピカチュウ" in result

    def test_removes_leading_dot_bullet(self):
        text = "・ エルフーンがおいかぜを使った！"
        result = _clean_commentary(text)
        assert not result.startswith("・")

    def test_removes_kakko(self):
        text = "「ピカチュウ」が倒れた！"
        result = _clean_commentary(text)
        assert "「" not in result
        assert "」" not in result

    def test_keeps_only_first_two_sentences(self):
        text = "一文目。二文目。三文目。四文目。"
        result = _clean_commentary(text)
        # 2文まで（。で区切り）
        assert result.count("。") <= 2

    def test_empty_string(self):
        assert _clean_commentary("") == ""

    def test_plain_text_unchanged(self):
        text = "エルフーンがおいかぜを使った！ゴリランダーに追い風が吹く！"
        result = _clean_commentary(text)
        assert "エルフーン" in result


# ═══════════════════════════════════════════════════════════════════════════════
# _is_battle_screen
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsBattleScreen:

    def test_empty_results_returns_true(self):
        assert _is_battle_screen([]) is True

    def test_battle_keyword_present_returns_true(self):
        results = [_ocr("HP")]
        assert _is_battle_screen(results) is True

    def test_non_battle_keyword_returns_false(self):
        results = [_ocr("バトルスタジアム")]
        assert _is_battle_screen(results) is False

    def test_offline_keyword_returns_false(self):
        results = [_ocr("オフライン")]
        assert _is_battle_screen(results) is False

    def test_normal_battle_text_returns_true(self):
        results = [_ocr("ピカチュウ"), _ocr("たたかう")]
        assert _is_battle_screen(results) is True

    def test_union_circle_returns_false(self):
        results = [_ocr("ユニオンサークル")]
        assert _is_battle_screen(results) is False


# ═══════════════════════════════════════════════════════════════════════════════
# _ocr_results_to_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestOcrResultsToText:

    def test_empty_results(self):
        assert _ocr_results_to_text([]) == "（テキスト未検出）"

    def test_low_confidence_filtered(self):
        results = [_ocr("ピカチュウ", confidence=0.3)]
        assert _ocr_results_to_text(results) == "（テキスト未検出）"

    def test_high_confidence_included(self):
        results = [_ocr("ピカチュウ", confidence=0.9)]
        assert "ピカチュウ" in _ocr_results_to_text(results)

    def test_multiple_texts_joined_with_slash(self):
        results = [_ocr("ピカチュウ"), _ocr("エルフーン")]
        result = _ocr_results_to_text(results)
        assert " / " in result

    def test_max_chars_truncated(self):
        long_text = "あ" * 200
        results = [_ocr(long_text)]
        result = _ocr_results_to_text(results)
        assert len(result) <= 120  # OCR_MAX_CHARS

    def test_threshold_boundary_exactly_04(self):
        # confidence == 0.4 → 含まれる（< 0.4 が除外条件）
        results = [_ocr("ピカチュウ", confidence=0.4)]
        assert "ピカチュウ" in _ocr_results_to_text(results)

    def test_with_classifier_excludes_move(self):
        """classifier が渡された場合、技名は除外される。"""
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "move"
        mock_clf.classify.return_value = mock_result

        results = [_ocr("まもる")]
        text = _ocr_results_to_text(results, classifier=mock_clf)
        assert "まもる" not in text

    def test_with_classifier_keeps_battle_message(self):
        """助詞を含むテキスト（バトルメッセージ）は技名でも除外しない。"""
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "move"
        mock_clf.classify.return_value = mock_result

        results = [_ocr("ピカチュウのかみなりを使った！")]
        text = _ocr_results_to_text(results, classifier=mock_clf)
        # 「を」を含むので除外されない
        assert "ピカチュウのかみなりを使った" in text


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_structured_info
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractStructuredInfo:

    def _opponent_ocr(self, text, conf=0.9):
        """相手エリア（y < _PLAYER_Y_THRESHOLD）の OCR 結果。"""
        return _ocr(text, conf, y_center=200.0)

    def _player_ocr(self, text, conf=0.9):
        """自分エリア（_PLAYER_Y_THRESHOLD <= y < _COMMAND_Y_MIN）の OCR 結果。"""
        return _ocr(text, conf, y_center=600.0)

    def _command_ocr(self, text, conf=0.9):
        """コマンドメニューエリア（y >= _COMMAND_Y_MIN）の OCR 結果。"""
        return _ocr(text, conf, y_center=800.0)

    def test_hp_extracted(self):
        results = [_ocr("176/176")]
        info = _extract_structured_info(results)
        assert "176/176" in info["hp_values"]

    def test_hp_low_denom_excluded_as_pp(self):
        """分母 < 50 は PP 値として除外する。"""
        results = [_ocr("8/8")]
        info = _extract_structured_info(results)
        assert len(info["hp_values"]) == 0

    def test_hp_assigned_to_player_side_by_y(self):
        results = [self._player_ocr("150/200")]
        info = _extract_structured_info(results)
        assert "150/200" in info["hp_values_player"]
        assert len(info["hp_values_opponent"]) == 0

    def test_hp_assigned_to_opponent_side_by_y(self):
        results = [self._opponent_ocr("100/200")]
        info = _extract_structured_info(results)
        assert "100/200" in info["hp_values_opponent"]
        assert len(info["hp_values_player"]) == 0

    def test_command_menu_items_excluded(self):
        results = [self._command_ocr("まもる")]
        info = _extract_structured_info(results)
        assert "まもる" not in info["name_candidates_player"]
        assert "まもる" not in info["name_candidates_opponent"]

    def test_ui_words_excluded(self):
        results = [self._player_ocr("たたかう")]
        info = _extract_structured_info(results)
        assert "たたかう" not in info["name_candidates_player"]

    def test_lv_prefix_excluded(self):
        results = [self._player_ocr("Lv50")]
        info = _extract_structured_info(results)
        assert "Lv50" not in info["name_candidates_player"]

    def test_text_ending_no_excluded(self):
        results = [self._player_ocr("ピカチュウの")]
        info = _extract_structured_info(results)
        assert "ピカチュウの" not in info["name_candidates_player"]

    def test_status_panel_skips_name_collection(self):
        """「戦闘中」テキストがある場合、名前候補収集をスキップする。"""
        results = [
            _ocr("戦闘中"),
            self._player_ocr("ピカチュウ"),
        ]
        info = _extract_structured_info(results)
        assert len(info["name_candidates_player"]) == 0

    def test_with_classifier_pokemon_included(self):
        """PokeClassifier がポケモン名と判定したテキストは名前候補に含まれる。"""
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "pokemon"
        mock_result.canonical_ja = "ピカチュウ"
        mock_clf.classify.return_value = mock_result

        results = [self._player_ocr("ピカチュウ")]
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "ピカチュウ" in info["name_candidates_player"]

    def test_with_classifier_non_pokemon_excluded(self):
        """PokeClassifier が技名と判定したテキストは除外される。"""
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "move"
        mock_clf.classify.return_value = mock_result

        results = [self._player_ocr("まもる")]
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "まもる" not in info["name_candidates_player"]

    def test_max_5_candidates_per_side(self):
        """各サイドの候補は最大 5 件に制限される。"""
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "pokemon"
        mock_result.canonical_ja = "テスト"
        mock_clf.classify.return_value = mock_result

        results = [self._player_ocr(f"ポケモン{i}") for i in range(10)]
        info = _extract_structured_info(results, classifier=mock_clf)
        assert len(info["name_candidates_player"]) <= 5

    def test_max_2_hp_per_side(self):
        """ダブルバトル: 各サイドの HP は最大 2 件に制限される。"""
        results = [
            self._player_ocr("100/200"),
            self._player_ocr("80/160"),
            self._player_ocr("50/100"),
        ]
        info = _extract_structured_info(results)
        assert len(info["hp_values_player"]) <= 2

    def test_chinese_name_added_to_opponent(self):
        """相手エリアの中国語テキストはポケモン名候補として登録される。"""
        mock_clf = MagicMock()
        unknown_result = MagicMock()
        unknown_result.category = "unknown"
        mock_clf.classify.return_value = unknown_result

        results = [self._opponent_ocr("卡咪龟")]  # 中国語のカメックス
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "卡咪龟" in info["name_candidates_opponent"]


# ═══════════════════════════════════════════════════════════════════════════════
# BattlePhaseClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestBattlePhaseClassifier:

    def setup_method(self):
        self.clf = BattlePhaseClassifier(debounce_seconds=0.0)

    def _ocr_list(self, *texts):
        return [{"text": t, "confidence": 0.9} for t in texts]

    def test_empty_returns_unknown(self):
        assert self.clf.classify([]) == "unknown"

    def test_tatakau_is_command_select(self):
        assert self.clf.classify(self._ocr_list("たたかう")) == "command_select"

    def test_dousuru_is_command_select(self):
        assert self.clf.classify(self._ocr_list("どうする")) == "command_select"

    def test_batsugun_is_animation(self):
        assert self.clf.classify(self._ocr_list("バツグンだ")) == "animation"

    def test_faint_phase_hp_zero(self):
        assert self.clf.classify(self._ocr_list("0/100")) == "faint"

    def test_faint_not_triggered_for_low_denom(self):
        """分母 < 50 は PP 値なので faint にならない。"""
        assert self.clf.classify(self._ocr_list("0/8")) != "faint"

    def test_switch_select(self):
        assert self.clf.classify(self._ocr_list("こうたい")) == "switch_select"

    def test_battle_end(self):
        assert self.clf.classify(self._ocr_list("勝負に勝った")) == "battle_end"

    def test_battle_end_loss(self):
        assert self.clf.classify(self._ocr_list("勝負に負けた")) == "battle_end"

    def test_battle_end_split_ocr(self):
        """OCRが「勝負に」+「勝った！」に分割した場合でも検知できる。"""
        assert self.clf.classify(self._ocr_list("勝負に", "勝った！")) == "battle_end"

    def test_battle_end_split_ocr_surrender(self):
        """OCRが「降参が」+「選ばれました」に分割した場合でも検知できる。"""
        assert self.clf.classify(self._ocr_list("降参が", "選ばれました")) == "battle_end"

    def test_battle_end_result_waiting_screen(self):
        """成績更新待ち画面（正常決着後のフォールバック）を検知できる。"""
        assert self.clf.classify(self._ocr_list("成績が", "更新されるまで", "少し時間がかかります")) == "battle_end"

    def test_selection_screen(self):
        assert self.clf.classify(self._ocr_list("ポケモンを選んで")) == "selection_screen"

    def test_priority_battle_end_over_command(self):
        """battle_end は command_select より優先。"""
        result = self.clf.classify(self._ocr_list("たたかう", "勝負に勝った"))
        assert result == "battle_end"

    def test_detect_battle_start_on_first_command(self):
        event = self.clf.detect(self._ocr_list("たたかう"))
        assert event == "battle_start"

    def test_detect_battle_start_only_once(self):
        self.clf.detect(self._ocr_list("たたかう"))  # battle_start
        event = self.clf.detect(self._ocr_list("たたかう"))  # 同じフェーズ継続
        assert event is None

    def test_detect_move_used_on_communication_to_animation(self):
        """move_used は 通信待機中→バトルアニメーション の遷移で発火する（Champions方式）。

        通信フェーズ平滑化の仕様に合わせて clock 注入で時間を進める:
          入場 = 連続 _COMM_ENTRY_SEC（0.7秒）以上の検出で確定
          退場 = _COMM_EXIT_GRACE_SEC（3秒）以上検出が途切れて確定
        """
        t = {"now": 0.0}
        clf = BattlePhaseClassifier(debounce_seconds=0.0, clock=lambda: t["now"])
        clf.detect(self._ocr_list("たたかう"))    # battle_start
        t["now"] = 10.0
        clf.detect(self._ocr_list("通信中"))      # communication 検出開始
        t["now"] = 11.0
        clf.detect(self._ocr_list("通信中"))      # 連続0.7秒以上 → 通信フェーズ確定
        t["now"] = 15.0                            # 退出猶予3秒を超えて通信表示が消える
        event = clf.detect(self._ocr_list("バツグンだ"))  # move_used
        assert event == "move_used"

    def test_detect_switch_on_command_to_switch_select(self):
        self.clf.detect(self._ocr_list("たたかう"))
        event = self.clf.detect(self._ocr_list("こうたい"))
        assert event == "switch"

    def test_detect_faint_on_hp_zero(self):
        self.clf.detect(self._ocr_list("たたかう"))
        self.clf.detect(self._ocr_list("バツグンだ"))  # animation
        event = self.clf.detect(self._ocr_list("0/100"))
        assert event == "faint"

    def test_detect_battle_end(self):
        """battle_start → animation → battle_end で battle_end が発火する。"""
        self.clf.detect(self._ocr_list("たたかう"))          # battle_start
        self.clf.detect(self._ocr_list("バツグンだ"))         # animation (unknown)
        event = self.clf.detect(self._ocr_list("勝負に勝った"))
        assert event == "battle_end"

    def test_battle_started_resets_on_selection_screen(self):
        """選出画面を通ると battle_started がリセットされ、次の command_select で battle_start が発火する。"""
        self.clf.detect(self._ocr_list("たたかう"))  # battle_start
        self.clf.detect(self._ocr_list("選出"))      # selection_screen → reset
        event = self.clf.detect(self._ocr_list("たたかう"))
        assert event == "battle_start"

    def test_debounce_suppresses_duplicate_move_used(self):
        """move_used は5秒デバウンスが有効。同一ターンで通信終了が2回誤検知されても1回のみ発火する。"""
        clf = BattlePhaseClassifier(debounce_seconds=60.0)
        clf.detect(self._ocr_list("たたかう"))   # battle_start
        clf.detect(self._ocr_list("通信中"))     # communication
        clf.detect(self._ocr_list("バツグンだ")) # move_used → 記録
        clf.detect(self._ocr_list("通信中"))     # communication (再び)
        event = clf.detect(self._ocr_list("バツグンだ"))  # 5秒デバウンス → 抑制
        assert event is None

    def test_processing_flag_suppresses_events(self):
        """処理中フラグ ON 時は battle_end 以外を抑制する。"""
        self.clf.detect(self._ocr_list("たたかう"))  # battle_start
        self.clf.set_processing(True)
        event = self.clf.detect(self._ocr_list("バツグンだ"))
        assert event is None

    def test_processing_flag_allows_battle_end(self):
        """処理中でも battle_end は割り込み検知する。"""
        self.clf.detect(self._ocr_list("たたかう"))
        self.clf.set_processing(True)
        event = self.clf.detect(self._ocr_list("勝負に勝った"))
        assert event == "battle_end"


# ═══════════════════════════════════════════════════════════════════════════════
# BattleStateTracker
# ═══════════════════════════════════════════════════════════════════════════════

def _make_game_state(
    player_names=None,
    opponent_names=None,
    hp_player=None,
    hp_opponent=None,
    hp_values=None,
    status="なし",
    ocr_text="",
):
    return {
        "name_candidates_player":   player_names or [],
        "name_candidates_opponent": opponent_names or [],
        "hp_values_player":         hp_player or [],
        "hp_values_opponent":       hp_opponent or [],
        "hp_values":                hp_values or [],
        "status":                   status,
        "ocr_text":                 ocr_text,
        # スロット別HP（x座標ソート）: テストでは座標なしのためhp_playerと同値で代用
        "hp_player_by_slot":        hp_player or [],
        "hp_opponent_by_slot":      hp_opponent or [],
        "name_player_with_cx":      [],
        "name_opponent_with_cx":    [],
    }


class TestBattleStateTracker:

    def setup_method(self):
        self.tracker = BattleStateTracker()

    def test_initial_state_empty(self):
        ctx = self.tracker.to_context()
        assert ctx["turn"] == 0
        assert ctx["player_field"] == "情報収集中"
        assert ctx["opponent_field"] == "情報収集中"

    def test_update_creates_player_slot(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs, "move_used")
        assert any(s.name == "ピカチュウ" for s in self.tracker._player)

    def test_update_creates_opponent_slot(self):
        gs = _make_game_state(opponent_names=["エルフーン"])
        self.tracker.update(gs, "move_used")
        assert any(s.name == "エルフーン" for s in self.tracker._opponent)

    def test_pokemon_marked_on_field_when_seen(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.on_field is True

    def test_turn_increments_on_update(self):
        gs = _make_game_state()
        self.tracker.update(gs, "move_used")
        assert self.tracker.turn == 1
        self.tracker.update(gs, "move_used")
        assert self.tracker.turn == 2

    def test_cap_on_field_max_2(self):
        """ダブルバトル制約: 場に出せるのは最大 2 匹。"""
        gs = _make_game_state(player_names=["A", "B", "C"])
        self.tracker.update(gs, "move_used")
        on_field = [s for s in self.tracker._player if s.on_field]
        assert len(on_field) <= 2

    def test_opponent_not_added_to_player_side(self):
        """相手側に登録済みのポケモンは自分側に混入しない。"""
        gs1 = _make_game_state(opponent_names=["エルフーン"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["エルフーン"])
        self.tracker.update(gs2, "move_used")
        assert not any(s.name == "エルフーン" for s in self.tracker._player)

    def test_hp_assigned_to_on_field_player(self):
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["150/176"])
        self.tracker.update(gs2, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.hp == "150/176"

    def test_faint_event_marks_pokemon_fainted(self):
        """faint イベント + HP=0 でポケモンが気絶扱いになる。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["0/176"])
        self.tracker.update(gs2, "faint")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.fainted is True
        assert slot.on_field is False

    def test_non_faint_event_does_not_faint_hp_zero(self):
        """faint イベント以外では HP=0 でも気絶しない（誤分類対策）。"""
        gs1 = _make_game_state(player_names=["ゴリランダー"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["ゴリランダー"], hp_player=["0/200"])
        self.tracker.update(gs2, "switch")  # faint ではない
        slot = next(s for s in self.tracker._player if s.name == "ゴリランダー")
        assert slot.fainted is False

    def test_max_slots_4_per_side(self):
        """各サイド最大 4 スロット。超えた分は無視される。"""
        for i in range(6):
            gs = _make_game_state(player_names=[f"ポケモン{i}"])
            self.tracker.update(gs, "move_used")
        assert len(self.tracker._player) <= BattleStateTracker.MAX_SLOTS

    def test_update_move_records_to_slot(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs, "move_used")
        self.tracker.update_move("ピカチュウ", "かみなり")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert "かみなり" in slot.moves_used

    def test_update_move_no_duplicate(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs, "move_used")
        self.tracker.update_move("ピカチュウ", "かみなり")
        self.tracker.update_move("ピカチュウ", "かみなり")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.moves_used.count("かみなり") == 1

    def test_update_move_max_4_moves(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs, "move_used")
        for move in ["技A", "技B", "技C", "技D", "技E"]:
            self.tracker.update_move("ピカチュウ", move)
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert len(slot.moves_used) <= 4

    def test_set_not_on_field_exact_match(self):
        gs = _make_game_state(player_names=["ゴリランダー"])
        self.tracker.update(gs, "move_used")
        result = self.tracker.set_not_on_field("ゴリランダー")
        assert result is True
        slot = next(s for s in self.tracker._player if s.name == "ゴリランダー")
        assert slot.on_field is False

    def test_set_not_on_field_partial_match(self):
        """OCR 誤読で部分一致する場合も対応する。"""
        gs = _make_game_state(player_names=["ゴリランダー"])
        self.tracker.update(gs, "move_used")
        result = self.tracker.set_not_on_field("ゴリランダ")  # 末尾1文字欠落
        assert result is True

    def test_set_not_on_field_returns_false_if_not_found(self):
        result = self.tracker.set_not_on_field("存在しないポケモン")
        assert result is False

    def test_pokemon_removed_from_field_after_miss_threshold(self):
        """_ON_FIELD_MISS_THRESHOLD ターン以上不検出なら場から降ろす。"""
        gs = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs, "move_used")
        # 不検出状態で何ターンも進める
        gs_empty = _make_game_state()
        for _ in range(BattleStateTracker._ON_FIELD_MISS_THRESHOLD + 1):
            self.tracker.update(gs_empty, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.on_field is False

    def test_to_context_shows_field_and_bench(self):
        """場のポケモンと控えが正しく分離して出力される。"""
        gs1 = _make_game_state(player_names=["ピカチュウ", "エルフーン"])
        self.tracker.update(gs1, "move_used")
        # エルフーンを場から降ろす
        self.tracker.set_not_on_field("エルフーン")
        ctx = self.tracker.to_context()
        assert "ピカチュウ" in ctx["player_field"]
        assert "エルフーン" in ctx["player_bench"]

    def test_to_context_fainted_shown_as_hinshi(self):
        """気絶したポケモンは控えに「(ひんし)」付きで表示される。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["0/176"])
        self.tracker.update(gs2, "faint")
        ctx = self.tracker.to_context()
        assert "ひんし" in ctx["player_bench"]

    def test_to_context_event_log_appended(self):
        gs = _make_game_state(ocr_text="ピカチュウのかみなりを")
        self.tracker.update(gs, "move_used")
        ctx = self.tracker.to_context()
        assert "T1:move_used" in ctx["event_log"]

    def test_to_context_hp_pinch_marker(self):
        """HP が 25% 以下のポケモンに★ピンチが付く。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["30/200"])
        self.tracker.update(gs2, "move_used")
        ctx = self.tracker.to_context()
        assert "★ピンチ" in ctx["player_field"]

    def test_status_updated_from_game_state(self):
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(player_names=["ピカチュウ"], status="まひ")
        self.tracker.update(gs2, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.status == "まひ"

    def test_hp_fallback_uses_all_hp_values(self):
        """hp_values_player/opponent が空の場合、hp_values からフォールバックする。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        self.tracker.update(gs1, "move_used")
        gs2 = _make_game_state(
            player_names=["ピカチュウ"],
            hp_values=["100/200", "80/160"],  # フォールバック用
        )
        self.tracker.update(gs2, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.hp is not None


# ═══════════════════════════════════════════════════════════════════════════════
# FieldPokemon dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestFieldPokemon:

    def test_default_values(self):
        p = FieldPokemon(name="ピカチュウ")
        assert p.name == "ピカチュウ"
        assert p.hp is None
        assert p.status is None
        assert p.moves_used == []
        assert p.on_field is False
        assert p.fainted is False
        assert p.confidence == 0

    def test_moves_used_not_shared_between_instances(self):
        """デフォルト引数の罠: field(default_factory=list) で共有されないことを確認。"""
        p1 = FieldPokemon(name="A")
        p2 = FieldPokemon(name="B")
        p1.moves_used.append("かみなり")
        assert "かみなり" not in p2.moves_used
