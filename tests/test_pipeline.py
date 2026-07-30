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
    _build_bedrock_context,
    _clean_commentary,
    _detect_battle_result,
    _detect_glitch_cause,
    _extract_structured_info,
    _is_battle_screen,
    _ocr_results_to_text,
    _replace_glitch_commentary,
    _GLITCH_CAUSE_KEYWORDS,
    _GLITCH_TEMPLATES,
    BattleMessageParser,
    BattlePhaseClassifier,
    BattleStateTracker,
    FieldPokemon,
    Pipeline,
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
# 勝敗検出（battle_end実況の勝敗明言・2026-07-30視聴fb#4）
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectBattleResult:

    def test_win(self):
        assert _detect_battle_result("勝負に勝った!") == "勝ち"

    def test_lose(self):
        assert _detect_battle_result("勝負に負けた!") == "負け"

    def test_ocr_split_with_spaces(self):
        """OCRが「勝負に」と「負けた!」に分割してスペース結合されるケース。"""
        assert _detect_battle_result("〇〇は 勝負に 負けた!") == "負け"

    def test_unknown_returns_none(self):
        assert _detect_battle_result("降参が選ばれました") is None
        assert _detect_battle_result("") is None

    def test_context_passthrough(self):
        """game_stateのbattle_resultが_build_bedrock_contextに乗る（battle_endのみ
        注入されるため通常イベントでは空文字）。"""
        gs = {"ocr_text": "", "hp_values": [], "balls_remaining": [],
              "name_candidates_player": [], "name_candidates_opponent": [],
              "status": "なし", "battle_result": "勝ち"}
        ctx = _build_bedrock_context(gs, "battle_end", None, None, [])
        assert ctx["battle_result"] == "勝ち"
        gs2 = dict(gs)
        del gs2["battle_result"]
        assert _build_bedrock_context(gs2, "move_used", None, None, [])["battle_result"] == ""


# ═══════════════════════════════════════════════════════════════════════════════
# AIグリッチ差し替え（Bedrock保留・困惑応答対策・2026-07-29決定）
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlitchCommentary:

    def test_contradiction_keywords(self):
        assert _detect_glitch_cause("データが矛盾していて実況できません") == "データがちぐはぐさん"
        assert _detect_glitch_cause("情報がちぐはぐで判断できないよ") == "データがちぐはぐさん"

    def test_visibility_keywords(self):
        assert _detect_glitch_cause("画面が見えにくくて…") == "画面がチカチカしてた"
        assert _detect_glitch_cause("HPが読み取れないの") == "画面がチカチカしてた"

    def test_pending_keywords(self):
        assert _detect_glitch_cause("まだ確定できてないの") == "情報がまだ揃ってない"
        assert _detect_glitch_cause("次のフレーム更新をお待ちください") == "情報がまだ揃ってない"

    def test_fallback_keywords(self):
        assert _detect_glitch_cause("なんだかモヤモヤするなあ") == "ナゾのノイズ"
        assert _detect_glitch_cause("誰か教えてほしいな") == "ナゾのノイズ"
        assert _detect_glitch_cause("状況を教えてもらえると助かる") == "ナゾのノイズ"
        assert _detect_glitch_cause("今は実況できないよ") == "ナゾのノイズ"

    def test_normal_commentary_not_detected(self):
        assert _detect_glitch_cause("ガブリアスのじしんが炸裂！大ダメージ！") is None
        assert _detect_glitch_cause("") is None

    def test_replace_returns_formatted_template(self):
        replaced = _replace_glitch_commentary("データが矛盾していて実況できません")
        expected = [t.format(cause="データがちぐはぐさん") for t in _GLITCH_TEMPLATES]
        assert replaced in expected

    def test_replace_passes_through_normal_text(self):
        text = "ペリッパーのぼうふう！すごい風だ！"
        assert _replace_glitch_commentary(text) == text

    def test_templates_do_not_trigger_detection(self):
        """テンプレート側が検出キーワードを含まないこと（キーワード拡張時の回帰ガード）。
        原因文言は意図的にキーワードを含む（例:「データがちぐはぐさん」）が、
        差し替えは実行時に1回しか通らないため実害はない。"""
        for template in _GLITCH_TEMPLATES:
            assert _detect_glitch_cause(template.format(cause="テスト")) is None

    def test_templates_have_no_emoji(self):
        """絵文字（U+1F300-1FAFF）はMeiryo字幕で豆腐化する既知地雷（♪♡は可）。"""
        import re as _re
        pat = _re.compile("[\U0001F300-\U0001FAFF]")
        for template in _GLITCH_TEMPLATES:
            assert not pat.search(template)
        for _, cause in _GLITCH_CAUSE_KEYWORDS:
            assert not pat.search(cause)


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
        """自分エリア（_PLAYER_Y_THRESHOLD <= y < _COMMAND_Y_MIN）の OCR 結果。
        名前候補としては中間帯（メッセージ/選出/パネル）のため採用されない。"""
        return _ocr(text, conf, y_center=600.0)

    def _player_nameplate_ocr(self, text, conf=0.9):
        """自分ネームプレート帯（y > _PLAYER_NAME_Y_MIN）の OCR 結果。"""
        return _ocr(text, conf, y_center=960.0)

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

        results = [self._player_nameplate_ocr("ピカチュウ")]
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "ピカチュウ" in info["name_candidates_player"]

    def test_message_band_name_excluded_from_player(self):
        """メッセージ帯（500<y<=930）のポケモン名は自分側候補に入らない。
        相手の繰り出し・技メッセージ内の相手名が自分側に混入し、
        新規登録ヒステリシスを貫通する問題（2026-07-08）のリグレッションガード。"""
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "pokemon"
        mock_result.canonical_ja = "ガブリアス"
        mock_clf.classify.return_value = mock_result

        results = [_ocr("ガブリアスを", y_center=819.0)]  # メッセージボックス内
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "ガブリアス" not in info["name_candidates_player"]
        assert "ガブリアス" not in info["name_candidates_opponent"]

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

        results = [self._player_nameplate_ocr(f"ポケモン{i}") for i in range(10)]
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
        # OCR HP割当は行動選択画面（command_cy 300-450）のみ有効（2026-04-22導入）。
        # テストでは行動選択画面のフレームを模す
        "command_cy":               380.0,
    }


def _register(tracker, gs, event="move_used"):
    """新規名の登録ヒステリシス（低信頼経路は2サイクル連続目撃で登録・
    2026-07-07導入）を満たすため、同じ game_state で2回 update する。"""
    tracker.update(gs, event)
    tracker.update(gs, event)


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
        _register(self.tracker, gs)
        assert any(s.name == "ピカチュウ" for s in self.tracker._player)

    def test_update_creates_opponent_slot(self):
        gs = _make_game_state(opponent_names=["エルフーン"])
        _register(self.tracker, gs)
        assert any(s.name == "エルフーン" for s in self.tracker._opponent)

    def test_pokemon_marked_on_field_when_seen(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs)
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
        gs = _make_game_state(player_names=["AAA", "BBB", "CCC"])
        _register(self.tracker, gs)
        on_field = [s for s in self.tracker._player if s.on_field]
        assert len(on_field) <= 2

    def test_opponent_not_added_to_player_side(self):
        """相手側に登録済みのポケモンは自分側に混入しない。"""
        gs1 = _make_game_state(opponent_names=["エルフーン"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["エルフーン"])
        _register(self.tracker, gs2)
        assert not any(s.name == "エルフーン" for s in self.tracker._player)

    def test_hp_assigned_to_on_field_player(self):
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["150/176"])
        self.tracker.update(gs2, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.hp == "150/176"

    def test_faint_event_marks_pokemon_fainted(self):
        """faint イベント + HP=0 でポケモンが気絶扱いになる。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["0/176"])
        self.tracker.update(gs2, "faint")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.fainted is True
        assert slot.on_field is False

    def test_non_faint_event_does_not_faint_hp_zero(self):
        """faint イベント以外では HP=0 でも気絶しない（誤分類対策）。"""
        gs1 = _make_game_state(player_names=["ゴリランダー"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ゴリランダー"], hp_player=["0/200"])
        self.tracker.update(gs2, "switch")  # faint ではない
        slot = next(s for s in self.tracker._player if s.name == "ゴリランダー")
        assert slot.fainted is False

    def test_max_slots_4_per_side(self):
        """各サイド最大 4 スロット。超えた分は無視される。"""
        for i in range(6):
            gs = _make_game_state(player_names=[f"ポケモン{i}"])
            _register(self.tracker, gs)
        assert len(self.tracker._player) <= BattleStateTracker.MAX_SLOTS

    def test_update_move_records_to_slot(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs)
        self.tracker.update_move("ピカチュウ", "かみなり")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert "かみなり" in slot.moves_used

    def test_update_move_no_duplicate(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs)
        self.tracker.update_move("ピカチュウ", "かみなり")
        self.tracker.update_move("ピカチュウ", "かみなり")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.moves_used.count("かみなり") == 1

    def test_update_move_max_4_moves(self):
        gs = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs)
        for move in ["技A", "技B", "技C", "技D", "技E"]:
            self.tracker.update_move("ピカチュウ", move)
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert len(slot.moves_used) <= 4

    def test_set_not_on_field_exact_match(self):
        gs = _make_game_state(player_names=["ゴリランダー"])
        _register(self.tracker, gs)
        result = self.tracker.set_not_on_field("ゴリランダー")
        assert result is True
        slot = next(s for s in self.tracker._player if s.name == "ゴリランダー")
        assert slot.on_field is False

    def test_set_not_on_field_partial_match(self):
        """OCR 誤読で部分一致する場合も対応する。"""
        gs = _make_game_state(player_names=["ゴリランダー"])
        _register(self.tracker, gs)
        result = self.tracker.set_not_on_field("ゴリランダ")  # 末尾1文字欠落
        assert result is True

    def test_set_not_on_field_returns_false_if_not_found(self):
        result = self.tracker.set_not_on_field("存在しないポケモン")
        assert result is False

    def test_set_not_on_field_mirror_ambiguous_does_nothing(self):
        """同名ミラーで両陣営とも場に出ている場合、陣営を示す手がかりが
        テキストに無いため誤ベンチ化を避けて何もしない（両方とも on_field のまま）。"""
        gs = _make_game_state(player_names=["イダイトウ"], opponent_names=["イダイトウ"])
        _register(self.tracker, gs)
        result = self.tracker.set_not_on_field("イダイトウ")
        assert result is False
        assert all(s.on_field for s in self.tracker._player if s.name == "イダイトウ")
        assert all(s.on_field for s in self.tracker._opponent if s.name == "イダイトウ")

    def test_set_not_on_field_mirror_one_benched_still_works(self):
        """同名ミラーでも片方が既にベンチにいれば、場に出ている方だけを一意に降ろせる。"""
        gs = _make_game_state(player_names=["イダイトウ"], opponent_names=["イダイトウ"])
        _register(self.tracker, gs)
        opp_slot = next(s for s in self.tracker._opponent if s.name == "イダイトウ")
        opp_slot.on_field = False
        result = self.tracker.set_not_on_field("イダイトウ")
        assert result is True
        player_slot = next(s for s in self.tracker._player if s.name == "イダイトウ")
        assert player_slot.on_field is False

    def test_pokemon_removed_from_field_after_miss_threshold(self):
        """_ON_FIELD_MISS_THRESHOLD ターン以上不検出なら場から降ろす。"""
        gs = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs)
        # 不検出状態で実ゲームターンを進める（閾値は game_turn 基準・
        # game_turn は PipelineRunner が turn_start で加算するためテストでは手動加算）
        gs_empty = _make_game_state()
        for _ in range(BattleStateTracker._ON_FIELD_MISS_THRESHOLD + 1):
            self.tracker.game_turn += 1
            self.tracker.update(gs_empty, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.on_field is False

    def test_to_context_shows_field_and_bench(self):
        """場のポケモンと控えが正しく分離して出力される。"""
        gs1 = _make_game_state(player_names=["ピカチュウ", "エルフーン"])
        _register(self.tracker, gs1)
        # エルフーンを場から降ろす
        self.tracker.set_not_on_field("エルフーン")
        ctx = self.tracker.to_context()
        assert "ピカチュウ" in ctx["player_field"]
        assert "エルフーン" in ctx["player_bench"]

    def test_to_context_fainted_shown_as_hinshi(self):
        """気絶したポケモンは控えに「(ひんし)」付きで表示される。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["0/176"])
        self.tracker.update(gs2, "faint")
        ctx = self.tracker.to_context()
        assert "ひんし" in ctx["player_bench"]

    def test_to_context_event_log_appended(self):
        gs = _make_game_state(ocr_text="ピカチュウのかみなりを")
        self.tracker.update(gs, "move_used")
        ctx = self.tracker.to_context()
        assert "T1:move_used" in ctx["event_log"]

    def test_to_context_turn_history_default_nashi(self):
        ctx = self.tracker.to_context()
        assert ctx["turn_history"] == "なし"

    def test_record_turn_snapshot_appends_field_state(self):
        gs = _make_game_state(player_names=["ピカチュウ"], hp_player=["80/100"])
        _register(self.tracker, gs)
        self.tracker.game_turn = 2
        self.tracker.record_turn_snapshot()
        ctx = self.tracker.to_context()
        assert "T2" in ctx["turn_history"]
        assert "ピカチュウ" in ctx["turn_history"]

    def test_record_turn_snapshot_trims_to_max_history(self):
        for i in range(BattleStateTracker.MAX_TURN_HISTORY + 3):
            self.tracker.game_turn = i
            self.tracker.record_turn_snapshot()
        assert len(self.tracker._turn_history) == BattleStateTracker.MAX_TURN_HISTORY
        # 古い方から捨てられ、最新のターンが残っている
        assert f"T{BattleStateTracker.MAX_TURN_HISTORY + 2}" in self.tracker._turn_history[-1]

    def test_to_context_hp_pinch_marker(self):
        """HP が 25% 以下のポケモンに★ピンチが付く。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["30/200"])
        self.tracker.update(gs2, "move_used")
        ctx = self.tracker.to_context()
        assert "★ピンチ" in ctx["player_field"]

    def test_status_updated_from_game_state(self):
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ピカチュウ"], status="まひ")
        self.tracker.update(gs2, "move_used")
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        assert slot.status == "まひ"

    def test_hp_fallback_uses_all_hp_values(self):
        """hp_values_player/opponent が空の場合、hp_values からフォールバックする。"""
        gs1 = _make_game_state(player_names=["ピカチュウ"])
        _register(self.tracker, gs1)
        # フォールバック値2件はスロット番号で割り当てられるため slot_index を確定させる
        slot = next(s for s in self.tracker._player if s.name == "ピカチュウ")
        slot.slot_index = 0
        gs2 = _make_game_state(
            player_names=["ピカチュウ"],
            hp_values=["100/200", "80/160"],  # フォールバック用
        )
        self.tracker.update(gs2, "move_used")
        assert slot.hp is not None


# ═══════════════════════════════════════════════════════════════════════════════
# BattleMessageParser（同名ミラー戦のサイド誤帰属の回帰ガード）
# ═══════════════════════════════════════════════════════════════════════════════

def _msg_ocr(*texts):
    """メッセージボックスROI内（cx 120-900, cy 740-930）のOCR結果リストを作る。"""
    results = []
    for i, t in enumerate(texts):
        x = 150 + i * 120
        results.append({
            "text": t, "confidence": 0.9,
            "bbox": [[x, 800], [x + 100, 800], [x + 100, 830], [x, 830]],
        })
    return results


class TestBattleMessageParser:

    def setup_method(self):
        self.parser = BattleMessageParser()

    def _types(self, events):
        return [(e["type"], e["pokemon"]) for e in events]

    def test_faint_player_side_no_prefix(self):
        events = self.parser.parse(_msg_ocr("オオニューラは", "たおれた!"))
        assert ("faint", "オオニューラ") in self._types(events)

    def test_faint_opponent_with_prefix(self):
        events = self.parser.parse(_msg_ocr("相手の", "イダイトウは", "たおれた!"))
        assert ("opponent_faint", "イダイトウ") in self._types(events)

    def test_faint_mangled_opponent_prefix_not_player(self):
        """「相手の イダイトウは」の崩れ読み「あい 手の イトウは」を自分側と誤判定しない。
        （実機: 同名ミラー戦で生存中の自分イダイトウ139/201が誤ひんし化した回帰ガード）"""
        events = self.parser.parse(_msg_ocr("あい", "手の", "イトウは", "たおれたー"))
        types = self._types(events)
        assert ("opponent_faint", "イトウ") in types
        assert all(t != "faint" for t, _ in types)

    def test_faint_cross_dedup_after_opponent(self):
        """相手側として発火済みの名前は、プレフィックス欠けの再読でfaintを発行しない。"""
        self.parser.parse(_msg_ocr("相手の", "イダイトウは", "たおれた!"))
        events = self.parser.parse(_msg_ocr("イダイトウは", "たおれた!"))
        assert all(t != "faint" for t, _ in self._types(events))

    def test_hikkometa_is_opponent_switch_out(self):
        """「(トレーナー名)は 〇〇を 引っこめた」は相手側の交代イベントとして発行される。"""
        events = self.parser.parse(_msg_ocr("rixohは", "オオニューラを", "引っこめた!"))
        types = self._types(events)
        assert ("opponent_switch_out", "オオニューラ") in types
        assert all(t != "switch_out" for t, _ in types)

    def test_modore_is_player_switch_out(self):
        """「〇〇 戻れ！」は自分側の交代イベントとして発行される。"""
        events = self.parser.parse(_msg_ocr("オオニューラ", "戻れ!"))
        assert ("switch_out", "オオニューラ") in self._types(events)

    def _statuses(self, events):
        return [(e["pokemon"], e["status"]) for e in events if e["type"] == "status"]

    def test_status_kanji_sleep_with_ocr_fragment(self):
        """Champions漢字形式「眠ってしまった」を検出し「ねむり」に正規化する。
        名前と状態語の間のOCR断片ノイズ「ねむ」も許容する
        （実機: 16-14-39 T2「ペリッパーは ねむ 眠ってしまった!」のねむり漏れの回帰ガード）。"""
        events = self.parser.parse(_msg_ocr("ペリッパーは", "ねむ", "眠ってしまった!"))
        assert ("ペリッパー", "ねむり") in self._statuses(events)

    def test_status_kanji_freeze(self):
        """「凍りついた」→「こおり」に正規化される。"""
        events = self.parser.parse(_msg_ocr("オオニューラは", "凍りついた!"))
        assert ("オオニューラ", "こおり") in self._statuses(events)

    def test_status_kanji_poison(self):
        """「毒を あびた」→「どく」に正規化される。"""
        events = self.parser.parse(_msg_ocr("イダイトウは", "毒を", "あびた!"))
        assert ("イダイトウ", "どく") in self._statuses(events)

    def test_status_hiragana_still_works(self):
        """従来のひらがな形式（SV）は引き続き検出される。"""
        events = self.parser.parse(_msg_ocr("ピカチュウは", "まひじょうたいになった!"))
        assert ("ピカチュウ", "まひ") in self._statuses(events)

    def test_status_not_triggered_by_imahitotsu(self):
        """「効果は いまひとつだ」の「(い)まひ」を状態異常と誤検出しない。
        ノイズスキップをひらがな状態語側に入れると実ログ53本で大量誤爆した回帰ガード。"""
        events = self.parser.parse(_msg_ocr("相手の", "イダイトウに", "効果は", "いまひとつだ"))
        assert self._statuses(events) == []

    def test_titled_opponent_switch_in_not_truncated_by_msg_x_max(self):
        """称号付き交代メッセージ（例:「rixohは ランクマスタ ガブリアスを 繰り出した!」）で
        文末「繰り出した!」がMSG_X_MAXを超えて欠落しない（実機07-00-19: ガブリアス消失バグの
        回帰ガード）。実際のbbox座標（診断JSONLフレーム5965）を再現。"""
        events = self.parser.parse([
            {"text": "rixohは", "confidence": 1.0,
             "bbox": [[328, 818], [408, 818], [408, 848], [328, 848]]},
            {"text": "ランクマスタ", "confidence": 0.673,
             "bbox": [[376, 883], [456, 883], [456, 913], [376, 913]]},
            {"text": "ガブリアスを", "confidence": 0.636,
             "bbox": [[677, 883], [757, 883], [757, 913], [677, 913]]},
            {"text": "繰り出した!", "confidence": 1.0,
             "bbox": [[931, 884], [1011, 884], [1011, 914], [931, 914]]},
        ])
        assert ("opponent_switch_in", "ガブリアス") in self._types(events)


class TestMarkBenchBySide:
    """mark_bench_by_name の side 限定（同名ミラー戦の誤ベンチ化防止）"""

    def setup_method(self):
        self.tracker = BattleStateTracker()
        self.mine = FieldPokemon(name="オオニューラ", on_field=True)
        self.theirs = FieldPokemon(name="オオニューラ", on_field=True)
        self.tracker._player.append(self.mine)
        self.tracker._opponent.append(self.theirs)

    def test_opponent_side_does_not_bench_player(self):
        assert self.tracker.mark_bench_by_name("オオニューラ", side="opponent") is True
        assert self.theirs.on_field is False
        assert self.mine.on_field is True  # 自分側は無傷

    def test_player_side_does_not_bench_opponent(self):
        assert self.tracker.mark_bench_by_name("オオニューラ", side="player") is True
        assert self.mine.on_field is False
        assert self.theirs.on_field is True


class TestMoveUserSide:
    """瞬間ログの陣営タグ判定（timeline.jsonl side・2026-07-30）。
    20-14-17の同名ミラー戦（相手もイダイトウ）でフィラーが陣営を推測で
    外した問題への対策。誤タグよりタグ無し（None）が安全という方針。"""

    def setup_method(self):
        self.tracker = BattleStateTracker()

    def test_player_only_name(self):
        self.tracker._player.append(FieldPokemon(name="オオニューラ", on_field=True))
        assert self.tracker.move_user_side("オオニューラ") == "自分"

    def test_opponent_only_name(self):
        self.tracker._opponent.append(FieldPokemon(name="キラフロル", on_field=True))
        assert self.tracker.move_user_side("キラフロル") == "相手"

    def test_is_opponent_flag_wins(self):
        assert self.tracker.move_user_side("ガブリアス", is_opponent=True) == "相手"

    def test_unknown_name_returns_none(self):
        assert self.tracker.move_user_side("ミュウツー") is None

    def test_mirror_prefers_on_field_side(self):
        """同名ミラー: 場に出ている側を使い手と判定（自分イダイトウがベンチ・
        相手イダイトウが場、のときの技は相手側）。"""
        self.tracker._player.append(FieldPokemon(name="イダイトウ", on_field=False))
        self.tracker._opponent.append(FieldPokemon(name="イダイトウ", on_field=True))
        assert self.tracker.move_user_side("イダイトウ") == "相手"

    def test_mirror_both_on_field_is_ambiguous(self):
        self.tracker._player.append(FieldPokemon(name="イダイトウ", on_field=True))
        self.tracker._opponent.append(FieldPokemon(name="イダイトウ", on_field=True))
        assert self.tracker.move_user_side("イダイトウ") is None


class TestOnFieldMissThresholdUsesGameTurn:
    """_ON_FIELD_MISS_THRESHOLD は self.turn（内部イベントカウンター）ではなく
    self.game_turn（実ターン数）で判定する回帰ガード。

    実機（07-00-19）で、メガシンカ・道具発動・毒ダメージ等のメッセージが
    立て込む区間で update() が同一ゲームターン内に何度も呼ばれ、内部イベント
    カウンターだけで閾値判定すると実際は1ターンも経っていないのに誤って
    場から降ろされていた（自分のオオニューラ・イダイトウが同時に誤って
    場外扱いになり、片方は試合終了まで復帰しなかった）。
    """

    def setup_method(self):
        self.tracker = BattleStateTracker()
        self.mine = FieldPokemon(name="オオニューラ", on_field=True, last_seen_turn=0)
        self.tracker._player.append(self.mine)

    def test_many_updates_within_same_turn_does_not_demote(self):
        """同一game_turn内でupdate()が何度呼ばれても場から降ろされない。"""
        gs = _make_game_state()  # 自分側名前候補なし（名前が見えないフレーム想定）
        for _ in range(10):
            self.tracker.update(gs, "move_used")  # game_turn は増やさない
        assert self.mine.on_field is True

    def test_real_turns_elapsed_still_demotes(self):
        """実際にゲームターンが閾値を超えて進めば従来通り正しく場から降ろされる。"""
        gs = _make_game_state()
        for _ in range(self.tracker._ON_FIELD_MISS_THRESHOLD + 1):
            self.tracker.game_turn += 1
            self.tracker.update(gs, "move_used")
        assert self.mine.on_field is False

    def test_hp_pixel_tracking_counts_as_seen(self):
        """名前OCRが再検出されなくても、HPpxが実測できていれば場に残り続ける。

        実機（07-00-19）で、オオニューラの名前が繰り出し後ほぼOCR再検出されない
        まま HPpx だけは継続的に読めていたのに、名前だけを見る旧ロジックでは
        game_turn 経過で機械的に場から降ろされ、二度と復帰しなかった。
        """
        self.mine.slot_index = 0
        gs = _make_game_state()  # 自分側名前候補なし
        for _ in range(self.tracker._ON_FIELD_MISS_THRESHOLD + 3):
            self.tracker.game_turn += 1
            self.tracker.update_pixel_hp({"player_0": 0.5})  # HPpxは継続して読める
            self.tracker.update(gs, "move_used")
        assert self.mine.on_field is True


class TestUpdateSwitchOut:
    """_update_switch_out のパターン1（「〜は戻っていく」検出）の誤爆防止回帰ガード。

    実機（2026-07-09・20-14-17と07-00-19の2本）で、「しろいハーブで ステータスを
    元に戻した」（アイテム回復メッセージ）が旧ゲート（「戻」1文字含有）に誤反応し、
    交代と無関係なポケモンを set_not_on_field の無条件両側検索で誤ベンチ化していた。
    """

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)  # __init__ を経由せず属性だけ用意
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._classifier = None
        self.mine = FieldPokemon(name="オオニューラ", on_field=True)
        self.runner._battle_tracker._player.append(self.mine)

    def test_shiroi_herb_message_does_not_bench(self):
        """「しろいハーブで元に戻した」はとんぼがえり誤検出させない。"""
        events = [
            _ocr("オオニューラは"), _ocr("しろいハーブで"),
            _ocr("もと"), _ocr("もど"),
            _ocr("ステータスを"), _ocr("元に戻した!"),
        ]
        Pipeline._update_switch_out(self.runner, events)
        assert self.mine.on_field is True

    def test_mamoru_success_message_does_not_bench(self):
        """OCR誤読「戻る」を含む「攻撃から身を守った」も誤検出させない。"""
        events = [
            _ocr("戻る"), _ocr("あいて"), _ocr("相手の"), _ocr("オオニューラは"),
            _ocr("こうげき"), _ocr("まも"), _ocr("攻撃から"), _ocr("身を守った!"),
        ]
        Pipeline._update_switch_out(self.runner, events)
        assert self.mine.on_field is True

    def test_tonbogaeri_still_detected(self):
        """本来の「〜は 戻っていく」（とんぼがえり）は引き続き検出される。"""
        events = [_ocr("オオニューラは"), _ocr("ともの元へ"), _ocr("戻っていく")]
        Pipeline._update_switch_out(self.runner, events)
        assert self.mine.on_field is False


class TestTryRegisterRosterFallback:
    """_try_register の使用者名見切れ検証（ロスター前方一致救済／未一致は棄却）の回帰ガード。

    実機で「トムの」「キの」等のOCR見切れ断片が分類器で弾かれずそのまま_move_log に
    登録され、「信頼度高」技としてBedrockに渡っていたバグの再発防止。
    """

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)  # __init__ を経由せず属性だけ用意
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._tentative_opponent_moves = []
        self.runner._battle_active = True
        self.runner._dense_scan_start_turn = None
        self.runner._move_log = []
        self.runner._MAX_MOVE_LOG = 8
        self.runner._dense_scan_remaining = 0
        self.runner._last_full_ocr_results = []

    def _set_classifier(self, moves, pokemon):
        """moves/pokemon に完全一致する文字列だけを該当カテゴリと判定する簡易分類器モック。"""
        class _Result:
            def __init__(self, category, score, canonical_ja):
                self.category = category
                self.score = score
                self.canonical_ja = canonical_ja

        def classify(text):
            if text in moves:
                return _Result("move", 95, text)
            if text in pokemon:
                return _Result("pokemon", 95, pokemon[text])
            return _Result("other", 0, text)

        clf = MagicMock()
        clf.classify.side_effect = classify
        self.runner._classifier = clf

    def test_unrecognized_truncated_name_without_roster_match_is_rejected(self):
        """ロスターにも図鑑にも一致しない見切れ断片（例:「トム」）は技ログへ登録しない。"""
        self._set_classifier(moves={"テクノバスター"}, pokemon={})
        events = [_ocr("トムの", y_center=800.0), _ocr("テクノバスター", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == []

    def test_truncated_roster_name_is_corrected_via_suffix_match(self):
        """既知ロスター「ドドゲザン」への見切れ断片「ドゲザン」はロスター名に補正して登録される。"""
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ドドゲザン", on_field=True)
        )
        self._set_classifier(moves={"ドゲザン"}, pokemon={})
        events = [_ocr("ドゲザンの", y_center=800.0), _ocr("ドゲザン", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ドドゲザンのドゲザン"]

    def test_opponent_move_from_unregistered_pokemon_calibrates_roster(self):
        """技ログにだけ記録されロスター未登録の相手ポケモンは、技検出と同時に
        ロスターへ校正登録される（実機07-00-19: 繰り出しメッセージのOCR取りこぼしで
        ガブリアスがロスター未登録のまま技ログにだけ記録され、move_logとロスターの
        食い違いがBedrockへの矛盾したcontextとして露呈し「保留」応答を誘発したバグの
        再発防止）。修正前は技ログには載るがロスターには一切現れなかった。"""
        self._set_classifier(moves={"じだんだ"}, pokemon={"ガブリアス": "ガブリアス"})
        events = [
            _ocr("あいて", y_center=800.0),
            _ocr("相手の", y_center=800.0),
            _ocr("ガブリアスの", y_center=800.0),
            _ocr("じだんだ!", y_center=800.0),
        ]
        assert self.runner._battle_tracker._opponent == []  # 事前状態: ロスター未登録
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのじだんだ"]
        assert any(s.name == "ガブリアス" for s in self.runner._battle_tracker._opponent)


class TestOpponentAttackFallbackAmbiguity:
    """_get_active_opponent_name の「場の1匹目」フォールバックの曖昧さ対策。

    実機で「ソーラービーム→フシギバナ」等、ダブルバトルで2匹とも場に出ている時に
    技の使い手を特定できないケースで決め打ちの1匹目に誤帰属していたバグの再発防止。
    使い手が一意に絞れない場合は登録を諦める（誤タグよりタグ無しの方が安全）。
    """

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._tentative_opponent_moves = []
        self.runner._battle_active = True
        self.runner._dense_scan_start_turn = None
        self.runner._move_log = []
        self.runner._MAX_MOVE_LOG = 8
        self.runner._dense_scan_remaining = 0
        self.runner._last_full_ocr_results = []

    def _set_classifier(self, moves, pokemon):
        class _Result:
            def __init__(self, category, score, canonical_ja):
                self.category = category
                self.score = score
                self.canonical_ja = canonical_ja

        def classify(text):
            if text in moves:
                return _Result("move", 95, text)
            if text in pokemon:
                return _Result("pokemon", 95, pokemon[text])
            return _Result("other", 0, text)

        clf = MagicMock()
        clf.classify.side_effect = classify
        self.runner._classifier = clf

    # 「相手の」の直後に使い手名トークンが無く、_find_attacker_from_full_ocr が
    # 空振りするメッセージ（実機の「相手の[技名]」形式の圧縮表示を想定）
    _AMBIGUOUS_EVENTS = [
        _ocr("あいて", y_center=800.0),
        _ocr("相手の", y_center=800.0),
        _ocr("じだんだ!", y_center=800.0),
    ]

    def test_single_on_field_still_registers_tentatively(self):
        """場に1匹しかいなければ、その1匹に仮登録する（従来通りの挙動を維持）。"""
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ガブリアス", on_field=True)
        )
        self._set_classifier(moves={"じだんだ"}, pokemon={})
        Pipeline._update_move_log(self.runner, self._AMBIGUOUS_EVENTS, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのじだんだ"]

    def test_two_on_field_does_not_guess(self):
        """ダブルバトルで2匹とも場に出ている場合、使い手を決め打ちせず未登録のままにする。"""
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="フシギバナ", on_field=True)
        )
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ガブリアス", on_field=True)
        )
        self._set_classifier(moves={"じだんだ"}, pokemon={})
        Pipeline._update_move_log(self.runner, self._AMBIGUOUS_EVENTS, is_main_ocr=True)
        assert self.runner._move_log == []

    def test_two_on_field_but_one_fainted_is_unambiguous(self):
        """2匹登録済みでも片方が気絶済みなら、場にいるのは実質1匹なので登録する。"""
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="フシギバナ", on_field=False, fainted=True)
        )
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ガブリアス", on_field=True)
        )
        self._set_classifier(moves={"じだんだ"}, pokemon={})
        Pipeline._update_move_log(self.runner, self._AMBIGUOUS_EVENTS, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのじだんだ"]


class TestResetBattleState:
    """_reset_battle_state: battle_start／遅延起動共通のリセット処理。
    遅延起動が前試合ロスターのまま走り新試合の繰り出しで eviction 連発していた
    （実機 08-15-22: 目撃53回のフラエッテまで削除）ため、遅延起動でも
    battle_start と同様に前試合の残骸を全て捨てる。"""

    def _make_runner(self):
        runner = Pipeline.__new__(Pipeline)
        runner._video_now = 100.0
        runner._hpbar_analyzer = MagicMock()
        old_tracker = BattleStateTracker()
        old_tracker._player.append(FieldPokemon(name="フラエッテ", confidence=53))
        old_tracker._opponent.append(FieldPokemon(name="スピアー"))
        runner._battle_tracker = old_tracker
        runner._battle_active = False
        runner._end_screen_count = 2
        runner._commentary_history = ["前試合の実況"]
        runner._move_log = ["T1:スピアーのどくづき"]
        runner._last_ball_yolo = object()
        runner._last_ability_msg = {"opp": "あめうけざら"}
        return runner, old_tracker

    def test_clears_previous_battle_roster(self):
        runner, old_tracker = self._make_runner()
        runner._reset_battle_state()
        assert runner._battle_tracker is not old_tracker
        assert runner._battle_tracker._player == []
        assert runner._battle_tracker._opponent == []

    def test_activates_and_clears_battle_scoped_state(self):
        runner, _ = self._make_runner()
        runner._reset_battle_state()
        assert runner._battle_active is True
        assert runner._battle_active_since == 100.0
        assert runner._end_screen_count == 0
        assert runner._commentary_history == []
        assert runner._move_log == []
        assert runner._last_ball_yolo is None
        assert runner._last_ability_msg == {}

    def test_analyzer_reset_and_slot_callback_rewired(self):
        runner, _ = self._make_runner()
        runner._reset_battle_state()
        runner._hpbar_analyzer.reset.assert_called_once()
        assert runner._battle_tracker.slot_reset_cb == runner._hpbar_analyzer.reset_slot


class TestMoveLogDisplay:
    """_move_log_display: 後付け未修正の仮確定エントリに「（推定）」を付けてBedrockへ渡す。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._move_log = ["T1:オオニューラのわるだくみ", "T2:リキキリンのけたぐり"]

    def test_confirmed_entry_has_no_marker(self):
        self.runner._tentative_opponent_moves = []
        assert self.runner._move_log_display(5) == [
            "T1:オオニューラのわるだくみ", "T2:リキキリンのけたぐり",
        ]

    def test_unconfirmed_tentative_entry_gets_marked(self):
        """使い手フォールバックのまま後付け修正されていないエントリのみ「（推定）」が付く。"""
        self.runner._tentative_opponent_moves = [
            {"old_entry": "T2:リキキリンのけたぐり", "move_name": "けたぐり",
             "turn_label": "2", "fallback_pokemon": "リキキリン"},
        ]
        assert self.runner._move_log_display(5) == [
            "T1:オオニューラのわるだくみ", "T2:リキキリンのけたぐり（推定）",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# スロット番号割り当て（_assign_slot_indices・HPスロット反転バグの回帰ガード）
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlotIndexAssignment:
    """ネームプレートcxによる物理スロット（0=左, 1=右）の割り当て。
    バー中心x（player: 292/688・opponent: 1336/1732）は hpbar_analyzer 実測値。
    """

    def setup_method(self):
        self.tracker = BattleStateTracker()

    def _add(self, side, name):
        slots = self.tracker._player if side == "player" else self.tracker._opponent
        p = FieldPokemon(name=name, on_field=True)
        slots.append(p)
        return p

    def test_single_candidate_right_bar_gets_slot1(self):
        """両スロット空き＋候補1匹が右バー位置 → スロット1。
        旧実装は zip で cx 無視のスロット0割当となり、後続の左側ポケモンと
        HP表示が丸ごと入れ替わった（06-25-46 実機で確認）。"""
        vana = self._add("opponent", "フシギバナ")
        self.tracker._assign_slot_indices(
            self.tracker._opponent, [("フシギバナ", 1672.0)], "opponent")
        assert vana.slot_index == 1

    def test_single_candidate_left_bar_gets_slot0(self):
        liza = self._add("opponent", "リザードン")
        self.tracker._assign_slot_indices(
            self.tracker._opponent, [("リザードン", 1275.0)], "opponent")
        assert liza.slot_index == 0

    def test_inversion_scenario_regression(self):
        """06-25-46 実機シナリオ: フシギバナ(右)が先に見え、リザードン(左)が後続。
        フシギバナ→1・リザードン→0 で反転しないこと。"""
        vana = self._add("opponent", "フシギバナ")
        self.tracker._assign_slot_indices(
            self.tracker._opponent, [("フシギバナ", 1672.0)], "opponent")
        liza = self._add("opponent", "リザードン")
        self.tracker._assign_slot_indices(
            self.tracker._opponent, [("リザードン", 1275.0)], "opponent")
        assert vana.slot_index == 1
        assert liza.slot_index == 0

    def test_single_candidate_far_from_bars_deferred(self):
        """選出リスト等バー位置と無関係な cx は割り当てず保留する。"""
        vana = self._add("opponent", "フシギバナ")
        self.tracker._assign_slot_indices(
            self.tracker._opponent, [("フシギバナ", 250.0)], "opponent")
        assert vana.slot_index is None

    def test_player_side_single_candidate(self):
        """自分側バー中心（688）近傍の候補1匹 → スロット1。"""
        p = self._add("player", "ピカチュウ")
        self.tracker._assign_slot_indices(
            self.tracker._player, [("ピカチュウ", 655.0)], "player")
        assert p.slot_index == 1

    def test_two_candidates_relative_order(self):
        """2匹同時は従来通り相対x順（左→0, 右→1）。"""
        a = self._add("player", "ピカチュウ")
        b = self._add("player", "ゴリランダー")
        self.tracker._assign_slot_indices(
            self.tracker._player,
            [("ゴリランダー", 650.0), ("ピカチュウ", 250.0)], "player")
        assert a.slot_index == 0
        assert b.slot_index == 1

    def test_one_candidate_one_free_slot_forced(self):
        """片方使用中＋候補1匹は残りの空きスロットへ（従来挙動維持）。"""
        vana = self._add("opponent", "フシギバナ")
        vana.slot_index = 1
        liza = self._add("opponent", "リザードン")
        self.tracker._assign_slot_indices(
            self.tracker._opponent, [("リザードン", 1275.0)], "opponent")
        assert liza.slot_index == 0


# ═══════════════════════════════════════════════════════════════════════════════
# assign_hp_from_ocr（定期OCR経路の数値HP補完割当）
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssignHpFromOcr:
    """定期OCRからの数値HP割当（位置ゲート＋バー中心cx近接＋2回一致ヒステリシス）。
    実機: 07-00-19 終盤のイダイトウ 7/201 が画面で正読されていたのに
    イベント経路にしか数値HP割当がなく、HPpx物理限界の古い47%が表示され続けた。
    座標は実測値（自分側数値=cy1000-1049・cx292/688近傍、相手側%=cy100-151・
    cx1336/1732近傍）に基づく。"""

    # 実測に基づく正当な表示位置
    P0 = (292.0, 1028.0)   # 自分スロット0の数値位置
    P1 = (688.0, 1028.0)   # 自分スロット1の数値位置
    O0 = (1350.0, 147.0)   # 相手スロット0のHP%位置
    O1 = (1750.0, 147.0)   # 相手スロット1のHP%位置

    def setup_method(self):
        self.tracker = BattleStateTracker()
        self.p = FieldPokemon(name="イダイトウ", on_field=True, slot_index=0)
        self.tracker._player.append(self.p)

    def _p(self, hp, pos=None):
        cx, cy = pos or self.P0
        return [(hp, cx, cy)]

    def test_assigns_after_two_consistent_reads(self):
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        assert self.p.hp is None  # 1回目は保留
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        assert self.p.hp == "7/201"

    def test_single_or_flickering_read_not_assigned(self):
        """単発誤読（1/205等）や毎サイクル値が変わる読みは割り当てない。"""
        self.tracker.assign_hp_from_ocr(self._p("1/205"), [])
        self.tracker.assign_hp_from_ocr(self._p("101/205"), [])
        assert self.p.hp is None

    def test_unreadable_cycle_does_not_reset_pending(self):
        """HP未読サイクル（アニメ・パネル等）を挟んでも保留値は維持され、
        次の同値読みで確定する（実機 07-00-19: 7/201→空→7/201 の交互列で
        確定しなかった回帰ガード）。"""
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        self.tracker.assign_hp_from_ocr([], [])
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        assert self.p.hp == "7/201"

    def test_hazard_band_rejected(self):
        """交換選択パネル等の危険帯（自分側 cy 750-799・相手側 cy 350-399）の
        HP風数値は割り当てない（実測: '201/201' cx787/cy766・'100%' cx813/cy386）。"""
        self.tracker.assign_hp_from_ocr(self._p("117/155", (787.0, 766.0)), [])
        self.tracker.assign_hp_from_ocr(self._p("117/155", (787.0, 766.0)), [])
        assert self.p.hp is None
        opp = FieldPokemon(name="ガブリアス", on_field=True, slot_index=0)
        self.tracker._opponent.append(opp)
        self.tracker.assign_hp_from_ocr([], [("100%", 813.0, 386.0)])
        self.tracker.assign_hp_from_ocr([], [("100%", 813.0, 386.0)])
        assert opp.hp is None

    def test_cx_far_from_bar_center_rejected(self):
        """バー中心からcx許容（200px）超の数値は割り当てない（画面中央の
        bboxなしフォールバック座標 cx=960 等）。"""
        self.tracker.assign_hp_from_ocr(self._p("7/201", (960.0, 999.0)), [])
        self.tracker.assign_hp_from_ocr(self._p("7/201", (960.0, 999.0)), [])
        assert self.p.hp is None

    def test_zero_hp_not_assigned(self):
        """0/X は誤気絶防止のため定期OCR経路でも割り当てない。"""
        self.tracker.assign_hp_from_ocr(self._p("0/201"), [])
        self.tracker.assign_hp_from_ocr(self._p("0/201"), [])
        assert self.p.hp is None

    def test_stale_px_loses_to_fresh_numeric_display(self):
        """HPpxが物理限界（<6.6%）で更新できない時、後から読めた数値HPが表示に勝つ。"""
        self.p.hp_pct_pixel = 0.473
        self.p.hp_px_turn = self.tracker.turn
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        assert "HP:7/201" in self.tracker._format_pokemon(self.p)

    def test_held_px_value_does_not_mask_numeric(self):
        """アナライザーが保持値を返し続けても hp_px_turn は再スタンプされず、
        数値HPが表示に勝ち続ける（実機 07-00-19: T6で7/201→T7で47%(px)に
        戻ってしまった再発バグの回帰ガード）。"""
        self.tracker.update_pixel_hp({"player_0": 0.473})
        first_stamp = self.p.hp_px_turn
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        self.tracker.assign_hp_from_ocr(self._p("7/201"), [])
        # 次のイベントで turn が進み、アナライザーは保持値 0.473 を返し続ける
        self.tracker.turn += 1
        self.tracker.update_pixel_hp({"player_0": 0.473})
        assert self.p.hp_px_turn == first_stamp  # 保持値では再スタンプしない
        assert "HP:7/201" in self.tracker._format_pokemon(self.p)

    def test_px_change_still_stamps_fresh(self):
        """値が実際に変わったpx読みは従来通り鮮度スタンプされ表示に勝つ。"""
        self.tracker.assign_hp_from_ocr(self._p("100/201"), [])
        self.tracker.assign_hp_from_ocr(self._p("100/201"), [])
        self.tracker.turn += 1
        self.tracker.update_pixel_hp({"player_0": 0.25})
        assert "HP:25%(px)" in self.tracker._format_pokemon(self.p)

    def test_opponent_side_slot_by_cx(self):
        """相手側はHP%のcxからバー中心近接でスロットを直接決定する
        （1匹しか読めないフレームでも取り違えない）。%形式は3回一致で確定。"""
        opp = FieldPokemon(name="ガブリアス", on_field=True, slot_index=1)
        self.tracker._opponent.append(opp)
        self.tracker.assign_hp_from_ocr([], [("64%", *self.O1)])
        self.tracker.assign_hp_from_ocr([], [("64%", *self.O1)])
        assert opp.hp is None  # %形式は2回では確定しない
        self.tracker.assign_hp_from_ocr([], [("64%", *self.O1)])
        assert opp.hp == "64%"

    def test_pct_needs_three_reads_against_digit_drop(self):
        """%形式の桁欠け誤読（72%→2%）が2回連続しても確定しない回帰ガード
        （実機 06-25-46: ガブリアス2%★ピンチがBedrockスナップショットに漏れた）。"""
        opp = FieldPokemon(name="ガブリアス", on_field=True, slot_index=0)
        self.tracker._opponent.append(opp)
        for _ in range(3):
            self.tracker.assign_hp_from_ocr([], [("72%", *self.O0)])
        assert opp.hp == "72%"
        self.tracker.assign_hp_from_ocr([], [("2%", *self.O0)])
        self.tracker.assign_hp_from_ocr([], [("2%", *self.O0)])
        assert opp.hp == "72%"  # 2回の誤読では上書きされない

    def test_single_digit_pct_rejected_when_known_hp_high(self):
        """1桁%はUI遮蔽の先頭桁欠け（72%→「2%」がconf1.0で14秒継続を実測）と
        区別がつかないため、既知HPが高いうちは何回一致しても受け付けない。"""
        opp = FieldPokemon(name="ガブリアス", on_field=True, slot_index=0)
        self.tracker._opponent.append(opp)
        for _ in range(3):
            self.tracker.assign_hp_from_ocr([], [("72%", *self.O0)])
        for _ in range(4):
            self.tracker.assign_hp_from_ocr([], [("2%", *self.O0)])
        assert opp.hp == "72%"

    def test_single_digit_pct_accepted_when_already_low(self):
        """既知HPがすでに低い（≤25%）場合、本物の低HP進行（13%→2%）は通る。"""
        opp = FieldPokemon(name="ユキメノコ", on_field=True, slot_index=0)
        self.tracker._opponent.append(opp)
        for _ in range(3):
            self.tracker.assign_hp_from_ocr([], [("13%", *self.O0)])
        assert opp.hp == "13%"
        for _ in range(3):
            self.tracker.assign_hp_from_ocr([], [("2%", *self.O0)])
        assert opp.hp == "2%"


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


# ═══════════════════════════════════════════════════════════════════════════════
# to_panel_state（実況動画の戦況パネル用スナップショット・v2b）
# ═══════════════════════════════════════════════════════════════════════════════

class TestToPanelState:

    def test_on_field_sorted_by_slot(self):
        """場のポケモンがスロット順で返る（気絶・控えは含まない）。"""
        tracker = BattleStateTracker()
        tracker._player.append(FieldPokemon(name="オオニューラ", on_field=True, slot_index=1))
        tracker._player.append(FieldPokemon(name="イダイトウ", on_field=True, slot_index=0))
        tracker._player.append(FieldPokemon(name="ペリッパー", on_field=False))
        tracker._opponent.append(FieldPokemon(name="リザードン", on_field=True, fainted=True))
        state = tracker.to_panel_state()
        assert [p["name"] for p in state["player"]] == ["イダイトウ", "オオニューラ"]
        assert state["opponent"] == []

    def test_hp_pct_prefers_fresher_source(self):
        """HPは_format_pokemonと同じ鮮度比較（pxが新しければpx・数値が新しければ数値）。"""
        tracker = BattleStateTracker()
        px_fresh = FieldPokemon(name="A", on_field=True, slot_index=0,
                                hp="100/200", hp_turn=1, hp_pct_pixel=0.47, hp_px_turn=2)
        num_fresh = FieldPokemon(name="B", on_field=True, slot_index=1,
                                 hp="7/201", hp_turn=3, hp_pct_pixel=0.47, hp_px_turn=2)
        tracker._player += [px_fresh, num_fresh]
        state = tracker.to_panel_state()
        assert state["player"][0]["hp_pct"] == 47
        assert state["player"][0]["hp_text"] == "47%"
        assert state["player"][1]["hp_pct"] == 3   # 7/201 ≒ 3.5% → round=3
        assert state["player"][1]["hp_text"] == "7/201"

    def test_alive_counts_fall_back_to_known(self):
        """ボール数未取得時は既知の非気絶数を残数にする。"""
        tracker = BattleStateTracker()
        tracker._player.append(FieldPokemon(name="A", on_field=True, slot_index=0))
        tracker._player.append(FieldPokemon(name="B", fainted=True))
        tracker._opponent.append(FieldPokemon(name="C", on_field=True, slot_index=0))
        state = tracker.to_panel_state()
        assert state["alive_player"] == 1
        assert state["alive_opponent"] == 1

    def test_status_included(self):
        tracker = BattleStateTracker()
        tracker._player.append(FieldPokemon(name="ペリッパー", on_field=True,
                                            slot_index=0, status="ねむり"))
        state = tracker.to_panel_state()
        assert state["player"][0]["status"] == "ねむり"
