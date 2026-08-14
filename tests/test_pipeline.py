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
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートを sys.path に追加（pytest がルートから実行されない場合の保険）
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# conftest.py で重いモジュールがモック済みなのでここで安全にインポートできる
from src.pipeline import (
    _build_bedrock_context,
    _check_end_screen_ocr,
    _clean_commentary,
    _detect_battle_result,
    _detect_glitch_cause,
    _detect_result_from_win_lose_ocr,
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

    def test_removes_emoji_block(self):
        # 字幕豆腐化バグ対策（U+1F300-1FAFF）。パス1検証で25本中83件と頻発判明→自動除去化
        text = "クレッフィが混乱するのもありかも💦♪"
        result = _clean_commentary(text)
        assert "💦" not in result
        assert "クレッフィ" in result

    def test_keeps_musical_note_and_heart(self):
        # ♪♡等（U+2600-27BF）はMeiryoにグリフがあるので除去対象外
        text = "いい勝負だったね♪えらいえらい♡"
        result = _clean_commentary(text)
        assert "♪" in result
        assert "♡" in result

    def test_removes_multiple_emoji(self):
        text = "やったね🎉🔥すごい試合だった！"
        result = _clean_commentary(text)
        assert "🎉" not in result
        assert "🔥" not in result

    def test_removes_code_fence(self):
        # パス1検証で発見（2026-08-12・2026-04-14_21-40-01 66.0s）: Phi-3が生コード片を
        # 出力してそのままVOICEVOXに渡っていた
        text = ("相手は何を打ってきたかしら？ 緊張感MAXだよ～   "
                "```python # Python側での処理例: move_used[（テキスト未検出）] "
                "の部分に、対戦相手の技の情報を記述 move_used[でんきショック / プテラ ] ```")
        result = _clean_commentary(text)
        assert "```" not in result
        assert "move_used" not in result
        assert "緊張感MAX" in result

    def test_removes_html_tags(self):
        # 同実例で"</span>"も一緒に混入していた（コードフェンス除去とは別経路の漏れ）
        text = "ロトムが出たから、マニューラで一気に攻め込むのもありだよね </td>"
        result = _clean_commentary(text)
        assert "</td>" not in result
        assert "マニューラ" in result

    def test_removes_various_html_tags(self):
        text = "威力も高そうで怖いよぉ～ </br> </br>"
        result = _clean_commentary(text)
        assert "<" not in result
        assert ">" not in result


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

    def test_ocr_fragments_joined_with_slash(self):
        """実際の_ocr_results_to_textは断片を" / "（スラッシュ）区切りで結合するため、
        スペース除去だけでは「勝負に」と「勝った!」の間に"/"が残り判定できなかった
        （実機07-03-23-34-29のocr_text「bennyとの / 勝負に / 勝った!」で再現・
        battle_result恒久未検出の真因）。"""
        assert _detect_battle_result("bennyとの / 勝負に / 勝った!") == "勝ち"

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


class TestCheckEndScreenOcr:
    """終了画面連続確認の複数フレーム分OCRを結合して判定する_check_end_screen_ocr。
    battle_result未検出バグ（07-03-23-34-29）は3回目確認フレーム1枚のOCRだけに
    依存していたため「勝負に勝った/負けた」がフェードイン中で拾えないと永久にNoneのままだった。
    複数フレームの結合判定でこれを緩和する。"""

    def test_keyword_and_result_in_last_frame_only(self):
        """1・2枚目はまだ文字が出ておらず、3枚目でようやく出揃うケース（フェードイン想定）。"""
        matched, result = _check_end_screen_ocr(["", "選ばれ", "勝負に勝った!"])
        assert matched is True
        assert result == "勝ち"

    def test_result_split_across_frames(self):
        """「勝負に」と「負けた!」が別フレームに分かれて出るケース。"""
        matched, result = _check_end_screen_ocr(["勝負に", "負けた!"])
        assert matched is True
        assert result == "負け"

    def test_no_keyword_in_any_frame_is_not_matched(self):
        matched, result = _check_end_screen_ocr(["", "", ""])
        assert matched is False
        assert result is None

    def test_keyword_matched_but_result_undetermined(self):
        """「選ばれました」（降参等）はキーワード一致するが勝敗は不明のままNone。"""
        matched, result = _check_end_screen_ocr(["降参が選ばれました"])
        assert matched is True
        assert result is None


class TestDetectResultFromWinLoseOcr:
    """降参終了時、「降参が選ばれました」の約10秒後に出るWIN/LOSEロゴ画面から
    勝敗を判定する_detect_result_from_win_lose_ocr（2026-08-12実機フレーム確認で追加）。
    自分は常に画面右半分に表示される仕様。"""

    FRAME_W = 1920

    def _bbox(self, cx: float):
        # 中心x座標cxの適当な矩形bboxを作る（幅100想定）
        return [[cx - 50, 500], [cx + 50, 500], [cx + 50, 560], [cx - 50, 560]]

    def test_win_on_right_means_self_wins(self):
        ocr = [{"text": "WIN", "bbox": self._bbox(1500), "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) == "勝ち"

    def test_win_on_left_means_self_loses(self):
        """自分（右）がLOSE側だと、WINは相手（左）に出る。"""
        ocr = [{"text": "WIN", "bbox": self._bbox(400), "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) == "負け"

    def test_lose_on_right_means_self_loses(self):
        ocr = [{"text": "LOSE", "bbox": self._bbox(1500), "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) == "負け"

    def test_lose_on_left_means_self_wins(self):
        ocr = [{"text": "LOSE", "bbox": self._bbox(400), "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) == "勝ち"

    def test_both_win_and_lose_present_uses_first_match(self):
        """通常は両方同時に出る（自分側と相手側）。どちらから見ても矛盾しない結果になる。"""
        ocr = [
            {"text": "LOSE", "bbox": self._bbox(400), "confidence": 0.9},
            {"text": "WIN", "bbox": self._bbox(1500), "confidence": 0.9},
        ]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) == "勝ち"

    def test_no_win_lose_text_returns_none(self):
        ocr = [{"text": "ヒシン", "bbox": self._bbox(400), "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) is None

    def test_empty_ocr_returns_none(self):
        assert _detect_result_from_win_lose_ocr([], self.FRAME_W) is None

    def test_missing_bbox_is_skipped(self):
        ocr = [{"text": "WIN", "bbox": None, "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) is None

    def test_case_insensitive(self):
        ocr = [{"text": "win", "bbox": self._bbox(1500), "confidence": 0.9}]
        assert _detect_result_from_win_lose_ocr(ocr, self.FRAME_W) == "勝ち"


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

    def test_instruction_echo_keywords(self):
        """2026-08-04発見: Bedrockが指示への「了解しました」を実況の代わりに返す事故（安全網）。"""
        assert _detect_glitch_cause(
            "了解しました！ 花圓くれぴとして、ポケモン対戦実況を担当させていただきます♪"
        ) == "指示書を読みすぎちゃった"
        assert _detect_glitch_cause("**性格・口調の確認:** 元気で甘えん坊") == "指示書を読みすぎちゃった"
        assert _detect_glitch_cause("**実況時の重要ルール:**") == "指示書を読みすぎちゃった"

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

    def test_move_detail_panel_skips_name_collection(self):
        """「能力」「ステータス」タブ（技の詳細・つよさの表示パネル）がある場合、
        名前候補収集をスキップする。

        実機（07-03-23-34-29）で、このパネルは自分の全ポケモン（場・控え問わず）を
        画面左側に縦一列で表示し、上位2匹（場のポケモン=ガオガエン・メタグロス）が
        y<_PLAYER_Y_THRESHOLD の「相手」帯に入り込んでいた。これにより本物の相手
        ポケモン（フシギバナ・リザードン）が「新顔が登場した」と誤判定されて
        即時evictされるバグが発生した（quick_threshold=1の直接証拠チェックへの
        誤入力）。状態確認パネルと同様に名前候補収集自体をスキップして対策する。
        """
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "pokemon"
        mock_result.canonical_ja = "ガオガエン"
        mock_clf.classify.return_value = mock_result

        results = [
            _ocr("能力", y_center=130.0),
            _ocr("ステータス", y_center=130.0),
            self._opponent_ocr("ガオガエン"),  # 本来は自分のポケモンだが相手帯に誤表示される
        ]
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "ガオガエン" not in info["name_candidates_opponent"]

    def test_move_detail_panel_tolerates_ocr_noise_in_tab_labels(self):
        """タブ文字列にOCRノイズ（前後の余分な文字）が混じっても検出できる。

        実機の再検証（2回目のパス1実行）で、この判定を完全一致
        （`"能力" in all_texts`という集合の要素一致）で実装したところ、ログ上は
        「能力」「ステータス」が明確に見えているのに検出されず誤evictが再現した。
        `is_status_panel`（"戦闘中" in t という部分一致方式）と同じ書き方に
        揃えることで、周辺文字混入に耐性を持たせて解決した。
        """
        mock_clf = MagicMock()
        mock_result = MagicMock()
        mock_result.category = "pokemon"
        mock_result.canonical_ja = "ガオガエン"
        mock_clf.classify.return_value = mock_result

        results = [
            _ocr("L能力", y_center=130.0),       # Lボタンアイコンのテキストが混入
            _ocr("ステータスR", y_center=130.0),  # Rボタンアイコンのテキストが混入
            self._opponent_ocr("ガオガエン"),
        ]
        info = _extract_structured_info(results, classifier=mock_clf)
        assert "ガオガエン" not in info["name_candidates_opponent"]

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

    def test_no_move_used_before_battle_started_on_communication_exit(self):
        """バトル開始前の「対戦準備中」画面で move_used が誤発火しない（2026-08-12修正）。

        「対戦準備中」画面の各プレイヤー準備完了ステータス「待機中」が単独OCR片として
        検出され、_COMM_RE の `^待機中$` 枝に一致して communication フェーズと誤判定
        →直後のポケモン入場演出への遷移で move_used が誤発火していた
        （実機フレーム＋実OCRで確認: 2026-04-13_06-34-11・2026-04-13_21-46-08で実証。
        パス1検証①の「ゴウカザーが地面技を」「バンギラスがいばるを連発」捏造NGの根本原因）。
        battle_start（初回command_select）より前はこの誤発火を起こさないことを確認する。
        """
        t = {"now": 0.0}
        clf = BattlePhaseClassifier(debounce_seconds=0.0, clock=lambda: t["now"])
        # battle_start前: 「対戦準備中」画面の「待機中」ラベル（communication誤判定）
        clf.detect(self._ocr_list("待機中"))
        t["now"] = 1.0
        clf.detect(self._ocr_list("待機中"))  # 連続0.7秒以上 → communication確定
        t["now"] = 5.0
        # ポケモン入場演出へ遷移（「待機中」表示が消える）
        event = clf.detect(self._ocr_list("ヒシン"))
        assert event is None  # battle_started=False なので move_used は発火しない

    def test_move_used_still_fires_after_battle_started_with_taikichu_alone(self):
        """battle_start後は「待機中」単独一致でも通常通りmove_usedが発火する（回帰防止）。"""
        t = {"now": 0.0}
        clf = BattlePhaseClassifier(debounce_seconds=0.0, clock=lambda: t["now"])
        clf.detect(self._ocr_list("たたかう"))    # battle_start
        t["now"] = 10.0
        clf.detect(self._ocr_list("待機中"))
        t["now"] = 11.0
        clf.detect(self._ocr_list("待機中"))      # 連続0.7秒以上 → communication確定
        t["now"] = 15.0
        event = clf.detect(self._ocr_list("バツグンだ"))
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

    def test_fainted_names_splits_by_side(self):
        gs1 = _make_game_state(player_names=["ピカチュウ"], opponent_names=["エルフーン"])
        _register(self.tracker, gs1)
        gs2 = _make_game_state(player_names=["ピカチュウ"], hp_player=["0/176"])
        self.tracker.update(gs2, "faint")
        player_fainted, opponent_fainted = self.tracker.fainted_names()
        assert player_fainted == {"ピカチュウ"}
        assert opponent_fainted == set()

    def test_diff_fainted_side_player(self):
        prev = (set(), set())
        curr = ({"ピカチュウ"}, set())
        assert BattleStateTracker.diff_fainted_side(prev, curr) == "player"

    def test_diff_fainted_side_opponent(self):
        prev = (set(), set())
        curr = (set(), {"エルフーン"})
        assert BattleStateTracker.diff_fainted_side(prev, curr) == "opponent"

    def test_diff_fainted_side_none_when_unchanged(self):
        prev = ({"ピカチュウ"}, set())
        curr = ({"ピカチュウ"}, set())
        assert BattleStateTracker.diff_fainted_side(prev, curr) is None

    def test_diff_fainted_side_none_when_both_sides_new(self):
        """同時ダウン等、両陣営同時に新規気絶がある場合は判定不能としてNone。"""
        prev = (set(), set())
        curr = ({"ピカチュウ"}, {"エルフーン"})
        assert BattleStateTracker.diff_fainted_side(prev, curr) is None

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
# 場のコンディション（天候・壁・トリックルーム・おいかぜ）2026-08-04新規
# ═══════════════════════════════════════════════════════════════════════════════

class TestTurnsLeft:
    def setup_method(self):
        self.tracker = BattleStateTracker()

    def test_none_start_turn_returns_zero(self):
        assert self.tracker._turns_left(None, 5) == 0

    def test_full_duration_at_start_turn(self):
        self.tracker.game_turn = 3
        assert self.tracker._turns_left(3, 5) == 5

    def test_decreases_as_turns_pass(self):
        self.tracker.game_turn = 6
        assert self.tracker._turns_left(3, 5) == 2

    def test_clamped_to_zero_when_expired(self):
        self.tracker.game_turn = 20
        assert self.tracker._turns_left(3, 5) == 0


class TestToContextConditions:
    def setup_method(self):
        self.tracker = BattleStateTracker()
        self.tracker.game_turn = 2

    def test_weather_included_while_active(self):
        self.tracker._weather = "あまごい"
        self.tracker._weather_start_turn = 2
        ctx = self.tracker.to_context()
        assert ctx["weather"] == "あまごい"
        assert ctx["weather_turns_left"] == 5

    def test_weather_omitted_when_expired(self):
        self.tracker._weather = "あまごい"
        self.tracker._weather_start_turn = 2
        self.tracker.game_turn = 20
        ctx = self.tracker.to_context()
        assert "weather" not in ctx

    def test_screens_included_while_active(self):
        # 残りターン数もctx["screens"]に持たせる（2026-08-07修正・以前は名前だけで
        # 「あと○ターン」が実況に出せなかった実装漏れがあった）。
        self.tracker._screens = {"player": ("リフレクター", 2)}
        ctx = self.tracker.to_context()
        assert ctx["screens"] == {"player": ("リフレクター", 5)}

    def test_expired_screen_excluded(self):
        self.tracker._screens = {"player": ("リフレクター", 2), "opponent": ("ひかりのかべ", -20)}
        ctx = self.tracker.to_context()
        assert ctx["screens"] == {"player": ("リフレクター", 5)}

    def test_trick_room_included_while_active(self):
        self.tracker._trick_room_start_turn = 2
        ctx = self.tracker.to_context()
        assert ctx["trick_room_turns_left"] == 5

    def test_tailwind_per_side(self):
        self.tracker._tailwind_start_turn = {"opponent": 2}
        ctx = self.tracker.to_context()
        assert ctx["tailwind"] == {"opponent": 4}

    def test_no_conditions_means_no_keys(self):
        ctx = self.tracker.to_context()
        for key in ("weather", "screens", "trick_room_turns_left", "tailwind"):
            assert key not in ctx


class TestConditionMessageSide:
    def test_opponent_prefix_detected(self):
        ocr = [{"text": "あいての リフレクターの効果で", "confidence": 0.9}]
        assert Pipeline._condition_message_side(ocr) == "opponent"

    def test_no_prefix_defaults_to_player(self):
        ocr = [{"text": "リフレクターの効果で", "confidence": 0.9}]
        assert Pipeline._condition_message_side(ocr) == "player"


class TestUpdateBattleConditions:
    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._battle_tracker.game_turn = 3

    def _ocr(self, *texts):
        return [{"text": t, "confidence": 0.9} for t in texts]

    def test_weather_detected_by_move_name(self):
        # 2026-08-07修正: 演出フレーバー文の推測ではなく技名/特性名そのものを直接
        # マッチする方式に変更（renders/2026-07-03-23-26-22の実機ログで、旧キーワード
        # 「ひざしが」「つよく」が実際の文言「ひざ日差しがつよ強くなった」に一致しない
        # バグを確認）。
        Pipeline._update_battle_conditions(self.runner, self._ocr("アシレーヌの", "あまごい！"))
        assert self.runner._battle_tracker._weather == "あまごい"
        assert self.runner._battle_tracker._weather_start_turn == 3

    def test_weather_detected_by_ability_name(self):
        # 技だけでなく特性発動（例: ペリッパーのあめふらし）でも同じキーワードで拾える
        # ことを確認する（天候は技/特性どちらでも発動メッセージが共通のため）。
        Pipeline._update_battle_conditions(self.runner, self._ocr("ペリッパーの", "あめふらし"))
        assert self.runner._battle_tracker._weather == "あまごい"
        assert self.runner._battle_tracker._weather_start_turn == 3

    def test_sunny_day_detected_by_move_or_ability(self):
        Pipeline._update_battle_conditions(self.runner, self._ocr("コータスの", "ひでり"))
        assert self.runner._battle_tracker._weather == "にほんばれ"

    def test_screen_detected_with_side(self):
        Pipeline._update_battle_conditions(
            self.runner, self._ocr("あいての", "リフレクターの効果で"))
        assert self.runner._battle_tracker._screens["opponent"] == ("リフレクター", 3)

    def test_screen_defaults_to_player_side(self):
        Pipeline._update_battle_conditions(self.runner, self._ocr("リフレクターの効果で"))
        assert self.runner._battle_tracker._screens["player"] == ("リフレクター", 3)

    def test_trick_room_detected(self):
        # 技名そのものを直接検出する方式（2026-08-07修正）。フレーバー文の推測
        # キーワードだと実際のゲーム文言と食い違い一度も検出できないバグがあった。
        Pipeline._update_battle_conditions(self.runner, self._ocr("バンギラスの", "トリックルーム！"))
        assert self.runner._battle_tracker._trick_room_start_turn == 3

    def test_tailwind_detected(self):
        # トリックルームと同様、技名そのものを直接検出する方式に統一（2026-08-07）。
        # おいかぜは現状特性発動がないため技名検出のみで足りる（ユーザー判断）。
        Pipeline._update_battle_conditions(self.runner, self._ocr("エルフーンの", "おいかぜ！"))
        assert self.runner._battle_tracker._tailwind_start_turn["player"] == 3

    def test_tailwind_detected_opponent_side(self):
        Pipeline._update_battle_conditions(
            self.runner, self._ocr("あいての", "エルフーンの", "おいかぜ！"))
        assert self.runner._battle_tracker._tailwind_start_turn["opponent"] == 3

    def test_no_match_leaves_state_untouched(self):
        Pipeline._update_battle_conditions(self.runner, self._ocr("メタグロスのアイアンヘッド"))
        assert self.runner._battle_tracker._weather is None
        assert self.runner._battle_tracker._screens == {}
        assert self.runner._battle_tracker._trick_room_start_turn is None


class TestComputeSpeedStageHint:
    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._move_log = []

    def _add(self, side, name, on_field=True):
        slots = self.runner._battle_tracker._player if side == "player" else self.runner._battle_tracker._opponent
        slots.append(FieldPokemon(name=name, on_field=on_field))

    def test_no_move_log_returns_none(self):
        assert self.runner._compute_speed_stage_hint() is None

    def test_speed_lowering_move_reported(self):
        self._add("player", "ペリッパー")
        self._add("opponent", "ドドゲザン")
        self.runner._move_log = ["T1:ペリッパーのこごえるかぜ"]
        hint = self.runner._compute_speed_stage_hint()
        assert hint == "ドドゲザンの素早さが1段階下がっている"

    def test_two_stage_move_reported(self):
        self._add("player", "ペリッパー")
        self._add("opponent", "ドドゲザン")
        self.runner._move_log = ["T1:ペリッパーのわたほうし"]
        hint = self.runner._compute_speed_stage_hint()
        assert hint == "ドドゲザンの素早さが2段階下がっている"

    def test_stacking_moves_accumulate(self):
        self._add("player", "ペリッパー")
        self._add("opponent", "ドドゲザン")
        self.runner._move_log = ["T1:ペリッパーのこごえるかぜ", "T2:ペリッパーのこごえるかぜ"]
        hint = self.runner._compute_speed_stage_hint()
        assert hint == "ドドゲザンの素早さが2段階下がっている"

    def test_clamped_to_six_stages(self):
        self._add("player", "ペリッパー")
        self._add("opponent", "ドドゲザン")
        self.runner._move_log = [f"T{i}:ペリッパーのわたほうし" for i in range(5)]
        hint = self.runner._compute_speed_stage_hint()
        assert hint == "ドドゲザンの素早さが6段階下がっている"

    def test_non_speed_move_ignored(self):
        self._add("player", "ペリッパー")
        self._add("opponent", "ドドゲザン")
        self.runner._move_log = ["T1:ペリッパーのウェザーボール"]
        assert self.runner._compute_speed_stage_hint() is None

    def test_unknown_user_side_ignored(self):
        self.runner._move_log = ["T1:謎のポケモンのこごえるかぜ"]
        assert self.runner._compute_speed_stage_hint() is None


class TestComputeConditionHint:
    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._move_log = []

    def test_empty_context_returns_none(self):
        assert self.runner._compute_condition_hint({}) is None

    def test_weather_line(self):
        hint = self.runner._compute_condition_hint(
            {"weather": "あまごい", "weather_turns_left": 4})
        assert hint == "あまごいが4ターン継続中"

    def test_screens_lines_per_side(self):
        # screensの値は(名前, 残りターン)のtuple（2026-08-07〜。残りターン数が
        # 表示されない実装漏れを修正・renders/07-03-23-34-29_condition_checkで確認）。
        hint = self.runner._compute_condition_hint(
            {"screens": {"player": ("リフレクター", 3), "opponent": ("ひかりのかべ", 2)}})
        assert "自分側にリフレクターが張られている（あと3ターン）" in hint
        assert "相手側にひかりのかべが張られている（あと2ターン）" in hint

    def test_trick_room_line(self):
        hint = self.runner._compute_condition_hint({"trick_room_turns_left": 3})
        assert "トリックルーム中（あと3ターン" in hint

    def test_tailwind_line(self):
        hint = self.runner._compute_condition_hint({"tailwind": {"player": 2}})
        assert "自分側におい風（あと2ターン" in hint

    def test_combines_with_speed_stage_hint(self):
        self.runner._battle_tracker._player.append(FieldPokemon(name="ペリッパー", on_field=True))
        self.runner._battle_tracker._opponent.append(FieldPokemon(name="ドドゲザン", on_field=True))
        self.runner._move_log = ["T1:ペリッパーのこごえるかぜ"]
        hint = self.runner._compute_condition_hint({"weather": "あまごい", "weather_turns_left": 4})
        assert "あまごいが4ターン継続中" in hint
        assert "ドドゲザンの素早さが1段階下がっている" in hint


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


class TestOpponentQuickEvictionRequiresReplacementEvidence:
    """相手側の「今フレームで名前が見えなければ即座に降ろす」（quick_threshold=1）の
    回帰ガード。ダブルバトルで2匹目の名前が単に1フレーム読めなかっただけで
    「交代」と誤判定して即時evictし、二度と復帰しなくなるバグ（オーロンゲ消失・
    実機07-03-23-34-29のstates.jsonl/timeline.jsonlで確認: オーロンゲがリフレクター
    使用直後、交代メッセージなしに同ターン中に消えたまま試合終了まで戻らなかった）
    の再発防止。「新しい名前が登場した」という直接証拠がある時だけ即時evictする。
    """

    def setup_method(self):
        self.tracker = BattleStateTracker()

    def test_doubles_partner_only_miss_does_not_evict(self):
        """ダブルバトルでパートナーの名前しか見えないフレームでは即座に降ろされない
        （新顔が登場していないため交代の直接証拠がない）。"""
        gs_both = _make_game_state(opponent_names=["アシレーヌ", "オーロンゲ"])
        _register(self.tracker, gs_both)
        oorondge = next(s for s in self.tracker._opponent if s.name == "オーロンゲ")
        assert oorondge.on_field is True

        gs_partner_only = _make_game_state(opponent_names=["アシレーヌ"])
        self.tracker.update(gs_partner_only, "move_used")
        assert oorondge.on_field is True

    def test_doubles_new_name_appearing_still_evicts_immediately(self):
        """新しい名前が登場した場合は本物の交代なので従来通り即座に降ろされる。"""
        gs_both = _make_game_state(opponent_names=["アシレーヌ", "オーロンゲ"])
        _register(self.tracker, gs_both)
        oorondge = next(s for s in self.tracker._opponent if s.name == "オーロンゲ")

        gs_switched = _make_game_state(opponent_names=["アシレーヌ", "フシギバナ"])
        self.tracker.update(gs_switched, "move_used")
        assert oorondge.on_field is False

    def test_singles_absence_still_evicts_immediately(self):
        """単体バトル（場に1匹だけ）では新顔登場と同義なので従来通り即座に降ろされる。"""
        gs = _make_game_state(opponent_names=["ドドゲザン"])
        _register(self.tracker, gs)
        mon = next(s for s in self.tracker._opponent if s.name == "ドドゲザン")

        gs_switched = _make_game_state(opponent_names=["ガブリアス"])
        self.tracker.update(gs_switched, "move_used")
        assert mon.on_field is False


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
            def __init__(self, category, score, canonical_ja, confident=True):
                self.category = category
                self.score = score
                self.canonical_ja = canonical_ja
                self.confident = confident

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
        """既知ロスター「ドドゲザン」への見切れ断片「ドゲザン」はロスター名に補正して登録される。

        ただし断片一致幽霊技対策（2026-07-30）により、ロスター名前方一致救済を経由した
        登録は tentative（「推定」）扱いになる——「ドドゲザンのドゲザン」は個別には
        正当に見えても、OCR断片が偶然どちらも実在名として通ってしまった可能性を
        否定できないため。
        """
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ドドゲザン", on_field=True)
        )
        self._set_classifier(moves={"ドゲザン"}, pokemon={})
        events = [_ocr("ドゲザンの", y_center=800.0), _ocr("ドゲザン", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ドドゲザンのドゲザン"]
        assert len(self.runner._tentative_opponent_moves) == 1
        assert self.runner._tentative_opponent_moves[0]["fallback_pokemon"] == "ドドゲザン"

    def test_direct_match_with_unlearnable_move_becomes_tentative(self):
        """ロスター前方一致救済を経由せず直接分類できた場合でも、そのポケモンが
        学習できない技だと判明したら tentative 扱いにする（断片一致幽霊技対策の
        もう一方の柱: pokemon_moves による学習可能技チェック）。"""
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ガブリアス", on_field=True)
        )
        self._set_classifier(moves={"ハイドロポンプ"}, pokemon={"ガブリアス": "ガブリアス"})
        self.runner._classifier.is_move_learnable.return_value = False
        events = [_ocr("ガブリアスの", y_center=800.0), _ocr("ハイドロポンプ", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのハイドロポンプ"]
        assert len(self.runner._tentative_opponent_moves) == 1

    def test_direct_match_with_learnable_move_is_not_tentative(self):
        """直接分類できて学習可能技リストにも合致する通常ケースは tentative 扱いにしない
        （回帰ガード: 断片一致幽霊技対策が普通の登録まで過剰にtentative化しないこと）。"""
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ガブリアス", on_field=True)
        )
        self._set_classifier(moves={"じだんだ"}, pokemon={"ガブリアス": "ガブリアス"})
        self.runner._classifier.is_move_learnable.return_value = True
        events = [_ocr("ガブリアスの", y_center=800.0), _ocr("じだんだ", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのじだんだ"]
        assert self.runner._tentative_opponent_moves == []

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

    def test_false_positive_opponent_flag_corrected_for_known_player_pokemon(self):
        """2026-08-14: 陣営判定クロスチェック。「相手」のOCR手がかり（is_opponent=True）
        があっても、解決済みポケモン名が自分ロスターにのみ登録済み（相手ロスターには
        居ない）と確定している場合はOCR誤検出と判断して自分側に補正する。

        実機由来のNG（2026-07-03_23-48-45）: is_opponent判定が直前OCRトークンの
        「相手/あいて」文字列一致だけの弱いロジックのため、自分のポケモンの技が
        「相手の技」として実況される陣営逆転と、相手ロスターへの誤登録が同時に
        発生していた（is_opponent=Trueで即座にregister_opponent_on_fieldするため）。
        """
        self.runner._battle_tracker._player.append(
            FieldPokemon(name="ガブリアス", on_field=True)
        )
        self._set_classifier(moves={"じだんだ"}, pokemon={"ガブリアス": "ガブリアス"})
        events = [
            _ocr("あいて", y_center=800.0),
            _ocr("相手の", y_center=800.0),
            _ocr("ガブリアスの", y_center=800.0),
            _ocr("じだんだ!", y_center=800.0),
        ]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのじだんだ"]
        # 相手ロスターへ誤登録されない（回帰ガード: 陣営逆転バグの副作用）
        assert self.runner._battle_tracker._opponent == []
        # 陣営判定は自分のまま（補正が効いている）
        assert self.runner._battle_tracker.move_user_side("ガブリアス") == "自分"

    def test_mirror_name_in_both_rosters_not_corrected(self):
        """同名ミラー戦（両陣営に存在）の場合は既存のis_opponentヒューリスティックを
        尊重し、クロスチェック補正は発動しない（move_user_side側のNone判定に委ねる
        設計を変えない）。"""
        self.runner._battle_tracker._player.append(
            FieldPokemon(name="ガブリアス", on_field=True))
        self.runner._battle_tracker._opponent.append(
            FieldPokemon(name="ガブリアス", on_field=True))
        self._set_classifier(moves={"じだんだ"}, pokemon={"ガブリアス": "ガブリアス"})
        events = [
            _ocr("あいて", y_center=800.0),
            _ocr("相手の", y_center=800.0),
            _ocr("ガブリアスの", y_center=800.0),
            _ocr("じだんだ!", y_center=800.0),
        ]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        assert self.runner._move_log == ["T0:ガブリアスのじだんだ"]
        # 両陣営とも1匹のまま（誤って重複登録されない）
        assert len(self.runner._battle_tracker._opponent) == 1
        assert len(self.runner._battle_tracker._player) == 1


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
            def __init__(self, category, score, canonical_ja, confident=True):
                self.category = category
                self.score = score
                self.canonical_ja = canonical_ja
                self.confident = confident

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


class TestMoveSingleDispatch:
    """技ごとの実況（move_single）: _update_move_log が技エントリを確定登録した瞬間に
    _dispatch_commentary を event_type="move_single" で呼ぶことの回帰ガード。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._tentative_opponent_moves = []
        self.runner._battle_active = True
        self.runner._dense_scan_start_turn = None
        self.runner._move_log = []
        self.runner._move_effectiveness = {}
        self.runner._MAX_MOVE_LOG = 8
        self.runner._dense_scan_remaining = 0
        self.runner._last_full_ocr_results = []
        self.runner._video_now = 123.0
        self.runner._ec2_url = "http://fake-ec2:5000"
        self.runner._dispatch_commentary = MagicMock()

    def _set_classifier(self, moves, pokemon, learnable=True):
        class _Result:
            def __init__(self, category, score, canonical_ja, confident=True):
                self.category = category
                self.score = score
                self.canonical_ja = canonical_ja
                self.confident = confident

        def classify(text):
            if text in moves:
                return _Result("move", 95, text)
            if text in pokemon:
                return _Result("pokemon", 95, pokemon[text])
            return _Result("other", 0, text)

        clf = MagicMock()
        clf.classify.side_effect = classify
        clf.is_move_learnable.return_value = learnable
        clf.get_move_effect.return_value = None
        self.runner._classifier = clf

    def test_confirmed_move_dispatches_move_single(self):
        self.runner._battle_tracker._player.append(FieldPokemon(name="ガブリアス", on_field=True))
        self._set_classifier(moves={"じしん"}, pokemon={"ガブリアス": "ガブリアス"})
        events = [_ocr("ガブリアスの", y_center=800.0), _ocr("じしん!", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)

        assert self.runner._move_log == ["T0:ガブリアスのじしん"]
        self.runner._dispatch_commentary.assert_called_once()
        args, kwargs = self.runner._dispatch_commentary.call_args
        event_type, frame, game_state, battle_context, move_log, attempt_bedrock = args
        assert event_type == "move_single"
        assert game_state["move_focus"] == "自分のガブリアスのじしん"
        assert attempt_bedrock is True
        assert kwargs["event_time"] == 123.0

    def test_move_effect_hint_propagated_to_battle_context(self):
        """2026-08-14: move_single専用のdispatch経路（_process_eventを経由しない）でも
        技効果ヒントRAGがbattle_contextに配線されること（type_hintと同じ2箇所目の
        注入経路の回帰ガード）。"""
        self.runner._battle_tracker._player.append(FieldPokemon(name="フシギバナ", on_field=True))
        self._set_classifier(moves={"おいかぜ"}, pokemon={"フシギバナ": "フシギバナ"})
        self.runner._classifier.get_move_effect.side_effect = (
            lambda name: "味方全員の素早さをあげる。" if name == "おいかぜ" else None)
        events = [_ocr("フシギバナの", y_center=800.0), _ocr("おいかぜ!", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)

        _, _, _, battle_context, _, _ = self.runner._dispatch_commentary.call_args.args
        assert battle_context.get("move_effect_hint") == "おいかぜ: 味方全員の素早さをあげる。"

    def test_tentative_move_still_dispatches(self):
        """断片一致救済等でtentative扱いになった技も実況の対象にする（ユーザー決定）。"""
        self.runner._battle_tracker._opponent.append(FieldPokemon(name="ドドゲザン", on_field=True))
        self._set_classifier(moves={"ドゲザン"}, pokemon={})
        events = [_ocr("ドゲザンの", y_center=800.0), _ocr("ドゲザン", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)

        assert len(self.runner._tentative_opponent_moves) == 1
        self.runner._dispatch_commentary.assert_called_once()
        _, _, game_state, _, _, _ = self.runner._dispatch_commentary.call_args.args
        assert game_state["move_focus"] == "相手のドドゲザンのドゲザン"

    def test_duplicate_entry_does_not_redispatch(self):
        """直近3件と同一エントリは_move_log自体に追加されないため、実況も再ディスパッチされない。"""
        self.runner._battle_tracker._player.append(FieldPokemon(name="ガブリアス", on_field=True))
        self._set_classifier(moves={"じしん"}, pokemon={"ガブリアス": "ガブリアス"})
        events = [_ocr("ガブリアスの", y_center=800.0), _ocr("じしん!", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)

        assert self.runner._move_log == ["T0:ガブリアスのじしん"]
        self.runner._dispatch_commentary.assert_called_once()

    def test_partial_pipeline_without_ec2_url_does_not_raise(self):
        """_ec2_url未設定（他の既存テストの部分構築Pipeline）では何もせず早期returnする。"""
        del self.runner._ec2_url
        self.runner._battle_tracker._player.append(FieldPokemon(name="ガブリアス", on_field=True))
        self._set_classifier(moves={"じしん"}, pokemon={"ガブリアス": "ガブリアス"})
        events = [_ocr("ガブリアスの", y_center=800.0), _ocr("じしん!", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)  # raise しなければOK

        assert self.runner._move_log == ["T0:ガブリアスのじしん"]
        self.runner._dispatch_commentary.assert_not_called()

    def test_flushes_pending_battle_start_before_own_dispatch(self):
        """move_singleは_process_event経由のbattle_startとは別経路のため、保留中の
        battle_startがあれば自分の実況より先にflushする必要がある。

        2026-08-08発見: このflushが漏れていたため、保留中battle_startが
        move_singleでは解消されず、次に_process_eventを通るイベント（多くの場合
        battle_end）まで持ち越され、進行しきった戦況でbattle_start実況が
        生成される事故があった（実機2026-06-03 22-57-11）。
        """
        self.runner._battle_tracker._player.append(FieldPokemon(name="ガブリアス", on_field=True))
        self._set_classifier(moves={"じしん"}, pokemon={"ガブリアス": "ガブリアス"})
        self.runner._pending_battle_start_time = 41.0
        self.runner._pending_battle_start_frame = None
        self.runner._pending_battle_start_game_state = {"event_type": "battle_start"}
        self.runner._pending_battle_start_move_log = []
        self.runner._pending_battle_start_attempt_bedrock = False

        events = [_ocr("ガブリアスの", y_center=800.0), _ocr("じしん!", y_center=800.0)]
        Pipeline._update_move_log(self.runner, events, is_main_ocr=True)

        assert self.runner._dispatch_commentary.call_count == 2
        flush_call, own_call = self.runner._dispatch_commentary.call_args_list
        assert flush_call.args[0] == "battle_start"
        assert flush_call.kwargs["event_time"] == 41.0  # 検知時点のまま（flush実行時刻ではない）
        assert own_call.args[0] == "move_single"
        assert self.runner._pending_battle_start_time is None  # 保留状態はクリアされる


class TestTrackNewFaints:
    """_track_new_faints: 気絶実況の重複防止と合成実況対象の選別。

    通常のfaint実況はOCRの0%表示フレーム（_HP_ZERO_RE）でのみ発火するため、
    2Hzサンプリングから漏れると気絶が一度も実況されない（実機2026-06-07
    12-48-22のリキキリン）。faintイベントを経ずに確定した相手の気絶を
    「現在の気絶−実況済み」方式で拾い、合成実況の対象として返す。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._announced_faints = set()

    def test_faint_event_registers_both_sides_and_returns_empty(self):
        """OCRの0%表示由来の気絶は既存faint経路が実況するため、登録のみ行う。"""
        curr = ({"ガブリアス"}, {"リキキリン"})
        result = Pipeline._track_new_faints(self.runner, curr, "faint")
        assert result == []
        assert self.runner._announced_faints == {"ガブリアス", "リキキリン"}

    def test_non_faint_event_returns_unannounced_opponent_faints(self):
        """ボール数減少推定（update()内）で確定した相手の気絶を合成対象として返す。"""
        result = Pipeline._track_new_faints(
            self.runner, (set(), {"リキキリン"}), "turn_start")
        assert result == ["リキキリン"]

    def test_message_derived_faint_between_events_is_picked_up(self):
        """「たおれた」メッセージ由来（_apply_message_events経由）の気絶は
        _battle_tracker.update()の外＝イベント間に立つため、「更新前後のdiff」方式では
        次のイベント時点でprev/curr両方に含まれてしまい常に取りこぼしていた
        （実機2026-06-07 12-48-22の再検証1回目で発覚）。「現在の気絶−実況済み」方式なら
        どの経路で立った気絶でも次のイベント処理時に必ず検出される。"""
        # メッセージ由来でfaintedが立った後の、通常イベント処理時点の状態を再現
        # （prev相当のスナップショットにも既に含まれている状況）
        result = Pipeline._track_new_faints(
            self.runner, (set(), {"リキキリン"}), "move_used")
        assert result == ["リキキリン"]
        # 実況合成後（呼び出し側で登録）は二度と返さない
        self.runner._announced_faints.add("リキキリン")
        result = Pipeline._track_new_faints(
            self.runner, (set(), {"リキキリン"}), "battle_end")
        assert result == []

    def test_already_announced_not_returned(self):
        """faintイベントで実況済みのポケモンは合成対象にしない（二重言及防止）。"""
        self.runner._announced_faints = {"リキキリン"}
        result = Pipeline._track_new_faints(
            self.runner, (set(), {"リキキリン"}), "move_used")
        assert result == []

    def test_player_side_not_synthesized(self):
        """自分側の気絶は合成対象外（相手側のみ・スコープ決定済み）。"""
        result = Pipeline._track_new_faints(
            self.runner, ({"ガブリアス"}, set()), "turn_start")
        assert result == []

    def test_no_faints_returns_empty(self):
        result = Pipeline._track_new_faints(self.runner, (set(), set()), "move_used")
        assert result == []


class TestFaintInferredDispatch:
    """_dispatch_faint_inferred: ボール数減少推定で確定した気絶の合成実況イベント。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_active = True
        self.runner._ec2_url = "http://fake-ec2:5000"
        self.runner._move_log = []
        self.runner._move_effectiveness = {}
        self.runner._tentative_opponent_moves = []
        self.runner._video_now = 456.0
        self.runner._announced_faints = set()
        self.runner._dispatch_commentary = MagicMock()

    def test_dispatches_synthesized_faint_event(self):
        game_state = {"event_type": "turn_start", "ocr_text": "コマンド画面"}
        battle_context = {"player_field": "ガブリアス"}
        Pipeline._dispatch_faint_inferred(
            self.runner, ["リキキリン"], None, game_state, battle_context)

        self.runner._dispatch_commentary.assert_called_once()
        args, kwargs = self.runner._dispatch_commentary.call_args
        event_type, _frame, sent_state, sent_context, _move_log, attempt_bedrock = args
        assert event_type == "faint"
        assert sent_state["event_type"] == "faint"  # コピー元のturn_startを上書きすること
        assert sent_state["faint_focus"] == "相手のリキキリン"
        assert sent_context["faint_side"] == "opponent"  # 表情連動（manifest経由）用
        assert attempt_bedrock is True
        assert kwargs["event_time"] == 456.0

    def test_does_not_mutate_caller_dicts(self):
        """現行イベント（コピー元）のgame_state/battle_contextを汚さないこと。"""
        game_state = {"event_type": "turn_start"}
        battle_context = {"player_field": "ガブリアス"}
        Pipeline._dispatch_faint_inferred(
            self.runner, ["リキキリン"], None, game_state, battle_context)
        assert game_state == {"event_type": "turn_start"}
        assert battle_context == {"player_field": "ガブリアス"}

    def test_multiple_names_joined(self):
        """2匹同時倒れ（ボールが一段階しか減らないケースの後追い確定含む）。"""
        Pipeline._dispatch_faint_inferred(
            self.runner, ["ヘイラッシャ", "リキキリン"], None,
            {"event_type": "turn_start"}, {})
        sent_state = self.runner._dispatch_commentary.call_args.args[2]
        assert sent_state["faint_focus"] == "相手のヘイラッシャとリキキリン"

    def test_partial_pipeline_without_ec2_url_does_not_raise(self):
        """_ec2_url未設定（他の既存テストの部分構築Pipeline）では何もせず早期returnする。"""
        del self.runner._ec2_url
        Pipeline._dispatch_faint_inferred(
            self.runner, ["リキキリン"], None, {"event_type": "turn_start"}, {})
        self.runner._dispatch_commentary.assert_not_called()

    def test_not_battle_active_early_returns(self):
        self.runner._battle_active = False
        Pipeline._dispatch_faint_inferred(
            self.runner, ["リキキリン"], None, {"event_type": "turn_start"}, {})
        self.runner._dispatch_commentary.assert_not_called()


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
        runner._end_screen_ocr_texts = ["選ばれ"]
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
        assert runner._end_screen_ocr_texts == []
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
        self.runner._move_effectiveness = {}

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

    def test_effectiveness_tag_shown(self):
        """2026-08-04: _update_move_effectivenessが記録した効果タグの表示。"""
        self.runner._tentative_opponent_moves = []
        self.runner._move_effectiveness = {"T2:リキキリンのけたぐり": "バツグン"}
        assert self.runner._move_log_display(5) == [
            "T1:オオニューラのわるだくみ", "T2:リキキリンのけたぐり（バツグン）",
        ]

    def test_effectiveness_and_tentative_tags_combine(self):
        self.runner._tentative_opponent_moves = [
            {"old_entry": "T2:リキキリンのけたぐり", "move_name": "けたぐり",
             "turn_label": "2", "fallback_pokemon": "リキキリン"},
        ]
        self.runner._move_effectiveness = {"T2:リキキリンのけたぐり": "バツグン"}
        assert self.runner._move_log_display(5) == [
            "T1:オオニューラのわるだくみ", "T2:リキキリンのけたぐり（バツグン）（推定）",
        ]


class TestUpdateMoveEffectiveness:
    """_update_move_effectiveness: OCRの効果テキストを直近の技ログエントリに紐付ける
    （改善ロードマップ・戦況推論強化 2026-08-04）。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._move_log = ["T1:メタグロスのアイアンヘッド"]
        self.runner._move_effectiveness = {}

    def test_batsugun_tags_latest_entry(self):
        ocr = [{"text": "バツグンだ！", "confidence": 0.9}]
        Pipeline._update_move_effectiveness(self.runner, ocr)
        assert self.runner._move_effectiveness == {"T1:メタグロスのアイアンヘッド": "バツグン"}

    def test_no_keyword_leaves_untouched(self):
        ocr = [{"text": "メタグロスのアイアンヘッド", "confidence": 0.9}]
        Pipeline._update_move_effectiveness(self.runner, ocr)
        assert self.runner._move_effectiveness == {}

    def test_empty_move_log_is_noop(self):
        self.runner._move_log = []
        ocr = [{"text": "バツグンだ！", "confidence": 0.9}]
        Pipeline._update_move_effectiveness(self.runner, ocr)
        assert self.runner._move_effectiveness == {}

    def test_imamahitotsu_not_tagged_yet(self):
        """いまひとつ/こうかなしは技選択UIにも出るため、Phase 1では対象外
        （_EFFECTIVENESS_TAGSにバツグンだのみ収録）。誤タグ付けを防ぐための意図的な仕様。"""
        ocr = [{"text": "いまひとつのようだ", "confidence": 0.9}]
        Pipeline._update_move_effectiveness(self.runner, ocr)
        assert self.runner._move_effectiveness == {}

    def test_second_move_tags_new_latest_entry_independently(self):
        ocr1 = [{"text": "バツグンだ！", "confidence": 0.9}]
        Pipeline._update_move_effectiveness(self.runner, ocr1)
        self.runner._move_log.append("T2:ペリッパーのウェザーボール")
        ocr2 = [{"text": "なにも起きなかった", "confidence": 0.9}]
        Pipeline._update_move_effectiveness(self.runner, ocr2)
        assert self.runner._move_effectiveness == {"T1:メタグロスのアイアンヘッド": "バツグン"}


class TestComputeTypeHint:
    """_compute_type_hint: 場のポケモンのタイプ相性ヒントを計算する
    （Cicero型アーキテクチャ・戦況推論強化 2026-08-04）。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._move_log = []
        self._types = {}
        self._move_types = {}
        self._move_categories = {}
        clf = MagicMock()
        clf.get_pokemon_types.side_effect = lambda name: self._types.get(name)
        clf.get_move_type.side_effect = lambda name: self._move_types.get(name)
        # 既定は物理/特殊扱い（変化技でないもの）。変化技を明示的にテストする箇所だけ
        # self._move_categories["技名"] = "変化" を設定する。
        clf.get_move_category.side_effect = lambda name: self._move_categories.get(name, "物理")
        self.runner._classifier = clf

    def _add(self, side, name, types, on_field=True):
        slots = self.runner._battle_tracker._player if side == "player" else self.runner._battle_tracker._opponent
        slots.append(FieldPokemon(name=name, on_field=on_field))
        self._types[name] = types

    def test_no_classifier_returns_none(self):
        self.runner._classifier = None
        assert self.runner._compute_type_hint() is None

    def test_no_on_field_pokemon_returns_none(self):
        assert self.runner._compute_type_hint() is None

    def test_super_effective_matchup_reported(self):
        # はがね技はいわに2倍（バツグン）
        self._add("player", "メタグロス", ["はがね", "エスパー"])
        self._add("opponent", "イワーク", ["いわ"])
        hint = self.runner._compute_type_hint()
        assert "メタグロスの技はイワークにバツグン" in hint

    def test_neutral_matchup_omitted(self):
        self._add("player", "ピカチュウ", ["でんき"])
        self._add("opponent", "カビゴン", ["ノーマル"])
        assert self.runner._compute_type_hint() is None

    def test_opponent_side_matchup_also_included(self):
        """自分→相手だけでなく相手→自分方向の相性（脅威）も返す。"""
        self._add("player", "コータス", ["ほのお"])
        self._add("opponent", "ペリッパー", ["みず", "ひこう"])
        hint = self.runner._compute_type_hint()
        assert "ペリッパーの技はコータスにバツグン" in hint

    def test_fainted_pokemon_excluded(self):
        self.runner._battle_tracker._player.append(
            FieldPokemon(name="メタグロス", on_field=True, fainted=True))
        self._types["メタグロス"] = ["はがね"]
        self._add("opponent", "コータス", ["ほのお"])
        assert self.runner._compute_type_hint() is None

    def test_unknown_pokemon_type_skipped_gracefully(self):
        self._add("player", "図鑑に無いポケモン", [])  # get_pokemon_typesがNoneを返す想定
        self._types["図鑑に無いポケモン"] = None
        self._add("opponent", "コータス", ["ほのお"])
        assert self.runner._compute_type_hint() is None

    def test_lines_capped_at_four(self):
        for i in range(3):
            self._add("player", f"炎{i}", ["ほのお"])
        for i in range(3):
            self._add("opponent", f"草{i}", ["くさ"])
        hint = self.runner._compute_type_hint()
        assert hint is not None
        assert len(hint.split(" / ")) <= 4

    def test_covering_move_type_prioritized_over_own_type_guess(self):
        """2026-08-04実機で発見: メタグロス(はがね/エスパー)のじだんだ（じめん技）が
        ドドゲザン（あく/はがね）にバツグンのはずが、メタグロス自身のタイプ基準の
        ヒントしか無くLLMが「いまひとつ」と誤答した実例の再発防止。
        実際に使われた技（move_log）のタイプを優先して先頭に出す。"""
        self._add("player", "メタグロス", ["はがね", "エスパー"])
        self._add("opponent", "ドドゲザン", ["あく", "はがね"])
        self.runner._move_log = ["T3:メタグロスのじだんだ"]
        self._move_types["じだんだ"] = "じめん"
        hint = self.runner._compute_type_hint()
        assert hint.startswith("（実際に使った）メタグロスのじだんだはドドゲザンにバツグン")

    def test_covering_move_hint_omitted_when_move_type_unknown(self):
        self._add("player", "メタグロス", ["はがね", "エスパー"])
        self._add("opponent", "コータス", ["ほのお"])
        self.runner._move_log = ["T3:メタグロスの謎の技"]
        # self._move_types に登録しない → get_move_typeがNoneを返す想定
        hint = self.runner._compute_type_hint()
        assert "実際に使った" not in hint

    def test_status_move_omitted_from_type_hint(self):
        """2026-08-07発見: 変化技（リフレクター等）はダメージを与えないため
        タイプ相性という概念自体が無意味なのに、判定なしで計算すると「フシギバナの
        リフレクターはメタグロスに4分の1」のような意味不明な文言をBedrockに渡して
        しまい、壁が弱まった/消えたと誤解釈するハルシネーションを誘発していた
        （renders/07-03-23-34-29_condition_check_fixの実機検証で確認）。
        変化技は_latest_move_type_hintから除外することを確認する。"""
        self._add("player", "メタグロス", ["はがね", "エスパー"])
        self._add("opponent", "フシギバナ", ["くさ", "どく"])
        self.runner._move_log = ["T2:フシギバナのリフレクター"]
        self._move_types["リフレクター"] = "エスパー"
        self._move_categories["リフレクター"] = "変化"
        hint = self.runner._compute_type_hint()
        assert hint is None or "実際に使った" not in hint

    def test_covering_move_hint_omitted_when_no_move_log(self):
        self._add("player", "メタグロス", ["はがね", "エスパー"])
        self._add("opponent", "イワーク", ["いわ"])
        hint = self.runner._compute_type_hint()
        assert "実際に使った" not in hint
        assert "メタグロスの技はイワークにバツグン" in hint

    def test_weather_ball_type_overridden_by_weather(self):
        """2026-08-08発見: ウェザーボールは天候で技タイプが変わるが、天候情報
        （condition_hint）を渡すだけではLLMが自力で結びつけてくれず「水技」等と
        誤答していた（renders/2026-06-07_12-48-22実機検証）。DBのベース値
        （ノーマル）ではなく、Python側で天候から確定計算したタイプを使う。"""
        self._add("player", "ペリッパー", ["みず", "ひこう"])
        self._add("opponent", "フシギバナ", ["くさ", "どく"])
        self.runner._battle_tracker._weather = "にほんばれ"
        self.runner._move_log = ["T1:ペリッパーのウェザーボール"]
        self._move_types["ウェザーボール"] = "ノーマル"  # DBのベース値（無天候時）
        hint = self.runner._compute_type_hint()
        assert "天候「にほんばれ」によりウェザーボールはほのおタイプになっている" in hint
        assert "フシギバナにバツグン" in hint  # ほのお技はくさにバツグン

    def test_weather_ball_type_disclosed_even_on_neutral_matchup(self):
        """タイプ相性が「等倍」で相性行が出ない場合でも、天候で変わった
        技タイプ自体は必ず伝える（省略すると誤ったタイプのまま実況されるため）。"""
        self._add("player", "ペリッパー", ["みず", "ひこう"])
        self._add("opponent", "カビゴン", ["ノーマル"])  # ほのおはノーマルに等倍
        self.runner._battle_tracker._weather = "にほんばれ"
        self.runner._move_log = ["T1:ペリッパーのウェザーボール"]
        self._move_types["ウェザーボール"] = "ノーマル"
        hint = self.runner._compute_type_hint()
        assert "天候「にほんばれ」によりウェザーボールはほのおタイプになっている" in hint

    def test_weather_ball_uses_db_type_without_weather(self):
        """無天候時はDBのベース値（ノーマル）のまま・上書きしない。"""
        self._add("player", "ペリッパー", ["みず", "ひこう"])
        self._add("opponent", "リキキリン", ["でんき"])
        self.runner._battle_tracker._weather = None
        self.runner._move_log = ["T1:ペリッパーのウェザーボール"]
        self._move_types["ウェザーボール"] = "ノーマル"
        hint = self.runner._compute_type_hint()
        assert "天候" not in (hint or "")

    def test_covering_move_hint_neutral_falls_back_to_own_type_lines(self):
        """実際の技が等倍だった場合は（情報量が無いので）own-type由来の相性行を使う。"""
        self._add("player", "メタグロス", ["はがね", "エスパー"])
        self._add("opponent", "イワーク", ["いわ"])
        self.runner._move_log = ["T3:メタグロスの10まんボルト"]
        self._move_types["10まんボルト"] = "でんき"  # イワークに等倍（でんきはいわに特に効果なし）
        hint = self.runner._compute_type_hint()
        assert "実際に使った" not in hint
        assert "メタグロスの技はイワークにバツグン" in hint


class TestLatestMoveEffectHint:
    """_latest_move_effect_hint: 技効果ヒントRAG新設（2026-08-14）。パス1検証で
    累計最頻のNGパターンだった「技の効果に関する事実誤認」（おいかぜ等の変化技を
    ダメージ技として説明する等）対策。_latest_move_type_hintと同じ「直近の技ログ
    1件だけを見る」設計だが、変化技も対象に含む点が異なる。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._move_log = []
        self._move_effects = {}
        self.clf = MagicMock()
        self.clf.get_move_effect.side_effect = lambda name: self._move_effects.get(name)

    def test_no_move_log_returns_none(self):
        assert self.runner._latest_move_effect_hint(self.clf) is None

    def test_known_move_returns_effect_text(self):
        self.runner._move_log = ["T1:フシギバナのおいかぜ"]
        self._move_effects["おいかぜ"] = "味方全員の素早さをあげる。"
        hint = self.runner._latest_move_effect_hint(self.clf)
        assert hint == "おいかぜ: 味方全員の素早さをあげる。"

    def test_unknown_effect_returns_none(self):
        """effectがDBに無い（backfill未取得）技はNone（type_hint同様、単に
        ヒントが出ないだけで実害なし）。"""
        self.runner._move_log = ["T1:フシギバナのつるのムチ"]
        hint = self.runner._latest_move_effect_hint(self.clf)
        assert hint is None

    def test_uses_latest_entry_only(self):
        """直近1件だけを見る（古いターンの技には反応しない）。"""
        self.runner._move_log = ["T1:フシギバナのおいかぜ", "T2:メタグロスのじだんだ"]
        self._move_effects["おいかぜ"] = "味方全員の素早さをあげる。"
        self._move_effects["じだんだ"] = "地面を思いきり踏みつける。"
        hint = self.runner._latest_move_effect_hint(self.clf)
        assert hint == "じだんだ: 地面を思いきり踏みつける。"

    def test_malformed_entry_returns_none(self):
        self.runner._move_log = ["これは形式に合わない文字列"]
        assert self.runner._latest_move_effect_hint(self.clf) is None


class TestUpdateMegaEvolution:
    """_update_mega_evolution: 「〜はメガ〜にメガシンカした!」検出でmega_evolvedフラグを
    立てる（改善ロードマップ「戦況推論強化」続き・2026-08-04）。

    ⚠️旧パターン「(ポケモン)の メガシンカ」は推測に留まり実機OCRの実文言と食い違って
    一度も発火していなかったバグを2026-08-07に修正（renders/18-12-45_condition_checkの
    実機ログで確認した実文言に合わせ「は」〜「に」構造へ変更）。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._battle_tracker._player.append(FieldPokemon(name="リザードン", on_field=True))
        self.runner._battle_tracker._opponent.append(FieldPokemon(name="ミュウツー", on_field=True))

    def _ocr(self, *texts):
        return [{"text": t, "confidence": 0.9} for t in texts]

    def test_mega_evolution_flag_set_on_match(self):
        self.runner._update_mega_evolution(
            self._ocr("リザードンは", "メガリザードンに", "メガシンカした！"))
        slot = self.runner._battle_tracker._player[0]
        assert slot.mega_evolved is True

    def test_opponent_side_also_matched(self):
        self.runner._update_mega_evolution(
            self._ocr("ミュウツーは", "メガミュウツーに", "メガシンカした！"))
        slot = self.runner._battle_tracker._opponent[0]
        assert slot.mega_evolved is True

    def test_no_match_leaves_flag_false(self):
        self.runner._update_mega_evolution(self._ocr("リザードンのかえんほうしゃ"))
        assert self.runner._battle_tracker._player[0].mega_evolved is False

    def test_unknown_pokemon_does_not_crash(self):
        self.runner._update_mega_evolution(
            self._ocr("謎のポケモンは", "メガ謎のポケモンに", "メガシンカした！"))
        assert self.runner._battle_tracker._player[0].mega_evolved is False


class TestEffectivePokemonTypes:
    """_effective_pokemon_types: メガシンカ後のタイプ上書き（2026-08-04新規）。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.clf = MagicMock()
        self.clf.get_pokemon_types.side_effect = lambda name: {
            "リザードン": ["ほのお", "ひこう"], "ピカチュウ": ["でんき"],
        }.get(name)

    def test_non_mega_uses_normal_types(self):
        p = FieldPokemon(name="リザードン", on_field=True, mega_evolved=False)
        assert self.runner._effective_pokemon_types(p, self.clf) == ["ほのお", "ひこう"]

    def test_mega_with_override_uses_mega_types(self):
        p = FieldPokemon(name="リザードン", on_field=True, mega_evolved=True)
        assert self.runner._effective_pokemon_types(p, self.clf) == ["ほのお", "ドラゴン"]

    def test_mega_without_override_falls_back_to_normal(self):
        """メガシンカしてもタイプ変化の登録が無い種（例: ピカチュウは実際はメガ進化不可だが
        テスト用に仮定）は通常タイプのまま。"""
        p = FieldPokemon(name="ピカチュウ", on_field=True, mega_evolved=True)
        assert self.runner._effective_pokemon_types(p, self.clf) == ["でんき"]


class TestTypeHintUsesMegaEvolution:
    """_compute_type_hint: メガシンカ後はタイプ相性計算にも反映される（統合テスト）。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._battle_tracker = BattleStateTracker()
        self.runner._move_log = []
        self._types = {"リザードン": ["ほのお", "ひこう"], "オンバーン": ["ひこう", "ドラゴン"]}
        clf = MagicMock()
        clf.get_pokemon_types.side_effect = lambda name: self._types.get(name)
        clf.get_move_type.side_effect = lambda name: None
        self.runner._classifier = clf

    def test_mega_charizard_x_becomes_weak_to_dragon(self):
        # メガリザードンX（ほのお/ドラゴン）はオンバーン（ひこう/ドラゴン）のドラゴン技にバツグン
        self.runner._battle_tracker._player.append(
            FieldPokemon(name="リザードン", on_field=True, mega_evolved=True))
        self.runner._battle_tracker._opponent.append(FieldPokemon(name="オンバーン", on_field=True))
        hint = self.runner._compute_type_hint()
        assert "オンバーンの技はリザードンにバツグン" in hint


class TestRecordSituationSnapshot:
    """_record_situation_snapshot: データウェアハウスの箱への記録（2026-08-04新規）。
    記録失敗が実況生成を止めないこと・match_id無しでは記録しないことを確認する。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)

    def test_no_match_id_skips_recording(self):
        with patch("src.pipeline.record_situation") as mock_record:
            self.runner._record_situation_snapshot(None, {"event_type": "move_used"})
        mock_record.assert_not_called()

    def test_records_with_expected_fields(self):
        # game_state の hp_player_by_slot/hp_opponent_by_slot は実際には list[str]
        # （例: ["87%", "45%"]）。2026-08-06実機検証で、旧実装が生の hp_values
        # （list・自他混在）をそのまま渡していたため sqlite3 が
        # 「type 'list' is not supported」で全件記録失敗していたことが発覚（修正済み）。
        ev = {
            "event_time": 12.3,
            "event_type": "move_used",
            "battle_context": {
                "turn": 2, "player_pokemon": "場: A", "opponent_pokemon": "場: B",
                "type_hint": "Aの技はBにバツグン",
            },
            "game_state": {
                "hp_player_by_slot": ["150/200"],
                "hp_opponent_by_slot": ["80/120", "60/90"],
            },
        }
        with patch("src.pipeline.record_situation") as mock_record:
            self.runner._record_situation_snapshot("match1", ev)
        mock_record.assert_called_once()
        snapshot = mock_record.call_args[0][0]
        assert snapshot["match_id"] == "match1"
        assert snapshot["turn"] == 2
        assert snapshot["type_hint"] == "Aの技はBにバツグン"
        assert snapshot["hp_player"] == "150/200"
        assert snapshot["hp_opponent"] == "80/120 / 60/90"

    def test_records_none_hp_when_slots_empty(self):
        ev = {
            "event_type": "move_used",
            "battle_context": {"turn": 1},
            "game_state": {},
        }
        with patch("src.pipeline.record_situation") as mock_record:
            self.runner._record_situation_snapshot("match1", ev)
        snapshot = mock_record.call_args[0][0]
        assert snapshot["hp_player"] is None
        assert snapshot["hp_opponent"] is None

    def test_screens_tuple_extracted_to_name_only(self):
        # battle_context["screens"]の値は(名前, 残りターン)のtuple（2026-08-07〜）。
        # hp_playerと同種のバグ（TEXT列にtuple/listを直接バインドしてsqlite3エラー）を
        # 防ぐため、名前だけ取り出して渡すことを確認する。
        ev = {
            "event_type": "move_used",
            "battle_context": {
                "turn": 3,
                "screens": {"player": ("リフレクター", 4), "opponent": ("ひかりのかべ", 2)},
            },
            "game_state": {},
        }
        with patch("src.pipeline.record_situation") as mock_record:
            self.runner._record_situation_snapshot("match1", ev)
        snapshot = mock_record.call_args[0][0]
        assert snapshot["screens_player"] == "リフレクター"
        assert snapshot["screens_opponent"] == "ひかりのかべ"

    def test_missing_battle_context_does_not_crash(self):
        with patch("src.pipeline.record_situation") as mock_record:
            self.runner._record_situation_snapshot(
                "match1", {"event_type": "move_used", "game_state": {}})
        mock_record.assert_called_once()

    def test_exception_from_record_is_swallowed(self):
        with patch("src.pipeline.record_situation", side_effect=RuntimeError("boom")):
            # 例外が外に漏れなければOK
            self.runner._record_situation_snapshot("match1", {"event_type": "move_used"})


class TestRenderContextFaintSide:
    """_render_context: 改善ロードマップ③（表情連動）用の faint_side 伝播。
    manifest.jsonl の context.faint_side に「自分/相手どちらが倒れたか」を
    載せ、VMC操作スクリプトが表情を選び分けられるようにする。"""

    def setup_method(self):
        self.runner = Pipeline.__new__(Pipeline)
        self.runner._move_log = []
        self.runner._move_effectiveness = {}
        self.runner._tentative_opponent_moves = []
        self.runner._render_sink = MagicMock()  # None でなければ良い（ダミー）

    def test_faint_side_included_when_present(self):
        ctx = self.runner._render_context({"turn": 3, "faint_side": "player"})
        assert ctx["faint_side"] == "player"

    def test_faint_side_omitted_when_not_a_faint_event(self):
        """faint以外のイベント（battle_contextにfaint_sideキーが無い）では含めない。"""
        ctx = self.runner._render_context({"turn": 3})
        assert "faint_side" not in ctx

    def test_faint_side_none_when_ambiguous(self):
        ctx = self.runner._render_context({"turn": 3, "faint_side": None})
        assert ctx["faint_side"] is None

    def test_battle_result_included_when_present(self):
        ctx = self.runner._render_context({"turn": 10, "battle_result": "勝ち"})
        assert ctx["battle_result"] == "勝ち"

    def test_battle_result_omitted_when_not_battle_end(self):
        ctx = self.runner._render_context({"turn": 3})
        assert "battle_result" not in ctx

    def test_type_hint_included_when_present(self):
        """2026-08-04: 戦況推論強化のtype_hintをmanifest.jsonlで実機確認できるように伝播する。"""
        ctx = self.runner._render_context(
            {"turn": 3, "type_hint": "メタグロスの技はコータスにいまひとつ"})
        assert ctx["type_hint"] == "メタグロスの技はコータスにいまひとつ"

    def test_type_hint_omitted_when_absent(self):
        ctx = self.runner._render_context({"turn": 3})
        assert "type_hint" not in ctx

    def test_returns_none_without_render_sink(self):
        self.runner._render_sink = None
        assert self.runner._render_context({"faint_side": "player"}) is None


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
