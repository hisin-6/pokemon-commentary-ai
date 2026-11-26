"""
scripts/fix_annotations.py の単体テスト

テスト対象（純粋関数のみ・ファイル入出力の main() は対象外）:
  - normalize()                       空白・全角スペース除去
  - chars_differ_only_by_ocr_error()  OCR誤読だけの差かどうか
  - find_correction()                 書き起こし正解セットからの安全な修正
  - is_noise()                        ノイズ行の判定

ケースの一部は旧 scripts/test_ocr_fix.py / test_ocr_fix2.py（print で目視するだけの
使い捨てスクリプト・2026-11-26削除）から assert 付きで移植したもの。
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from fix_annotations import (  # noqa: E402
    KNOWN_OCR_ERRORS,
    chars_differ_only_by_ocr_error,
    find_correction,
    is_noise,
    normalize,
)


# ─── normalize ───────────────────────────────────────────────────────────────

class TestNormalize:

    def test_removes_halfwidth_and_fullwidth_spaces(self):
        assert normalize("ゆけ っ　！") == "ゆけっ！"

    def test_removes_tabs_and_newlines(self):
        assert normalize("ピカ\tチュウ\n") == "ピカチュウ"

    def test_no_change_when_no_space(self):
        assert normalize("イダイトウ") == "イダイトウ"


# ─── chars_differ_only_by_ocr_error ──────────────────────────────────────────

def _differ(ocr: str, correct: str) -> bool:
    return chars_differ_only_by_ocr_error(normalize(ocr), normalize(correct))


class TestCharsDifferOnlyByOcrError:

    @pytest.mark.parametrize("ocr, correct", [
        ("オオニユーラの", "オオニューラの"),      # ユ→ュ（小文字カタカナ）
        ("ペリツパーの", "ペリッパーの"),          # ツ→ッ
        ("さいきよう", "さいきょう"),              # よ→ょ（小文字ひらがな）
        ("こだわゆスカーフ", "こだわりスカーフ"),  # ゆ→り（OCRの混同）
        ("ねこだまし」", "ねこだまし！"),          # 」→！（句点誤読）
        ("ソーラービーム」", "ソーラービーム！"),
        ("だくりゆう!", "だくりゅう！"),           # ゆ→ゅ と !→！ の複合
    ])
    def test_ocr_misreads_are_allowed(self, ocr, correct):
        assert _differ(ocr, correct) is True

    @pytest.mark.parametrize("ocr, correct", [
        ("リザードンを", "リザードンは"),  # 末尾の助詞違いは別テキスト
        ("ピカチュウと", "ピカチュウの"),  # 「と」も助詞扱い（旧コピーには無かった）
    ])
    def test_trailing_particle_change_is_rejected(self, ocr, correct):
        assert _differ(ocr, correct) is False

    def test_unrelated_char_change_is_rejected(self):
        assert _differ("ピカチュウ", "ピカチョウ") is False

    def test_reverse_direction_is_rejected(self):
        # 小文字→大文字の向きは誤読補正ではない
        assert _differ("オオニューラの", "オオニユーラの") is False

    def test_different_length_is_rejected(self):
        assert _differ("イダイトウ」", "イダイトウ") is False

    def test_identical_texts(self):
        assert _differ("イダイトウ", "イダイトウ") is True


# ─── find_correction ─────────────────────────────────────────────────────────

_CORRECT_SET = [
    "イダイトウ",
    "オオニューラの",
    "ねこだまし！",
    "ソーラービーム！",
    "だくりゅう！",
    "ペリッパーの",
    "こだわりスカーフ",
    "さいきょう",
    "ゆけっ！",
]


@pytest.fixture
def correct_norm_map():
    return {normalize(t): t for t in _CORRECT_SET}


class TestFindCorrection:

    @pytest.mark.parametrize("ocr, expected, method", [
        ("イダイトウ」", "イダイトウ", "strip_kagikakko"),
        ("オオニユーラの", "オオニューラの", "ocr_char_fix"),
        ("ねこだまし」", "ねこだまし！", "ocr_char_fix"),
        ("ソーラービーム」", "ソーラービーム！", "ocr_char_fix"),
        ("だくりゆう!", "だくりゅう！", "ocr_char_fix"),
        ("ペリツパーの", "ペリッパーの", "ocr_char_fix"),
        ("ゆけっ ！", "ゆけっ！", "exact_match"),
    ])
    def test_safe_corrections(self, correct_norm_map, ocr, expected, method):
        assert find_correction(ocr, correct_norm_map) == (expected, method)

    @pytest.mark.parametrize("ocr", ["こだわゆスカーフ", "さいきよう", "ゆけつ!"])
    def test_known_error_dict_takes_priority(self, correct_norm_map, ocr):
        result, method = find_correction(ocr, correct_norm_map)
        assert method == "known_error_dict"
        assert result == KNOWN_OCR_ERRORS[normalize(ocr)]

    def test_known_error_dict_works_without_correct_set(self):
        assert find_correction("さいきよう", {}) == ("さいきょう", "known_error_dict")

    def test_particle_difference_is_not_corrected(self, correct_norm_map):
        local = dict(correct_norm_map)
        local["リザードンは"] = "リザードンは"
        assert find_correction("リザードンを", local) == (None, None)

    def test_already_correct_returns_none(self, correct_norm_map):
        assert find_correction("イダイトウ", correct_norm_map) == (None, None)

    def test_unknown_text_returns_none(self, correct_norm_map):
        assert find_correction("まったく別の文", correct_norm_map) == (None, None)


# ─── is_noise ────────────────────────────────────────────────────────────────

class TestIsNoise:

    @pytest.mark.parametrize("text, conf, reason", [
        ("a", 0.9, "2文字未満"),
        ("　ア ", 0.9, "2文字未満"),          # 空白除去後に1文字
        ("~~@@ab", 0.1, "低信頼度+記号多数"),
        ("AB12", 0.1, "低信頼度+英数字記号のみ"),
        ("ピカチュウ", 0.01, "信頼度0.02未満"),
    ])
    def test_noise(self, text, conf, reason):
        assert is_noise(text, conf) == (True, reason)

    @pytest.mark.parametrize("text, conf", [
        ("ab", 0.9),
        ("AB12", 0.5),        # 英数字のみでも信頼度0.3以上なら残す
        ("ピカチュウ", 0.5),
        ("ピカチュウ", 0.05),  # 低信頼度でも日本語で0.02以上なら残す
    ])
    def test_not_noise(self, text, conf):
        assert is_noise(text, conf) == (False, "")
