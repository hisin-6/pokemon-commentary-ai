#!/usr/bin/env python3
"""
末尾の」を除去するケース、長さが1違うケース、など追加ケーステスト
"""
import re

def normalize(text):
    return re.sub(r'[　\s]+', '', text)

SMALL_KANA_MAP = {
    'ユ': 'ュ', 'ア': 'ァ', 'イ': 'ィ', 'エ': 'ェ', 'オ': 'ォ',
    'ヤ': 'ャ', 'ヨ': 'ョ', 'ツ': 'ッ',
    'ゆ': 'ゅ', 'あ': 'ぁ', 'い': 'ぃ', 'え': 'ぇ', 'お': 'ぉ',
    'や': 'ゃ', 'よ': 'ょ', 'つ': 'っ', 'わ': 'ゎ',
}
PUNCT_MAP = {
    '」': '！', '!': '！',
}

def chars_differ_only_by_ocr_error(ocr_norm, correct_norm):
    if len(ocr_norm) != len(correct_norm):
        return False
    for pos, (oc, cc) in enumerate(zip(ocr_norm, correct_norm)):
        if oc == cc:
            continue
        is_last = (pos == len(ocr_norm) - 1)
        if oc in PUNCT_MAP and PUNCT_MAP[oc] == cc:
            continue
        if oc in SMALL_KANA_MAP and SMALL_KANA_MAP[oc] == cc:
            continue
        if oc == 'ゆ' and cc == 'り':
            continue
        if is_last and oc in 'をはにのでがもへ' and cc in 'をはにのでがもへ':
            return False
        return False
    return True

def find_correction(ocr_text, correct_norm_map):
    """
    書き起こし正解セットからの修正を探す。
    安全な修正（OCR誤読のみ）のみ返す。
    """
    on = normalize(ocr_text)

    # 1. 完全一致
    if on in correct_norm_map:
        orig = correct_norm_map[on]
        if orig != ocr_text:
            return orig, 'exact_match'
        return None, None

    # 2. 末尾の」を除去して一致チェック
    if on.endswith('」'):
        stripped = on[:-1]
        if stripped in correct_norm_map:
            return correct_norm_map[stripped], 'strip_kagikakko'

    # 3. 同じ長さの正解候補でOCR誤読のみ異なるもの
    for cnorm, corig in correct_norm_map.items():
        if len(cnorm) == len(on):
            if chars_differ_only_by_ocr_error(on, cnorm):
                return corig, 'ocr_char_fix'

    return None, None

# テスト用の正解セット（書き起こしから抽出されたもの）
correct_set = [
    'イダイトウ',
    'オオニューラの',
    'ねこだまし！',
    'ソーラービーム！',
    'だくりゅう！',
    'ペリッパーの',
    'こだわりスカーフ',
    'さいきょう',
    'ゆけっ！',
]
correct_norm_map = {normalize(t): t for t in correct_set}

test_cases = [
    'イダイトウ」',      # -> イダイトウ (末尾」除去)
    'オオニユーラの',    # -> オオニューラの (ユ->ュ)
    'ねこだまし」',      # -> ねこだまし！ (」->！)
    'ソーラービーム」',  # -> ソーラービーム！ (」->！)
    'だくりゆう!',       # -> だくりゅう！ (ゆ->ゅ + !->！)
    'ペリツパーの',      # -> ペリッパーの (ツ->ッ)
    'こだわゆスカーフ',  # -> こだわりスカーフ (ゆ->り)
    'さいきよう',        # -> さいきょう (よ->ょ)
    'リザードンを',      # -> None (助詞違いは修正しない)
    'ゆけつ!',           # -> ゆけっ！ (つ->っ + !->！)
]

print("=== find_correction テスト ===")
for tc in test_cases:
    result, method = find_correction(tc, correct_norm_map)
    print(f"  [{method or '-':20s}] {repr(tc):30s} -> {repr(result)}")
