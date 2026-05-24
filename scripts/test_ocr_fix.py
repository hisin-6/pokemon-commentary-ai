#!/usr/bin/env python3
import re
from difflib import SequenceMatcher

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
        # 句点・記号の修正
        if oc in PUNCT_MAP and PUNCT_MAP[oc] == cc:
            continue
        # 小文字カタカナの修正
        if oc in SMALL_KANA_MAP and SMALL_KANA_MAP[oc] == cc:
            continue
        # ゆ -> り (こだわゆ -> こだわり)
        if oc == 'ゆ' and cc == 'り':
            continue
        # 末尾の助詞変化はNG
        if is_last and oc in 'をはにのでがもへ' and cc in 'をはにのでがもへ':
            return False
        return False
    return True

test_pairs = [
    ('オオニユーラの', 'オオニューラの'),
    ('ねこだまし」', 'ねこだまし！'),
    ('リザードンを', 'リザードンは'),
    ('ソーラービーム」', 'ソーラービーム！'),
    ('だくりゆう!', 'だくりゅう！'),
    ('ペリツパーの', 'ペリッパーの'),
    ('こだわゆスカーフ', 'こだわりスカーフ'),
    ('さいきよう', 'さいきょう'),
]

names = [
    ('オオニユーラの', 'オオニューラの'),
    ('ねこだまし」', 'ねこだまし！'),
    ('リザードンを', 'リザードンは'),
    ('ソーラービーム」', 'ソーラービーム！'),
    ('だくりゆう!', 'だくりゅう！'),
    ('ペリツパーの', 'ペリッパーの'),
    ('こだわゆスカーフ', 'こだわりスカーフ'),
    ('さいきよう', 'さいきょう'),
]

print("=== chars_differ_only_by_ocr_error テスト ===")
for (ocr, correct), (n1, n2) in zip(test_pairs, names):
    on = normalize(ocr)
    cn = normalize(correct)
    if len(on) == len(cn):
        result = chars_differ_only_by_ocr_error(on, cn)
        print(f"  {'OK' if result else 'NG'} {repr(n1):35s} -> {repr(n2)}")
    else:
        print(f"  -- (len {len(on)} vs {len(cn)}) {repr(n1):35s} -> {repr(n2)}")
