#!/usr/bin/env python3
"""
annotations.json を書き起こしデータと照合して修正するスクリプト。
出力: data/ocr_crops/annotations_fixed.json

修正ルール:
  - OCR誤読（小文字カタカナ/ひらがな、句点誤読）のみを修正
  - 助詞の違い・prefix追加などは修正しない（別テキストと混同しないため）
  - conf < 0.02 または 記号/英数字のみのノイズ行は削除
  - テキスト2文字未満も削除
"""
import json
import re
from collections import Counter

TRANSCRIPT_FILES = [
    "/mnt/c/Users/rotat/AITuberProject/records/2026-04-12 16-14-39書き起こし.md",
    "/mnt/c/Users/rotat/AITuberProject/records/2026-04-12 18-12-45書き起こし.md",
    "/mnt/c/Users/rotat/AITuberProject/records/2026-04-13 06-25-46書き起こし.md",
    "/mnt/c/Users/rotat/AITuberProject/records/2026-04-13 07-00-19書き起こし.md",
    "/mnt/c/Users/rotat/AITuberProject/records/2026-04-14 08-15-22書き起こし.md",
    "/mnt/c/Users/rotat/AITuberProject/records/2026-04-14 20-14-17書き起こし.md",
]

INPUT_PATH  = "/mnt/c/Users/rotat/AITuberProject/data/ocr_crops/annotations.json"
OUTPUT_PATH = "/mnt/c/Users/rotat/AITuberProject/data/ocr_crops/annotations_fixed.json"


# ---------------------------------------------------------------------------
# 既知誤読辞書（書き起こしにはないが確実にわかる誤読）
# ---------------------------------------------------------------------------

# 正規化後のOCRテキスト -> 正しいテキスト
KNOWN_OCR_ERRORS = {
    # ゆ -> り (こだわゆスカーフ: 最頻出68件)
    'こだわゆスカーフ': 'こだわりスカーフ',
    # よ -> ょ (さいきょう)
    'さいきよう': 'さいきょう',
    'きんちよう': 'きんちょう',
    'ぼうぎよ': 'ぼうぎょ',
    'めいちゆうりつ': 'めいちゅうりつ',
    # ちょうバツグン誤読各種
    'ちようバツグンだ': 'ちょうバツグンだ',
    'ちようバツグンだ!': 'ちょうバツグンだ！',
    'ちようバツグンだ!!': 'ちょうバツグンだ！！',
    'ちようバツグンだり': 'ちょうバツグンだ！',
    'ちようバツグンだ川': 'ちょうバツグンだ！！',
    'ちようバツクンだ川': 'ちょうバツグンだ！！',
    # 末尾の句点誤読 (」 -> ！) で書き起こしに存在しない行
    'まもる」': 'まもる！',
    '身を守つた」': '身を守った！',
    '下がつた」': '下がった！',
    '下がつたり': '下がった！',
    '下がったり': '下がった！',
    '決まらなかつた」': '決まらなかった！',
    '決まらなかつたし': '決まらなかった！',
    '守つている」': '守っている！',
    '食べられなくなつた」': '食べられなくなった！',
    '食べられなくなつた!': '食べられなくなった！',
    '入つた!': '入った！',
    '入つた」': '入った！',
    '入つた!': '入った！',
    # バツグン誤読
    'バツグンた」': 'バツグンだ！',
    # ゆけつ -> ゆけっ
    'ゆけつ!': 'ゆけっ！',
    # インファイト誤読
    'インフアイト」': 'インファイト！',
    # ダブルウィング誤読
    'ダブルウイング」': 'ダブルウィング！',
}


# ---------------------------------------------------------------------------
# OCR誤読補正テーブル
# ---------------------------------------------------------------------------

# 大文字->小文字カタカナ/ひらがなの誤読
SMALL_KANA_MAP = {
    'ユ': 'ュ', 'ア': 'ァ', 'イ': 'ィ', 'エ': 'ェ', 'オ': 'ォ',
    'ヤ': 'ャ', 'ヨ': 'ョ', 'ツ': 'ッ',
    'ゆ': 'ゅ', 'あ': 'ぁ', 'い': 'ぃ', 'え': 'ぇ', 'お': 'ぉ',
    'や': 'ゃ', 'よ': 'ょ', 'つ': 'っ', 'わ': 'ゎ',
}

# 句点・記号の誤読
PUNCT_MAP = {
    '」': '！',
    '!': '！',
}


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def normalize(text):
    """空白・全角スペースを除去して正規化"""
    return re.sub(r'[　\s]+', '', text)


def chars_differ_only_by_ocr_error(ocr_norm, correct_norm):
    """
    2つの（正規化済み）テキストが、OCR誤読のみで異なるかチェック。
    長さが同じであること前提。
    許可される変化:
      - 大文字カタカナ -> 小文字カタカナ (ユ->ュ, ツ->ッ など)
      - 大文字ひらがな -> 小文字ひらがな (よ->ょ, つ->っ など)
      - ゆ -> り (OCRがゆとりを混同する誤読)
      - 句点・記号誤読 (」->！, !->！)
    禁止される変化:
      - 末尾の助詞の変化 (を->は, は->に など) -> 別のテキストを指す
    """
    if len(ocr_norm) != len(correct_norm):
        return False
    for pos, (oc, cc) in enumerate(zip(ocr_norm, correct_norm)):
        if oc == cc:
            continue
        is_last = (pos == len(ocr_norm) - 1)
        # 句点の修正
        if oc in PUNCT_MAP and PUNCT_MAP[oc] == cc:
            continue
        # 小文字カタカナ/ひらがなの修正
        if oc in SMALL_KANA_MAP and SMALL_KANA_MAP[oc] == cc:
            continue
        # ゆ -> り の誤読補正
        if oc == 'ゆ' and cc == 'り':
            continue
        # 末尾の助詞違いはNG（別テキスト）
        if is_last and oc in 'をはにのでがもへと' and cc in 'をはにのでがもへと':
            return False
        # その他の変化は不明 -> NG
        return False
    return True


def find_correction(ocr_text, correct_norm_map):
    """
    書き起こし正解セットからの安全な修正を探す。
    戻り値: (修正後テキスト or None, 修正理由 or None)
    """
    on = normalize(ocr_text)

    # 0. 既知誤読辞書チェック
    if on in KNOWN_OCR_ERRORS:
        return KNOWN_OCR_ERRORS[on], 'known_error_dict'

    # 1. 正規化後の完全一致
    if on in correct_norm_map:
        orig = correct_norm_map[on]
        if orig != ocr_text:
            return orig, 'exact_match'
        return None, None  # すでに正しい

    # 2. 末尾の「」を除去して正解セットに一致
    if on.endswith('」'):
        stripped = on[:-1]
        if stripped in correct_norm_map:
            return correct_norm_map[stripped], 'strip_kagikakko'
        # 既知誤読辞書でも確認
        if stripped in KNOWN_OCR_ERRORS:
            return KNOWN_OCR_ERRORS[stripped], 'known_error_dict+strip'

    # 3. 同じ長さの正解候補でOCR誤読のみ異なるもの
    for cnorm, corig in correct_norm_map.items():
        if len(cnorm) == len(on):
            if chars_differ_only_by_ocr_error(on, cnorm):
                return corig, 'ocr_char_fix'

    return None, None


# ---------------------------------------------------------------------------
# Step 1: 書き起こしから正解テキストセットを構築
# ---------------------------------------------------------------------------

def extract_correct_texts(filepath):
    """【実際のメッセージ】セクションからゲーム画面テキストを抽出する"""
    texts = set()
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    msg_sections = re.findall(
        r'【実際のメッセージ】(.*?)(?:={20,}|$)', content, re.DOTALL
    )

    for section in msg_sections:
        for line in section.split('\n'):
            line = line.strip()
            if not line:
                continue
            # セクション見出し除外
            if re.match(r'^・?T\d+', line):
                continue
            if re.match(r'^【|^↑|^=', line):
                continue
            # 注釈行除外
            if '※' in line:
                continue
            if '（ここで' in line or '(ここで' in line:
                continue

            # [ player:道具表示]XXX 形式 -> ] 以降を取得
            tag_match = re.match(r'\[.*?\](.*)', line)
            if tag_match:
                game_text = tag_match.group(1).strip().rstrip(']').strip()
                clean = normalize(game_text)
                if len(clean) >= 2:
                    texts.add(clean)
                continue

            # 通常行
            clean = normalize(line)
            if len(clean) >= 2:
                texts.add(clean)

    return texts


all_correct = set()
for fp in TRANSCRIPT_FILES:
    all_correct |= extract_correct_texts(fp)

print(f"[Step 1] 正解テキスト数: {len(all_correct)}")

# 正規化キー -> 正規化後テキスト（= 書き起こしの正解）
correct_norm_map = {t: t for t in all_correct}  # すでに正規化済み


# ---------------------------------------------------------------------------
# Step 2: ノイズ判定
# ---------------------------------------------------------------------------

def is_noise(text, conf):
    """削除すべきノイズかどうかを判定"""
    clean = normalize(text)

    # 2文字未満は常に削除
    if len(clean) < 2:
        return True, '2文字未満'

    if conf < 0.3:
        # 記号が多すぎる
        noise_chars = re.findall(r'[~@》《\[\]|【】\.,:;/\\^_+=<>!\'"#$%&*(){}`]', clean)
        if noise_chars and len(noise_chars) / len(clean) > 0.3:
            return True, '低信頼度+記号多数'
        # 英数字・記号のみ
        if re.match(r'^[0-9A-Za-z/:\-\.@~\s]+$', clean):
            return True, '低信頼度+英数字記号のみ'
        # 信頼度が極端に低い
        if conf < 0.02:
            return True, '信頼度0.02未満'

    return False, ''


# ---------------------------------------------------------------------------
# Step 3: メイン処理
# ---------------------------------------------------------------------------

with open(INPUT_PATH, encoding='utf-8') as f:
    data = json.load(f)

print(f"[Step 2] 元データ件数: {len(data)}")

fixed = []
deleted_entries = []  # (file, text, conf, reason)
modified_entries = []  # (file, before, after, method)

for entry in data:
    text = entry['ocr_text']
    conf = entry['conf']
    fname = entry['file']

    # ノイズ判定
    noise, reason = is_noise(text, conf)
    if noise:
        deleted_entries.append((fname, text, conf, reason))
        continue

    # 書き起こしと照合して修正
    correction, method = find_correction(text, correct_norm_map)
    if correction is not None and correction != text:
        modified_entries.append((fname, text, correction, method))
        entry = dict(entry)
        entry['ocr_text'] = correction
        entry['corrected'] = True

    fixed.append(entry)

# 結果保存
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(fixed, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 完了報告
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"完了報告")
print(f"{'='*60}")
print(f"元件数:       {len(data)}")
print(f"最終件数:     {len(fixed)}")
print(f"削除件数:     {len(deleted_entries)}")
print(f"修正件数:     {len(modified_entries)}")
print(f"変更なし:     {len(data) - len(deleted_entries) - len(modified_entries)}")

reasons = Counter(r for _, _, _, r in deleted_entries)
print(f"\n--- 削除理由内訳 ---")
for reason, count in reasons.most_common():
    print(f"  {reason}: {count}件")

methods = Counter(m for _, _, _, m in modified_entries)
print(f"\n--- 修正方法内訳 ---")
for method, count in methods.most_common():
    print(f"  {method}: {count}件")

print(f"\n--- 代表的な修正例 (最大20件) ---")
for fname, before, after, method in modified_entries[:20]:
    print(f"  [{method:18s}] {repr(before):35s} -> {repr(after)}")

print(f"\n--- 削除サンプル (最大15件) ---")
for fname, text, conf, reason in deleted_entries[:15]:
    print(f"  conf={conf:.3f} [{reason}] {repr(text)}")

print(f"\n出力ファイル: {OUTPUT_PATH}")
