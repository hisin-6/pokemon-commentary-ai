"""プレイヤーの「吹き出し」ツッコミ演出（プロトタイプ・2026-08-16）。

AIVTuber実況にプレイヤー本人（文字のみ出演）が時々ツッコミを入れる演出のための
吹き出し描画。ボイロ実況的な「掛け合い」を、台本を書く人間なしで実現する方向性の
一部——AIの実況ミスをそのまま茶番のネタにする用途を想定（「AIグリッチ」を隠すのでは
なく持ちネタにする発想と同系統）。

現時点では見た目のプロトタイプのみ。パス2本編への統合（--bubbles引数等）・
入力ファイル形式の設計は次のステップ。

使い方（プレビュー生成）:
    python3 scripts/render_bubble_overlay.py --frame <1920x1080のPNG> \\
        --text "それ違うよ〜" --out preview.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from generate_thumbnail import _wrap_to_lines  # noqa: E402

_FONT_PATH = "/mnt/c/Windows/Fonts/meiryob.ttc"

# 「プレイヤーの生の言葉」を示すための配色（暖色・付箋メモ風）。
# ゲーム画面のシアン枠・戦況パネルの寒色系とはっきり区別し、
# 「AIとは別の声が割り込んでいる」ことが一目で分かるようにする狙い。
_BUBBLE_FILL      = (255, 248, 225, 245)   # クリーム地（付箋メモ風）
_BUBBLE_BORDER    = (230, 140, 40, 255)    # 温かみのあるオレンジ
_BUBBLE_TEXT      = (90, 60, 20, 255)      # こげ茶（クリーム地でも視認性を確保）
_TAG_FILL         = (230, 140, 40, 255)    # ラベル（「トレーナー」）の背景
_TAG_TEXT         = (255, 255, 255, 255)
_SHADOW_COLOR     = (0, 0, 0, 90)

_TAG_LABEL = "トレーナー"
_BUBBLE_MAX_WIDTH = 460
_BUBBLE_PADDING = 24
_TAIL_SIZE = 22

# パス2（動画本編）への統合用（2026-08-16）
_DEFAULT_BUBBLE_DURATION = 4.5  # 1つの吹き出しを表示し続ける秒数
_BUBBLE_DEFAULT_X = 30          # ゲーム画面左上・観客席の絵の上に重ねる（プロトタイプで確認済み）
_BUBBLE_DEFAULT_Y = 20
_CANVAS_SIZE = (1920, 1080)     # biimレイアウトの出力解像度に合わせる


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = _FONT_PATH if Path(_FONT_PATH).exists() else None
    return ImageFont.truetype(path, size=size) if path else ImageFont.load_default()


def draw_speech_bubble(base: Image.Image, text: str, x: int, y: int,
                        max_width: int = _BUBBLE_MAX_WIDTH,
                        tail: str = "bottom-left") -> Image.Image:
    """base（RGBA推奨）の(x, y)を吹き出し左上として、プレイヤーのツッコミ吹き出しを
    合成した新しいImageを返す（baseは変更しない）。

    tail: 吹き出しの尻尾の向き（"bottom-left"=左下・実況帯側を指す。既定）。
    """
    if base.mode != "RGBA":
        base = base.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    tag_font = _font(24)
    body_font = _font(32)

    lines = _wrap_to_lines(draw, text, body_font, max_width - _BUBBLE_PADDING * 2, max_lines=3)
    line_h = body_font.size + 10
    tag_h = tag_font.size + 16

    bubble_w = max_width
    bubble_h = _BUBBLE_PADDING * 2 + tag_h + 8 + line_h * len(lines)

    # 影（本体より少し右下にずらした半透明の角丸矩形）
    shadow_offset = 6
    draw.rounded_rectangle(
        [x + shadow_offset, y + shadow_offset, x + bubble_w + shadow_offset, y + bubble_h + shadow_offset],
        radius=22, fill=_SHADOW_COLOR,
    )
    # 本体
    draw.rounded_rectangle(
        [x, y, x + bubble_w, y + bubble_h], radius=22,
        fill=_BUBBLE_FILL, outline=_BUBBLE_BORDER, width=4,
    )

    # 尻尾（吹き出しの下端から三角形を突き出す。「話者が画面の外にいる」ことを示す）
    if tail == "bottom-left":
        tail_x = x + 48
        draw.polygon(
            [(tail_x, y + bubble_h - 2), (tail_x + _TAIL_SIZE * 2, y + bubble_h - 2),
             (tail_x, y + bubble_h + _TAIL_SIZE)],
            fill=_BUBBLE_FILL, outline=_BUBBLE_BORDER,
        )
        # 縁取りの継ぎ目を隠すため下辺のボーダーを上書き
        draw.line([(tail_x - 2, y + bubble_h - 2), (tail_x + _TAIL_SIZE * 2 + 2, y + bubble_h - 2)],
                   fill=_BUBBLE_FILL, width=6)

    # 「トレーナー」ラベル（ピル型）
    tag_w = draw.textlength(_TAG_LABEL, font=tag_font) + 28
    tag_x, tag_y = x + _BUBBLE_PADDING, y + _BUBBLE_PADDING
    draw.rounded_rectangle(
        [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], radius=tag_h // 2, fill=_TAG_FILL,
    )
    draw.text((tag_x + 14, tag_y + 8), _TAG_LABEL, font=tag_font, fill=_TAG_TEXT)

    # 本文
    text_y = tag_y + tag_h + 10
    for ln in lines:
        draw.text((x + _BUBBLE_PADDING, text_y), ln, font=body_font, fill=_BUBBLE_TEXT)
        text_y += line_h

    return Image.alpha_composite(base, overlay)


def load_bubbles(render_dir: Path) -> list[dict]:
    """bubbles.jsonl（プレイヤーのツッコミ吹き出し・任意）を読み込む。

    1行1件のJSON: {"time": 275.8, "text": "...", "duration": 4.5（省略可）}。
    NGチェック（review_checklist.md）で見つけた面白い/分かりやすい間違いを厳選して
    手書きする想定（LLMには書かせない・生の言葉としての強みを残すため）。
    ファイルが無ければ空リスト（既存動画の合成フローに一切影響しない）。
    """
    path = render_dir / "bubbles.jsonl"
    if not path.exists():
        return []
    bubbles = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                bubbles.append(json.loads(line))
    bubbles.sort(key=lambda b: b["time"])
    return bubbles


def render_bubble_pngs(bubbles: list[dict], out_dir: Path,
                        x: int = _BUBBLE_DEFAULT_X, y: int = _BUBBLE_DEFAULT_Y) -> list[dict]:
    """各ツッコミを、吹き出し部分以外は透明な1920x1080のPNGとして書き出す
    （パス2のffmpeg overlayフィルタでそのまま画面に重ねられる形式・2026-08-16）。

    戻り値: [{"path": Path, "start": float, "end": float, "text": str}, ...]
    （event_time順で呼ぶ側がそのまま`build_ffmpeg_command_biim(bubbles=...)`に渡せる）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, b in enumerate(bubbles):
        canvas = Image.new("RGBA", _CANVAS_SIZE, (0, 0, 0, 0))
        composed = draw_speech_bubble(canvas, b["text"], x, y)
        png_path = out_dir / f"bubble_{i:04d}.png"
        composed.save(png_path)  # RGBAのまま保存（アルファチャンネルを保持）
        start = float(b["time"])
        duration = float(b.get("duration", _DEFAULT_BUBBLE_DURATION))
        results.append({"path": png_path, "start": start, "end": start + duration,
                        "text": b["text"]})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="プレイヤー吹き出しのプレビュー生成")
    parser.add_argument("--frame", type=Path, required=True, help="背景に使う1920x1080フレームPNG")
    parser.add_argument("--text", default="それ違うよ〜", help="吹き出しの本文")
    parser.add_argument("--x", type=int, default=30)
    parser.add_argument("--y", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("bubble_preview.png"))
    args = parser.parse_args(argv)

    base = Image.open(args.frame)
    result = draw_speech_bubble(base, args.text, args.x, args.y)
    result.convert("RGB").save(args.out)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
