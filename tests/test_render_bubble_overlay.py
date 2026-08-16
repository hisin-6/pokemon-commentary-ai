"""
render_bubble_overlay.py（プレイヤーのツッコミ吹き出し・v2d・2026-08-16新設）の単体テスト

テスト対象:
  - load_bubbles       bubbles.jsonl（任意）の読み込み
  - render_bubble_pngs 透過PNGの書き出し・タイミング計算
  - draw_speech_bubble  見た目の合成（サイズ・アルファ保持のみ軽く検証）
"""

import importlib.util
from pathlib import Path

from PIL import Image

_SCRIPT = Path(__file__).parent.parent / "scripts" / "render_bubble_overlay.py"
_spec = importlib.util.spec_from_file_location("render_bubble_overlay", _SCRIPT)
rbo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbo)


class TestLoadBubbles:
    def test_missing_file_returns_empty(self, tmp_path):
        assert rbo.load_bubbles(tmp_path) == []

    def test_loads_and_sorts_by_time(self, tmp_path):
        path = tmp_path / "bubbles.jsonl"
        path.write_text(
            '{"time": 100.0, "text": "後の方"}\n'
            '{"time": 10.0, "text": "先の方"}\n',
            encoding="utf-8",
        )
        bubbles = rbo.load_bubbles(tmp_path)
        assert [b["text"] for b in bubbles] == ["先の方", "後の方"]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "bubbles.jsonl"
        path.write_text('{"time": 1.0, "text": "a"}\n\n\n', encoding="utf-8")
        assert len(rbo.load_bubbles(tmp_path)) == 1


class TestRenderBubblePngs:
    def test_returns_start_end_from_time_and_duration(self, tmp_path):
        results = rbo.render_bubble_pngs(
            [{"time": 10.0, "text": "テスト", "duration": 3.0}], tmp_path)
        assert results[0]["start"] == 10.0
        assert results[0]["end"] == 13.0
        assert results[0]["text"] == "テスト"

    def test_default_duration_used_when_omitted(self, tmp_path):
        results = rbo.render_bubble_pngs([{"time": 5.0, "text": "テスト"}], tmp_path)
        assert results[0]["end"] == 5.0 + rbo._DEFAULT_BUBBLE_DURATION

    def test_writes_transparent_canvas_sized_png(self, tmp_path):
        results = rbo.render_bubble_pngs([{"time": 0.0, "text": "テスト"}], tmp_path)
        png_path = results[0]["path"]
        assert png_path.exists()
        img = Image.open(png_path)
        assert img.size == rbo._CANVAS_SIZE
        assert img.mode == "RGBA"
        # キャンバス右下（吹き出しが絶対置かれない位置）は透明のまま
        assert img.getpixel((1900, 1070))[3] == 0

    def test_creates_out_dir(self, tmp_path):
        out_dir = tmp_path / "nested" / "bubbles"
        rbo.render_bubble_pngs([{"time": 0.0, "text": "テスト"}], out_dir)
        assert out_dir.exists()

    def test_multiple_bubbles_numbered_sequentially(self, tmp_path):
        results = rbo.render_bubble_pngs(
            [{"time": 0.0, "text": "A"}, {"time": 10.0, "text": "B"}], tmp_path)
        assert results[0]["path"].name == "bubble_0000.png"
        assert results[1]["path"].name == "bubble_0001.png"


class TestDrawSpeechBubble:
    def test_output_size_matches_input(self, tmp_path):
        base = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        result = rbo.draw_speech_bubble(base, "テスト", x=10, y=10)
        assert result.size == (200, 200)

    def test_bubble_pixels_are_opaque_background_transparent(self):
        base = Image.new("RGBA", (600, 300), (0, 0, 0, 0))
        result = rbo.draw_speech_bubble(base, "テスト", x=10, y=10)
        # 吹き出しの内側（ラベル・本文が乗る領域）は不透明
        assert result.getpixel((60, 70))[3] > 200
        # 右下の遠い領域は依然として透明
        assert result.getpixel((590, 290))[3] == 0
