"""
generate_thumbnail.py（実況動画のサムネイル自動生成・改善ロードマップ⑥）の単体テスト

テスト対象:
  - _collect_event_candidates    manifestからKO/battle_end候補を抽出
  - _collect_hp_swing_candidates statesからHP急変候補を抽出
  - select_thumbnail_moment      候補からの最終選択（優先度: battle_end>faint>HP急変）
  - build_extract_frame_command  ffmpegコマンド組み立て
  - compose_thumbnail            フレームへのテキスト焼き込み（実PILで検証）
"""

import importlib.util
import sys
from pathlib import Path

from PIL import Image

# scripts/ はパッケージではないためファイルパスから直接ロードする
_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_thumbnail.py"
_spec = importlib.util.spec_from_file_location("generate_thumbnail", _SCRIPT)
gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)


def _manifest_entry(event_time, event_type, commentary="コメント"):
    return {"event_time": event_time, "event_type": event_type, "commentary": commentary}


def _state(time, player=None, opponent=None):
    return {"time": time, "player": player or [], "opponent": opponent or []}


def _mon(name, hp_pct):
    return {"name": name, "hp_pct": hp_pct, "hp_text": f"{hp_pct}%", "status": "なし"}


class TestCollectEventCandidates:
    def test_faint_and_battle_end_collected(self):
        manifest = [
            _manifest_entry(10.0, "move_used"),
            _manifest_entry(50.0, "faint", "たおれた！"),
            _manifest_entry(90.0, "battle_end", "勝った！"),
        ]
        candidates = gt._collect_event_candidates(manifest)
        reasons = {c["reason"] for c in candidates}
        assert reasons == {"faint", "battle_end"}

    def test_battle_end_scores_higher_than_faint(self):
        manifest = [_manifest_entry(10.0, "faint"), _manifest_entry(20.0, "battle_end")]
        candidates = gt._collect_event_candidates(manifest)
        by_reason = {c["reason"]: c["score"] for c in candidates}
        assert by_reason["battle_end"] > by_reason["faint"]

    def test_irrelevant_event_types_ignored(self):
        manifest = [_manifest_entry(10.0, "move_used"), _manifest_entry(20.0, "turn_start")]
        assert gt._collect_event_candidates(manifest) == []


class TestCollectHpSwingCandidates:
    def test_large_drop_detected(self):
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(10.0, player=[_mon("ピカチュウ", 40)]),  # 60pt減
        ]
        candidates = gt._collect_hp_swing_candidates(states, threshold=30.0)
        assert len(candidates) == 1
        assert candidates[0]["time"] == 10.0
        assert candidates[0]["reason"] == "hp_swing"

    def test_small_drop_below_threshold_ignored(self):
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(10.0, player=[_mon("ピカチュウ", 85)]),  # 15pt減
        ]
        assert gt._collect_hp_swing_candidates(states, threshold=30.0) == []

    def test_hp_swing_score_capped_below_ko_scores(self):
        """HP急変のスコアはKO系イベント（80/100）を絶対に上回らない。"""
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(10.0, player=[_mon("ピカチュウ", 0)]),  # 100pt減
        ]
        candidates = gt._collect_hp_swing_candidates(states, threshold=30.0)
        assert candidates[0]["score"] < gt._EVENT_SCORE["faint"]

    def test_missing_hp_pct_skipped(self):
        states = [
            _state(0.0, player=[{"name": "ピカチュウ", "hp_pct": None, "hp_text": "?", "status": "なし"}]),
            _state(10.0, player=[_mon("ピカチュウ", 10)]),
        ]
        assert gt._collect_hp_swing_candidates(states, threshold=30.0) == []

    def test_different_side_tracked_independently(self):
        """自分と相手の同名ポケモンは別トラック（同名ミラーでの誤検出防止）。"""
        states = [
            _state(0.0, player=[_mon("イダイトウ", 100)], opponent=[_mon("イダイトウ", 30)]),
            _state(10.0, player=[_mon("イダイトウ", 90)], opponent=[_mon("イダイトウ", 25)]),
        ]
        # 自分側10pt減（閾値未満）・相手側5pt減（閾値未満）→ どちらも候補なし
        assert gt._collect_hp_swing_candidates(states, threshold=30.0) == []


class TestSelectThumbnailMoment:
    def test_prefers_battle_end_over_faint_and_hp_swing(self):
        manifest = [_manifest_entry(50.0, "faint"), _manifest_entry(90.0, "battle_end")]
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(30.0, player=[_mon("ピカチュウ", 0)]),
        ]
        moment = gt.select_thumbnail_moment(manifest, states)
        assert moment["reason"] == "battle_end"
        assert moment["time"] == 90.0

    def test_falls_back_to_hp_swing_when_no_ko_events(self):
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(30.0, player=[_mon("ピカチュウ", 20)]),
        ]
        moment = gt.select_thumbnail_moment([], states)
        assert moment["reason"] == "hp_swing"

    def test_raises_when_no_candidates(self):
        try:
            gt.select_thumbnail_moment([], [])
            assert False, "ValueError が発生するはず"
        except ValueError:
            pass


class TestBuildExtractFrameCommand:
    def test_command_contains_time_and_paths(self, tmp_path):
        video = tmp_path / "battle.mp4"
        out = tmp_path / "frame.png"
        cmd = gt.build_extract_frame_command("ffmpeg", video, 123.456, out)
        assert cmd[0] == "ffmpeg"
        assert str(video) in cmd
        assert str(out) in cmd
        assert "123.456" in cmd

    def test_negative_time_clamped_to_zero(self, tmp_path):
        cmd = gt.build_extract_frame_command("ffmpeg", tmp_path / "v.mp4", -5.0, tmp_path / "f.png")
        assert "0.0" in cmd


class TestTruncateLabel:
    def test_short_text_unchanged(self):
        assert gt._truncate_label("たおれた！", max_chars=28) == "たおれた！"

    def test_long_text_truncated_with_ellipsis(self):
        text = "あ" * 40
        result = gt._truncate_label(text, max_chars=28)
        assert len(result) == 28
        assert result.endswith("…")


class TestComposeThumbnail:
    def test_output_image_created_with_same_size(self, tmp_path):
        frame = tmp_path / "frame.png"
        Image.new("RGB", (1920, 1080), color=(50, 80, 120)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "テスト実況テキスト！")
        assert out.exists()
        with Image.open(out) as img:
            assert img.size == (1920, 1080)

    def test_bottom_bar_darkens_pixels(self, tmp_path):
        """下部帯オーバーレイで元フレームより暗くなっていることを確認。"""
        frame = tmp_path / "frame.png"
        Image.new("RGB", (400, 300), color=(200, 200, 200)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "")
        with Image.open(out) as img:
            # 下部帯領域（下から10px上）のピクセルは暗い下地で覆われているはず
            r, g, b = img.convert("RGB").getpixel((10, 295))
            assert (r, g, b) != (200, 200, 200)

    def test_long_text_wraps_without_overflowing_width(self, tmp_path):
        """回帰テスト: 長い実況テキストがフォント巨大化で右端からはみ出さないこと
        （実機フレームで「わぁ、試合終了だ〜！♪…」が画面外まで見切れた不具合の再発防止）。"""
        frame = tmp_path / "frame.png"
        w, h = 1920, 1080
        Image.new("RGB", (w, h), color=(0, 0, 0)).save(frame)
        out = tmp_path / "thumb.png"
        long_label = "わぁ、試合終了だ〜！♪ この激熱なダブルバトルもここまでか〜、どちらが勝利をつかんだのかな〜？"
        gt.compose_thumbnail(frame, out, long_label)

        from PIL import ImageDraw, ImageFont
        img = Image.open(out).convert("RGB")
        bar_h = int(h * 0.26)
        font_size = max(1, int(bar_h / gt._LABEL_MAX_LINES * 0.62))
        font_path = gt._FONT_PATH if Path(gt._FONT_PATH).exists() else None
        font = ImageFont.truetype(font_path, size=font_size) if font_path else ImageFont.load_default()
        draw = ImageDraw.Draw(img)
        pad_x = int(w * 0.04)
        max_width = w - 2 * pad_x
        text = gt._truncate_label(long_label, max_chars=gt._LABEL_MAX_CHARS)
        lines = gt._wrap_to_lines(draw, text, font, max_width)
        assert len(lines) <= gt._LABEL_MAX_LINES
        for line in lines:
            assert draw.textlength(line, font=font) <= max_width
