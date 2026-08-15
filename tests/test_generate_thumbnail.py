"""
generate_thumbnail.py（実況動画のサムネイル自動生成・改善ロードマップ⑥）の単体テスト

テスト対象:
  - _collect_event_candidates    manifestからKO（・battle_end）候補を抽出
  - _collect_hp_swing_candidates statesからHP急変候補を抽出
  - select_thumbnail_moment      候補からの最終選択（既定はfaint>HP急変・battle_end除外・
                                    allow_result_spoiler=Trueでbattle_end>faint>HP急変）
  - build_extract_frame_command  ffmpegコマンド組み立て
  - _collect_roster              statesから陣営の登場ポケモン名を収集（2026-08-04）
  - _resolve_pokemon_id/fetch_pokemon_icon  図鑑DB参照・アイコン取得キャッシュ（2026-08-04）
  - build_avatar_face_command    アバター顔クロップffmpegコマンド組み立て（2026-08-04）
  - compose_thumbnail            フレームへのテキスト・バッジ・顔・構築アイコン焼き込み
"""

import importlib.util
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

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
    def test_faint_collected_battle_end_excluded_by_default(self):
        """2026-08-04: battle_end（試合結果）は既定でネタバレ防止のため除外する。"""
        manifest = [
            _manifest_entry(10.0, "move_used"),
            _manifest_entry(50.0, "faint", "たおれた！"),
            _manifest_entry(90.0, "battle_end", "勝った！"),
        ]
        candidates = gt._collect_event_candidates(manifest)
        reasons = {c["reason"] for c in candidates}
        assert reasons == {"faint"}

    def test_battle_end_included_when_spoiler_allowed(self):
        manifest = [
            _manifest_entry(50.0, "faint", "たおれた！"),
            _manifest_entry(90.0, "battle_end", "勝った！"),
        ]
        candidates = gt._collect_event_candidates(manifest, allow_result_spoiler=True)
        reasons = {c["reason"] for c in candidates}
        assert reasons == {"faint", "battle_end"}

    def test_battle_end_scores_higher_than_faint(self):
        manifest = [_manifest_entry(10.0, "faint"), _manifest_entry(20.0, "battle_end")]
        candidates = gt._collect_event_candidates(manifest, allow_result_spoiler=True)
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
    def test_battle_end_excluded_by_default(self):
        """2026-08-04: 既定ではbattle_endを見せず、faintが選ばれる。"""
        manifest = [_manifest_entry(50.0, "faint"), _manifest_entry(90.0, "battle_end")]
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(30.0, player=[_mon("ピカチュウ", 0)]),
        ]
        moment = gt.select_thumbnail_moment(manifest, states)
        assert moment["reason"] == "faint"
        assert moment["time"] == 50.0

    def test_prefers_battle_end_when_spoiler_allowed(self):
        manifest = [_manifest_entry(50.0, "faint"), _manifest_entry(90.0, "battle_end")]
        states = [
            _state(0.0, player=[_mon("ピカチュウ", 100)]),
            _state(30.0, player=[_mon("ピカチュウ", 0)]),
        ]
        moment = gt.select_thumbnail_moment(manifest, states, allow_result_spoiler=True)
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

    def test_wrap_to_lines_respects_explicit_newline(self, tmp_path):
        """2026-08-15対応: ロゴテキストに明示的な改行(\\n)があれば、_wrap_jpの
        句読点ベース折り返しに関わらずその位置で行分割する。改行が無いと
        「ポケモンダブルバトルAI自動実況」が「ポケモンダブルバトルA」/「I自動実況」
        のように英単語の途中で機械的に割れてしまう実例があった。"""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_path = gt._FONT_PATH if Path(gt._FONT_PATH).exists() else None
        font = (ImageFont.truetype(font_path, size=60) if font_path
               else ImageFont.load_default())
        lines = gt._wrap_to_lines(draw, "ポケモンダブルバトル\nAI自動実況", font, max_width=1800)
        assert lines == ["ポケモンダブルバトル", "AI自動実況"]

    def test_avatar_face_composited_top_right(self, tmp_path):
        """avatar_face_png指定時、右上付近が元フレームと変わっていること。"""
        frame = tmp_path / "frame.png"
        w, h = 800, 600
        Image.new("RGB", (w, h), color=(30, 30, 30)).save(frame)
        face = tmp_path / "face.png"
        Image.new("RGBA", (400, 400), color=(255, 0, 0, 255)).save(face)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "", avatar_face_png=face)
        with Image.open(out) as img:
            # face_w = int(h*0.34) = 204, fx = w-204-20 = 576, fy = int(h*0.03) = 18
            # → (700, 60) は貼り付けたRGBA画像の内側に確実に収まる
            r, g, b = img.convert("RGB").getpixel((700, 60))
            assert (r, g, b) != (30, 30, 30)

    def test_avatar_face_scale_enlarges_composited_region(self, tmp_path):
        """2026-08-15新設: avatar_face_scaleを大きくすると顔の表示幅が広がる
        （既定0.34だと存在感が薄いとのfbで調整可能にした）。"""
        frame = tmp_path / "frame.png"
        w, h = 800, 600
        Image.new("RGB", (w, h), color=(30, 30, 30)).save(frame)
        face = tmp_path / "face.png"
        Image.new("RGBA", (400, 400), color=(255, 0, 0, 255)).save(face)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "", avatar_face_png=face, avatar_face_scale=0.6)
        with Image.open(out) as img:
            # face_w = int(h*0.6) = 360, fx = w-360-20 = 420 → x=450は貼り付け範囲内
            # （既定0.34だとfx=576なのでx=450は元フレームの背景色のまま残るはず）
            r, g, b = img.convert("RGB").getpixel((450, 60))
            assert (r, g, b) != (30, 30, 30)

    def test_missing_avatar_face_png_skipped_gracefully(self, tmp_path):
        frame = tmp_path / "frame.png"
        Image.new("RGB", (400, 300), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "", avatar_face_png=tmp_path / "does_not_exist.png")
        assert out.exists()

    def test_roster_icons_composited_above_caption_bar(self, tmp_path):
        frame = tmp_path / "frame.png"
        w, h = 800, 600
        Image.new("RGB", (w, h), color=(30, 30, 30)).save(frame)
        icon = tmp_path / "icon.png"
        Image.new("RGBA", (96, 96), color=(0, 255, 0, 255)).save(icon)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "", roster_icon_pngs=[icon, icon, icon])
        bar_h = int(h * 0.26)
        icon_size = int(h * 0.09)
        strip_h = int(icon_size * 1.3)
        strip_y = h - bar_h - strip_h
        with Image.open(out) as img:
            r, g, b = img.convert("RGB").getpixel((w // 2, strip_y + strip_h // 2))
            assert (r, g, b) != (30, 30, 30)

    def test_empty_label_shows_big_ai_logo_text(self, tmp_path, monkeypatch):
        """label=""（盛り上がりシーンの字幕なし）指定時、実況キャプションの代わりに
        大きな「AI自動実況」ロゴテキストが描画されること（2026-08-14新設）。"""
        calls = []
        real_text = ImageDraw.ImageDraw.text

        def spy_text(self, xy, text, *args, **kwargs):
            calls.append(text)
            return real_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
        frame = tmp_path / "frame.png"
        Image.new("RGB", (1920, 1080), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "")
        assert "AI自動実況" in calls

    def test_big_logo_text_custom_and_wraps_to_two_lines(self, tmp_path, monkeypatch):
        """big_logo_textを指定した場合そのテキストが使われ、画面幅に収まらない
        長さなら自動的に2行へ折り返されること（2026-08-14）。"""
        calls = []
        real_text = ImageDraw.ImageDraw.text

        def spy_text(self, xy, text, *args, **kwargs):
            calls.append(text)
            return real_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
        frame = tmp_path / "frame.png"
        Image.new("RGB", (1920, 1080), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "", big_logo_text="ポケモンダブルバトルAI自動実況")
        joined = "".join(calls)
        assert "ポケモンダブルバトル" in joined
        assert "AI自動実況" in joined
        # 1文字列の描画呼び出しには収まらず複数行に分割されているはず
        assert "ポケモンダブルバトルAI自動実況" not in calls

    def test_big_logo_text_explicit_newline_splits_on_word_boundary(self, tmp_path, monkeypatch):
        """big_logo_textに"\\n"を含めた場合、自動折り返しではなく指定位置で
        割れること（"AI"のような英字トークンの途中で割れる不自然さの回避・2026-08-14）。"""
        calls = []
        real_text = ImageDraw.ImageDraw.text

        def spy_text(self, xy, text, *args, **kwargs):
            calls.append(text)
            return real_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
        frame = tmp_path / "frame.png"
        Image.new("RGB", (1920, 1080), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "",
                             big_logo_text="ポケモンダブルバトル\nAI自動実況")
        assert "ポケモンダブルバトル" in calls
        assert "AI自動実況" in calls

    def test_non_empty_label_does_not_show_big_ai_logo_text(self, tmp_path, monkeypatch):
        """通常通りlabelがある場合は大きなロゴテキストを出さない（既存動作の回帰防止）。"""
        calls = []
        real_text = ImageDraw.ImageDraw.text

        def spy_text(self, xy, text, *args, **kwargs):
            calls.append(text)
            return real_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
        frame = tmp_path / "frame.png"
        Image.new("RGB", (1920, 1080), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "テスト実況テキスト")
        assert "AI自動実況" not in calls

    def test_character_name_defaults_to_kurepi(self, tmp_path, monkeypatch):
        """character_name省略時は従来通り花圓くれぴ表記のまま（回帰防止）。"""
        calls = []
        real_text = ImageDraw.ImageDraw.text

        def spy_text(self, xy, text, *args, **kwargs):
            calls.append(text)
            return real_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
        frame = tmp_path / "frame.png"
        Image.new("RGB", (400, 300), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "")
        assert gt._CHARACTER_NAME in calls

    def test_character_name_neutral_override(self, tmp_path, monkeypatch):
        """character_name="VOICEVOX：四国めたん"指定時、花圓くれぴ表記が出ないこと
        （2026-08-14・persona="neutral"でキャラ名誤表示になる不具合の再発防止）。"""
        calls = []
        real_text = ImageDraw.ImageDraw.text

        def spy_text(self, xy, text, *args, **kwargs):
            calls.append(text)
            return real_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
        frame = tmp_path / "frame.png"
        Image.new("RGB", (400, 300), color=(30, 30, 30)).save(frame)
        out = tmp_path / "thumb.png"
        gt.compose_thumbnail(frame, out, "", character_name=gt._CHARACTER_NAME_NEUTRAL)
        assert gt._CHARACTER_NAME_NEUTRAL in calls
        assert gt._CHARACTER_NAME not in calls


class TestCollectRoster:
    def test_dedup_preserves_first_appearance_order(self):
        states = [
            _state(0.0, player=[_mon("コノヨザル", 100)]),
            _state(10.0, player=[_mon("コノヨザル", 90), _mon("メタグロス", 100)]),
            _state(20.0, player=[_mon("メタグロス", 80)]),
        ]
        assert gt._collect_roster(states, "player") == ["コノヨザル", "メタグロス"]

    def test_sides_are_independent(self):
        states = [
            _state(0.0, player=[_mon("コノヨザル", 100)], opponent=[_mon("イッカネズミ", 100)]),
        ]
        assert gt._collect_roster(states, "player") == ["コノヨザル"]
        assert gt._collect_roster(states, "opponent") == ["イッカネズミ"]

    def test_empty_states_returns_empty_list(self):
        assert gt._collect_roster([], "player") == []


class TestResolvePokemonId:
    def _make_pokedb(self, tmp_path) -> Path:
        db_path = tmp_path / "pokedb.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE pokemon (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, "
                     "name_en TEXT NOT NULL)")
        conn.execute("INSERT INTO pokemon (id, name_ja, name_en) VALUES (376, 'メタグロス', 'Metagross')")
        conn.commit()
        conn.close()
        return db_path

    def test_known_name_resolves_to_id(self, tmp_path):
        db_path = self._make_pokedb(tmp_path)
        assert gt._resolve_pokemon_id("メタグロス", pokedb_path=db_path) == 376

    def test_unknown_name_returns_none(self, tmp_path):
        db_path = self._make_pokedb(tmp_path)
        assert gt._resolve_pokemon_id("存在しないポケモン", pokedb_path=db_path) is None

    def test_missing_db_returns_none(self, tmp_path):
        assert gt._resolve_pokemon_id("メタグロス", pokedb_path=tmp_path / "no_such.sqlite") is None


class TestFetchPokemonIcon:
    def _make_pokedb(self, tmp_path) -> Path:
        db_path = tmp_path / "pokedb.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE pokemon (id INTEGER PRIMARY KEY, name_ja TEXT NOT NULL, "
                     "name_en TEXT NOT NULL)")
        conn.execute("INSERT INTO pokemon (id, name_ja, name_en) VALUES (376, 'メタグロス', 'Metagross')")
        conn.commit()
        conn.close()
        return db_path

    def test_downloads_and_caches_on_first_call(self, tmp_path):
        db_path = self._make_pokedb(tmp_path)
        cache_dir = tmp_path / "icons"

        def _fake_urlretrieve(url, out_path):
            Path(out_path).write_bytes(b"fake-png-bytes")

        with patch("generate_thumbnail.urllib.request.urlretrieve", side_effect=_fake_urlretrieve) as mock_dl:
            result = gt.fetch_pokemon_icon("メタグロス", cache_dir=cache_dir, pokedb_path=db_path)
            assert result == cache_dir / "376.png"
            assert result.read_bytes() == b"fake-png-bytes"
            assert mock_dl.call_count == 1

    def test_second_call_uses_cache_without_network(self, tmp_path):
        db_path = self._make_pokedb(tmp_path)
        cache_dir = tmp_path / "icons"
        cache_dir.mkdir(parents=True)
        (cache_dir / "376.png").write_bytes(b"cached-bytes")

        with patch("generate_thumbnail.urllib.request.urlretrieve") as mock_dl:
            result = gt.fetch_pokemon_icon("メタグロス", cache_dir=cache_dir, pokedb_path=db_path)
            assert result.read_bytes() == b"cached-bytes"
            mock_dl.assert_not_called()

    def test_network_failure_returns_none(self, tmp_path):
        db_path = self._make_pokedb(tmp_path)
        cache_dir = tmp_path / "icons"

        with patch("generate_thumbnail.urllib.request.urlretrieve", side_effect=OSError("boom")):
            assert gt.fetch_pokemon_icon("メタグロス", cache_dir=cache_dir, pokedb_path=db_path) is None

    def test_unknown_pokemon_returns_none_without_network_call(self, tmp_path):
        db_path = self._make_pokedb(tmp_path)
        cache_dir = tmp_path / "icons"

        with patch("generate_thumbnail.urllib.request.urlretrieve") as mock_dl:
            result = gt.fetch_pokemon_icon("存在しないポケモン", cache_dir=cache_dir, pokedb_path=db_path)
            assert result is None
            mock_dl.assert_not_called()


class TestBuildAvatarFaceCommand:
    def test_command_contains_crop_and_chromakey_filter(self, tmp_path):
        avatar_video = tmp_path / "avatar.mp4"
        out = tmp_path / "face.png"
        cmd = gt.build_avatar_face_command("ffmpeg", avatar_video, 71.5, "400:400:800:250", out)
        assert cmd[0] == "ffmpeg"
        assert str(avatar_video) in cmd
        assert str(out) in cmd
        vf = cmd[cmd.index("-vf") + 1]
        assert "crop=400:400:800:250" in vf
        assert "chromakey=" in vf
        assert "despill=" in vf
        assert "format=rgba" in vf

    def test_negative_time_clamped_to_zero(self, tmp_path):
        cmd = gt.build_avatar_face_command(
            "ffmpeg", tmp_path / "a.mp4", -3.0, "400:400:800:250", tmp_path / "f.png")
        assert "0.0" in cmd


class TestCliDefaults:
    """2026-08-15確定運用: ユーザー承認済みの標準サムネ仕様（構築アイコン無し・
    実況テキストの代わりに固定タイトルロゴ・アバターは上半身まで大きく・
    persona=neutral）がオプション無指定でも再現できることの回帰テスト。"""

    def test_main_defaults_match_confirmed_thumbnail_spec(self, tmp_path):
        render_dir = tmp_path / "renders" / "sample"
        render_dir.mkdir(parents=True)
        with patch.object(gt, "generate_thumbnail") as mock_gen:
            mock_gen.return_value = {"out": "x", "reason": "faint", "time": 1.0,
                                     "score": 80.0, "label": ""}
            gt.main([str(render_dir)])
        _, kwargs = mock_gen.call_args
        assert kwargs["label_override"] == ""
        assert kwargs["team"] is None
        assert kwargs["persona"] == "neutral"
        assert kwargs["big_logo_text"] == "ポケモンダブルバトル\nAI自動実況"
        assert kwargs["avatar_crop"] == "663:765:631:314"
        assert kwargs["avatar_face_scale"] == 0.62
