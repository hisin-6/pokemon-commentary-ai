"""
render_commentary_video.py（実況動画合成・ADR-009 パス2）の単体テスト

テスト対象:
  - dedupe_entries          近接時間内の同一実況文の除去
  - schedule_entries        後ろ倒しスケジューリング（かぶり解決・遅延超過破棄）
  - resolve_video_path      Windowsパス→WSLパス変換
  - build_commentary_track  実況トラックWAVの配置・長さ
"""

import importlib.util
import io
import json
import wave
from pathlib import Path

import pytest

# scripts/ はパッケージではないためファイルパスから直接ロードする
_SCRIPT = Path(__file__).parent.parent / "scripts" / "render_commentary_video.py"
_spec = importlib.util.spec_from_file_location("render_commentary_video", _SCRIPT)
rcv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcv)


def _entry(event_time, duration=10.0, commentary="実況", event_type="move_used",
           wav="wav/0001_move_used.wav", seq=1):
    return {"seq": seq, "event_time": event_time, "event_type": event_type,
            "commentary": commentary, "wav": wav, "duration": duration}


class TestDedupeEntries:
    def test_identical_text_within_window_dropped(self):
        """近接時間の同一実況文は後の方が破棄される（faint/switch同文の実測ケース）。"""
        entries = [_entry(133.2, commentary="同じ実況", event_type="faint"),
                   _entry(144.0, commentary="同じ実況", event_type="switch")]
        kept, dropped = rcv.dedupe_entries(entries)
        assert len(kept) == 1
        assert kept[0]["event_time"] == 133.2  # 早い方が残る
        assert len(dropped) == 1
        assert dropped[0]["event_type"] == "switch"

    def test_identical_text_outside_window_kept(self):
        """時間が離れた同一文（定型実況など）は両方残る。"""
        entries = [_entry(10.0, commentary="同じ実況"),
                   _entry(10.0 + rcv._DEDUPE_WINDOW_SEC + 1, commentary="同じ実況")]
        kept, dropped = rcv.dedupe_entries(entries)
        assert len(kept) == 2
        assert dropped == []

    def test_different_text_kept(self):
        entries = [_entry(10.0, commentary="実況A"), _entry(12.0, commentary="実況B")]
        kept, dropped = rcv.dedupe_entries(entries)
        assert len(kept) == 2
        assert dropped == []


class TestScheduleEntries:
    def test_no_overlap_keeps_event_time(self):
        """かぶりが無ければイベント時刻ちょうどに配置される。"""
        entries = [_entry(10.0, duration=5.0), _entry(30.0, duration=5.0)]
        scheduled, dropped = rcv.schedule_entries(entries)
        assert [e["start"] for e in scheduled] == [10.0, 30.0]
        assert all(e["delay"] == 0.0 for e in scheduled)
        assert dropped == []

    def test_overlap_pushed_back_with_gap(self):
        """かぶったら前の実況の終了+gapまで後ろ倒しされる。"""
        entries = [_entry(10.0, duration=10.0), _entry(15.0, duration=5.0)]
        scheduled, _ = rcv.schedule_entries(entries, gap=0.5)
        assert scheduled[1]["start"] == pytest.approx(20.5)
        assert scheduled[1]["delay"] == pytest.approx(5.5)

    def test_chained_pushback(self):
        """後ろ倒しが連鎖する（3件が数珠つなぎ）。"""
        entries = [_entry(10.0, duration=10.0), _entry(11.0, duration=10.0),
                   _entry(12.0, duration=10.0)]
        scheduled, _ = rcv.schedule_entries(entries, gap=0.5)
        assert scheduled[1]["start"] == pytest.approx(20.5)
        assert scheduled[2]["start"] == pytest.approx(31.0)

    def test_exceeding_max_delay_dropped(self):
        """遅延がmax_delayを超える実況は破棄され、後続はかぶらなければ定刻。"""
        entries = [_entry(10.0, duration=30.0), _entry(12.0, duration=5.0),
                   _entry(50.0, duration=5.0)]
        scheduled, dropped = rcv.schedule_entries(entries, gap=0.5, max_delay=20.0)
        assert len(dropped) == 1
        assert dropped[0]["event_time"] == 12.0
        assert dropped[0]["delay"] == pytest.approx(28.5)
        # 破棄されたエントリはカーソルを進めない
        assert [e["start"] for e in scheduled] == [10.0, 50.0]

    def test_unsorted_input_sorted_by_event_time(self):
        """manifest保存順（faintフラッシュで時刻順と不一致）でも時刻順に整列。"""
        entries = [_entry(144.0, event_type="switch"), _entry(133.2, event_type="faint")]
        scheduled, _ = rcv.schedule_entries(entries)
        assert [e["event_time"] for e in scheduled] == [133.2, 144.0]


def _filler(event_time, duration=8.0, commentary="フィラー", seq=1):
    return {"seq": seq, "event_time": event_time, "event_type": "filler",
            "commentary": commentary, "wav": f"wav/f{seq:04d}_filler.wav",
            "duration": duration}


class TestFitFillers:
    def test_placed_at_desired_time_when_free(self):
        """空いていれば希望時刻ちょうどに置かれる。"""
        scheduled = [dict(_entry(10.0, duration=10.0), start=10.0),
                     dict(_entry(100.0, duration=10.0), start=100.0)]
        placed, dropped = rcv.fit_fillers(scheduled, [_filler(50.0)])
        assert len(placed) == 1
        assert placed[0]["start"] == 50.0
        assert dropped == []

    def test_pushed_past_event_when_shift_within_max(self):
        """イベント実況と重なるフィラーはイベントを動かさず後ろへずれる
        （ずれがmax_shift以内なら採用）。"""
        scheduled = [dict(_entry(50.0, duration=10.0), start=50.0)]
        placed, _ = rcv.fit_fillers(scheduled, [_filler(48.0)],
                                    gap=0.5, max_shift=15.0)
        assert placed[0]["start"] == pytest.approx(60.5)

    def test_dropped_when_shift_exceeds_max(self):
        """max_shiftを超えるずれが必要なフィラーは破棄される
        （既定12秒: 長いイベント実況を飛び越える配置は基本却下）。"""
        scheduled = [dict(_entry(50.0, duration=13.0), start=50.0)]
        placed, dropped = rcv.fit_fillers(scheduled, [_filler(48.0)], gap=0.5)
        assert placed == []
        assert len(dropped) == 1
        assert dropped[0]["delay"] == pytest.approx(15.5)

    def test_fillers_do_not_overlap_each_other(self):
        """フィラー同士も重ならない（先に置いた方が占有区間になる）。"""
        placed, _ = rcv.fit_fillers([], [_filler(10.0, duration=8.0, seq=1),
                                         _filler(12.0, duration=8.0, seq=2)],
                                    gap=0.5)
        assert placed[0]["start"] == 10.0
        assert placed[1]["start"] == pytest.approx(18.5)

    def test_event_before_filler_not_pushed(self):
        """フィラーの後ろ倒しでも後続イベント実況は動かない。"""
        scheduled = [dict(_entry(10.0, duration=10.0), start=10.0),
                     dict(_entry(30.0, duration=5.0), start=30.0)]
        # 28秒に8秒フィラー → 30秒のイベントとぶつかる → イベント後の35.5へ
        placed, _ = rcv.fit_fillers(scheduled, [_filler(28.0, duration=8.0)],
                                    gap=0.5)
        assert placed[0]["start"] == pytest.approx(35.5)
        # イベント側は不変
        assert [e["start"] for e in scheduled] == [10.0, 30.0]


class TestLoadFillers:
    def test_missing_file_returns_empty(self, tmp_path):
        assert rcv.load_fillers(tmp_path) == []

    def test_sorted_by_event_time(self, tmp_path):
        lines = [json.dumps(_filler(100.0, seq=2), ensure_ascii=False),
                 json.dumps(_filler(50.0, seq=1), ensure_ascii=False)]
        (tmp_path / "fillers.jsonl").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
        fillers = rcv.load_fillers(tmp_path)
        assert [f["event_time"] for f in fillers] == [50.0, 100.0]


class TestResolveVideoPath:
    def test_windows_drive_path_converted_to_wsl(self, tmp_path, monkeypatch):
        """存在しないWindowsパスは /mnt/<drive>/ 形式に変換して探す。"""
        monkeypatch.setattr(rcv.sys, "platform", "linux")
        # /mnt/x/... は存在しないので変換候補も見つからず元パスが返る
        result = rcv.resolve_video_path("X:\\foo\\bar.mp4")
        assert str(result) == "X:\\foo\\bar.mp4"

    def test_existing_path_returned_as_is(self, tmp_path):
        video = tmp_path / "battle.mp4"
        video.write_bytes(b"")
        assert rcv.resolve_video_path(str(video)) == video


def _write_wav(path, seconds, rate=24000, value=1000):
    """一定振幅のモノラル16bit WAVを生成する。"""
    frames = int(seconds * rate)
    data = value.to_bytes(2, "little", signed=True) * frames
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


class TestBuildCommentaryTrack:
    def test_clips_placed_at_scheduled_time(self, tmp_path):
        """各WAVがstart秒の位置に配置され、間は無音になる。"""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        _write_wav(wav_dir / "0001_a.wav", 1.0)
        _write_wav(wav_dir / "0002_b.wav", 1.0)
        scheduled = [
            {"start": 2.0, "wav": "wav/0001_a.wav"},
            {"start": 5.0, "wav": "wav/0002_b.wav"},
        ]
        out = tmp_path / "track.wav"
        track_len = rcv.build_commentary_track(tmp_path, scheduled, out)
        assert track_len == pytest.approx(6.0)

        with wave.open(str(out), "rb") as w:
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())

        def sample_at(sec):
            i = int(sec * rate) * 2
            return int.from_bytes(raw[i:i + 2], "little", signed=True)

        assert sample_at(1.0) == 0       # 配置前は無音
        assert sample_at(2.5) == 1000    # クリップ1の途中
        assert sample_at(4.0) == 0       # クリップ間は無音
        assert sample_at(5.5) == 1000    # クリップ2の途中

    def test_min_duration_pads_track(self, tmp_path):
        """min_duration指定でトラックが動画長まで無音パディングされる。"""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        _write_wav(wav_dir / "0001_a.wav", 1.0)
        scheduled = [{"start": 0.0, "wav": "wav/0001_a.wav"}]
        out = tmp_path / "track.wav"
        track_len = rcv.build_commentary_track(tmp_path, scheduled, out,
                                               min_duration=10.0)
        assert track_len == pytest.approx(10.0)

    def test_format_mismatch_raises(self, tmp_path):
        """WAVフォーマット不一致は明示的にエラーにする。"""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        _write_wav(wav_dir / "0001_a.wav", 1.0, rate=24000)
        _write_wav(wav_dir / "0002_b.wav", 1.0, rate=48000)
        scheduled = [
            {"start": 0.0, "wav": "wav/0001_a.wav"},
            {"start": 2.0, "wav": "wav/0002_b.wav"},
        ]
        with pytest.raises(ValueError, match="フォーマット不一致"):
            rcv.build_commentary_track(tmp_path, scheduled, tmp_path / "track.wav")

    def test_empty_schedule_raises(self, tmp_path):
        with pytest.raises(ValueError, match="0件"):
            rcv.build_commentary_track(tmp_path, [], tmp_path / "track.wav")


class TestWrapJp:
    def test_short_text_unchanged(self):
        assert rcv._wrap_jp("短い文") == "短い文"

    def test_long_text_wrapped_within_width(self):
        """全行がwidth+2（禁則ぶら下げ許容）以内・文字の欠落なし。"""
        text = "「インファイト、ねむりごな。相手のフシギバナがペリッパーを眠らせてきました。眠り状態は大きなハンデです。」"
        wrapped = rcv._wrap_jp(text)
        lines = wrapped.split("\\N")
        assert len(lines) >= 2
        assert all(len(l) <= rcv._SUBTITLE_WRAP_CHARS + 2 for l in lines)
        assert "".join(lines) == text

    def test_punctuation_preferred_break(self):
        text = "あ" * 30 + "。" + "い" * 20
        assert rcv._wrap_jp(text).split("\\N")[0].endswith("。")

    def test_closing_bracket_not_at_line_head(self):
        """閉じ括弧が行頭に来ない（禁則ぶら下げ）。"""
        text = "あ" * (rcv._SUBTITLE_WRAP_CHARS * 2) + "」"
        for line in rcv._wrap_jp(text).split("\\N"):
            assert not line.startswith("」")


class TestBuildAss:
    def test_time_format(self):
        assert rcv._ass_time(0.0) == "0:00:00.00"
        assert rcv._ass_time(63.0) == "0:01:03.00"
        assert rcv._ass_time(3723.456) == "1:02:03.46"

    def test_escape_override_tags(self):
        assert rcv._ass_escape("a{b}c\nd") == "a｛b｝c\\Nd"

    def test_dialogue_lines_with_styles(self, tmp_path):
        """イベント=Event・フィラー=Fillerスタイルで、音声区間+余韻の字幕が出る。"""
        scheduled = [dict(_entry(63.0, duration=10.0, commentary="開幕だ"), start=63.0),
                     dict(_filler(100.0, duration=8.0, commentary="つなぎ"), start=100.0)]
        out = tmp_path / "c.ass"
        rcv.build_ass(scheduled, out)
        text = out.read_text(encoding="utf-8")
        assert "Dialogue: 0,0:01:03.00,0:01:14.50,Event,,0,0,0,,「開幕だ」" in text
        assert "Dialogue: 0,0:01:40.00,0:01:49.50,Filler,,0,0,0,,「つなぎ」" in text

    def test_linger_capped_by_next_entry(self, tmp_path):
        """余韻表示は次の実況開始の0.1秒前まででカットされる。"""
        scheduled = [dict(_entry(10.0, duration=5.0), start=10.0),
                     dict(_entry(15.5, duration=5.0), start=15.5)]
        out = tmp_path / "c.ass"
        rcv.build_ass(scheduled, out)
        first = [l for l in out.read_text(encoding="utf-8").splitlines()
                 if l.startswith("Dialogue")][0]
        assert ",0:00:10.00,0:00:15.40," in first


class TestBiimCommand:
    def test_contains_layout_filters_and_encode(self, tmp_path):
        cmd = rcv.build_ffmpeg_command_biim(
            Path("in.mp4"), Path("track.wav"), Path("out.mp4"),
            tmp_path / "c.ass", gain=1.4, duck_threshold=0.03, duck_ratio=8.0)
        joined = " ".join(cmd)
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "scale=1440:810" in fc
        assert "pad=1920:1080:16:12" in fc
        assert "subtitles=" in fc and "fontsdir=" in fc
        assert "sidechaincompress" in fc  # 音声チェインはplainと同一
        assert "-c:v libx264" in joined   # レイアウト焼き込みのため再エンコード
        assert "[vout]" in joined and "[aout]" in joined

    def test_tail_pad_inserts_tpad_before_scale(self, tmp_path):
        """末尾実況が動画終端をまたぐ場合の映像延長（tpad）。字幕・パネル描画の
        前段に入ること（後段だと延長区間で字幕が最終フレームごと固まる）。"""
        cmd = rcv.build_ffmpeg_command_biim(
            Path("in.mp4"), Path("track.wav"), Path("out.mp4"),
            tmp_path / "c.ass", gain=1.4, duck_threshold=0.03, duck_ratio=8.0,
            tail_pad=6.563)
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "tpad=stop_mode=clone:stop_duration=6.563" in fc
        assert fc.index("tpad=") < fc.index("scale=1440:810")

    def test_tail_pad_zero_omits_tpad(self, tmp_path):
        """tail_pad=0（既定）ではtpadを挿入しない（従来コマンドと同一）。"""
        cmd = rcv.build_ffmpeg_command_biim(
            Path("in.mp4"), Path("track.wav"), Path("out.mp4"),
            tmp_path / "c.ass", gain=1.4, duck_threshold=0.03, duck_ratio=8.0)
        fc = cmd[cmd.index("-filter_complex") + 1]
        # 注: "tpad" 単体だとtmp_pathのテスト名（..._tpad0）に誤マッチする
        assert "tpad=" not in fc


def _panel_state(t, turn=3, player=None, opponent=None, ap=2, ao=2):
    return {"time": t, "turn": turn,
            "player": player if player is not None else
            [{"name": "イダイトウ", "hp_pct": 47, "hp_text": "47%", "status": None}],
            "opponent": opponent if opponent is not None else [],
            "alive_player": ap, "alive_opponent": ao}


class TestPanelEvents:
    def test_hp_bar_color_thresholds(self):
        assert rcv._hp_bar_color(100) == rcv._hp_bar_color(51)   # 緑
        assert rcv._hp_bar_color(50) == rcv._hp_bar_color(21)    # 黄
        assert rcv._hp_bar_color(20) == rcv._hp_bar_color(1)     # 赤
        assert len({rcv._hp_bar_color(100), rcv._hp_bar_color(50),
                    rcv._hp_bar_color(10)}) == 3

    def test_bar_width_scales_with_pct(self):
        """HPバーの塗り幅がhp_pctに比例する。"""
        lines = rcv._panel_dialogues(10.0, 20.0, _panel_state(10.0), {})
        fill = [l for l in lines if "l 141 0" in l]  # 300*47% = 141
        assert len(fill) == 1

    def test_state_selected_per_keyframe(self):
        """各キーフレームで直近の状態が使われ、区間が次のキーフレームまで続く。"""
        states = [_panel_state(100.0, turn=3), _panel_state(200.0, turn=4)]
        dialogues = rcv.build_panel_events(states, [], video_end=300.0)
        text = "\n".join(dialogues)
        assert "0:01:40.00,0:03:20.00" in text  # 100→200秒の区間
        assert "ターン 3" in text and "ターン 4" in text
        assert "残り" not in text  # ゲーム画面と重複するため表示しない

    def test_moves_revealed_at_moment_time(self):
        """技表示は瞬間ログの時刻で?が埋まる（前は?のまま）。"""
        states = [_panel_state(50.0)]
        moments = [{"time": 100.0, "kind": "move", "text": "T2:イダイトウのだくりゅう"}]
        dialogues = rcv.build_panel_events(states, moments, video_end=200.0)
        before = [l for l in dialogues if l.startswith("Dialogue: 1,0:00:50.00")]
        after = [l for l in dialogues if l.startswith("Dialogue: 1,0:01:40.00")]
        assert any("技:?/?" in l for l in before)
        assert any("技:だくりゅう/?" in l for l in after)

    def test_moves_by_pokemon_dedupes(self):
        moments = [{"time": float(i), "kind": "move",
                    "text": f"T1:ピカチュウの技{i % 5}"} for i in range(10)]
        moves = rcv._moves_by_pokemon(moments, until=100.0)
        # 重複なしで(技名, 陣営)を保持。表示4枠キャップは_moves_for_side側
        assert [mv for mv, _ in moves["ピカチュウ"]] == ["技0", "技1", "技2", "技3", "技4"]
        assert rcv._moves_for_side(moves, "ピカチュウ", "自分") == ["技0", "技1", "技2", "技3"]

    def test_moves_for_side_filters_mirror_moves(self):
        """陣営タグ付きは一致側のみ・タグ無し（旧データ）は両側に出る
        （同名ミラーで両陣営の技が同じ欄に混ざる問題の対策・2026-07-30）。"""
        moments = [
            {"time": 1.0, "kind": "move", "text": "T1:イダイトウのだくりゅう", "side": "自分"},
            {"time": 2.0, "kind": "move", "text": "T1:イダイトウのおはかまいり", "side": "相手"},
            {"time": 3.0, "kind": "move", "text": "T2:イダイトウのこごえるかぜ"},  # 旧形式
        ]
        moves = rcv._moves_by_pokemon(moments, until=100.0)
        assert rcv._moves_for_side(moves, "イダイトウ", "自分") == ["だくりゅう", "こごえるかぜ"]
        assert rcv._moves_for_side(moves, "イダイトウ", "相手") == ["おはかまいり", "こごえるかぜ"]

    def test_moves_lines_format(self):
        line1, line2 = rcv._moves_lines(["インファイト", "ねこだまし", "まもる"])
        assert line1 == "技:インファイト/ねこだまし"
        assert line2 == "　　まもる/?"

    def test_no_states_returns_empty(self):
        """statesが無ければパネルは描画しない（技だけの表示はしない）。"""
        moments = [{"time": 10.0, "kind": "move", "text": "T1:Aのたいあたり"}]
        assert rcv.build_panel_events([], moments, video_end=100.0) == []

    def test_empty_side_shows_placeholder(self):
        lines = rcv._panel_dialogues(0.0, 10.0, _panel_state(0.0, opponent=[]), {})
        assert any("情報収集中" in l for l in lines)


class TestAvatarCommand:
    def test_without_avatar_has_two_inputs(self, tmp_path):
        cmd = rcv.build_ffmpeg_command_biim(
            Path("in.mp4"), Path("t.wav"), Path("o.mp4"), tmp_path / "c.ass",
            gain=1.4, duck_threshold=0.03, duck_ratio=8.0)
        assert cmd.count("-i") == 2
        assert "chromakey" not in " ".join(cmd)

    def test_with_avatar_adds_chromakey_overlay(self, tmp_path):
        """アバター指定時は第3入力＋クロマキー＋右下overlay＋末尾静止。"""
        cmd = rcv.build_ffmpeg_command_biim(
            Path("in.mp4"), Path("t.wav"), Path("o.mp4"), tmp_path / "c.ass",
            gain=1.4, duck_threshold=0.03, duck_ratio=8.0,
            avatar_video=Path("avatar.mp4"), avatar_offset=1.5)
        joined = " ".join(cmd)
        assert cmd.count("-i") == 3
        # -ss がアバター入力の直前に付く（頭合わせ・並びは "-ss 1.5 -i avatar.mp4"）
        idx = cmd.index("avatar.mp4")
        assert cmd[idx - 3:idx] == ["-ss", "1.5", "-i"]
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "chromakey=0x00FF00:0.25" in fc
        assert "scale=344:-2" in fc
        assert "eof_action=repeat" in fc
        # 字幕の上にアバターが乗る（subtitles → overlay の順）
        assert fc.index("subtitles=") < fc.index("eof_action=repeat")

    def test_negative_offset_rejected(self, tmp_path):
        """負のオフセットは非対応（録画を先に開始する運用で統一）。"""
        with pytest.raises(ValueError, match="0以上"):
            rcv.build_ffmpeg_command_biim(
                Path("in.mp4"), Path("t.wav"), Path("o.mp4"), tmp_path / "c.ass",
                gain=1.4, duck_threshold=0.03, duck_ratio=8.0,
                avatar_video=Path("avatar.mp4"), avatar_offset=-1.0)


class TestLoadStates:
    def test_missing_file_returns_empty(self, tmp_path):
        assert rcv.load_states(tmp_path) == []

    def test_sorted_by_time(self, tmp_path):
        lines = [json.dumps(_panel_state(200.0), ensure_ascii=False),
                 json.dumps(_panel_state(100.0), ensure_ascii=False)]
        (tmp_path / "states.jsonl").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
        states = rcv.load_states(tmp_path)
        assert [s["time"] for s in states] == [100.0, 200.0]


class TestLoadManifest:
    def test_sorted_by_event_time(self, tmp_path):
        """保存順が時刻順と違っても event_time 昇順で返る。"""
        lines = [
            json.dumps(_entry(144.0, seq=2), ensure_ascii=False),
            json.dumps(_entry(133.2, seq=3), ensure_ascii=False),
            json.dumps(_entry(63.0, seq=1), ensure_ascii=False),
        ]
        (tmp_path / "manifest.jsonl").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        entries = rcv.load_manifest(tmp_path)
        assert [e["event_time"] for e in entries] == [63.0, 133.2, 144.0]
