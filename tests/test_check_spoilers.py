"""
check_spoilers.py（フィラーのネタバレ検査・パス1.5→パス2間の必須工程）の単体テスト

テスト対象:
  - find_spoilers  未来の技名に言及するフィラーの検出
"""

import importlib.util
from pathlib import Path

# scripts/ はパッケージではないためファイルパスから直接ロードする
_SCRIPT = Path(__file__).parent.parent / "scripts" / "check_spoilers.py"
_spec = importlib.util.spec_from_file_location("check_spoilers", _SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _filler(t, text):
    return {"seq": 1, "event_time": t, "event_type": "filler",
            "commentary": text, "wav": "wav/x.wav", "duration": 13.0}


def _moment(t, text):
    return {"time": t, "kind": "move", "text": text}


class TestFindSpoilers:
    def test_future_move_mention_flagged(self):
        """まだ画面に映っていない技名への言及を検出する（実測ケース）。"""
        fillers = [_filler(233.3, "シャカシャカほうの展開、相手の新しい戦術です")]
        moments = [_moment(256.9, "T4:ヤバソチャのシャカシャカほう")]
        flags = cs.find_spoilers(fillers, moments, events=[])
        assert len(flags) == 1
        assert flags[0]["move"] == "シャカシャカほう"
        assert flags[0]["known_at"] == 256.9

    def test_past_move_mention_not_flagged(self):
        """既に映った技への言及は正常（初出時刻より後ならOK）。"""
        fillers = [_filler(215.0, "ふいうちが決まった！")]
        moments = [_moment(205.4, "T3:ドドゲザンのふいうち")]
        assert cs.find_spoilers(fillers, moments, events=[]) == []

    def test_repeated_move_uses_first_occurrence(self):
        """同じ技が複数回使われる場合、初出時刻を基準にする（誤フラグ防止）。"""
        fillers = [_filler(190.0, "だくりゅうが再度くり出される")]
        moments = [_moment(105.7, "T1:イダイトウのだくりゅう"),
                   _moment(242.9, "T4:イダイトウのだくりゅう")]
        assert cs.find_spoilers(fillers, moments, events=[]) == []

    def test_event_commentary_makes_move_known_earlier(self):
        """イベント実況が先に技名に触れていれば、その後のフィラー言及はOK。"""
        fillers = [_filler(120.0, "しおふきの威力はすさまじい")]
        moments = [_moment(233.7, "T4:カメックスのしおふき")]
        events = [{"event_time": 100.0, "event_type": "move_used",
                   "commentary": "カメックスのしおふきが炸裂だ！"}]
        assert cs.find_spoilers(fillers, moments, events) == []

    def test_tolerance_near_moment_time(self):
        """瞬間時刻の直前（許容誤差内）は検知タイミングの微差としてフラグしない。"""
        fillers = [_filler(123.0, "ソーラービームだ")]
        moments = [_moment(123.4, "T1:リザードンのソーラービーム")]
        assert cs.find_spoilers(fillers, moments, events=[]) == []
