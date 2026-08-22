"""
scripts/play_and_animate_avatar.py（改善ロードマップ③・表情連動）の単体テスト

テスト対象:
  - expression_for              イベント種別/contextからのブレンドシェイプ決定
  - build_expression_timeline   schedule.jsonのscheduledからの発火時刻リスト構築
  - _axis_angle_quat/send_idle_pose  T-pose対策の腕ボーン姿勢OSC送信
  - send_reaction/play_pose_reaction  表情変化時のポーズ遷移（explore_avatar_poses.POSES統合）
"""

import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

# pythonosc は Windows側の実行にのみ必要（WSLテスト環境には未導入のためモック）。
# sounddevice/soundfile は conftest.py で既にモック済み。
sys.modules.setdefault("pythonosc", MagicMock())
sys.modules.setdefault("pythonosc.udp_client", MagicMock())

_SCRIPT = Path(__file__).parent.parent / "scripts" / "play_and_animate_avatar.py"
_spec = importlib.util.spec_from_file_location("play_and_animate_avatar", _SCRIPT)
paa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paa)


class TestExpressionFor:
    def test_battle_start_is_fun(self):
        assert paa.expression_for("battle_start", {}) == "Fun"

    def test_move_used_with_no_commentary_is_none(self):
        assert paa.expression_for("move_used", {}) is None

    def test_filler_with_no_commentary_is_none(self):
        """キーワードにヒットしなければfillerも表情変更なし。"""
        assert paa.expression_for("filler", {}) is None

    def test_filler_reacts_to_commentary_sentiment(self):
        """2026-08-22〜: fillerも他イベントと同じくキーワード判定で反応する
        （閾値引き下げでフィラーが発話時間の大半を占めるようになり、従来の
        「雑談なので無反応」方針だとポーズの体感頻度が下がりすぎるため変更）。"""
        assert paa.expression_for("filler", {}, "バツグンだ！決まった！") == "Fun"
        assert paa.expression_for("filler", {}, "うわ、ピンチかも…削られちゃった") == "Sorrow"

    def test_battle_start_ignores_commentary_sentiment(self):
        """battle_startは_EVENT_EXPRESSIONの明示値"Fun"を常に優先する。"""
        assert paa.expression_for("battle_start", {}, "ピンチ…厳しい展開になりそう") == "Fun"

    def test_move_single_positive_commentary_is_fun(self):
        assert paa.expression_for("move_single", {}, "バツグンだ！炸裂したね〜！") == "Fun"

    def test_move_single_negative_commentary_is_sorrow(self):
        assert paa.expression_for("move_single", {}, "うわ、ピンチかも…削られちゃった") == "Sorrow"

    def test_move_single_neutral_commentary_no_change(self):
        assert paa.expression_for("move_single", {}, "淡々とターンが進んでいます") is None

    def test_switch_uses_sentiment_too(self):
        assert paa.expression_for("switch", {}, "急所に決まった〜！") == "Fun"

    def test_faint_player_side_is_sorrow(self):
        assert paa.expression_for("faint", {"faint_side": "player"}) == "Sorrow"

    def test_faint_opponent_side_is_joy(self):
        assert paa.expression_for("faint", {"faint_side": "opponent"}) == "Joy"

    def test_faint_ambiguous_side_no_change(self):
        assert paa.expression_for("faint", {"faint_side": None}) is None
        assert paa.expression_for("faint", {}) is None

    def test_faint_side_on_move_used_is_sorrow(self):
        """pipeline.py側のfaint統合バグ修正により、faint_sideはevent_type=="faint"
        単独ではなく次のmove_usedに統合されたcontextとして届く（通常経路）。"""
        assert paa.expression_for("move_used", {"faint_side": "player"}) == "Sorrow"

    def test_faint_side_on_move_used_opponent_is_joy(self):
        assert paa.expression_for("move_used", {"faint_side": "opponent"}) == "Joy"

    def test_battle_end_win_is_joy(self):
        assert paa.expression_for("battle_end", {"battle_result": "勝ち"}) == "Joy"

    def test_battle_end_lose_is_sorrow(self):
        assert paa.expression_for("battle_end", {"battle_result": "負け"}) == "Sorrow"

    def test_battle_end_unknown_result_no_change(self):
        assert paa.expression_for("battle_end", {}) is None

    def test_unknown_event_type_no_change(self):
        assert paa.expression_for("something_else", {}) is None


class TestBuildExpressionTimeline:
    def test_only_events_with_expression_included(self):
        scheduled = [
            {"start": 10.0, "event_type": "battle_start", "context": {}},
            {"start": 20.0, "event_type": "move_used", "context": {}},
            {"start": 30.0, "event_type": "faint", "context": {"faint_side": "opponent"}},
        ]
        timeline = paa.build_expression_timeline(scheduled)
        assert timeline == [(10.0, "Fun", "battle_start"), (30.0, "Joy", "faint")]

    def test_sorted_by_start_time(self):
        scheduled = [
            {"start": 30.0, "event_type": "faint", "context": {"faint_side": "player"}},
            {"start": 10.0, "event_type": "battle_start", "context": {}},
        ]
        timeline = paa.build_expression_timeline(scheduled)
        assert [t for t, _, _ in timeline] == [10.0, 30.0]

    def test_missing_context_treated_as_empty(self):
        scheduled = [{"start": 5.0, "event_type": "battle_start"}]
        assert paa.build_expression_timeline(scheduled) == [(5.0, "Fun", "battle_start")]

    def test_empty_schedule_returns_empty_timeline(self):
        assert paa.build_expression_timeline([]) == []


class TestAxisAngleQuat:
    def test_zero_degrees_is_identity(self):
        assert paa._axis_angle_quat("z", 0.0) == (0.0, 0.0, 0.0, 1.0)

    def test_z_axis_rotation_only_sets_z_and_w(self):
        qx, qy, qz, qw = paa._axis_angle_quat("z", 90.0)
        assert qx == 0.0 and qy == 0.0
        assert math.isclose(qz, math.sin(math.radians(45)), rel_tol=1e-9)
        assert math.isclose(qw, math.cos(math.radians(45)), rel_tol=1e-9)

    def test_x_axis_rotation_only_sets_x_and_w(self):
        qx, qy, qz, qw = paa._axis_angle_quat("x", 90.0)
        assert qy == 0.0 and qz == 0.0
        assert qx != 0.0

    def test_quaternion_is_unit_length(self):
        qx, qy, qz, qw = paa._axis_angle_quat("z", 37.0)
        assert math.isclose(qx**2 + qy**2 + qz**2 + qw**2, 1.0, rel_tol=1e-9)


class TestIdleBoneDeg:
    def test_zero_elapsed_is_zero(self):
        """sin(0)=0なので位相0のボーンは経過0秒で角度0。"""
        spine = paa._IDLE_BONES[0]
        assert spine.phase_rad == 0.0
        assert paa._idle_bone_deg(spine, 0.0) == 0.0

    def test_amplitude_does_not_exceed_configured_max(self):
        """主周期の半周期ごとにサンプリングして、振れ幅が主振幅+副振幅の和を超えないことを確認。"""
        for cfg in paa._IDLE_BONES:
            max_deg = max(
                abs(paa._idle_bone_deg(cfg, t))
                for t in [i * cfg.period_sec / 40 for i in range(41)]
            )
            assert max_deg <= cfg.amplitude_deg + cfg.sub_amplitude_deg + 1e-6

    def test_matches_manual_two_sine_sum(self):
        """レイヤードサイン=主周期サイン＋副周期サインの単純な合計であることを確認。"""
        cfg = paa._IDLE_BONES[0]
        t = 1.234
        expected = (
            cfg.amplitude_deg * math.sin(2 * math.pi * t / cfg.period_sec + cfg.phase_rad)
            + cfg.sub_amplitude_deg * math.sin(
                2 * math.pi * t / cfg.sub_period_sec + cfg.phase_rad * 1.7)
        )
        assert math.isclose(paa._idle_bone_deg(cfg, t), expected, abs_tol=1e-9)

    def test_without_sub_period_is_pure_sine_and_periodic(self):
        """sub_period_secが0（未指定）のボーン設定は主周期のみの単純な正弦波になる。"""
        cfg = paa._IdleBoneConfig("Test", "y", 4.0, 5.0, 0.0, 0.0, 0.0)
        a = paa._idle_bone_deg(cfg, 1.234)
        b = paa._idle_bone_deg(cfg, 1.234 + cfg.period_sec)
        assert math.isclose(a, b, abs_tol=1e-9)


class TestSwayQuat:
    def test_zero_elapsed_is_identity(self):
        assert paa._sway_quat(0.0) == (0.0, 0.0, 0.0, 1.0)


class TestRunIdleSway:
    def test_sends_to_idle_bones_and_stops_on_event(self):
        client = MagicMock()
        stop_event = __import__("threading").Event()

        def _stop_soon():
            import time as _t
            _t.sleep(paa._SWAY_UPDATE_INTERVAL_SEC * 2.5)
            stop_event.set()

        t = __import__("threading").Thread(target=_stop_soon)
        t.start()
        paa.run_idle_sway(client, stop_event, __import__("time").monotonic())
        t.join()

        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert bones_sent == {cfg.bone for cfg in paa._IDLE_BONES}
        assert client.send_message.call_count >= len(paa._IDLE_BONES) * 2


class TestRunIdleGestures:
    def test_fires_gesture_on_head_and_stops_cleanly(self):
        client = MagicMock()
        stop_event = __import__("threading").Event()
        rng = __import__("random").Random(0)

        def _stop_soon():
            import time as _t
            _t.sleep(0.05)
            stop_event.set()

        # rngのuniformを毎回ほぼ0にして即座に仕草が発火するようにする
        rng.uniform = lambda a, b: 0.0

        t = __import__("threading").Thread(target=_stop_soon)
        t.start()
        paa.run_idle_gestures(client, stop_event, rng)
        t.join()

        sent = [call.args[1] for call in client.send_message.call_args_list]
        assert all(s[0] == paa._GESTURE_BONE for s in sent)
        # 最後は停止時のニュートラル復帰
        assert tuple(sent[-1][4:8]) == (0.0, 0.0, 0.0, 1.0)


class TestSendReaction:
    def test_unknown_expression_falls_back_to_default_nod(self):
        """2026-08-03滑らか化: フォールバックのうなずきはキーフレーム間をSLERP補間する
        多フレームアニメーションになったため、送信回数はキーフレーム数より多くなる。"""
        client = MagicMock()
        paa.send_reaction(client, "Angry")
        sent = [call.args[1] for call in client.send_message.call_args_list]
        assert len(sent) > len(paa._NOD_SEQUENCE_DEG)
        assert all(s[0] == paa._NOD_BONE for s in sent)
        assert tuple(sent[-1][4:8]) == (0.0, 0.0, 0.0, 1.0)

    def test_fun_transitions_to_head_tilt_curious_pose(self):
        client = MagicMock()
        paa.send_reaction(client, "Fun")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["head_tilt_curious"]) <= bones_sent

    def test_sorrow_transitions_to_bow_apologetic_pose(self):
        client = MagicMock()
        paa.send_reaction(client, "Sorrow")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["bow_apologetic"]) <= bones_sent

    def test_joy_transitions_to_victory_arms_up_pose(self):
        client = MagicMock()
        paa.send_reaction(client, "Joy")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["victory_arms_up"]) <= bones_sent

    def test_joy_faint_variant_uses_fist_pump_right(self):
        """2026-08-15: 1匹倒す度に毎回victory_arms_up（腕を大きく振り上げ、
        --avatar-cropからはみ出しやすい）を送っていたのを、faint variantでは
        軽めのfist_pump_rightに切り替えて頻度を落とした。"""
        client = MagicMock()
        paa.send_reaction(client, "Joy", "faint")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["fist_pump_right"]) <= bones_sent

    def test_sorrow_sentiment_variant_uses_thinking_chin(self):
        client = MagicMock()
        paa.send_reaction(client, "Sorrow", "sentiment")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["thinking_chin"]) <= bones_sent

    def test_fun_battle_start_variant_uses_lean_back_confident(self):
        client = MagicMock()
        paa.send_reaction(client, "Fun", "battle_start")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["lean_back_confident"]) <= bones_sent

    def test_unknown_variant_falls_back_to_default_pose(self):
        client = MagicMock()
        paa.send_reaction(client, "Joy", "some_unmapped_variant")
        bones_sent = {call.args[1][0] for call in client.send_message.call_args_list}
        assert set(paa.eap.POSES["victory_arms_up"]) <= bones_sent


class TestPoseVariantKey:
    def test_faint_side_in_context_wins_regardless_of_event_type(self):
        """faint_sideはfaint単独dispatchだけでなくmove_used統合経路にも乗る
        （expression_forと同じ優先順）。"""
        assert paa._pose_variant_key("move_used", {"faint_side": "opponent"}) == "faint"
        assert paa._pose_variant_key("faint", {"faint_side": "player"}) == "faint"

    def test_battle_start(self):
        assert paa._pose_variant_key("battle_start", {}) == "battle_start"

    def test_sentiment_event_types(self):
        for et in ("move_used", "move_single", "switch"):
            assert paa._pose_variant_key(et, {}) == "sentiment"

    def test_battle_end_is_default(self):
        assert paa._pose_variant_key("battle_end", {"battle_result": "勝ち"}) == "default"


class TestSendSmoothKeyframes:
    def test_ends_exactly_at_last_keyframe(self):
        client = MagicMock()
        paa._send_smooth_keyframes(client, "Neck", "x", (0.0, 10.0, 0.0), 0.05)
        sent = [call.args[1] for call in client.send_message.call_args_list]
        assert all(s[0] == "Neck" for s in sent)
        expected_last = paa._axis_angle_quat("x", 0.0)
        assert tuple(sent[-1][4:8]) == expected_last

    def test_single_keyframe_sends_nothing(self):
        client = MagicMock()
        paa._send_smooth_keyframes(client, "Neck", "x", (5.0,), 0.05)
        client.send_message.assert_not_called()


class TestPlayPoseReaction:
    def test_returns_arms_to_idle_pose_angle_not_neutral(self):
        """最後は必ずidle_downへ戻すので、腕はsend_idle_poseと同じ角度に戻る
        （0度=T-poseに逆戻りしてはいけない）。"""
        client = MagicMock()
        paa.play_pose_reaction(client, "victory_arms_up", hold_sec=0.01, transition_sec=0.03)
        sent = [call.args[1] for call in client.send_message.call_args_list]
        last_by_bone = {}
        for s in sent:
            last_by_bone[s[0]] = s
        for bone in paa._IDLE_POSE_BONES:
            expected_deg = -paa._IDLE_POSE_DEG if "Right" in bone else paa._IDLE_POSE_DEG
            expected_qz = paa._axis_angle_quat(paa._IDLE_POSE_AXIS, expected_deg)[2]
            assert math.isclose(last_by_bone[bone][6], expected_qz, abs_tol=1e-6)

    def test_suspends_pose_bones_during_playback_and_resumes_after(self):
        client = MagicMock()
        assert paa._suspended_bones == set()
        paa.play_pose_reaction(client, "bow_apologetic", hold_sec=0.01, transition_sec=0.03)
        assert paa._suspended_bones == set()


class TestSuspendedBones:
    def test_run_idle_sway_skips_suspended_bone(self):
        client = MagicMock()
        stop_event = __import__("threading").Event()
        bone = paa._IDLE_BONES[0].bone
        paa._suspend_bones({bone})
        try:
            def _stop_soon():
                import time as _t
                _t.sleep(paa._SWAY_UPDATE_INTERVAL_SEC * 2.5)
                stop_event.set()

            t = __import__("threading").Thread(target=_stop_soon)
            t.start()
            paa.run_idle_sway(client, stop_event, __import__("time").monotonic())
            t.join()
        finally:
            paa._resume_bones({bone})

        # 停止時の無条件ニュートラル復帰（末尾1回）を除けば、サスペンド中の
        # ボーンには揺れの値が一切送られていないはず。
        sent_for_bone = [call.args[1] for call in client.send_message.call_args_list
                         if call.args[1][0] == bone]
        assert len(sent_for_bone) == 1
        assert tuple(sent_for_bone[0][4:8]) == (0.0, 0.0, 0.0, 1.0)


class TestSendIdlePose:
    def test_sends_pose_for_each_configured_bone(self):
        client = MagicMock()
        paa.send_idle_pose(client)
        sent_bones = [call.args[1][0] for call in client.send_message.call_args_list]
        assert sent_bones == paa._IDLE_POSE_BONES

    def test_left_and_right_arm_use_mirrored_sign(self):
        """実機確認済み: LeftUpperArmと同じ符号だとRightは逆方向に曲がるため、
        Rightを含むボーン名は符号を反転して送る（左右対称の下げポーズにするため）。"""
        client = MagicMock()
        paa.send_idle_pose(client)
        sent = {call.args[1][0]: call.args[1] for call in client.send_message.call_args_list}
        # [name, x, y, z, qx, qy, qz, qw] のインデックス6がqz
        left_z = sent["LeftUpperArm"][6]
        right_z = sent["RightUpperArm"][6]
        assert math.isclose(left_z, -right_z, rel_tol=1e-9)
        assert left_z != 0.0
