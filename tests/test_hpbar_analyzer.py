"""HpBarAnalyzer のテスト

セグメント密度検証（攻撃アニメーション中の背景映り込み棄却）と
安定化フィルタの基本動作を確認する。

実測に基づく背景:
- 本物のHPバーはベタ塗りで着色密度1.00
- 攻撃アニメーション中はHUDが消え、ROIに映り込んだ炎エフェクト等の散点が
  _GAP_TOLERANCE の橋渡しで偽セグメントになる（実測: 密度0.33の15pxで7%誤確定）
"""

import cv2
import numpy as np
import pytest

from src.capture.hpbar_analyzer import HpBarAnalyzer, SlotConfig

# テスト用スロット: 幅100px・player_0 相当の位置
_SLOT = {"player_0": SlotConfig(x_left=100, x_right=200, y_top=50, y_bottom=55, label="player_0")}
_GREEN = (0, 255, 0)  # HSV hue=60 → 緑レンジ（42-85）に入る


def _blank_frame():
    return np.zeros((100, 300, 3), dtype=np.uint8)


def _solid_bar_frame(fill_px: int):
    """左端から fill_px 分をベタ塗り緑にしたフレーム（本物のバー相当）"""
    frame = _blank_frame()
    frame[50:55, 100:100 + fill_px] = _GREEN
    return frame


def _scattered_frame(span_px: int, step: int = 6):
    """span_px の範囲に step 間隔で1px幅の緑を散らしたフレーム
    （攻撃アニメ中の炎エフェクト映り込み相当。gap=step-1 ≤ _GAP_TOLERANCE で
    1本の偽セグメントに橋渡しされるが密度は 1/step と低い）"""
    frame = _blank_frame()
    for x in range(100, 100 + span_px, step):
        frame[50:55, x:x + 1] = _GREEN
    return frame


def _analyze_n(analyzer, frame, n=6):
    """安定化ウィンドウ分（デフォルト6フレーム）同一フレームを流して最終結果を返す"""
    result = None
    for _ in range(n):
        result = analyzer.analyze(frame)
    return result


class TestSegmentDensity:

    def setup_method(self):
        self.an = HpBarAnalyzer(slots=dict(_SLOT))

    def test_solid_full_bar_returns_100pct(self):
        result = _analyze_n(self.an, _solid_bar_frame(100))
        assert result["player_0"] == pytest.approx(1.0, abs=0.01)

    def test_solid_partial_bar_returns_ratio(self):
        result = _analyze_n(self.an, _solid_bar_frame(60))
        assert result["player_0"] == pytest.approx(0.6, abs=0.02)

    def test_scattered_segment_rejected(self):
        """散点セグメント（密度 ≈ 0.17 < 0.85）は棄却され None のまま。"""
        result = _analyze_n(self.an, _scattered_frame(60))
        assert result["player_0"] is None

    def test_scattered_does_not_overwrite_stable_value(self):
        """確定済みの値は、後続の映り込みフレームで上書きされず保持される。
        （実機: いわなだれアニメ中に リザードン100% が 7% に誤確定した回帰ガード）"""
        _analyze_n(self.an, _solid_bar_frame(100))
        result = _analyze_n(self.an, _scattered_frame(20))
        assert result["player_0"] == pytest.approx(1.0, abs=0.01)

    def test_tiny_segment_below_min_width_rejected(self):
        """_MIN_SEG_WIDTH 未満のセグメントはノイズとして棄却。"""
        result = _analyze_n(self.an, _solid_bar_frame(10))
        assert result["player_0"] is None


def _hsv_to_bgr(h, s, v):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


class TestHudAnchor:
    """デフォルト構成（1920x1080・アンカー有効）でのHUD表示検証。

    攻撃アニメーション中はHUDが消え、ROIに映り込んだベタ塗りの炎エフェクトは
    色・密度ではバーと区別できない（実測: 繰り出し演出の炎が56%を誤確定）。
    バーとセットで表示されるネームプレート帯の色をアンカーとして検証する。
    """

    def setup_method(self):
        self.an = HpBarAnalyzer()  # デフォルト = アンカー有効

    def _frame_with_player0_bar(self, with_band: bool):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[1005:1010, 178:405] = (0, 255, 0)  # player_0 バー満タン相当
        if with_band:
            # 自分側ネームバンド（実測中央値 H119 S143 V232）
            frame[950:974, 188:288] = _hsv_to_bgr(119, 143, 232)
        return frame

    def test_bar_with_name_band_accepted(self):
        frame = self._frame_with_player0_bar(with_band=True)
        result = None
        for _ in range(6):
            result = self.an.analyze(frame)
        assert result["player_0"] == pytest.approx(1.0, abs=0.01)

    def test_bar_without_name_band_rejected(self):
        """バー相当の色があってもネームバンドが無い（=HUD非表示のシーン映り込み）
        フレームは棄却され、読み値は付かない。"""
        frame = self._frame_with_player0_bar(with_band=False)
        result = None
        for _ in range(6):
            result = self.an.analyze(frame)
        assert result["player_0"] is None

    def test_scene_flame_does_not_overwrite_stable_value(self):
        """確定済みの値は、HUD非表示中のベタ塗り映り込みでも上書きされない。
        （実機: 繰り出し演出の炎で オオニューラ 56.2% が誤確定した回帰ガード）"""
        for _ in range(6):
            self.an.analyze(self._frame_with_player0_bar(with_band=True))
        flame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        flame[1000:1020, 100:500] = _hsv_to_bgr(15, 200, 220)  # 橙の炎ベタ塗り
        result = None
        for _ in range(6):
            result = self.an.analyze(flame)
        assert result["player_0"] == pytest.approx(1.0, abs=0.01)

    def test_custom_slots_disable_anchor(self):
        """カスタムslots指定時はアンカー無効（従来挙動・テスト互換）。"""
        an = HpBarAnalyzer(slots=dict(_SLOT))
        frame = _solid_bar_frame(100)
        result = None
        for _ in range(6):
            result = an.analyze(frame)
        assert result["player_0"] == pytest.approx(1.0, abs=0.01)


class TestStabilizer:

    def setup_method(self):
        self.an = HpBarAnalyzer(slots=dict(_SLOT))

    def test_unstable_window_not_confirmed(self):
        """フレームごとに値が動いている間（減少アニメ中）は確定しない。"""
        for fill in (100, 90, 80, 70, 60, 50):
            result = self.an.analyze(_solid_bar_frame(fill))
        assert result["player_0"] is None

    def test_reset_slot_clears_stable_value(self):
        """reset_slot 後は確定値がクリアされ None に戻る（占有者交代時の誤継承防止）。"""
        _analyze_n(self.an, _solid_bar_frame(100))
        self.an.reset_slot("player_0")
        result = self.an.analyze(_solid_bar_frame(60))
        assert result["player_0"] is None
