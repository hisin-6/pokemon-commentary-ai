"""
HPバーピクセル解析モジュール

1920x1080 の対戦画面から、HPバー色（緑/黄橙/赤）の
左端〜現在右端の幅を測定してHP%を推定する。

スロット座標（1920x1080 固定）
【SV（スカーレット・バイオレット）】
  player  slot0: x=106-385,  y=948-974
  player  slot1: x=432-711,  y=948-974
  opponent slot0: x=1270-1461, y=88-110
  opponent slot1: x=1532-1839, y=88-110

【チャンピオンズ】
  player  slot0: x=106-385,  y=948-974  ← SV と同じ（要再確認）
  player  slot1: x=432-711,  y=948-974  ← SV と同じ（要再確認）
  opponent slot0: x=1105-1450, y=110-130  ← 実測（2026-04-13）
  opponent slot1: x=1498-1846, y=110-130  ← 実測（2026-04-13）
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── HPバー色 HSV範囲 ────────────────────────────────────────────────
_HP_COLOR_RANGES: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array([18, 100, 120]), np.array([42, 255, 255])),   # 橙/黄
    (np.array([42, 100, 120]), np.array([85, 255, 255])),   # 緑
    (np.array([ 0, 120, 120]), np.array([10, 255, 255])),   # 赤
    (np.array([170, 120, 120]), np.array([180, 255, 255])), # 赤（折り返し）
]


@dataclass
class SlotConfig:
    """1スロット分のHPバー座標設定"""
    x_left: int        # バー左端（HP 100%時の左端 = 常に固定）
    x_right: int       # バー右端（HP 100%時の右端 = 満タン）
    y_top: int
    y_bottom: int
    label: str

    @property
    def full_width(self) -> int:
        return self.x_right - self.x_left


# ── デフォルト座標（1920x1080 基準） ──────────────────────────────
# 実測値（debug/ocr_turn_*.png の複数フレームで満タン幅を計測）:
#   player:   y=948-974, 満タン幅=279px（バドレックス100%フレームで確認）
#   opp_0:    y=110-130（チャンピオンズ実測 2026-04-13）
#   opp_1:    y=110-130（チャンピオンズ実測 2026-04-13）
_DEFAULT_SLOTS: dict[str, SlotConfig] = {
    "player_0":   SlotConfig(x_left=106,  x_right=385,  y_top=948, y_bottom=974, label="player_0"),
    "player_1":   SlotConfig(x_left=432,  x_right=711,  y_top=948, y_bottom=974, label="player_1"),
    "opponent_0": SlotConfig(x_left=1222, x_right=1450, y_top=110, y_bottom=130, label="opp_0"),
    "opponent_1": SlotConfig(x_left=1618, x_right=1846, y_top=110, y_bottom=130, label="opp_1"),
}


class HpBarAnalyzer:
    """
    フレームからHP%を推定する。

    使い方:
        analyzer = HpBarAnalyzer()
        result = analyzer.analyze(frame)
        print(result["player_0"])   # 0.0-1.0, None=未検出

    キャリブレーション:
        満タン幅が実測で既知の場合は set_full_width() で上書きできる。
        analyze() は観測幅が既知の満タン幅を超えたとき自動更新する。
    """

    # HPバーとHPテキストボックスを分離するギャップ許容値
    _GAP_TOLERANCE: int = 8
    # バー幅の最小有効ピクセル数（ノイズフィルタ）
    _MIN_SEG_WIDTH: int = 15
    # 安定化フィルタ: 確定値との差がこの値以内なら変化なし扱い
    _STABLE_TOL: float = 0.03
    # 安定化フィルタ: HP増加（アニメーションノイズ候補）を確定するのに必要な連続秒数
    _BOUNCE_SECS: float = 2.5
    # 安定化フィルタ: HP減少（ダメージ）を確定するのに必要な連続秒数
    _DECREASE_SECS: float = 0.8

    def __init__(self, slots: dict[str, SlotConfig] | None = None) -> None:
        self._slots = slots or dict(_DEFAULT_SLOTS)
        # 満タン幅の動的更新用（初期値 = 設定値）
        self._observed_max: dict[str, int] = {
            key: cfg.full_width for key, cfg in self._slots.items()
        }
        # 安定化フィルタ用ステート
        self._committed: dict[str, float | None] = {key: None for key in self._slots}
        self._cand_value: dict[str, float | None] = {key: None for key in self._slots}
        self._cand_start: dict[str, float | None] = {key: None for key in self._slots}

    def set_full_width(self, slot_key: str, width: int) -> None:
        """指定スロットの満タン幅を手動設定する。"""
        self._observed_max[slot_key] = width

    def analyze(self, frame: np.ndarray) -> dict[str, float | None]:
        """
        フレームを解析して各スロットのHP%を返す。

        Returns:
            { "player_0": 0.46, "player_1": 1.0, "opponent_0": 0.12, "opponent_1": None }
            値は 0.0-1.0 の float、検出不能なら None。
        """
        result: dict[str, float | None] = {}
        for key, cfg in self._slots.items():
            pct = self._measure_slot(frame, key, cfg)
            result[key] = pct
        return result

    # ── 内部メソッド ──────────────────────────────────────────────

    def _apply_stabilizer(self, key: str, raw: float | None) -> float | None:
        """
        非対称バウンスフィルタ（時間ベース）。

        - raw が None → 確定値をそのまま返す（一時的な検出失敗を無視）
        - raw ≈ 確定値（±STABLE_TOL）→ 確定値を返す（微細ノイズ除去）
        - raw < 確定値（HP減少）→ DECREASE_SECS 秒連続で安定したら確定
        - raw > 確定値（HP増加）→ BOUNCE_SECS 秒連続で安定したら確定
          （アニメーション中の一時的な増加はここで吸収される）
        """
        committed = self._committed[key]

        # 初回読み取り
        if committed is None:
            if raw is not None:
                self._committed[key] = raw
            return raw

        # 検出失敗時は確定値を保持
        if raw is None:
            return committed

        # 確定値と誤差範囲内なら変化なしとみなす
        if abs(raw - committed) <= self._STABLE_TOL:
            self._cand_value[key] = None
            self._cand_start[key] = None
            return committed

        now = time.monotonic()
        # 候補が今回の raw と連続しているか確認
        prev_cand = self._cand_value[key]
        if prev_cand is not None and abs(raw - prev_cand) <= self._STABLE_TOL:
            # 同じ候補が継続中 → タイマーはそのまま
            pass
        else:
            # 新しい候補に切り替え → タイマーリセット
            self._cand_value[key] = raw
            self._cand_start[key] = now

        required_secs = self._DECREASE_SECS if raw < committed else self._BOUNCE_SECS
        elapsed = now - (self._cand_start[key] or now)
        if elapsed >= required_secs:
            log.debug(
                "[HpBar] %s 確定値更新: %.1f%% → %.1f%% (%.1f秒安定)",
                key, committed * 100, raw * 100, elapsed,
            )
            self._committed[key] = raw
            self._cand_value[key] = None
            self._cand_start[key] = None

        return self._committed[key]

    def _measure_slot(self, frame: np.ndarray, key: str, cfg: SlotConfig) -> float | None:
        """1スロット分のHP%を計算する。"""
        roi = frame[cfg.y_top:cfg.y_bottom, cfg.x_left:cfg.x_right]
        if roi.size == 0:
            return self._apply_stabilizer(key, None)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _HP_COLOR_RANGES:
            mask |= cv2.inRange(hsv, lo, hi)

        # 列方向集約: いずれかの行で着色されていれば True
        col_colored = mask.any(axis=0)  # shape: (roi_width,)

        # 左端から最初の連続セグメントのみを HP バーとして取得
        seg_width = self._first_segment_width(col_colored)
        if seg_width < self._MIN_SEG_WIDTH:
            return self._apply_stabilizer(key, None)

        # 満タン幅の動的更新
        if seg_width > self._observed_max[key]:
            log.debug("[HpBar] %s 満タン幅更新: %d → %d", key, self._observed_max[key], seg_width)
            self._observed_max[key] = seg_width

        pct = round(min(seg_width / self._observed_max[key], 1.0), 3)
        return self._apply_stabilizer(key, pct)

    def _first_segment_width(self, col_colored: np.ndarray) -> int:
        """
        左端から最初の連続着色セグメント幅を返す。
        _GAP_TOLERANCE px 以内の空白は同一セグメントとみなす。
        """
        in_seg = False
        start = 0
        gap = 0
        for i, v in enumerate(col_colored):
            if v:
                if not in_seg:
                    start = i
                    in_seg = True
                gap = 0
            else:
                if in_seg:
                    gap += 1
                    if gap > self._GAP_TOLERANCE:
                        return i - gap - start
        if in_seg:
            return len(col_colored) - gap - start
        return 0
