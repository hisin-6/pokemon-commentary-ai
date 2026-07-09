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


@dataclass
class AnchorConfig:
    """HUD表示検証用アンカー（バー付近のネームプレート帯）の座標と色域。

    攻撃アニメーション中はHUDが消えてバーROIに背景シーンが映り込むが、
    炎エフェクト等のベタ塗りは色・密度ともバーと区別できない（実測: 繰り出し
    演出の炎がplayer_0を横切り56%を誤確定）。バーが表示される時は必ず
    ネームプレート帯が一緒に表示されるため、その色域画素率でHUD表示を判定する。
    HUDの表示はスロット単位（片側だけ表示されるフレームが実在する）。
    """
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    lo: np.ndarray     # HSV下限（cv2.inRange用）
    hi: np.ndarray     # HSV上限


# ── デフォルト座標（1920x1080 基準） ──────────────────────────────
# 実測値（2026-05-06 ピクセルスキャンで確認）:
#   player_0: x=178-405 y=1005-1010  バー実幅=227px（100%時 seg=227/227）
#   player_1: x=574-801 y=1005-1010  バー実幅=227px（87.7%時 seg=199/227）
#   opp_0:    y=110-130（チャンピオンズ実測 2026-04-13）
#   opp_1:    y=110-130（チャンピオンズ実測 2026-04-13）
_DEFAULT_SLOTS: dict[str, SlotConfig] = {
    "player_0":   SlotConfig(x_left=178,  x_right=405,  y_top=1005, y_bottom=1010, label="player_0"),
    "player_1":   SlotConfig(x_left=574,  x_right=801,  y_top=1005, y_bottom=1010, label="player_1"),
    "opponent_0": SlotConfig(x_left=1222, x_right=1450, y_top=110, y_bottom=130, label="opp_0"),
    "opponent_1": SlotConfig(x_left=1618, x_right=1846, y_top=110, y_bottom=130, label="opp_1"),
}

# ── HUDアンカー色域（2026-07-09 動画フレーム実測・06-25-46全域スキャンで検証） ──
# 自分側: バー上のネームバンド＝明るい青紫（実測中央値 H119 S143 V232）
_PLAYER_BAND_LO = np.array([105,  80, 190])
_PLAYER_BAND_HI = np.array([135, 210, 255])
# 相手側: バー上のネームプレート帯＝ビビッドピンク（実測中央値 H171 S212 V191）
_OPP_BAND_LO = np.array([165, 170, 120])
_OPP_BAND_HI = np.array([178, 255, 235])

# アンカー帯はバーROIの直上（自分: ネームバンドy950-974 / 相手: プレート帯y86-102）
_DEFAULT_ANCHORS: dict[str, AnchorConfig] = {
    "player_0":   AnchorConfig(188, 288, 950, 974, _PLAYER_BAND_LO, _PLAYER_BAND_HI),
    "player_1":   AnchorConfig(584, 684, 950, 974, _PLAYER_BAND_LO, _PLAYER_BAND_HI),
    "opponent_0": AnchorConfig(1232, 1332, 86, 102, _OPP_BAND_LO, _OPP_BAND_HI),
    "opponent_1": AnchorConfig(1628, 1728, 86, 102, _OPP_BAND_LO, _OPP_BAND_HI),
}


def slot_bar_centers(slots: dict[str, SlotConfig] = _DEFAULT_SLOTS) -> dict[str, tuple[float, float]]:
    """side（"player"/"opponent"）ごとの (slot0中心x, slot1中心x) を返す。

    HPバーROI座標(`_DEFAULT_SLOTS`)から算出するため、ROIを再キャリブレーションしても
    ネームプレート近接判定用の中心座標が自動的に追従する（手計算のハードコード値が
    ROI変更に追従できず静かにズレるのを防ぐ）。
    """
    centers: dict[str, tuple[float, float]] = {}
    for side in ("player", "opponent"):
        slot0 = slots[f"{side}_0"]
        slot1 = slots[f"{side}_1"]
        centers[side] = (
            (slot0.x_left + slot0.x_right) / 2,
            (slot1.x_left + slot1.x_right) / 2,
        )
    return centers


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
    # セグメント内の着色列密度の下限。本物のバーはベタ塗り（実測1.00）だが、
    # 攻撃アニメーション中はHUDが消えてROIに背景シーンが映り込み、散在する
    # 炎エフェクト等が _GAP_TOLERANCE の橋渡しで偽セグメントになる
    # （実測: いわなだれアニメ中に密度0.33の15px偽セグメント→7%を誤確定）
    _SEG_DENSITY_MIN: float = 0.85
    # アンカー帯の色域画素率の下限。06-25-46全域スキャン（4623サンプル）で
    # HUD表示中は≥0.5・非表示（シーン映り込み）は≤0.29に二峰分離することを確認
    _ANCHOR_MIN: float = 0.4
    # 安定化フィルタ: ウィンドウフレーム数（skip=15・30fps ≒ 2fps で約3秒）
    _STABILITY_WINDOW: int = 6
    # 安定化フィルタ: ウィンドウ内の max-min がこの値以内なら「減少停止（安定）」とみなす
    _STABILITY_DELTA: float = 0.02

    def __init__(
        self,
        slots: dict[str, SlotConfig] | None = None,
        anchors: dict[str, AnchorConfig] | None = None,
    ) -> None:
        # カスタムslots指定時はアンカーも明示指定がない限り無効
        # （テスト・別レイアウトのキャリブレーション用途で座標が一致しないため）
        if slots is None:
            self._anchors = anchors if anchors is not None else dict(_DEFAULT_ANCHORS)
        else:
            self._anchors = anchors or {}
        self._slots = slots or dict(_DEFAULT_SLOTS)
        # 満タン幅の動的更新用（初期値 = 設定値）
        self._observed_max: dict[str, int] = {
            key: cfg.full_width for key, cfg in self._slots.items()
        }
        # 安定化フィルタ用ステート
        self._buf: dict[str, list[float]] = {key: [] for key in self._slots}
        self._last_stable: dict[str, float | None] = {key: None for key in self._slots}

    def set_full_width(self, slot_key: str, width: int) -> None:
        """指定スロットの満タン幅を手動設定する。"""
        self._observed_max[slot_key] = width

    def reset_slot(self, slot_key: str) -> None:
        """スロットの安定化状態をリセットする（占有ポケモンの交代時に呼ぶ）。
        _last_stable はスロット（画面位置）に紐づくため、中のポケモンが交代しても
        前のポケモンの確定値を返し続ける。リセットしないと交代直後のアニメーション中
        （バー非表示）に新しいポケモンへ前任者のHP%が付与される（実機で確認）。
        """
        if slot_key in self._buf:
            self._buf[slot_key] = []
            self._last_stable[slot_key] = None

    def reset(self) -> None:
        """全スロットの安定化状態をリセットする（試合開始時に呼ぶ）。"""
        for key in self._slots:
            self.reset_slot(key)

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
        フレームウィンドウ安定化フィルタ。

        直近 _STABILITY_WINDOW フレームの読み取り値を保持し、
        ウィンドウ内の max-min が _STABILITY_DELTA 以内になった時点
        （= HPの減少/増加アニメーションが止まった瞬間）を確定タイミングとする。

        - raw が None → バッファを更新せず最後の確定値を返す
        - ウィンドウが埋まるまで → 最後の確定値を返す（未確定なら None）
        - ウィンドウ内の spread ≤ _STABILITY_DELTA → 平均値を新しい確定値として採用
        """
        if raw is None:
            return self._last_stable[key]

        buf = self._buf[key]
        buf.append(raw)
        if len(buf) > self._STABILITY_WINDOW:
            buf.pop(0)

        if len(buf) < self._STABILITY_WINDOW:
            return self._last_stable[key]

        spread = max(buf) - min(buf)
        if spread <= self._STABILITY_DELTA:
            new_val = round(sum(buf) / len(buf), 3)
            old_val = self._last_stable[key]
            if old_val is None or abs(new_val - old_val) > self._STABILITY_DELTA:
                log.debug(
                    "[HpBar] %s 確定: %.1f%% → %.1f%% (spread=%.1f%%)",
                    key, (old_val or 0) * 100, new_val * 100, spread * 100,
                )
                self._last_stable[key] = new_val

        return self._last_stable[key]

    def _anchor_visible(self, frame: np.ndarray, anc: AnchorConfig) -> bool:
        """アンカー帯（ネームプレート）の色域画素率でHUD表示を判定する。"""
        strip = frame[anc.y_top:anc.y_bottom, anc.x_left:anc.x_right]
        if strip.size == 0:
            return True  # フレームが小さい場合は検証不能 → フェイルオープン
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        frac = cv2.inRange(hsv, anc.lo, anc.hi).mean() / 255.0
        return frac >= self._ANCHOR_MIN

    def _measure_slot(self, frame: np.ndarray, key: str, cfg: SlotConfig) -> float | None:
        """1スロット分のHP%を計算する。"""
        # HUDアンカー検証: バー非表示（アニメーション中）のシーン映り込みを棄却
        anc = self._anchors.get(key)
        if anc is not None and not self._anchor_visible(frame, anc):
            return self._apply_stabilizer(key, None)

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
        seg_start, seg_width = self._first_segment(col_colored)
        if seg_width < self._MIN_SEG_WIDTH:
            return self._apply_stabilizer(key, None)

        # 着色密度検証: 虫食いセグメントは背景シーンの映り込みとして棄却
        density = float(col_colored[seg_start:seg_start + seg_width].mean())
        if density < self._SEG_DENSITY_MIN:
            log.debug("[HpBar] %s 密度不足で棄却: seg=%dpx density=%.2f", key, seg_width, density)
            return self._apply_stabilizer(key, None)

        # 満タン幅の動的更新
        if seg_width > self._observed_max[key]:
            log.debug("[HpBar] %s 満タン幅更新: %d → %d", key, self._observed_max[key], seg_width)
            self._observed_max[key] = seg_width

        pct = round(min(seg_width / self._observed_max[key], 1.0), 3)
        return self._apply_stabilizer(key, pct)

    def _first_segment(self, col_colored: np.ndarray) -> tuple[int, int]:
        """
        左端から最初の連続着色セグメントの (開始位置, 幅) を返す。
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
                        return start, i - gap - start
        if in_seg:
            return start, len(col_colored) - gap - start
        return 0, 0
