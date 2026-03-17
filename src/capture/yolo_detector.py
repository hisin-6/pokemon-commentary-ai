"""
Sprint 4: YOLOv8 アイコン検出
--------------------------------------------------
状態異常アイコン・ボール数を YOLOv8 で検出する。

カスタムモデルパスが未指定の場合:
  - yolov8n.pt（COCO事前学習済み）でパイプラインの動作確認のみ行う
  - 実際のアイコン検出はカスタム学習後に有効化

使用例:
  from src.capture.yolo_detector import YoloDetector, BattleState

  detector = YoloDetector()
  # または: YoloDetector(model_path="models/pokemon_icons.pt")

  frame = cv2.imread("debug/turn_001.png")
  state = detector.detect(frame)
  print(state.player_status)    # "やけど" など
  print(state.player_balls)     # BallCount(alive=3, faint=2)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ─── 定数 ───────────────────────────────────────────────────────────────────

# カスタム学習時のクラス定義（class_id → ラベル名）
CUSTOM_CLASS_NAMES: dict[int, str] = {
    0: "poison",       # どく
    1: "bad_poison",   # どくどく
    2: "burn",         # やけど
    3: "paralysis",    # まひ
    4: "freeze",       # こおり
    5: "sleep",        # ねむり
    6: "confusion",    # こんらん
    7: "ball_alive",   # 生存ポケモンのボール
    8: "ball_faint",   # 瀕死ポケモンのボール
    9: "ball_status",  # 状態異常ポケモンのボール（黄色）
}

BALL_LABELS = {"ball_alive", "ball_faint", "ball_status"}

# 日本語表示名マッピング
STATUS_JP: dict[str, str] = {
    "poison":     "どく",
    "bad_poison": "どくどく",
    "burn":       "やけど",
    "paralysis":  "まひ",
    "freeze":     "こおり",
    "sleep":      "ねむり",
    "confusion":  "こんらん",
}

# バトル画面の関心領域（画面サイズに対する比率: x1, y1, x2, y2）
# ポケモン SV / 剣盾 の 16:9 固定 UI を想定
# レイアウト:
#   上部: [opponent HP bars] [opponent_balls ←ここ]
#   下部: [player_balls ここ→] [player HP bars]
ROIS: dict[str, tuple[float, float, float, float]] = {
    "opponent_status": (0.57, 0.00, 1.00, 0.28),  # 右上: 相手のHP・状態
    "player_status":   (0.00, 0.69, 0.43, 1.00),  # 左下: 自分のHP・状態
    "opponent_balls":  (0.83, 0.01, 1.00, 0.20),  # 右上端: 相手のボール数（opponent_statusの右）
    "player_balls":    (0.00, 0.72, 0.10, 0.97),  # 左下端: 自分のボール数（player_statusの左）
}

CONFIDENCE_THRESHOLD = 0.5


# ─── データクラス ────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """単一の検出結果"""
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2（元画像座標）
    roi_name: str


@dataclass
class BallCount:
    """ボール数の集計結果"""
    alive: int = 0
    faint: int = 0

    @property
    def total(self) -> int:
        return self.alive + self.faint


@dataclass
class BattleState:
    """1フレームのバトル状態"""
    player_status: str | None = None
    opponent_status: str | None = None
    player_balls: BallCount = field(default_factory=BallCount)
    opponent_balls: BallCount = field(default_factory=BallCount)
    detections: list[Detection] = field(default_factory=list)
    mode: str = "pretrained_only"

    def summary(self) -> str:
        p_status = self.player_status or "正常"
        o_status = self.opponent_status or "正常"
        return (
            f"自分: {p_status} / ボール {self.player_balls.alive}匹生存 | "
            f"相手: {o_status} / ボール {self.opponent_balls.alive}匹生存"
        )


# ─── YoloDetector ────────────────────────────────────────────────────────────

class YoloDetector:
    """
    YOLOv8 を使って状態異常アイコン・ボール数を検出する。

    Args:
        model_path: カスタム学習済みモデルのパス（.pt）。
                    未指定の場合は yolov8n.pt（パイプライン確認用）を使用。
        device:     推論デバイス（"cuda" / "cpu"）。None で自動選択。
        conf:       信頼度閾値（デフォルト 0.5）。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        conf: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        from ultralytics import YOLO

        self._conf = conf
        self._custom_model = model_path is not None

        if model_path:
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            log.info(f"カスタムモデルをロード: {model_path}")
            self._model = YOLO(str(p))
            self._mode = "custom"
        else:
            log.warning(
                "カスタムモデルが未指定のため yolov8n.pt（COCO事前学習済み）を使用します。"
                "実際のアイコン検出は学習済みモデル指定後に有効になります。"
            )
            self._model = YOLO("yolov8n.pt")  # 自動DL（約6MB）
            self._mode = "pretrained_only"

        self._device = device or self._auto_device()
        log.info(f"YoloDetector 初期化完了 (mode={self._mode}, device={self._device})")

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ─── ROI ────────────────────────────────────────────────────────────────

    @staticmethod
    def crop_roi(
        frame: np.ndarray,
        roi: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """フレームから ROI 領域を切り出す。Returns (切り出し画像, (x_offset, y_offset))"""
        h, w = frame.shape[:2]
        x1 = int(w * roi[0])
        y1 = int(h * roi[1])
        x2 = int(w * roi[2])
        y2 = int(h * roi[3])
        return frame[y1:y2, x1:x2], (x1, y1)

    # ─── 推論 ────────────────────────────────────────────────────────────────

    def _run_on_full_frame(self, frame: np.ndarray) -> list[tuple]:
        """フル画像に対して推論を実行し生の検出結果を返す。"""
        results = self._model(
            frame,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        raw: list[tuple] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                if self._custom_model:
                    label = CUSTOM_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                else:
                    label = self._model.names.get(cls_id, f"class_{cls_id}")
                raw.append((cls_id, label, conf, x1, y1, x2, y2))
        return raw

    def _assign_roi(
        self,
        frame: np.ndarray,
        raw: list[tuple],
    ) -> list[Detection]:
        """検出結果をROI座標で振り分ける。
        ボール系クラスは _balls ROI を優先、状態異常クラスは _status ROI を優先する。
        （opponent_balls は opponent_status の内側に重なっているため順序制御が必要）
        """
        h, w = frame.shape[:2]
        detections: list[Detection] = []
        roi_order_balls  = ["opponent_balls", "player_balls", "opponent_status", "player_status"]
        roi_order_status = ["opponent_status", "player_status", "opponent_balls", "player_balls"]
        for cls_id, label, conf, x1, y1, x2, y2 in raw:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            priority = roi_order_balls if label in BALL_LABELS else roi_order_status
            roi_name = None
            for name in priority:
                rx1, ry1, rx2, ry2 = ROIS[name]
                if rx1 * w <= cx <= rx2 * w and ry1 * h <= cy <= ry2 * h:
                    roi_name = name
                    break
            if roi_name is None:
                roi_name = "other"
            detections.append(Detection(
                class_id=cls_id,
                label=label,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                roi_name=roi_name,
            ))
        return detections

    # ─── ROIクロップ推論 ─────────────────────────────────────────────────────

    def _run_on_rois(self, frame: np.ndarray) -> list[Detection]:
        """
        ROIごとにクロップして推論する。
        フルフレーム推論より小さいアイコンの検出精度が高い。
        ball系ラベルは _balls ROIからのみ、状態異常ラベルは _status ROIからのみ採用する。
        """
        all_detections: list[Detection] = []
        for roi_name, roi_ratio in ROIS.items():
            crop, (x_off, y_off) = self.crop_roi(frame, roi_ratio)
            if crop.size == 0:
                continue
            results = self._model(crop, conf=self._conf, device=self._device, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id  = int(box.cls[0])
                    conf    = float(box.conf[0])
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    label   = CUSTOM_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    # ROI種別とラベル種別が一致しない検出は除外
                    # （opponent_status ROI にボールが映っても無視する等）
                    if roi_name.endswith("_balls") != (label in BALL_LABELS):
                        continue
                    all_detections.append(Detection(
                        class_id=cls_id,
                        label=label,
                        confidence=conf,
                        bbox=(x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off),
                        roi_name=roi_name,
                    ))
        return all_detections

    # ─── メイン検出 ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> BattleState:
        """
        バトル画面フレームに対して検出を実行し BattleState を返す。
        カスタムモデル使用時はROIクロップ推論、事前学習モデルはフルフレーム推論。

        Args:
            frame: BGR 形式の numpy 配列（cv2.imread の出力）。
        """
        state = BattleState(mode=self._mode)

        if self._custom_model:
            all_detections = self._run_on_rois(frame)
        else:
            raw = self._run_on_full_frame(frame)
            all_detections = self._assign_roi(frame, raw)

        state.detections = all_detections

        if self._custom_model:
            state.player_status   = self._extract_status(all_detections, "player_status")
            state.opponent_status = self._extract_status(all_detections, "opponent_status")
            state.player_balls    = self._count_balls(all_detections, "player_balls")
            state.opponent_balls  = self._count_balls(all_detections, "opponent_balls")

        return state

    @staticmethod
    def _extract_status(detections: list[Detection], roi_name: str) -> str | None:
        """指定 ROI から状態異常ラベルを抽出（最高信頼度のもの）。"""
        candidates = [
            d for d in detections
            if d.roi_name == roi_name and d.label in STATUS_JP
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda d: d.confidence)
        return STATUS_JP.get(best.label)

    @staticmethod
    def _count_balls(detections: list[Detection], roi_name: str) -> BallCount:
        """指定 ROI からボール数を集計する。"""
        bc = BallCount()
        for d in detections:
            if d.roi_name != roi_name:
                continue
            if d.label in ("ball_alive", "ball_status"):
                bc.alive += 1
            elif d.label == "ball_faint":
                bc.faint += 1
        return bc

    # ─── デバッグ描画 ─────────────────────────────────────────────────────────

    def draw_detections(self, frame: np.ndarray, state: BattleState) -> np.ndarray:
        """検出結果をフレームに描画して返す（デバッグ用）。"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # ROI 枠
        for roi_name, roi_ratio in ROIS.items():
            x1 = int(w * roi_ratio[0])
            y1 = int(h * roi_ratio[1])
            x2 = int(w * roi_ratio[2])
            y2 = int(h * roi_ratio[3])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, roi_name, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 検出ボックス
        for det in state.detections:
            bx1, by1, bx2, by2 = det.bbox
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(vis, f"{det.label} {det.confidence:.2f}", (bx1, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return vis


# ─── CLI テスト ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8 アイコン検出テスト（Sprint 4）"
    )
    parser.add_argument("--image", required=True, metavar="PATH",
                        help="テスト用画像ファイルのパス")
    parser.add_argument("--model", metavar="PATH", default=None,
                        help="カスタム学習済みモデルのパス（省略で yolov8n.pt）")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="可視化画像の保存先（省略で debug/yolo_<元ファイル名>.png）")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"信頼度閾値（デフォルト: {CONFIDENCE_THRESHOLD}）")
    parser.add_argument("--cpu", action="store_true", help="CPU モードで実行")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    path = Path(args.image)
    if not path.exists():
        log.error(f"ファイルが見つかりません: {args.image}")
        sys.exit(1)

    frame = cv2.imread(str(path))
    if frame is None:
        log.error("画像の読み込みに失敗しました")
        sys.exit(1)

    log.info(f"画像サイズ: {frame.shape[1]}x{frame.shape[0]}")

    device = "cpu" if args.cpu else None
    detector = YoloDetector(model_path=args.model, device=device, conf=args.conf)
    state = detector.detect(frame)

    print("\n" + "=" * 50)
    print("【バトル状態】")
    print(state.summary())
    print(f"検出数: {len(state.detections)} 件 (mode={state.mode})")
    if state.detections:
        print("\n--- 検出詳細 ---")
        for d in state.detections:
            print(f"  [{d.roi_name}] {d.label} ({d.confidence:.1%})  bbox={d.bbox}")
    print("=" * 50 + "\n")

    save_path = args.save or f"debug/yolo_{path.stem}.png"
    Path(save_path).parent.mkdir(exist_ok=True)
    vis = detector.draw_detections(frame, state)
    cv2.imwrite(save_path, vis)
    log.info(f"可視化画像を保存: {save_path}")

    # ROIクロップをデバッグ保存
    debug_dir = Path(save_path).parent
    for roi_name, roi_ratio in ROIS.items():
        crop, _ = YoloDetector.crop_roi(frame, roi_ratio)
        crop_path = debug_dir / f"roi_{path.stem}_{roi_name}.png"
        cv2.imwrite(str(crop_path), crop)
        log.info(f"ROIクロップ保存: {crop_path}")


if __name__ == "__main__":
    main()
