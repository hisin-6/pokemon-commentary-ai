"""
ボール検出用 ROI クロップスクリプト
=====================================
フルフレームの学習データから opponent_balls / player_balls の ROI を切り出し、
ボール検出専用データセット（data/ball_dataset/）を作成する。

処理の流れ:
  1. data/yolo_dataset/labels/{train,val}/ にアノテーションがある画像のみ対象
  2. 各画像から 2 つの ROI（opponent_balls / player_balls）を切り出し保存
  3. 既存のボールアノテーション（クラスID 7/8/9）を ROI 座標系に変換して保存
     - ボールのない ROI はラベルファイルなし（= YOLO ネガティブサンプル）
  4. data/ball_dataset/data.yaml を生成

新クラス定義:
  0: ball_alive   （旧: 7）
  1: ball_faint   （旧: 8）
  2: ball_status  （旧: 9）

実行例:
  python scripts/crop_ball_rois.py
  python scripts/crop_ball_rois.py --dry-run   # ファイルを書き出さず件数だけ確認
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

# ─── ROI 定義（yolo_detector.py と同じ値）────────────────────────────────────
# (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
BALL_ROIS: dict[str, tuple[float, float, float, float]] = {
    "opponent": (0.83, 0.01, 1.00, 0.20),
    "player":   (0.00, 0.72, 0.10, 0.97),
}

# 旧クラスID → 新クラスID（ボール以外は無視）
OLD_TO_NEW: dict[int, int] = {
    7: 0,  # ball_alive
    8: 1,  # ball_faint
    9: 2,  # ball_status
}

# ─── パス定義 ─────────────────────────────────────────────────────────────────
SRC_ROOT = Path("data/yolo_dataset")
DST_ROOT = Path("data/ball_dataset")


def crop_roi(img, roi: tuple[float, float, float, float]):
    """画像から ROI を切り出す。(crop, x1_px, y1_px, roi_w_px, roi_h_px) を返す。"""
    h, w = img.shape[:2]
    x1 = int(w * roi[0])
    y1 = int(h * roi[1])
    x2 = int(w * roi[2])
    y2 = int(h * roi[3])
    roi_w = x2 - x1
    roi_h = y2 - y1
    return img[y1:y2, x1:x2], x1, y1, roi_w, roi_h


def convert_annotations(
    label_path: Path,
    roi: tuple[float, float, float, float],
) -> list[str]:
    """
    フルフレームのアノテーションを ROI 座標系に変換する。

    ボールクラス（7/8/9）のうち ROI 内に中心が入るものだけを変換して返す。
    ROI 外のボールや非ボールクラスは無視。
    """
    if not label_path.exists():
        return []

    rx1, ry1, rx2, ry2 = roi
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1

    lines = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue  # "ball_alive" 等のテキスト形式 → スキップ

        if cls_id not in OLD_TO_NEW:
            continue

        cx, cy, bw, bh = map(float, parts[1:])

        # 中心が ROI 内にあるか確認
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            continue

        # ROI 座標系に変換
        cx_new = (cx - rx1) / roi_w
        cy_new = (cy - ry1) / roi_h
        bw_new = bw / roi_w
        bh_new = bh / roi_h

        # [0, 1] にクリップ（ROI 境界にかかるボックスへの安全策）
        cx_new = max(0.0, min(1.0, cx_new))
        cy_new = max(0.0, min(1.0, cy_new))
        bw_new = max(0.0, min(1.0, bw_new))
        bh_new = max(0.0, min(1.0, bh_new))

        new_cls = OLD_TO_NEW[cls_id]
        lines.append(f"{new_cls} {cx_new:.6f} {cy_new:.6f} {bw_new:.6f} {bh_new:.6f}")

    return lines


def process_split(split: str, dry_run: bool, include_unlabeled: bool = False, date_prefix: str | None = None) -> dict[str, int]:
    """train または val の分割を処理する。統計を返す。

    Args:
        include_unlabeled: True の場合、ラベルなし画像も対象にする（新規収集画像の追加用）。
                           既に ball_dataset に存在するファイルはスキップする。
        date_prefix: 指定した場合、ファイル名がこのプレフィックスで始まる画像だけを対象にする
                     （例: "20260323" で今日収集分のみ）。
    """
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split
    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split

    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_src": 0,
        "crops_saved": 0,
        "labels_auto": 0,   # ボールアノテーションを自動変換したクロップ
        "labels_neg": 0,    # ボールなし確認済み（ラベルファイルなし）
        "ball_instances": 0,
        "skipped_exists": 0,  # 既存ファイルのスキップ数
    }

    if include_unlabeled:
        # ラベルあり・なし両方の画像を対象にする
        img_files = sorted(src_img_dir.glob("*.png")) + sorted(src_img_dir.glob("*.jpg"))
        if date_prefix:
            img_files = [p for p in img_files if p.name.startswith(date_prefix)]
        targets = [(p, src_lbl_dir / (p.stem + ".txt")) for p in img_files]
    else:
        # アノテーション済み画像のみ対象（従来の動作）
        label_files = sorted(src_lbl_dir.glob("*.txt"))
        targets = [(src_img_dir / (lbl.stem + ".png"), lbl) for lbl in label_files]
        # .jpg フォールバック
        targets = [
            (p if p.exists() else src_img_dir / (lbl.stem + ".jpg"), lbl)
            for p, lbl in targets
        ]
        targets = [(p, lbl) for p, lbl in targets if p.exists()]

    for img_path, lbl_path in targets:
        if not img_path.exists():
            continue

        stats["total_src"] += 1
        img = None  # 遅延ロード

        for roi_name, roi in BALL_ROIS.items():
            out_stem = f"{img_path.stem}_{roi_name}"
            out_img_path = dst_img_dir / f"{out_stem}.png"
            out_lbl_path = dst_lbl_dir / f"{out_stem}.txt"

            # 既存ファイルはスキップ（重複防止）
            if out_img_path.exists():
                stats["skipped_exists"] += 1
                continue

            # アノテーション変換（ラベルなし画像は空リスト）
            ann_lines = convert_annotations(lbl_path, roi) if lbl_path.exists() else []

            if not dry_run:
                if img is None:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  [WARN] 画像読み込み失敗: {img_path}")
                        break

                crop, *_ = crop_roi(img, roi)
                cv2.imwrite(str(out_img_path), crop)

                if ann_lines:
                    out_lbl_path.write_text("\n".join(ann_lines) + "\n", encoding="utf-8")

            stats["crops_saved"] += 1
            if ann_lines:
                stats["labels_auto"] += 1
                stats["ball_instances"] += len(ann_lines)
            else:
                stats["labels_neg"] += 1

    return stats


def write_data_yaml(dry_run: bool) -> None:
    yaml_path = DST_ROOT / "data.yaml"
    content = """\
# ボール検出専用データセット
# ROI クロップ学習でスケール問題を解消し、yolov8s で再学習する

path: data/ball_dataset
train: images/train
val:   images/val

nc: 3
names:
  0: ball_alive   # 生存ポケモンのボール（白）
  1: ball_faint   # 瀕死ポケモンのボール（グレー）
  2: ball_status  # 状態異常ポケモンのボール（黄色）
"""
    if not dry_run:
        DST_ROOT.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(content, encoding="utf-8")
    print(f"{'[DRY] ' if dry_run else ''}data.yaml → {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ボール検出用 ROI クロップデータセットを作成する")
    parser.add_argument("--dry-run", action="store_true", help="ファイルを書き出さず件数だけ確認")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="ラベルなし画像も対象にする（新規収集画像の追加用）。既存ファイルはスキップ。")
    parser.add_argument("--date", default=None, metavar="YYYYMMDD",
                        help="指定日付のファイルだけ対象にする（例: 20260323）。--include-unlabeled と併用。")
    args = parser.parse_args()

    dry = args.dry_run
    if dry:
        print("=== DRY RUN モード（ファイル書き出しなし）===\n")
    if args.include_unlabeled:
        date_str = f"（{args.date}）" if args.date else "（全日付）"
        print(f"=== ラベルなし画像も対象 {date_str} ===\n")

    splits = ("train",) if args.include_unlabeled else ("train", "val")
    for split in splits:
        print(f"── {split} ──")
        stats = process_split(split, dry_run=dry,
                              include_unlabeled=args.include_unlabeled,
                              date_prefix=args.date)
        print(f"  対象画像数      : {stats['total_src']} 枚")
        print(f"  生成クロップ数  : {stats['crops_saved']} 枚 (×2 ROI)")
        print(f"  自動ラベル変換  : {stats['labels_auto']} クロップ（ボール {stats['ball_instances']} 件）")
        print(f"  ネガティブ確認  : {stats['labels_neg']} クロップ（ラベルファイルなし）")
        if stats['skipped_exists']:
            print(f"  既存スキップ    : {stats['skipped_exists']} 枚")
        print()

    write_data_yaml(dry_run=dry)

    if not dry:
        print("\n完了！")
        print(f"出力先: {DST_ROOT.resolve()}")
        print("\n次のステップ:")
        print("  1. LabelImg で data/ball_dataset/images/train/ を開いてアノテーション確認・追加")
        print("     - 自動変換済みラベルはそのまま使用可能（位置がずれていたら修正）")
        print("     - ラベルなし画像にボールが映っていたらアノテーション追加")
        print("  2. 再学習:")
        print("     venv\\Scripts\\python.exe -m ultralytics train \\")
        print("         model=yolov8s.pt \\")
        print("         data=data/ball_dataset/data.yaml \\")
        print("         imgsz=640 \\")
        print("         epochs=100 \\")
        print("         batch=8")
    else:
        print("\n実際に実行する場合は --dry-run を外してください。")


if __name__ == "__main__":
    main()
