"""
Label Studio JSON エクスポート → YOLO 形式変換スクリプト

使い方:
    venv\\Scripts\\python.exe scripts/export_to_yolo.py annotations.json
    venv\\Scripts\\python.exe scripts/export_to_yolo.py annotations.json --output-dir data/ball_dataset/labels/train
"""

import argparse
import json
import os
from urllib.parse import parse_qs, urlparse, unquote


# Label Studio ラベル名 → YOLO クラスID
LABEL_MAP = {"ball_alive": 0, "ball_faint": 1, "ball_status": 2}


def _extract_stem(image_url: str) -> str:
    """Label Studio の画像URL からファイル名（拡張子なし）を取得する。

    /data/local-files/?d=Users%5Crotat%5C...%5Cimage.png 形式に対応。
    d パラメータは Windows パス（バックスラッシュ区切り）。
    """
    parsed = urlparse(image_url)
    qs = parse_qs(parsed.query)
    if "d" in qs:
        d_path = unquote(qs["d"][0])
        # Windows パスのバックスラッシュをスラッシュに変換してから basename 取得
        filename = d_path.replace("\\", "/").split("/")[-1]
    else:
        filename = os.path.basename(image_url.split("?")[0])
    return os.path.splitext(filename)[0]


def convert(annotations_file: str, output_dir: str) -> None:
    with open(annotations_file, encoding="utf-8") as f:
        tasks = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    exported = 0
    skipped = 0

    for task in tasks:
        image_url = task["data"].get("image", "")
        stem = _extract_stem(image_url)

        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        # 最新の承認済みアノテーションを使用
        results = annotations[0].get("result", [])
        lines = []

        for r in results:
            if r.get("type") != "rectanglelabels":
                continue

            v = r["value"]
            labels = v.get("rectanglelabels", [])
            if not labels:
                continue

            label = labels[0]
            cls_id = LABEL_MAP.get(label)
            if cls_id is None:
                print(f"  警告: 未知のラベル '{label}' をスキップ")
                continue

            # Label Studio: 左上基準・パーセント → YOLO: 中心基準・正規化
            x_pct = v["x"] / 100
            y_pct = v["y"] / 100
            w_pct = v["width"] / 100
            h_pct = v["height"] / 100

            x_center = x_pct + w_pct / 2
            y_center = y_pct + h_pct / 2

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_pct:.6f} {h_pct:.6f}")

        out_path = os.path.join(output_dir, f"{stem}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        exported += 1

    print(f"\n変換完了: {exported} 件エクスポート / {skipped} 件スキップ（未アノテーション）")
    print(f"出力先: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio JSON → YOLO 形式ラベル変換"
    )
    parser.add_argument("annotations_file", help="Label Studio からエクスポートした JSON ファイル")
    parser.add_argument(
        "--output-dir",
        default="data/ball_dataset/labels/train",
        help="YOLO ラベル出力先ディレクトリ (default: data/ball_dataset/labels/train)",
    )
    args = parser.parse_args()
    convert(args.annotations_file, args.output_dir)


if __name__ == "__main__":
    main()
