"""VMC Protocol経由でバーチャルモーションキャプチャーの表情ブレンドシェイプを
外部から操作できるか確認するための最小テストスクリプト（改善ロードマップ③の検証用）。

事前準備（Windows側）:
    venv\\Scripts\\pip.exe install python-osc

使い方（Windows側・VMC起動中に実行）:
    venv\\Scripts\\python.exe scripts\\test_vmc_expression.py --port 39540 --name Joy
    venv\\Scripts\\python.exe scripts\\test_vmc_expression.py --port 39541 --name Joy

--port は VMC の設定画面で確認した Receiver のポート番号（39540 or 39541）。
--name はVRM 0.x標準のブレンドシェイププリセット名（Joy/Angry/Sorrow/Fun/A/I/U/E/O/
Blink/Blink_L/Blink_R等）。まずは Joy で試し、アバターの表情が変化するか目視確認する。

3秒間 value=1.0 を維持したあと Neutral（全部0）に戻す。
"""

from __future__ import annotations

import argparse
import time

from pythonosc.udp_client import SimpleUDPClient

# VRM 0.x 標準ブレンドシェイププリセット名（大文字小文字はVMC側の実装依存の可能性あり）
_ALL_PRESETS = ["Joy", "Angry", "Sorrow", "Fun", "A", "I", "U", "E", "O",
                "Blink", "Blink_L", "Blink_R"]


def send_blend(client: SimpleUDPClient, name: str, value: float) -> None:
    client.send_message("/VMC/Ext/Blend/Val", [name, value])
    client.send_message("/VMC/Ext/Blend/Apply", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="VMC表情ブレンドシェイプのOSC操作テスト")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="VMCのReceiverポート番号")
    parser.add_argument("--name", default="Joy", help="ブレンドシェイプ名（既定: Joy）")
    parser.add_argument("--hold", type=float, default=3.0, help="表情を維持する秒数")
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    print(f"[送信] {args.host}:{args.port} へ {args.name}=1.0 を送信...")
    send_blend(client, args.name, 1.0)
    print(f"アバターの表情が変化したか確認してください（{args.hold}秒間維持）")
    time.sleep(args.hold)

    print("[送信] 全ブレンドシェイプを0にリセット...")
    for preset in _ALL_PRESETS:
        client.send_message("/VMC/Ext/Blend/Val", [preset, 0.0])
    client.send_message("/VMC/Ext/Blend/Apply", [])
    print("完了。")


if __name__ == "__main__":
    main()
