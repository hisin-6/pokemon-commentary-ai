"""play_and_animate_avatar.pyへ統合したポーズ遷移（2026-08-03・改善ロードマップ③続き）を
実機で動作確認するための検証スクリプト。play_and_animate_avatar.py自体の関数
（send_reaction/play_pose_reaction/run_idle_sway/run_idle_gestures）をそのまま呼び出すので、
「本番と同じコード」の動きを確認できる（このスクリプト独自のポーズ実装は無い）。

事前準備（Windows側）:
    venv\\Scripts\\pip.exe install python-osc

使い方（Windows側・VMC起動中・Receiver有効化チェック済みの状態で）:
    venv\\Scripts\\python.exe scripts\\test_vmc_pose_reaction.py --port 39540

流れ（Enterキーで1段階ずつ進む）:
  1. 腕を下ろした初期姿勢を送信（T-pose解消）
  2. 常時スウェイ（Spine/Chest）＋ランダム仕草（Head）を開始
     → 呼吸っぽい揺れが始まる（数秒〜十数秒に一度、頭の仕草も入る）
  3. Joy → Sorrow → Fun → Angry の順でsend_reactionを実行
     （Joy=victory_arms_up・Sorrow=bow_apologetic・Fun=head_tilt_curious・
     Angry=未マッピングなので標準うなずきにフォールバック）
     各ポーズについて確認したい点:
       - 遷移が滑らかか（急に飛ばないか）
       - ポーズ後、腕/首/上体が元の位置（T-poseではなくidle_downの下げ角度）に
         正しく戻るか
       - ポーズ再生中〜直後、常時スウェイ/ランダム仕草と喧嘩して不自然に
         ガクつかないか（`_suspended_bones`による除外が機能しているか）
  4. Enterでスウェイ/仕草を停止しNeutralへ戻して終了
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import play_and_animate_avatar as paa  # noqa: E402

from pythonosc.udp_client import SimpleUDPClient

_CHECK_ORDER = ["Joy", "Sorrow", "Fun", "Angry"]


def main() -> int:
    parser = argparse.ArgumentParser(description="ポーズ統合済みplay_and_animate_avatar.pyの実機検証")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="VMCのReceiverポート番号")
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    paa.send_expression(client, paa.NEUTRAL)
    paa.send_idle_pose(client)
    print("[1/4] 腕を下ろした初期姿勢を送信しました。VMCでT-poseが解消されているか確認してください。")
    input("Enterで常時スウェイ/ランダム仕草を開始します...")

    stop_event = threading.Event()
    sway_thread = threading.Thread(
        target=paa.run_idle_sway, args=(client, stop_event, time.monotonic()), daemon=True)
    sway_thread.start()
    gesture_thread = threading.Thread(
        target=paa.run_idle_gestures, args=(client, stop_event), daemon=True)
    gesture_thread.start()
    print("[2/4] 常時スウェイを開始しました（呼吸っぽい揺れ＋ランダムな首の仕草）。")

    for expr in _CHECK_ORDER:
        pose_name = paa._EXPRESSION_POSE.get(expr, "（未マッピング＝標準うなずき）")
        input(f"\nEnterで表情「{expr}」→ポーズ「{pose_name}」を再生します...")
        paa.send_expression(client, expr)
        paa.send_reaction(client, expr)
        print(f"[{expr}] 再生完了。遷移の滑らかさ／idle_downへの復帰／"
              "スウェイとの競合の有無を確認してください。")

    input("\n[3/4] 確認が終わったらEnterでスウェイ/仕草を停止します...")
    stop_event.set()
    sway_thread.join(timeout=1.0)
    gesture_thread.join(timeout=1.0)
    paa.send_expression(client, paa.NEUTRAL)
    print("[4/4] Neutralへ戻して終了しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
