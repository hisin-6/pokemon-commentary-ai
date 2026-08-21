"""予測→回収パス: 試合をまたぐ「予測」と「回収」の生成（2026-08-21新設・ADR-009追記）。

パス1の素材（manifest.jsonl）から
  - 「予測ポイント」: 場のコンディション（おいかぜ・壁・天候等）を陣営として
    初めて確立した move_single イベント（自分側/相手側それぞれ最大1件＝最大2件）
  - 「決定的な出来事」: 試合最後の faint イベント（無ければ battle_end）＝回収アンカー
を機械的に検出し、EC2 ``/api/script``（mode=predict/payoff）にテキストのみで
送信して予測文・回収文を生成、VOICEVOXでWAV化して ``predictions.jsonl`` として
素材ディレクトリに追加する。パス2（render_commentary_video.py）は
predictions.jsonl があれば fillers.jsonl と同様に自動でマージする。

的中/外れの判定はこのスクリプトが最終battle_resultから機械的に確定させる
（LLMには判定させず、演技だけをさせる。詳細:
docs/design/prediction-payoff-commentary-idea.md）。

battle_result が確定していない試合（降参・OCR未検出等）では予測は生成しない
（0件も正常な結果として許容する）。

使い方:
    python scripts/generate_predictions.py renders/16-14-39 --ec2-url http://<EC2>:5000
    （VOICEVOX起動が必要。--dry-run はBedrock生成結果の表示まで＝VOICEVOX不要）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

# 同ディレクトリのパス2実装からmanifest読み込みを再利用
sys.path.insert(0, str(Path(__file__).parent))
from render_commentary_video import load_manifest  # noqa: E402

# プロジェクトルート（src.* のインポート用）
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def find_prediction_candidates(entries: list) -> list:
    """陣営ごとに「場のコンディションを新たに確立したmove_singleイベント」を
    最大1件ずつ検出する（自分側/相手側で最大2件）。

    manifest.jsonlのcontext.condition_hintは「自分側におい風（あと4ターン・
    素早さ2倍）」のように陣営名で始まる文字列（src/pipeline.py の
    _compute_condition_hint 由来）。直前まで無かった/別内容だったのに
    move_singleイベントで初めてその陣営の条件として現れた瞬間を「予測しがいの
    ある場面」とみなす。既存のcondition_hint計算をそのまま再利用するだけで
    新規DB問い合わせは不要。

    ⚠️既知の限界: 両陣営が同時に条件を持つ場合の合成テキストまでは分解して
    いない（自分側/相手側どちらの文字列で始まるかのみで判定）。MVP実装として
    許容する。
    """
    candidates: list[dict] = []
    seen_sides: set[str] = set()
    prev_hint: str | None = None
    for e in entries:
        ctx = e.get("context") or {}
        hint = ctx.get("condition_hint")
        if e.get("event_type") == "move_single" and hint and hint != prev_hint:
            side = ("player" if hint.startswith("自分側") else
                    "opponent" if hint.startswith("相手側") else None)
            if side and side not in seen_sides:
                seen_sides.add(side)
                move_log = ctx.get("move_log") or []
                candidates.append({
                    "time": e["event_time"],
                    "side": side,
                    "move_text": move_log[-1] if move_log else "?",
                    "hint": hint,
                })
        prev_hint = hint
    candidates.sort(key=lambda c: c["time"])
    return candidates


def find_decisive_event(entries: list) -> dict | None:
    """回収アンカー（決定的な出来事）を探す。

    試合最後の faint イベント（event_time最大）を優先し、無ければ
    battle_end にフォールバックする（降参決着等でfaintが1件も無いケース）。
    """
    faints = [e for e in entries if e.get("event_type") == "faint"]
    if faints:
        return max(faints, key=lambda e: e["event_time"])
    ends = [e for e in entries if e.get("event_type") == "battle_end"]
    if ends:
        return max(ends, key=lambda e: e["event_time"])
    return None


def determine_battle_result(entries: list) -> str | None:
    """試合の最終battle_result（"勝ち"/"負け"）を取得する。確定できなければNone。"""
    for e in reversed(entries):
        result = (e.get("context") or {}).get("battle_result")
        if result:
            return result
    return None


def judge_hit(side: str, battle_result: str) -> bool:
    """予測が的中したかを機械的に判定する（LLMには渡さず事実として決定する）。"""
    if side == "player":
        return battle_result == "勝ち"
    return battle_result == "負け"


def _events_payload(entries: list) -> list:
    """/api/script に送る events ペイロードへ変換する（request_fillersと同形式）。"""
    return [
        {
            "time": e["event_time"],
            "event_type": e["event_type"],
            "commentary": e["commentary"],
            "context": e.get("context"),
        }
        for e in entries
    ]


def request_prediction(ec2_url: str, events_payload: list, candidate: dict,
                       persona: str = "kurepi", timeout: float = 120.0) -> str | None:
    """EC2 /api/script（mode=predict）を呼び、予測文を1件生成する。"""
    payload = {
        "mode": "predict",
        "events": events_payload,
        "gap": {"start": candidate["time"], "end": candidate["time"]},
        "persona": persona,
    }
    resp = requests.post(f"{ec2_url.rstrip('/')}/api/script", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"/api/script(predict) 失敗: {data.get('error')} {data.get('message')}")
    fillers = data.get("fillers") or []
    if not fillers:
        logger.warning("予測生成: t=%.1fs で0件応答", candidate["time"])
        return None
    logger.info("予測生成 t=%.1fs(%s): %s", candidate["time"], candidate["side"],
                fillers[0]["text"][:40])
    return fillers[0]["text"]


def request_payoff(ec2_url: str, prediction_text: str, hit: bool, outcome_summary: str,
                   payoff_time: float, persona: str = "kurepi",
                   timeout: float = 120.0) -> str | None:
    """EC2 /api/script（mode=payoff）を呼び、回収文を1件生成する。"""
    payload = {
        "mode": "payoff",
        "prediction_text": prediction_text,
        "hit": hit,
        "outcome_summary": outcome_summary,
        "time": payoff_time,
        "persona": persona,
    }
    resp = requests.post(f"{ec2_url.rstrip('/')}/api/script", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"/api/script(payoff) 失敗: {data.get('error')} {data.get('message')}")
    fillers = data.get("fillers") or []
    if not fillers:
        logger.warning("回収生成: t=%.1fs で0件応答", payoff_time)
        return None
    logger.info("回収生成 t=%.1fs(的中=%s): %s", payoff_time, hit, fillers[0]["text"][:40])
    return fillers[0]["text"]


def synthesize_predictions(render_dir: Path, items: list, voicevox_url: str,
                           speaker: int) -> list:
    """予測・回収セリフをVOICEVOXでWAV化し predictions.jsonl のエントリ列を返す。

    items: [{"time": .., "text": .., "event_type": "prediction"/"prediction_payoff"}, ...]
    """
    from src.output.render_sink import RenderSink
    from src.output.voicevox_client import VoicevoxClient

    client = VoicevoxClient(url=voicevox_url, speaker=speaker)
    wav_dir = render_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    entries = []
    for i, item in enumerate(items, 1):
        wav_bytes = client.generate_wav(item["text"])
        wav_name = f"p{i:04d}_{item['event_type']}.wav"
        (wav_dir / wav_name).write_bytes(wav_bytes)
        entry = {
            "seq": i,
            "event_time": item["time"],
            "event_type": item["event_type"],
            "commentary": item["text"],
            "wav": f"wav/{wav_name}",
            "duration": round(RenderSink.wav_duration(wav_bytes), 3),
        }
        entries.append(entry)
        logger.info("[予測/回収] #%d t=%.1fs (%.1f秒) %s", i, item["time"],
                    entry["duration"], item["text"][:30])
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="予測→回収パス（2026-08-21新設）")
    parser.add_argument("render_dir", help="パス1の素材ディレクトリ（renders/<動画名>）")
    parser.add_argument("--ec2-url", required=True, help="EC2 APIのURL（http://<IP>:5000）")
    parser.add_argument("--voicevox-url", default="http://localhost:50021",
                        help="VOICEVOXのURL（既定 http://localhost:50021）")
    parser.add_argument("--speaker", type=int, default=None,
                        help="VOICEVOX話者ID（省略時はrender_info.jsonの値）")
    parser.add_argument("--dry-run", action="store_true",
                        help="Bedrock生成結果の表示まで（VOICEVOX不要・predictions.jsonl未更新）")
    parser.add_argument("--persona", choices=["kurepi", "neutral"], default="kurepi",
                        help="キャラクター設定（既定kurepi。パス1と同じ値を指定すること）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    render_dir = Path(args.render_dir)
    if not (render_dir / "manifest.jsonl").exists():
        logger.error("manifest.jsonl が見つかりません: %s", render_dir)
        return 1

    entries = load_manifest(render_dir)

    candidates = find_prediction_candidates(entries)
    decisive = find_decisive_event(entries)
    battle_result = determine_battle_result(entries)

    if not candidates:
        logger.info("予測ポイント（場のコンディションを確立したmove_single）なし。予測生成は不要")
    if decisive is None:
        logger.info("回収アンカー（faint/battle_end）なし。予測生成は不要")
    if battle_result is None:
        logger.info("battle_result 未確定（降参・OCR未検出等）。予測生成は不要")

    if not candidates or decisive is None or battle_result is None:
        if not args.dry_run:
            (render_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        return 0

    events_payload = _events_payload(entries)
    items = []  # request_predictionが成功した候補ぶんの{time,text,event_type}
    for c in candidates:
        prediction_text = request_prediction(args.ec2_url, events_payload, c,
                                             persona=args.persona)
        if not prediction_text:
            continue
        hit = judge_hit(c["side"], battle_result)
        outcome_summary = decisive.get("commentary", "")
        payoff_text = request_payoff(args.ec2_url, prediction_text, hit, outcome_summary,
                                     decisive["event_time"], persona=args.persona)
        if not payoff_text:
            continue
        items.append({"time": c["time"], "text": prediction_text, "event_type": "prediction"})
        items.append({"time": decisive["event_time"], "text": payoff_text,
                      "event_type": "prediction_payoff"})
        print(f"  {c['time']:>7.1f}s [予測/{c['side']}]  {prediction_text}")
        print(f"  {decisive['event_time']:>7.1f}s [回収/的中={hit}]  {payoff_text}")

    if args.dry_run:
        logger.info("dry-run のためここまで（predictions.jsonl 未更新）")
        return 0

    if not items:
        (render_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        logger.info("予測/回収とも0件（生成失敗またはBedrock応答なし）")
        return 0

    with (render_dir / "render_info.json").open(encoding="utf-8") as fp:
        info = json.load(fp)
    speaker = args.speaker if args.speaker is not None else int(info.get("speaker", 2))

    written = synthesize_predictions(render_dir, items, args.voicevox_url, speaker)
    predictions_path = render_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fp:
        for e in written:
            fp.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info("予測/回収素材出力完了: %d件 → %s", len(written), predictions_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
