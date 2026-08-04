"""ローカルLLM候補モデルの実況文生成比較（改善ロードマップ⑦・2026-08-04）。

Ollama経由で複数の候補モデルに、くれぴの実況プロンプト（キャラ設定＋用語集＋
few-shot例文＋ダブル戦術知識。`src/api/server.py`の`_SLANG_GLOSSARY_LINES`等と
同内容を手動移植・要同期）＋同一の対戦状況を渡し、出力を並べて比較する。

Windows側（Ollama起動済み・対象モデルpull済み）で実行すること。VRAM予算の前提は
`docs/adr/ADR-003-local-llm-phi3.md`の2026-08-04追記を参照
（パス1実行中は最大約8.5GBがローカルLLM用に空いている想定）。

使い方:
    # 1. 比較したいモデルを事前にpull（初回のみ・ディスク/時間がかかる）
    ollama pull qwen2.5:7b
    ollama pull qwen2.5:14b
    ollama pull gemma2:9b
    ollama pull phi3.5
    # Llama-3.1-Swallow-8B-Instruct（東工大の日本語継続事前学習版）はOllama公式
    # ライブラリに無いが、Hugging FaceのGGUF（mmnga氏変換）を直接pullできる:
    ollama pull hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.1-gguf:Q4_K_M

    # 2. 比較実行（既定シナリオ・既定4モデル）
    venv\\Scripts\\python.exe scripts\\compare_local_llm_commentary.py

    # 3. renders/配下の実データのイベントで比較したい場合（Bedrockの実際の実況文を
    #    参照として一緒に表示する。何番目のイベントかは--event-indexで指定）
    venv\\Scripts\\python.exe scripts\\compare_local_llm_commentary.py ^
        --render-dir renders\\2026-07-03-23-26-22 --event-index 5

    # 4. モデルを絞る/変える場合
    venv\\Scripts\\python.exe scripts\\compare_local_llm_commentary.py --models qwen2.5:7b,gemma2:9b

    # 5. 結果をJSONで保存したい場合
    venv\\Scripts\\python.exe scripts\\compare_local_llm_commentary.py --out comparison.json

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from render_commentary_video import load_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.commentary.kurepi_persona import (  # noqa: E402
    CHARACTER_INTRO as _CHARACTER_INTRO,
    DOUBLES_TACTICS_LINES as _DOUBLES_TACTICS_LINES,
    OUTPUT_RULES_LINES as _OUTPUT_RULES_LINES,
    SLANG_GLOSSARY_LINES as _SLANG_GLOSSARY_LINES,
    TONE_EXAMPLES as _TONE_EXAMPLES,
    battle_result_line as _battle_result_line,
)

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
# 既定4候補（2026-08-04選定）。qwen2.5:14bも候補だが既定では7bのみ
# （--models qwen2.5:7b,qwen2.5:14b,... のように追加/差し替え可）。
_DEFAULT_MODELS = [
    "hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.1-gguf:Q4_K_M",
    "qwen2.5:7b",
    "gemma2:9b",
    "phi3.5",
]
_DEFAULT_TIMEOUT_SEC = 60

# くれぴプロンプトの共通要素は src/commentary/kurepi_persona.py が単一ソース
# （phi3_client.py のローカルLLMフォールバックとも共用。2026-08-04リファクタで集約）。


def _build_kurepi_prompt(situation: dict) -> str:
    """状況dictから、くれぴの実況プロンプト（全モデル共通）を組み立てる。

    situation:
        {"turn": int|str, "player": str, "opponent": str,
         "move_log": list[str], "event_type": str, "battle_result": str（任意・"勝ち"/"負け"）}
    """
    lines = [_CHARACTER_INTRO, ""]
    lines += _DOUBLES_TACTICS_LINES + [""]
    lines += _SLANG_GLOSSARY_LINES + [""]
    lines += _OUTPUT_RULES_LINES
    battle_result = situation.get("battle_result")
    if battle_result:
        lines.append(_battle_result_line(battle_result))
    lines.append("")
    lines += ["【口調のイメージ例（この試合とは無関係な架空例・内容は真似しなくてよい）】"]
    for cond, example in _TONE_EXAMPLES:
        lines += ["【状況】", f"（{cond}）", "【実況】", example, ""]
    lines += [
        "【今回の対戦状況】",
        f"今回のイベント種別: {situation.get('event_type', '不明')}",
        f"ターン数: {situation.get('turn', '不明')}",
        f"自分の場: {situation.get('player', '情報収集中')}",
        f"相手の場: {situation.get('opponent', '情報収集中')}",
    ]
    move_log = situation.get("move_log") or []
    if move_log:
        lines.append(f"直近の技ログ: {' / '.join(move_log[-5:])}")
    lines += ["", "実況文（1〜2文・日本語のみ）："]
    return "\n".join(lines)


def _default_scenario() -> dict:
    """renderデータが無い場合の組み込みサンプル状況。"""
    return {
        "event_type": "move_single",
        "turn": 1,
        "player": "メタグロス HP:91/157 技=[アイアンヘッド]",
        "opponent": "コータス HP:64% 技=[ふんか]",
        "move_log": ["T0:コノヨザルのコーチング", "T0:メタグロスのアイアンヘッド",
                     "T1:コータスのふんか"],
    }


def _scenario_from_manifest_entry(entry: dict) -> tuple[dict, str]:
    """manifest.jsonlの1エントリから (situation, reference_commentary) を作る。

    reference_commentary はBedrockが実際に生成した実況文（比較の参考表示用。
    プロンプトには含めない＝答えを教えてしまわないようにする）。
    """
    context = entry.get("context") or {}
    situation = {
        "event_type": entry.get("event_type", "不明"),
        "turn": context.get("turn", "不明"),
        "player": context.get("player", "情報収集中"),
        "opponent": context.get("opponent", "情報収集中"),
        "move_log": context.get("move_log") or [],
    }
    if context.get("battle_result"):
        situation["battle_result"] = context["battle_result"]
    return situation, entry.get("commentary", "")


def load_scenario(render_dir: Path | None, event_index: int) -> tuple[dict, str | None]:
    """render_dirが指定されていればmanifest.jsonlの該当イベントを、
    無ければ組み込みサンプルを返す。"""
    if render_dir is None:
        return _default_scenario(), None
    manifest = load_manifest(render_dir)
    if not manifest:
        raise ValueError(f"manifest.jsonl にイベントがありません: {render_dir}")
    if not (0 <= event_index < len(manifest)):
        raise IndexError(
            f"--event-index は 0〜{len(manifest) - 1} の範囲で指定してください（指定値: {event_index}）")
    situation, reference = _scenario_from_manifest_entry(manifest[event_index])
    return situation, reference


def call_ollama(model: str, prompt: str, host: str = OLLAMA_URL,
                timeout: int = _DEFAULT_TIMEOUT_SEC) -> dict:
    """1モデルに対してOllama /api/generate を呼び、結果をdictで返す。

    Returns: {"model": str, "text": str|None, "elapsed": float, "error": str|None}
    """
    t0 = time.perf_counter()
    try:
        response = requests.post(
            host,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 150, "temperature": 0.7},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json()["response"].strip()
        return {"model": model, "text": text, "elapsed": time.perf_counter() - t0, "error": None}
    except requests.exceptions.ConnectionError:
        return {"model": model, "text": None, "elapsed": time.perf_counter() - t0,
                "error": "Ollamaに接続できません（起動していますか？）"}
    except Exception as exc:  # noqa: BLE001 - 比較ツールなので1モデル失敗でも他を続ける
        return {"model": model, "text": None, "elapsed": time.perf_counter() - t0, "error": str(exc)}


def run_comparison(models: list[str], situation: dict, host: str = OLLAMA_URL,
                   timeout: int = _DEFAULT_TIMEOUT_SEC) -> list[dict]:
    prompt = _build_kurepi_prompt(situation)
    return [call_ollama(model, prompt, host=host, timeout=timeout) for model in models]


def format_results(results: list[dict], reference_commentary: str | None = None) -> str:
    lines = []
    if reference_commentary:
        lines += [f"【参考: Bedrockの実際の実況文】\n  {reference_commentary}\n"]
    for r in results:
        lines.append(f"── {r['model']} ({r['elapsed']:.1f}s) " + "─" * 20)
        if r["error"]:
            lines.append(f"  [エラー] {r['error']}")
        else:
            lines.append(f"  {r['text']}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="ローカルLLM候補モデルの実況文生成比較")
    parser.add_argument("--models", default=",".join(_DEFAULT_MODELS),
                        help=f"比較するOllamaモデルタグ・カンマ区切り（既定: {','.join(_DEFAULT_MODELS)}）")
    parser.add_argument("--render-dir", type=Path, default=None,
                        help="renders/<動画名> を指定すると実データのイベントで比較する（省略時は組み込みサンプル）")
    parser.add_argument("--event-index", type=int, default=0,
                        help="--render-dir指定時、manifest.jsonlの何番目のイベントを使うか（既定0）")
    parser.add_argument("--host", default=OLLAMA_URL, help="Ollama APIのURL")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_SEC, help="1モデルあたりのタイムアウト秒")
    parser.add_argument("--out", type=Path, default=None, help="結果をJSONで保存するパス（任意）")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    situation, reference = load_scenario(args.render_dir, args.event_index)

    logger.info("比較対象モデル: %s", ", ".join(models))
    logger.info("状況: %s\n", situation)

    results = run_comparison(models, situation, host=args.host, timeout=args.timeout)
    print(format_results(results, reference_commentary=reference))

    if args.out:
        args.out.write_text(
            json.dumps({"situation": situation, "reference": reference, "results": results},
                      ensure_ascii=False, indent=2),
            encoding="utf-8")
        logger.info("結果を保存しました: %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
