"""
ローカルLLM 実況文生成クライアント（Bedrockフォールバック用）
Ollama経由でローカルLLMを呼び出し、ポケモン対戦の実況文を生成する。

実行環境: Windows Python (venv/Scripts/python.exe)
Ollama URL: http://localhost:11434

⚠️ クラス名/ファイル名は歴史的経緯で`Phi3Client`/`phi3_client.py`のままだが、
既定モデルは2026-08-04に`gemma2:9b`へ変更した（ADR-003追記・改善ロードマップ⑦の
第一歩）。`src/pipeline.py`側のimport文（`from src.commentary.phi3_client import
Phi3Client`）・属性名（`self._phi3`）を変えない範囲での最小変更にしている。
プロンプトは`scripts/compare_local_llm_commentary.py`によるモデル比較の結果を踏まえ、
`src/commentary/kurepi_persona.py`（くれぴのキャラ設定・用語集・出力ルール共通ソース）
から組み立てる。
"""

from __future__ import annotations

import requests

from src.commentary import kurepi_persona as persona

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:9b"
HISTORY_SIZE = 3


class Phi3Client:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = MODEL_NAME,
        history_size: int = HISTORY_SIZE,
        timeout: int = 20,
        # Bedrock失敗時のフォールバックとして同期呼び出しされるため、メインループを
        # 長時間ブロックしないよう、プライマリ経路（Bedrock, timeout=15s）と近い値に
        # 抑える。旧値60sはOllamaハング時にメインループを最大60秒止めてしまっていた。
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.history_size = history_size
        self.timeout = timeout
        self._history: list[str] = []

    def generate_commentary(
        self,
        game_state: dict,
        bedrock_analysis: str | None = None,
        battle_context: dict | None = None,
    ) -> str:
        """
        実況文を生成する。

        Args:
            game_state: OCR/YOLOで取得した対戦状況（"event_type"/"status"/"ocr_text"等。
                pipeline.pyの`_ocr_results_to_text`等が組み立てる辞書）
            bedrock_analysis: Bedrock Vision分析結果テキスト（任意・補足情報として使う）
            battle_context: `BattleStateTracker.to_context()`の戦況サマリー（任意）。
                自分/相手の場・控え・ターン数等の構造化情報。渡せる場合は必ず渡すこと
                （2026-08-04のモデル比較で、これが無いとgemma2:9bでも精度が大きく落ちると
                判明したため）

        Returns:
            生成された実況テキスト
        """
        prompt = self._build_prompt(game_state, bedrock_analysis, battle_context)

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 150,  # 最大生成トークン数（1〜2文で十分）
                    "temperature": 0.7,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        commentary = response.json()["response"].strip()
        self._add_history(commentary)
        return commentary

    def _build_prompt(self, game_state: dict, bedrock_analysis: str | None,
                      battle_context: dict | None) -> str:
        lines = [persona.CHARACTER_INTRO, ""]
        lines += persona.DOUBLES_TACTICS_LINES + [""]
        lines += persona.SLANG_GLOSSARY_LINES + [""]
        lines += persona.OUTPUT_RULES_LINES
        result = (battle_context or {}).get("battle_result")
        if result:
            lines.append(persona.battle_result_line(result))
        lines.append("")

        lines += ["【今回の対戦状況】", f"今回のイベント種別: {game_state.get('event_type', '不明')}"]
        if battle_context:
            lines += [
                f"ターン数: {battle_context.get('turn', '不明')}",
                f"自分の場: {battle_context.get('player_pokemon', '情報収集中')}",
                f"相手の場: {battle_context.get('opponent_pokemon', '情報収集中')}",
            ]
            event_log = battle_context.get("event_log")
            if event_log:
                lines.append(f"直近の出来事: {event_log}")

        # Bedrock 分析結果がある場合のみ補足として使う
        if bedrock_analysis and not bedrock_analysis.startswith("（テキスト未検出）"):
            lines.append(f"画面の補足情報: {bedrock_analysis}")

        status = game_state.get("status", "なし")
        if status and status != "なし":
            lines.append(f"状態異常: {status}")

        if self._history:
            lines.append(f"直前の実況（繰り返さないこと）: {self._history[-1]}")

        lines.append("\n実況文（1〜2文・日本語のみ）：")
        return "\n".join(lines)

    def _add_history(self, commentary: str) -> None:
        self._history.append(commentary)
        if len(self._history) > self.history_size:
            self._history.pop(0)

    def clear_history(self) -> None:
        """試合開始時などに履歴をリセットする"""
        self._history = []


if __name__ == "__main__":
    client = Phi3Client()

    test_state = {"event_type": "move_used", "status": "なし"}
    test_battle_context = {
        "turn": 3,
        "player_pokemon": "場: ガブリアス / 控え: なし",
        "opponent_pokemon": "場: サーフゴー / 控え: なし",
        "event_log": "T3:ガブリアスのじしん",
    }

    print(f"{client.model} 接続テスト中...")
    try:
        result = client.generate_commentary(test_state, battle_context=test_battle_context)
        print(f"生成結果: {result}")
    except requests.exceptions.ConnectionError:
        print("エラー: Ollamaが起動していません。タスクトレイのOllamaアイコンを確認してください。")
    except Exception as e:
        print(f"エラー: {e}")
