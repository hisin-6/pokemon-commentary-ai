"""
Phi-3 mini 実況文生成クライアント
Ollama経由でローカルLLMを呼び出し、ポケモン対戦の実況文を生成する。

実行環境: Windows Python (venv/Scripts/python.exe)
Ollama URL: http://localhost:11434
"""

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"
HISTORY_SIZE = 3

SYSTEM_PROMPT = """あなたはポケモン対戦の実況者です。以下のルールで実況してください。
- 1〜2文で簡潔に実況する
- テンポよく、興奮感のある実況をする
- ポケモン名・技名はそのまま使う
- HPが低い時は緊張感を出す
- 日本語で出力する
- 【重要】画面から読み取れた情報だけを使う。不明な情報は絶対に創作しない
- 鉤括弧（「」）は使わない
- 余計な説明・質問・指示は出力しない。実況文だけ出力する
"""


class Phi3Client:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = MODEL_NAME,
        history_size: int = HISTORY_SIZE,
        timeout: int = 60,
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
    ) -> str:
        """
        実況文を生成する。

        Args:
            game_state: OCR/YOLOで取得した対戦状況
                {
                    "pokemon_player": str,
                    "hp_player": int,
                    "pokemon_opponent": str,
                    "hp_opponent": int,
                    "last_move": str,
                    "status": str,
                    "balls_remaining": [int, int],
                    "event_type": str,
                }
            bedrock_analysis: Bedrock Vision分析結果テキスト（任意）

        Returns:
            生成された実況テキスト
        """
        prompt = self._build_prompt(game_state, bedrock_analysis)

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 120,  # 最大生成トークン数（1〜2文で十分）
                    "temperature": 0.7,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        commentary = response.json()["response"].strip()
        self._add_history(commentary)
        return commentary

    def _build_prompt(self, game_state: dict, bedrock_analysis: str | None) -> str:
        lines = [SYSTEM_PROMPT, ""]

        # Bedrock 分析結果がある場合のみ使う
        if bedrock_analysis and not bedrock_analysis.startswith("（テキスト未検出）"):
            lines.append(f"現在の対戦状況（AI分析）: {bedrock_analysis}")
            lines.append("")

        # 値が確定している情報だけ渡す（不明・?・（OCR参照）は渡さない）
        ocr_text = game_state.get("ocr_text", "")
        if ocr_text and ocr_text != "（テキスト未検出）":
            lines.append(f"画面テキスト（OCR）: {ocr_text}")

        status = game_state.get("status", "なし")
        if status and status != "なし":
            lines.append(f"状態異常: {status}")

        balls = game_state.get("balls_remaining", [])
        if len(balls) == 2 and (balls[0] > 0 or balls[1] > 0):
            lines.append(f"残りボール: 自分{balls[0]}個 / 相手{balls[1]}個")

        if self._history:
            lines.append(f"直前の実況: {self._history[-1]}")

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

    test_state = {
        "pokemon_player": "ガブリアス",
        "hp_player": 85,
        "pokemon_opponent": "サーフゴー",
        "hp_opponent": 42,
        "last_move": "じしん",
        "status": "normal",
        "balls_remaining": [6, 4],
        "event_type": "move_used",
    }

    print("Phi-3 mini 接続テスト中...")
    try:
        result = client.generate_commentary(test_state)
        print(f"生成結果: {result}")
    except requests.exceptions.ConnectionError:
        print("エラー: Ollamaが起動していません。タスクトレイのOllamaアイコンを確認してください。")
    except Exception as e:
        print(f"エラー: {e}")
