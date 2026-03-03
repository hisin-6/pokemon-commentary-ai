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
"""


class Phi3Client:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = MODEL_NAME,
        history_size: int = HISTORY_SIZE,
        timeout: int = 30,
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
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        commentary = response.json()["response"].strip()
        self._add_history(commentary)
        return commentary

    def _build_prompt(self, game_state: dict, bedrock_analysis: str | None) -> str:
        lines = [SYSTEM_PROMPT, ""]

        if bedrock_analysis:
            lines.append(f"【画面分析】\n{bedrock_analysis}\n")

        lines.append("【対戦状況】")
        lines.append(f"自分のポケモン: {game_state.get('pokemon_player', '不明')} (HP: {game_state.get('hp_player', '?')}%)")
        lines.append(f"相手のポケモン: {game_state.get('pokemon_opponent', '不明')} (HP: {game_state.get('hp_opponent', '?')}%)")
        lines.append(f"使用した技: {game_state.get('last_move', '不明')}")
        lines.append(f"状態異常: {game_state.get('status', 'なし')}")

        balls = game_state.get("balls_remaining", [])
        if len(balls) == 2:
            lines.append(f"残りボール: 自分{balls[0]}個 / 相手{balls[1]}個")

        if self._history:
            lines.append("\n【直前の実況】")
            for past in self._history:
                lines.append(f"- {past}")

        lines.append("\n上記の状況を実況してください：")
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
