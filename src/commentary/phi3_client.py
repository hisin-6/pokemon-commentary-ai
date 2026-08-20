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

import logging

import requests

from src.commentary import kurepi_persona as persona

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:9b"
HISTORY_SIZE = 3

# 自分/相手の帰属を取り違えた言い回し（改善ロードマップ「戦況推論強化」2026-08-04・
# phi3.5の実機比較で「我がイッカネズミ」＝相手のポケモンを自分呼ばわりする事故を確認）。
_SELF_ATTRIBUTION_PREFIXES = ("自分の", "我が", "うちの", "味方の")
_OPPONENT_ATTRIBUTION_PREFIXES = ("相手の",)


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
        persona: str = "kurepi",
        # 2026-08-14: 3Dモデル一時差し替え検証用（--persona neutral）。
        # "kurepi"=花圓くれぴ（デフォルト・従来動作）/"neutral"=中立実況口調
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.history_size = history_size
        self.timeout = timeout
        # 属性名は`self.persona`にしない: このファイルは
        # `from src.commentary import kurepi_persona as persona` でモジュールを
        # `persona`という名前でimportしているため、同名だと紛らわしい
        self._persona = persona
        self._history: list[str] = []

    def generate_commentary(
        self,
        game_state: dict,
        bedrock_analysis: str | None = None,
        battle_context: dict | None = None,
        samples: int = 1,
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
            samples: 生成を試みる回数（既定1）。2以上の場合、自分/相手の帰属エラーが
                無い最初のサンプルを採用する「consistent action generation」
                （PokéLLMon論文に着想・2026-08-04）。呼び出しが線形に増えるため、
                即時性が求められないバッチ処理（動画モード後付け生成）でのみ使うこと。

        Returns:
            生成された実況テキスト
        """
        prompt = self._build_prompt(game_state, bedrock_analysis, battle_context)

        candidates: list[str] = []
        for _ in range(max(1, samples)):
            text = self._generate_once(prompt)
            candidates.append(text)
            if not self._has_attribution_error(text, battle_context):
                self._add_history(text)
                return text

        log.warning("生成%d回とも自分/相手の帰属エラーの疑いあり。先頭のサンプルを採用: 「%s」",
                   len(candidates), candidates[0])
        self._add_history(candidates[0])
        return candidates[0]

    def _generate_once(self, prompt: str) -> str:
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
        return response.json()["response"].strip()

    @staticmethod
    def _has_attribution_error(text: str, battle_context: dict | None) -> bool:
        """相手のポケモンを自分側の言い回しで（またはその逆で）言及していないかを
        簡易チェックする。battle_contextに`player_names`/`opponent_names`
        （`BattleStateTracker.to_context()`のRAG用フィールド）が無ければ判定不能として
        Falseを返す（過検出で全サンプル却下になるのを避ける）。"""
        if not battle_context:
            return False
        opponent_names = set(battle_context.get("opponent_names") or [])
        player_names = set(battle_context.get("player_names") or [])
        for name in opponent_names:
            if any(f"{prefix}{name}" in text for prefix in _SELF_ATTRIBUTION_PREFIXES):
                return True
        for name in player_names:
            if any(f"{prefix}{name}" in text for prefix in _OPPONENT_ATTRIBUTION_PREFIXES):
                return True
        return False

    def _build_prompt(self, game_state: dict, bedrock_analysis: str | None,
                      battle_context: dict | None) -> str:
        intro = persona.CHARACTER_INTRO if self._persona == "kurepi" else persona.NEUTRAL_CHARACTER_INTRO
        lines = [intro, ""]
        lines += persona.DOUBLES_TACTICS_LINES + [""]
        lines += persona.SLANG_GLOSSARY_LINES + [""]
        lines += persona.OUTPUT_RULES_LINES
        result = (battle_context or {}).get("battle_result")
        if result:
            lines.append(persona.battle_result_line(result))
        if (battle_context or {}).get("battle_surrendered"):
            lines.append(persona.battle_surrendered_line(result))
        lines.append("")

        lines += ["【今回の対戦状況】", f"今回のイベント種別: {game_state.get('event_type', '不明')}"]
        # move_single: ターン全体ではなく「今まさに使われた技1つ」だけに焦点を絞らせる。
        # move_focus はpipeline側で組み立てた「陣営の＋ポケモン名＋の＋技名」の文字列
        # （server.py の Bedrock 用プロンプトと同じ指示。従来ここが抜けており、
        # 直近の技ログ全体から古い技を今起きたことのように話してしまうバグがあった）
        if game_state.get("event_type") == "move_single" and game_state.get("move_focus"):
            lines.append(
                f"今まさに使われた技「{game_state['move_focus']}」1つだけに反応する"
                "短い実況をする（ターン全体のまとめや他の技には触れず、この技への即時リアクションに徹すること）"
            )
        # 合成faint（ボール数減少推定・パイプライン側で確定済み）: 画面にHP=0表示は
        # 映っていないため、確定済みの対象を直接指示する（server.py側と同じ文言。
        # move_focus/type_hintで過去2回あった片側配線漏れの再発防止として同時に配線）
        if game_state.get("event_type") == "faint" and game_state.get("faint_focus"):
            lines.append(
                f"「{game_state['faint_focus']}」が倒れたことが蓄積した戦況データから確定した"
                "（画面にHP=0の表示は映っていない）。倒れたことに今気づいた体で、"
                "このポケモンが倒れたことだけを実況する"
            )
        # 交代ヒント（2026-08-15・server.py側と同じ文言で配線）: switchイベントは
        # 交代選択画面の時点で発火するため、実際に繰り出されたポケモンを直接指示する
        if game_state.get("event_type") == "switch" and game_state.get("switch_focus"):
            lines.append(
                f"ポケモンの交代・繰り出しの場面。実際に繰り出されたのは「{game_state['switch_focus']}」"
                "（画面の繰り出しメッセージから確定）。この繰り出しだけを実況し、"
                "それより前の交代を今起きたかのように語らないこと"
            )
        elif game_state.get("switch_focus"):
            lines.append(
                f"直近で実際に繰り出されたポケモン（画面の繰り出しメッセージから確定）: "
                f"{game_state['switch_focus']}。交代・繰り出しに言及する場合は必ずこれに従うこと"
            )
        # move_used=新しいターンの攻防が始まる瞬間（2026-08-15・server.py側と同じ趣旨）:
        # 個別の技はmove_singleが都度実況するため、戦況全体に徹させる
        if game_state.get("event_type") == "move_used":
            lines.append(
                "コマンドが確定して新しいターンの攻防が始まる場面。戦況全体（HP状況・残り数・"
                "有利不利）とこのターンの注目ポイントを実況する。イベント履歴・技ログにある"
                "過去の技や交代を今起きたかのように実況し直さないこと"
            )
        if game_state.get("faint_context"):
            lines.append(
                f"直前に起きた気絶の時点の戦況（この直後に下記の技が使われた）: {game_state['faint_context']}"
            )
        if battle_context:
            lines += [
                f"ターン数: {battle_context.get('turn', '不明')}",
                f"自分の場: {battle_context.get('player_pokemon', '情報収集中')}",
                f"相手の場: {battle_context.get('opponent_pokemon', '情報収集中')}",
            ]
            event_log = battle_context.get("event_log")
            if event_log:
                lines.append(f"直近の出来事: {event_log}")

            # タイプ相性ヒント・場のコンディション（天候・壁・素早さ操作）:
            # 改善ロードマップ「戦況推論強化」（2026-08-04）でserver.py（Bedrock）側の
            # プロンプトには追加済みだったが、Phi3Client側は配線が漏れていた
            # （2026-08-08発見: 天候「にほんばれ」下でウェザーボールが炎技になる
            # ことをローカルLLMが知らず「水技」と誤って実況していた）。文言は
            # server.pyの_build_vision_promptと合わせている。
            type_hint = battle_context.get("type_hint")
            if type_hint:
                lines.append(
                    f"タイプ相性ヒント（Python側で計算済みの確定結果。"
                    f"信頼して有利不利の実況に使ってよい）: {type_hint}"
                )
            # 技効果ヒントRAG（2026-08-14・server.py側と同じ文言で配線）:
            # パス1検証の最頻NGパターン「技の効果に関する事実誤認」対策
            move_effect_hint = battle_context.get("move_effect_hint")
            if move_effect_hint:
                lines.append(
                    f"直近で使われた技の効果（PokeAPI由来のデータ。信頼して事実として"
                    f"扱ってよい）: {move_effect_hint}"
                )
                lines.append(
                    "※ ダメージを与えない変化技（能力変化・状態異常付与等）の場合、"
                    "「ダメージ」「効果ばつぐん」「〜に効いた」等の攻撃結果を表す言葉を使わないこと"
                )
                lines.append(
                    "※ 天候による技の威力アップは水/ほのお/こおり等一部タイプの技に限られる"
                    "特殊ルールで、上記の効果テキストにその記載が無い限り、天候を理由に"
                    "「威力が上がる/絶大」等と独自の知識で付け足さないこと（2026-08-16・"
                    "server.py側と同じ文言で配線）"
                )
            # 技の対象ヒント（2026-08-15・server.py側と同じ文言で配線。技の対象範囲
            # （自分自身/相手全体等）の合流は2026-08-16）:
            # move_single対象誤認（最頻NG）対策。技の仕様＋直後の観測に厳密に従わせる
            move_target_hint = battle_context.get("move_target_hint")
            if move_target_hint:
                lines.append(
                    f"この技の対象・結果に関する確定情報（技の仕様＋画面から観測された"
                    f"変化。Python側で照合済み）: {move_target_hint}"
                )
                lines.append(
                    "※ この技の対象・結果は必ず上記に従うこと。そこに登場しない"
                    "ポケモンをこの技の対象として実況しないこと"
                )
            condition_hint = battle_context.get("condition_hint")
            if condition_hint:
                lines.append(
                    f"場のコンディション（天候・壁・素早さ操作。Python側で計算済みの確定結果。"
                    f"信頼して有利不利の実況に使ってよい）: {condition_hint}"
                )
                lines.append(
                    "※ 上記に記載の無い天候・壁・トリックルーム・おいかぜは発生していない"
                    "ものとして扱い、独自の知識や一般的な戦術の連想で勝手に補って"
                    "言及しないこと（2026-08-16・server.py側と同じ文言で配線）"
                )
            else:
                lines.append(
                    "※ 天候・壁・トリックルーム・おいかぜは現在いずれも発生していない"
                    "（Python側で確認済み）。これらに言及しないこと（2026-08-16新設）"
                )

        # Bedrock 分析結果がある場合のみ補足として使う
        if bedrock_analysis and not bedrock_analysis.startswith("（テキスト未検出）"):
            lines.append(f"画面の補足情報: {bedrock_analysis}")

        status = game_state.get("status", "なし")
        if status and status != "なし":
            lines.append(f"状態異常: {status}")

        if self._history:
            lines.append(f"直前の実況（繰り返さないこと）: {self._history[-1]}")

        lines.append("\n実況文（1〜2文・日本語のみ）：")
        prompt = "\n".join(lines)
        # ローカルLLMに実際に送るプロンプト全文をデバッグログに出す（2026-08-21新設・
        # RAGヒント等が実際どう組み込まれたかを事後確認できるように）。既定ではroot
        # ロガーがINFOのため出力されない。確認時はログレベルをDEBUGに上げること。
        log.debug("[Phi3プロンプト]\n%s", prompt)
        return prompt

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
