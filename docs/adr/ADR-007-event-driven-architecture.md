# ADR-007: イベント駆動アーキテクチャによる処理トリガーを採用する

## ステータス
承認済み（2026-03-18 改訂）

## 日付
2026-02-25（初版）/ 2026-03-18（Sprint 5 実装反映）

## 文脈

毎フレームすべてのコンポーネントを動かすとCPU/GPU負荷が高く、APIコストも増大する。画面に変化がないターン中は処理をスキップし、イベント発生時にのみ各コンポーネントを起動するイベント駆動設計が必要。

「どの変化をイベントと見なすか」「イベントごとにどのコンポーネントを呼ぶか」の定義がなければ実装が発散するため、本ADRで明示的に定義する。

Sprint 5 実装でフェーズ遷移ベースのイベント検知（`BattlePhaseClassifier`）と戦況蓄積クラス（`BattleStateTracker`）を導入。初版のイベント定義から実装に合わせて改訂。

## 決定

以下のトリガー条件とコンポーネント呼び出しルールを採用する。

## イベント定義とトリガー条件（Sprint 5 実装版）

### バトルフェーズの分類

`BattlePhaseClassifier` が OCR テキストから現在フェーズを判定する。

| フェーズ | 判定キーワード |
|---------|-------------|
| `command_select` | 「たたかう」「どうする」 |
| `switch_select` | 「こうたい」「ポケモンをえらんで」 |
| `animation` | 「バツグンだ」「いまひとつ」「こうかなし」「きゅうしょ」等 |
| `faint` | `0/XX`（分母 50 以上）のHP値を検出 |
| `battle_end` | 「勝負に勝」「勝負に負」「降参が選ばれ」「降参」 |
| `selection_screen` | 「ポケモンを選んで」「選出」「きめる」「リーダー」「選出順」 |
| `unknown` | 上記いずれにも該当しない（演出中など） |

### フェーズ遷移からのイベント発火

| 遷移 | 発火イベント |
|------|-----------|
| 初回 `command_select` 出現 | `battle_start` |
| `command_select` → それ以外 | `move_used` |
| `command_select` → `switch_select` | `switch` |
| `faint` フェーズ出現（直前が非 faint）| `faint` |
| `switch_select` 出現（`command_select` 以外から）| `switch` |
| `battle_end` フェーズ出現 | `battle_end` |
| `selection_screen` 出現 | `battle_started` フラグをリセット（イベントは発火しない） |

### デバウンス

- `move_used`・`faint`・`switch` は同一イベントの連続発火を **10秒** 間抑制
- `battle_start`・`battle_end` はデバウンスなし（試合に1回のみ）

## イベント発生時のコンポーネント呼び出しルール

```
[1] OBS仮想カメラからキャプチャ（1秒ごと・cv2.VideoCapture）

[2] OpenCV 差分検出 → 変化がなければスキップ

[3] EasyOCR（差分あり時のみ）
    + YOLOv8（毎フレーム）

[4] BattlePhaseClassifier.detect()
    → フェーズ分類 → フェーズ遷移でイベント発火

[5] イベント発火時:
    ├─ バトル外画面判定（_NON_BATTLE_KEYWORDS）→ 該当すればスキップ
    ├─ OCR 件数チェック（2件未満はスキップ、battle_end は除外）
    ├─ game_state 構築（OCR y 座標で自分/相手 ポケモン名を分離）
    │   - y < 500 px  → 相手エリア
    │   - 500〜700 px → 自分エリア
    │   - y > 700 px  → コマンドメニュー（除外）
    ├─ BattleStateTracker.update() ← _battle_active = True の間のみ
    └─ OCRデバッグ画像を debug/ に保存

[6] _battle_active フラグによる Bedrock 呼び出し制御
    - battle_start イベント → _battle_active = True にセット → Bedrock 呼び出し
    - battle_end イベント  → Bedrock 呼び出し → _battle_active = False にリセット
    - _battle_active = False の間（選出画面等）→ Bedrock 呼び出しスキップ

[7] Bedrock Vision 呼び出し（EC2 API 経由）
    → BattleStateTracker.to_context() で蓄積した戦況も渡す
    失敗時 → Phi-3 mini フォールバック

[8] VOICEVOX で音声合成 → 再生

[9] S3 に実況ログ・スクリーンショット保存
```

## Bedrockを呼ぶイベント

| イベント | Bedrock呼び出し | 追加条件 |
|---------|--------------|---------|
| `battle_start` | ✅ あり | `_battle_active = True` にセット後 |
| `move_used` | ✅ あり | `_battle_active = True` の間のみ |
| `switch` | ✅ あり | `_battle_active = True` の間のみ |
| `faint` | ✅ あり | `_battle_active = True` の間のみ |
| `battle_end` | ✅ あり | 呼び出し後 `_battle_active = False` にリセット |

> **重要**: `_battle_active = False`（選出画面・試合外）の間は、イベントが検知されても Bedrock を呼び出さない。選出画面での誤実況を防ぐ。

## 戦況蓄積クラス（BattleStateTracker）

試合全体の戦況を蓄積し、Bedrock へのプロンプトに渡す。

```
BattleStateTracker
├─ _player: list[PokemonSlot]    # 自分の最大4匹
├─ _opponent: list[PokemonSlot]  # 相手の最大4匹
└─ _event_log: list[str]         # 直近8件のイベント履歴

PokemonSlot（dataclass）
├─ name: str
├─ confidence: int      # OCR で検出された回数（高いほど確実）
├─ hp_history: list     # 直近3件のHP値
├─ status: str | None   # 状態異常
└─ fainted: bool        # 気絶済みフラグ
```

- `battle_start` でリセット
- `name_candidates_player/opponent` の OCR 候補を信頼度付きで蓄積
- `faint` イベント時に HP 履歴末尾 `0/XXX` のスロットを気絶マーク
- `to_context()` で Bedrock へ渡す辞書を生成

## ポーリング間隔

- 画面キャプチャ間隔: **1秒ごと**（OBS仮想カメラ・cv2.VideoCapture）
- OpenCV差分検出: キャプチャごとに実行（CPU処理で軽量）
- OCR処理: 差分検出でイベントありと判定した場合のみ実行

## 処理の優先度と遅延目標

| ステップ | 目標遅延 | 実測（参考） |
|---------|---------|-----------|
| イベント検知〜Bedrock呼び出し | < 0.5秒 | — |
| Bedrock Vision分析 | < 5秒 | 2.5〜3.5秒 |
| VOICEVOX音声生成 | < 7秒 | 5〜7秒 |
| **合計（イベント〜音声再生開始）** | **< 15秒** | 8〜13秒 |

## 却下した選択肢

| 選択肢 | 却下理由 |
|--------|---------|
| 毎フレーム全処理 | CPU/GPU負荷過大・APIコスト爆発 |
| 固定インターバル呼び出し（例：毎5秒） | イベントのタイミングと合わないため実況が不自然になる |
| キューイング（Celery等） | Sprint初期段階では過剰。Sprint 4以降で検討 |
| HP 差分ベースのイベント検知 | 同種ポケモンが両サイドにいる場合に混在する問題 → フェーズ分類に切り替え |
| BoooleanフラグベースのBattleEventDetector | `_prev_command_visible` が脆弱で誤検知多発 → BattlePhaseClassifier に置換 |

## 今後の拡張

- `selection_screen` フェーズ中の OCR 結果を利用した事前パーティ情報収集
- 状態異常 YOLO 検出結果を BattleStateTracker に統合（現在は game_state 経由でのみ利用）
- S3 への実況ログ保存（`/api/log` エンドポイント）のパイプライン組み込み

## 結果

フェーズ遷移ベースの検知により、選出画面での誤実況・同一イベントの多重トリガーを防止。`BattleStateTracker` により複数ターンにわたる戦況（ポケモン名・HP・状態異常・気絶情報）を蓄積して Bedrock に渡せるようになり、実況品質が向上した。
