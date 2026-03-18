# システム設計書 - ポケモン対戦実況AI

**バージョン**: 1.0
**作成日**: 2026-02-25
**対象GPU**: NVIDIA GeForce RTX 3080 / VRAM 10GB

---

## 目次

1. [システム概要](#1-システム概要)
2. [コンポーネント図](#2-コンポーネント図)
3. [VRAM配分の詳細](#3-vram配分の詳細)
4. [データフロー図](#4-データフロー図)
5. [EC2 Flask API仕様](#5-ec2-flask-api仕様)
6. [コンポーネント詳細](#6-コンポーネント詳細)
7. [エラーハンドリング方針](#7-エラーハンドリング方針)

---

## 1. システム概要

ポケモン対戦画面をリアルタイムで認識し、3Dモデル（VTuberアバター）が音声で実況するAIシステム。

### 設計原則

- **VRAM 10GB制約**: ローカルGPUリソースをVRAM 10GB以内に収める
- **イベント駆動**: 画面変化を検知したときのみ処理を起動しAPIコストを抑制
- **ローカル優先**: 定型処理はローカルで完結させ、クラウドAPIは最小限に絞る
- **3秒以内**: イベント検知から音声再生まで3秒以内を目標とする

---

## 2. コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│  ローカルPC（Windows / RTX 3080 VRAM 10GB）                          │
│                                                                      │
│  ┌─────────────────┐                                                │
│  │   ゲーム画面      │                                                │
│  └────────┬────────┘                                                │
│           │ 1秒間隔でキャプチャ                                        │
│  ┌────────▼────────┐                                                │
│  │  mss            │  画面キャプチャ（CPU）                           │
│  │  スクリーンショット │                                                │
│  └────────┬────────┘                                                │
│           │                                                          │
│  ┌────────▼────────────────────────────┐                           │
│  │  イベント検知レイヤー（CPU / GPU軽量）  │                           │
│  │                                      │                           │
│  │  ┌──────────┐  ┌──────────┐        │                           │
│  │  │ OpenCV   │  │ EasyOCR  │        │ VRAM: 0〜0.5GB            │
│  │  │ 差分検出  │  │ テキスト  │        │                           │
│  │  │ ターン検知 │  │ HP/名前  │        │                           │
│  │  └──────────┘  └──────────┘        │                           │
│  │  ┌──────────┐                       │                           │
│  │  │ YOLOv8   │ 状態異常・ボール数      │ VRAM: 0.5〜1GB           │
│  │  └──────────┘                       │                           │
│  └────────┬────────────────────────────┘                           │
│           │ イベント発生時のみ                                          │
│  ┌────────▼────────────────────────────┐                           │
│  │  実況文生成レイヤー                    │                           │
│  │                                      │                           │
│  │  ┌─────────────────────────────┐   │                           │
│  │  │ Ollama + Phi-3 mini 4bit    │   │ VRAM: 2〜3GB              │
│  │  │ 毎ターン実況文を生成          │   │                           │
│  │  └─────────────────────────────┘   │                           │
│  └────────┬────────────────────────────┘                           │
│           │                                                          │
│  ┌────────▼────────────────────────────┐                           │
│  │  音声・映像出力レイヤー                │                           │
│  │                                      │                           │
│  │  ┌──────────┐  ┌──────────────────────┐  │                      │
│  │  │ VOICEVOX │  │ バーチャルモーション    │  │ VRAM: 4〜5GB         │
│  │  │ 音声生成  │  │ キャプチャー          │  │（3Dモデル描画）       │
│  │  │ (CPU)    │  │ VB-Audio連携         │  │                      │
│  │  └──────────┘  └──────────────────────┘  │                      │
│  └────────┬────────────────────────────┘                           │
│           │                                                          │
│  ┌────────▼────────┐                                                │
│  │  配信ソフト       │ OBS等（画面キャプチャで3Dモデルを合成）          │
│  └─────────────────┘                                                │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ 大きな状況変化時のみ（HTTPS）
┌──────────────────▼──────────────────────────────────────────────────┐
│  AWS クラウド                                                         │
│                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐                       │
│  │  EC2            │────▶│  Bedrock        │                       │
│  │  Flask API      │     │  Claude Haiku   │                       │
│  │  APIキー管理     │◀────│  Vision分析     │                       │
│  └────────┬────────┘     └─────────────────┘                       │
│           │                                                          │
│  ┌────────▼────────┐                                                │
│  │  S3             │                                                │
│  │  実況ログ保存    │                                                │
│  │  スクショ保存    │                                                │
│  └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. VRAM配分の詳細

### 常駐コンポーネント（常時ロード）

| コンポーネント | ツール | VRAM消費 | 備考 |
|-------------|-------|---------|------|
| 3Dモデル描画 | バーチャルモーションキャプチャー | 4〜5 GB | 最大消費コンポーネント |
| 実況文生成LLM | Phi-3 mini 4bit / Ollama | 2〜3 GB | 量子化でVRAM削減 |
| **常駐合計** | | **6〜8 GB** | |

### 処理時のみ使用（一時的）

| コンポーネント | ツール | VRAM消費 | 発生タイミング |
|-------------|-------|---------|------------|
| 物体検出 | YOLOv8 | 0.5〜1 GB | イベント検知時 |
| OCRエンジン | EasyOCR（PyTorch） | 0〜0.5 GB | イベント検知時 |
| **処理時追加** | | **0.5〜1.5 GB** | |

### VRAM配分サマリー

| 状態 | VRAM使用量 | 余裕 |
|-----|----------|-----|
| アイドル時（常駐のみ） | 6〜8 GB | 2〜4 GB |
| 処理時（最大） | 7.5〜9.5 GB | 0.5〜2.5 GB |
| **上限（RTX 3080）** | **10 GB** | |

### VRAM逼迫時の対応策

1. Phi-3 miniのコンテキスト長を短縮（VRAM削減）
2. バーチャルモーションキャプチャーの描画解像度・品質設定を下げる
3. EasyOCRのバッチサイズを1に固定する
4. それでも足りない場合は実況文生成をBedrock Claude Haikuに移行（ADR-003再検討条件）

---

## 4. データフロー図

### 4-1. メインフロー（イベント発生時）

```
[1] 画面キャプチャ
    cv2.VideoCapture（OBS仮想カメラ・カメラ番号3）
    間隔: 1秒

[2] 差分検出
    cv2.absdiff(前フレーム, 現フレーム)
    → 差分スコアが閾値を超えた場合のみ [3] へ進む
    → 超えなければ [1] に戻る

[3] OCR + YOLO（並列）
    ├── EasyOCR: OCR テキスト + bbox 取得
    └── YOLOv8: 状態異常アイコン・ボール数を検出（毎フレーム）

[4] BattlePhaseClassifier によるフェーズ分類・イベント検知
    OCR テキストからフェーズを判定:
      command_select / switch_select / animation / faint / battle_end / selection_screen / unknown
    フェーズ遷移でイベントを発火:
      battle_start / move_used / switch / faint / battle_end
    デバウンス 10秒で多重発火を防止

[5] バトル外判定（イベント発生時）
    _NON_BATTLE_KEYWORDS に該当 → スキップ
    OCR 件数 < 2（battle_end 除く）→ スキップ

[6] game_state 構築
    OCR bbox の y 座標でポケモン名を自分/相手に分類:
      y < 500px  → 相手エリア（name_candidates_opponent）
      500〜700px → 自分エリア（name_candidates_player）
      y > 700px  → コマンドメニュー（除外）
    {
      "hp_values": ["121/171", "175/175"],
      "name_candidates_player": ["ウーラオス", ...],
      "name_candidates_opponent": ["テラパゴス", ...],
      "ocr_text": "...",
      "status": "..."
    }

[7] BattleStateTracker 更新（_battle_active = True の間のみ）
    battle_start → トラッカーリセット → _battle_active = True
    各イベントで name/HP/状態異常/気絶フラグを蓄積
    to_context() で Bedrock へ渡す戦況サマリーを生成:
      {turn, player_pokemon, opponent_pokemon, event_log}

[8] Bedrock Vision 呼び出し判定
    _battle_active = False の場合（選出画面等）→ スキップ
    event_type が [battle_start / move_used / switch / faint / battle_end] の場合:
      POST {ec2_url}/api/vision
      画像 + game_state + BattleStateTracker コンテキスト
      → 実況文（commentary）と状況分析（analysis）を取得

    battle_end 処理後 → _battle_active = False にリセット

[9] 実況文決定
    Bedrock 成功 → commentary を使用
    Bedrock 失敗 → Phi-3 mini（フォールバック）で生成

[10] VOICEVOX で音声生成
     POST http://localhost:50021/audio_query + /synthesis
     → WAVファイル生成

[11] 音声再生
     CABLE Input（VB-Audio Virtual Cable）に出力
     → バーチャルモーションキャプチャーが自動で口パク

[12] S3に保存（/api/log エンドポイント経由）
     ├── 実況テキストログ（JSON）
     └── イベント時スクリーンショット（PNG）
```

### 4-2. 遅延タイムライン

```
t=0.0s  イベント検知（差分検出）
t=0.1s  OCR / YOLO処理完了
t=0.2s  game_state 構築・BattleStateTracker 更新
t=0.2s  Bedrock 呼び出し開始（battle_active かつ対象イベントの場合）
t=2.7〜3.5s  Bedrock Vision分析完了（実測 2.5〜3.5秒）
t=3.5〜9.5s  VOICEVOX音声生成完了（実測 5〜7秒）
t=9.5〜28s   音声再生完了（実測 10〜20秒）
```

> 注: 音声生成・再生が支配的。Bedrock のレイテンシは許容範囲内。

---

## 5. EC2 Flask API仕様

### 概要

EC2上で動作するFlaskサーバー。AWSのAPIキーをEC2側で管理し、ローカルPCにキーを持たせない構成。

### ベースURL

```
http://{EC2_HOST}:{PORT}
```

- ポート: `5000`（デフォルト、変更可）
- 認証: EC2のセキュリティグループでローカルPCのIPのみ許可（APIキーは不要）

---

### エンドポイント一覧

| メソッド | パス | 説明 |
|--------|------|------|
| GET | `/health` | ヘルスチェック |
| POST | `/api/vision` | 画面Vision分析 |
| POST | `/api/log` | S3へのログ保存 |

---

### GET `/health`

サーバーの死活確認。

**レスポンス**

```json
{
  "status": "ok",
  "timestamp": "2026-02-25T12:00:00Z"
}
```

---

### POST `/api/vision`

ゲーム画面のスクリーンショットをBedrock Claude Haiku Visionで分析し、状況テキストを返す。

**リクエスト**

```http
POST /api/vision
Content-Type: application/json
```

```json
{
  "image_base64": "<Base64エンコードされたPNG画像（800x450にリサイズ済み）>",
  "context": {
    "event_type": "move_used",
    "ocr_text": "ウーラオス / 123/175 / ...",
    "hp_values": "123/175 / 176/176",
    "name_candidates_player": "ウーラオス / テラパゴス",
    "name_candidates_opponent": "ガオガエン / ゴリランダ",
    "status_player": "なし",
    "status_opponent": "なし",
    "balls_remaining_player": "?",
    "balls_remaining_opponent": "?"
  },
  "history": [
    "ウーラオスが猛攻を仕掛けます！"
  ],
  "battle_state": {
    "turn": 3,
    "player_pokemon": "ウーラオス HP:123/175 / テラパゴス HP:219/219",
    "opponent_pokemon": "ガオガエン(気絶・場に出ていない) / ゴリランダ HP:176/176",
    "event_log": "T1:battle_start[...] | T2:move_used[...]"
  }
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|---|------|------|
| `image_base64` | string | ○ | PNG画像のBase64文字列（上限5MB） |
| `context` | object | ○ | OCR/YOLOで取得した状況情報 |
| `context.event_type` | string | ○ | `battle_start` / `move_used` / `switch` / `faint` / `battle_end` のいずれか |
| `history` | array[string] | - | 直前の実況テキスト（最大3件） |
| `battle_state` | object | - | BattleStateTracker が生成した複数ターン分の戦況サマリー |

**レスポンス（成功）**

```json
{
  "success": true,
  "analysis": "【状況】ウーラオスがゴリランダに攻撃中。",
  "commentary": "ウーラオスの猛攻がゴリランダを追い詰めます！",
  "usage": {
    "input_tokens": 1240,
    "output_tokens": 85
  },
  "latency_ms": 2710
}
```

**レスポンス（エラー）**

```json
{
  "success": false,
  "error": "bedrock_timeout",
  "message": "Bedrock APIのタイムアウト（5秒）"
}
```

| エラーコード | 説明 |
|-----------|------|
| `bedrock_timeout` | Bedrock APIが5秒以内に応答しなかった |
| `image_too_large` | 画像サイズが上限（5MB）を超えた |
| `invalid_event_type` | event_typeが不正な値 |

---

### POST `/api/log`

実況ログとスクリーンショットをS3に保存する。

**リクエスト**

```http
POST /api/log
Content-Type: application/json
```

```json
{
  "session_id": "20260225_120000",
  "turn": 5,
  "timestamp": "2026-02-25T12:01:23Z",
  "event_type": "turn_end",
  "context": { ... },
  "analysis": "Bedrockの分析テキスト（あれば）",
  "commentary": "ガブリアスのじしんが炸裂！サーフゴーはピンチです！",
  "image_base64": "<スクリーンショット（任意）>"
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|---|------|------|
| `session_id` | string | ○ | 試合セッションID（開始日時） |
| `turn` | int | ○ | ターン数 |
| `commentary` | string | ○ | Phi-3が生成した実況テキスト |
| `image_base64` | string | - | 省略時は画像保存しない |

**S3保存パス**

```
s3://{BUCKET}/logs/{session_id}/turn_{turn:03d}.json
s3://{BUCKET}/screenshots/{session_id}/turn_{turn:03d}.png
```

**レスポンス（成功）**

```json
{
  "success": true,
  "s3_log_path": "s3://bucket/logs/20260225_120000/turn_005.json",
  "s3_image_path": "s3://bucket/screenshots/20260225_120000/turn_005.png"
}
```

---

## 6. コンポーネント詳細

### 6-1. イベント検知モジュール

```
src/capture/
├── screen_capture.py    # OBS仮想カメラキャプチャ（cv2.VideoCapture）・EasyOCR
└── yolo_detector.py     # YOLOv8による状態異常・ボール検出

src/pipeline.py          # 以下のクラスを含む（Sprint 5 統合）
├── BattlePhaseClassifier    # OCR テキストからフェーズ分類・イベント検知
├── BattleStateTracker       # 複数ターンにわたる戦況蓄積
└── PokemonSlot（dataclass） # 1匹分の状態（名前・HP・状態異常・気絶）
```

**BattlePhaseClassifier の主要定数**

```python
_PLAYER_Y_THRESHOLD = 500   # y < 500px = 相手エリア、以上 = 自分エリア
_COMMAND_Y_MIN      = 700   # y > 700px = コマンドメニュー（名前候補から除外）
_HP_ZERO_RE = re.compile(r'\b0/([5-9]\d|\d{3})\b')  # 分母 50 以上のみ faint 判定
```

### 6-2. 実況生成モジュール

```
src/commentary/
└── phi3_client.py        # Ollama Phi-3 mini呼び出し（フォールバック用）

src/pipeline.py
└── _call_bedrock_vision()  # EC2 /api/vision 呼び出し（メイン実況生成）
```

**Bedrock へのリクエスト構造（`server.py` の `_build_vision_prompt`）**

```
[ロール設定] ポケモンSVダブルバトル実況者

[ダブルバトル基本知識] 技名・ダメージ表記・状態異常色の説明

[出力ルール]
  - 今回のイベント種別 + 実況指示（例: battle_start → 両者のポケモン紹介）
  - 【状況】1文 + 【実況】1〜2文で出力
  - 画面左上HPバー＝相手、右下HPバー＝自分

[蓄積された戦況] BattleStateTracker.to_context()
  - ターン数・自分/相手ポケモン名・HP・状態異常・気絶情報・イベント履歴

[現在フレームのOCR情報]
  - 画面テキスト・HP値・自分/相手ポケモン名候補・状態異常
```

### 6-3. 音声・映像同期モジュール

```
src/output/
├── voicevox_client.py   # VOICEVOX APIクライアント
└── audio_player.py      # WAV再生（出力先: CABLE Input）
```

**音声デバイス設定**

```python
# audio_player.py で CABLE Input デバイスに出力することで
# バーチャルモーションキャプチャーが自動でリップシンクを行う
AUDIO_OUTPUT_DEVICE = "CABLE Input (VB-Audio Virtual Cable)"
```

---

## 7. エラーハンドリング方針

| 障害 | 対応 |
|-----|------|
| Bedrock APIタイムアウト（>5秒） | スキップしてPhi-3のみで実況生成 |
| Phi-3生成エラー | 「{ポケモン名}の攻撃！」などテンプレート文を使用 |
| VOICEVOX接続失敗 | テキストのみコンソール出力・音声なしで継続 |
| VRAM不足（OOM） | YOLOをCPUモードに切り替えてリトライ |
| OCR誤認識 | ポケモン名辞書で後補正（完全一致しない場合は近似補正） |
| S3保存失敗 | ローカルにフォールバック保存・次回バッチで再送 |

---

## 関連ドキュメント

- [ADR-001](../adr/ADR-001-bedrock-claude-haiku.md) - Vision分析にBedrock Claude Haikuを使用
- [ADR-002](../adr/ADR-002-ocr-yolo-hybrid.md) - OCR+YOLOハイブリッド構成
- [ADR-003](../adr/ADR-003-local-llm-phi3.md) - 実況文生成にPhi-3 mini
- [ADR-004](../adr/ADR-004-voicevox.md) - 音声合成にVOICEVOX
- [ADR-005](../adr/ADR-005-aws-structure.md) - AWSのBedrock・EC2・S3構成
- [ADR-006](../adr/ADR-006-3d-model-rendering.md) - 3DモデルにVRoid+バーチャルモーションキャプチャー
- [ADR-007](../adr/ADR-007-event-driven-architecture.md) - イベント駆動アーキテクチャ
