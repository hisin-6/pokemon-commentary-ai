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
│  │  ┌──────────┐  ┌────────────────┐  │                           │
│  │  │ VOICEVOX │  │ VTube Studio   │  │ VRAM: 4〜5GB              │
│  │  │ 音声生成  │  │ 3Dモデル描画   │  │（3Dモデル描画）            │
│  │  │ (CPU)    │  │ VMC Protocol   │  │                           │
│  │  └──────────┘  └────────────────┘  │                           │
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
| 3Dモデル描画 | VTube Studio | 4〜5 GB | 最大消費コンポーネント |
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
2. VTube Studioの描画解像度・品質設定を下げる
3. EasyOCRのバッチサイズを1に固定する
4. それでも足りない場合は実況文生成をBedrock Claude Haikuに移行（ADR-003再検討条件）

---

## 4. データフロー図

### 4-1. メインフロー（イベント発生時）

```
[1] 画面キャプチャ
    mss.grab() → numpy配列（BGR）
    間隔: 1秒

[2] 差分検出
    cv2.absdiff(前フレーム, 現フレーム)
    → 差分スコアが閾値を超えた場合のみ [3] へ進む
    → 超えなければ [1] に戻る

[3] イベント種別の判定（並列実行）
    ├── EasyOCR: ポケモン名・HP数値・技名を取得
    └── YOLOv8: 状態異常アイコン・ボール数を検出

[4] 状況テキストの構築
    {
      "pokemon_player": "ガブリアス",
      "hp_player": 85,
      "pokemon_opponent": "サーフゴー",
      "hp_opponent": 42,
      "last_move": "じしん",
      "status": "normal",
      "balls_remaining": [6, 4],
      "event_type": "move_used"
    }

[5] Bedrock呼び出し判定
    event_type が [turn_end / switch / faint] → [6A] へ
    それ以外                                 → [6B] へ

[6A] Bedrock Vision分析（大きな状況変化時のみ）
     POST https://{ec2-host}/api/vision
     スクリーンショット（Base64） + 状況テキスト
     → Bedrockから状況説明テキストを取得
     → [7] へ

[6B] ローカル情報のみで実況
     状況テキスト [4] をそのまま使用
     → [7] へ

[7] Phi-3 mini で実況文生成（Ollama）
    プロンプト構成:
    - システムプロンプト（実況者キャラクター設定）
    - Bedrock分析結果（あれば）
    - OCR/YOLO状況テキスト
    - 直前3件の実況履歴
    → 実況テキスト生成

[8] VOICEVOX で音声生成
    POST http://localhost:50021/audio_query
    POST http://localhost:50021/synthesis
    → WAVファイル生成

[9] 音声再生 + 口パク同期（同時実行）
    ├── WAVファイルを再生
    └── VMC Protocol（UDP）でVTube Studioに音量データ送信
        → 3Dモデルが口パク

[10] S3に保存
     ├── 実況テキストログ（JSON）
     └── イベント時スクリーンショット（PNG）
```

### 4-2. 遅延タイムライン

```
t=0.0s  イベント検知（差分検出）
t=0.1s  OCR / YOLO処理完了
t=0.2s  状況テキスト構築完了
t=0.2s  Bedrock呼び出し開始（該当イベントの場合）
t=1.2s  Bedrock Vision分析完了（最大1秒）
t=1.3s  Phi-3 mini 実況文生成開始
t=2.3s  実況文生成完了（最大1秒）
t=2.4s  VOICEVOX音声生成開始
t=2.9s  音声生成完了・再生開始　← 目標3秒以内 ✓
```

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
  "image_base64": "<Base64エンコードされたPNG画像>",
  "context": {
    "pokemon_player": "ガブリアス",
    "hp_player": 85,
    "pokemon_opponent": "サーフゴー",
    "hp_opponent": 42,
    "last_move": "じしん",
    "status_player": "normal",
    "status_opponent": "normal",
    "balls_remaining_player": 6,
    "balls_remaining_opponent": 4,
    "event_type": "turn_end",
    "turn_count": 5
  },
  "history": [
    "ガブリアスのじしんがサーフゴーに命中！",
    "サーフゴーはHPの半分以上を削られた！"
  ]
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|---|------|------|
| `image_base64` | string | ○ | PNG画像のBase64文字列 |
| `context` | object | ○ | OCR/YOLOで取得した状況情報 |
| `context.event_type` | string | ○ | `turn_end` / `switch` / `faint` のいずれか |
| `history` | array[string] | - | 直前の実況テキスト（最大5件） |

**レスポンス（成功）**

```json
{
  "success": true,
  "analysis": "ガブリアスのじしんでサーフゴーのHPが残り42。サーフゴーは次のターンに交代か反撃か迫られている状況。テラスタルの使用タイミングも重要な局面。",
  "usage": {
    "input_tokens": 1240,
    "output_tokens": 85
  },
  "latency_ms": 820
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
├── screen_capture.py    # mssによる画面キャプチャ
├── diff_detector.py     # OpenCV差分検出・イベント判定
├── ocr_reader.py        # EasyOCRによるテキスト取得
└── yolo_detector.py     # YOLOv8による物体検出
```

**差分検出の閾値（暫定）**

```python
DIFF_THRESHOLD = 30      # ピクセル差分の平均値
DIFF_MIN_AREA = 1000     # 変化領域の最小面積（px²）
```

### 6-2. 実況生成モジュール

```
src/commentary/
├── situation_builder.py  # OCR/YOLO結果→状況テキスト変換
├── bedrock_client.py     # EC2 Flask APIクライアント
├── phi3_client.py        # Ollama Phi-3 mini呼び出し
└── history_manager.py    # 実況履歴管理（直前N件保持）
```

**Phi-3 miniへのプロンプト構造**

```
[システムプロンプト]
あなたはポケモン対戦の実況者です。テンポよく、専門用語を使って
ポケモン対戦を実況してください。1〜2文で簡潔に。

[状況情報]
{situation_text}

[Bedrock分析]
{bedrock_analysis}  ← ある場合のみ

[直前の実況]
{history[-3:]}

[指示]
次の実況文を生成してください。
```

### 6-3. 音声・映像同期モジュール

```
src/output/
├── voicevox_client.py   # VOICEVOX APIクライアント
├── audio_player.py      # WAV再生
└── vmc_sender.py        # VMC Protocol UDP送信（口パク同期）
```

**VMC Protocol設定**

```python
VMC_HOST = "127.0.0.1"
VMC_PORT = 39539         # VTube Studioデフォルトポート
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
- [ADR-006](../adr/ADR-006-3d-model-rendering.md) - 3DモデルにVRoid+VTube Studio
- [ADR-007](../adr/ADR-007-event-driven-architecture.md) - イベント駆動アーキテクチャ
