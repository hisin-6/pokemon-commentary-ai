# ポケモン対戦実況AI - Claude Code 引継ぎプロンプト

## プロジェクト概要

ポケモン対戦画面をリアルタイムで認識し、3Dモデルが音声で実況するAIシステム。
- OS: Windows / GPU: RTX 3080 VRAM 10GB / 言語: Python
- Claude Code は WSL2 で動作。実行は `venv\Scripts\python.exe`（Windows Python）

---

## システム構成（Sprint 5 時点）

```
OBS仮想カメラ（カメラ番号3・1920x1080）
　↓ 1秒ごとにキャプチャ
OpenCV 差分検出（静止フレームのOCRスキップ）
EasyOCR（差分あり時のみ）
　↓
BattlePhaseClassifier（フェーズ遷移ベースのイベント検知）
  フェーズ: command_select / switch_select / animation / faint / battle_end / selection_screen / unknown
  ├─ 初回 command_select 出現 → battle_start イベント
  ├─ command_select → それ以外  → move_used イベント
  ├─ faint フェーズ出現         → faint イベント
  ├─ battle_end フェーズ出現    → battle_end イベント
  └─ デバウンス 10秒（多重トリガー防止）
　↓ イベント検知時
YOLOv8（状態異常・ボール検出）
game_state 構築（OCR y 座標で自分/相手 ポケモン名を分離）
BattleStateTracker（戦況を複数ターンにわたって蓄積）
　↓ _battle_active = True の間のみ Bedrock を呼び出す
AWS Bedrock Claude Haiku（EC2 API経由・Vision分析 + 実況文生成）
　↓ Bedrock 失敗時フォールバック
Phi-3 mini 4bit / Ollama（ローカル実況文生成）
　↓
VOICEVOX（localhost:50021）→ 音声合成
AudioPlayer → 音声再生
　↓
AWS S3 → 実況ログ・スクリーンショット保存
```

> **重要変更（Sprint 5）**:
> - 実況文生成は Bedrock Claude Haiku が優先。Phi-3 mini は Bedrock 失敗時のフォールバック。
> - `BattleEventDetector`（Boolean フラグ方式）→ `BattlePhaseClassifier`（フェーズ遷移方式）に変更。
> - `BattleStateTracker` で自分/相手それぞれ最大4匹の戦況を蓄積し Bedrock に渡す。
> - `_battle_active` フラグで選出画面など試合外での Bedrock 呼び出しを抑制。

---

## アーキテクチャ決定記録（ADR）サマリー

| ADR | 決定内容 |
|-----|---------|
| ADR-001 | Vision分析はAWS Bedrock Claude Haiku |
| ADR-002 | 画面認識はOCR＋YOLOのハイブリッド |
| ADR-003 | 実況文生成はPhi-3 mini 4bit（フォールバック用） |
| ADR-004 | 音声合成はVOICEVOX |
| ADR-005 | AWSはBedrock＋EC2＋S3 |
| ADR-006 | 3DモデルはVRoid + バーチャルモーションキャプチャー + VB-Audio |
| ADR-007 | イベント駆動アーキテクチャ（BattlePhaseClassifier + BattleStateTracker） |

詳細: `docs/adr/`

---

## VRAM配分（合計 10GB 以内）

| コンポーネント | VRAM |
|-------------|------|
| 3Dモデル | 4〜5 GB |
| Phi-3 mini 4bit | 2〜3 GB |
| YOLO + EasyOCR | 0.5〜1.5 GB |
| **合計（最大）** | **9.5 GB以内** |

---

## Sprint 進捗

| Sprint | 内容 | 状態 |
|--------|------|------|
| Sprint 1 | 画面キャプチャ + EasyOCR | ✅ 完了 |
| Sprint 2 | Bedrock連携 + EC2 API + S3 | ✅ 完了 |
| Sprint 3 | VOICEVOX + バーチャルモーションキャプチャー | ✅ 完了 |
| Sprint 4 | YOLOv8学習・導入 | ✅ 完了 |
| Sprint 5 | パイプライン統合 | 🔄 動作確認中 |
| Sprint 6 | 実況品質向上（RAG + OCR改善） | 📋 計画中 |
| Sprint 7 | 検出精度向上（ボール検出 + 音声二重化） | 📋 計画中 |
| Sprint 8 | Fine-tuning（Phi-3 LoRA） | 📋 計画中 |
| Sprint 9 | キャラクター強化（音声・3Dモデル改善） | 📋 アイデア段階 |

---

## ロードマップ詳細

### Sprint 5 完了タスク（最優先）
- [ ] `src/api/server.py` を EC2 に WinSCP で転送 → gunicorn 再起動
- [ ] 統合テスト: 選出画面誤実況・battle_start/end 検知・BattleStateTracker 精度を確認

### Sprint 6: 実況品質向上（RAG + OCR改善）
- [ ] PokeAPI から全ポケモンDBを JSON に作成するスクリプト
- [ ] `pipeline.py` に DB 参照を組み込み（OCR検出名 → タイプ・特性・代表技を取得）
- [ ] `server.py` プロンプトに `pokemon_info` セクション追加
- [ ] OCR ノイズフィルター改善（アイテム名・UIテキストの混入対策）

### Sprint 7: 検出精度向上
- [ ] ボール検出改善（YOLO Recall=0 の根本原因調査 → 再学習 or 代替手法検討）
  - 現状: OBS映像からの検出が Recall=0（学習データと映像の差異が原因の可能性）
  - 候補手法: YOLO再学習 / OpenCV テンプレートマッチング / HSV色検出
- [ ] 音声出力二重化（Voicemeeter で CABLE Input + スピーカー同時出力）

### Sprint 8: Fine-tuning 準備・実行
- [ ] S3 ログフォーマット整備（Fine-tuning 用の入出力ペアを蓄積）
- [ ] ラベリングツール作成（Bedrock が生成した実況文を人間が修正・承認）
- [ ] Phi-3 mini LoRA fine-tuning 実行（RTX 3080 でローカル）
- [ ] AWS Bedrock Fine-tuning も候補（コスト要調査）

### Sprint 9: キャラクター強化（将来）
- [ ] **VOICEVOX 話者変更**: 現在ずんだもん（speaker=1）→ 別キャラクターに変更
  - 変更箇所: `--speaker` 引数（pipeline.py）/ VOICEVOX アプリで話者番号確認
- [ ] **3Dモデル口パク連携の動作確認**:
  - バーチャルモーションキャプチャー + VB-Audio Virtual Cable の連携が未確認
  - 音声出力を CABLE Input に向けると自動でリップシンクする設計（ADR-006）
  - 実際に動くか要検証
- [ ] VRM 0.x 変換問題: VRoid Studio 現行版（VRM 1.0）→ バーチャルモーションキャプチャー対応形式への変換ツール調査

---

## 実行コマンド

```powershell
# パイプライン起動（Bedrock あり）
venv\Scripts\python.exe src/pipeline.py --camera 3 --model runs/detect/train4/weights/best.pt --ec2-url http://13.211.11.202:5000

# パイプライン起動（Bedrock なし・Phi-3のみ）
venv\Scripts\python.exe src/pipeline.py --camera 3 --model runs/detect/train4/weights/best.pt
```

事前起動:
- OBS仮想カメラ ON
- Ollama（`ollama serve`）
- VOICEVOX（localhost:50021）

---

## EC2 デプロイ

- **WinSCP で直接ファイル転送**（git pull は使っていない）
- デプロイ先: `/home/app_admin/pokemon-api/server.py`
- gunicorn 再起動:
  ```bash
  kill $(pgrep -f "gunicorn.*server:app" | head -1)
  cd /home/app_admin/pokemon-api
  venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 server:app --daemon
  ```
- ヘルスチェック: `curl -UseBasicParsing http://13.211.11.202:5000/health`

---

## 次回作業（Sprint 5 完了 → Sprint 6 開始）

1. EC2 に `server.py` デプロイ → 統合テスト
2. テスト結果を確認して Sprint 6（RAG実装）へ

---

## 注意事項

- VRAM 合計が 10GB を超える変更は提案しない
- AWS API キーはローカルに書かない（EC2 IAMロール経由）
- EC2 の IP アドレスはコードにハードコードしない（`--ec2-url` で渡す）
- `cv2.CAP_DSHOW` は OBS 仮想カメラで黒フレームになるので使わない
