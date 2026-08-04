# ADR-001: Vision分析にAWS Bedrock（Claude Haiku）を使用する

## ステータス
改訂済み（2026-02-25）

## 日付
2026-02-24（初版） / 2026-02-25（改訂）

## 改訂内容

「実況文生成」の責務を本ADRのスコープから削除。実況文生成はADR-003（ローカルLLM Phi-3 mini）が担当する。本ADRはVision分析（画面全体の状況理解）のみを対象とする。

## 文脈

ゲーム画面全体の状況理解にVision AIが必要。ローカルで動作するVision LLM（LLaVAなど）はVRAM消費が6〜7GBと大きく、3Dモデルの描画と共存することが困難。またリアルタイム実況のためレスポンス速度も重要な要件となる。

## 決定

AWS BedrockのClaude Haiku Visionを**画面全体の状況理解（Vision分析）専用**として使用する。画面の変化があるイベント時（ターン切替・交代・技使用・気絶）のみ呼び出す。

取得した状況テキストはPhi-3 mini（ADR-003）に渡し、実況文の生成はローカルLLMが行う。

## 責務の明確化

| 処理 | 担当 |
|------|------|
| 画面全体の状況理解（Vision分析） | **本ADR: Bedrock Claude Haiku** |
| 実況文テキストの生成 | ADR-003: Phi-3 mini（ローカル） |

## 理由

- Claude Haikuは上位モデルと比べ安価かつ高速
- AWS Bedrockを使うことでAWS請求に一本化でき管理が容易
- イベント駆動で呼び出し頻度を制御することでコストを抑えられる
- 既存のAWS環境（EC2・S3）と統合しやすい

## 却下した選択肢

| 選択肢 | 却下理由 |
|--------|---------|
| ローカル LLaVA 7B | VRAM 6〜7GB消費で3Dモデルと共存不可 |
| GPT-4o（OpenAI API） | コストが高い・AWS請求と分離される |
| Gemini Flash | 検討余地はあるがClaude統一を優先 |

## コスト見込み

- Vision分析のみ：1試合あたり約30ターン × 0.3〜0.5円 = **10〜15円**
- 1日10試合で100〜150円程度

## 結果

VRAMを消費せずVision処理をクラウドに逃がすことができ、ローカルリソースを3DモデルとローカルLLMに集中させられる。実況文生成はローカルLLM（ADR-003）が担当することでAPIコストをさらに抑制できる。

## 追記（2026-07-12）: Claude Haiku 4.5への移行

AWSより`anthropic.claude-3-haiku-20240307-v1:0`のBedrockモデル廃止通知を受領。Legacy状態は2026-03-10開始済み、EOLは2026-09-10。

後継の`anthropic.claude-haiku-4-5-20251001-v1:0`（Claude Haiku 4.5）へ移行。「安価・高速なHaiku系列をBedrockで使う」という本ADRの決定自体に変更はないため、ADR自体の再改訂ではなく追記とする。

ただしap-southeast-2リージョンはClaude Haiku 4.5のIn-Region直接呼び出しに非対応のため、AU向けクロスリージョン推論プロファイルID `au.anthropic.claude-haiku-4-5-20251001-v1:0` を使用する（`src/api/server.py`の`BEDROCK_MODEL_ID`）。実体としてはシドニー/メルボルンにルーティングされるのみでAU圏内に閉じる。

**移行に必要な作業（ユーザー側）**:
- Bedrockコンソールでモデルアクセス有効化
- EC2 IAMロールが推論プロファイルARNへの`bedrock:InvokeModel`を許可しているか確認
- `server.py`更新版をEC2へWinSCPデプロイ

詳細手順: [Bedrock Claude Haiku 4.5 移行マニュアル](../manual/bedrock-haiku-4-5-migration.md)

## 追記（2026-08-04）: 実況文生成の責務も本ADR側に実質移動している

本ADRの「改訂内容」では「実況文生成はADR-003（Phi-3 mini）が担当する」としていたが、
現在の実装（`src/pipeline.py`の`_dispatch_commentary`）はBedrock Vision呼び出しが
返した`bedrock_commentary`を実況文としてそのまま優先採用し、**Phi-3 miniはBedrock呼び出しが
失敗した場合のみのフォールバック**という扱いになっている（ADR-003側にも追記済み）。
「Vision分析専用」としていた本ADRのスコープは実態としては実況文生成も含む形に広がっている。
責務表は将来的に本ADRを再改訂して統合するのが望ましいが、現時点では両ADRへの追記に留める。
