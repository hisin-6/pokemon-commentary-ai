# ポケモン対戦実況AI

## プロジェクト構成
- 言語: Python
- GPU: RTX 3080 / VRAM 10GB（配分に常に注意）
- クラウド: AWS Bedrock・EC2・S3

## コーディングルール
- VRAM合計が10GBを超える変更を提案しない
- AWS APIキーはEC2経由で管理・ローカルに書かない
- ADRに反する設計変更は必ずADRを先に更新する

## ADRの場所
docs/adr/ を参照すること
```

---

### Skills（自動読み込みの知識ファイル）

Skillsはタスクに関連すると判断したときだけClaudeが自動で読み込む知識ファイルで、常にコンテキストに入れておく必要がない情報を管理するのに適しています。 

このプロジェクトなら以下が有効です。
```
.claude/skills/
├── aws-bedrock/SKILL.md       # Bedrock呼び出しパターン・コスト管理
├── vram-budget/SKILL.md       # VRAM配分ルールと制約
├── pokemon-domain/SKILL.md    # ポケモン対戦の専門用語・ルール
├── ocr-finetune/SKILL.md      # EasyOCRファインチューニングサイクル・コマンド集
└── video-commentary/SKILL.md  # 実況動画合成（パス1→1.5→2 biim）の手順・地雷・検証
```

OCRファインチューニング（collect_ocr_crops / fix_annotations / finetune_ocr / pokemon_g2 等）に関するタスクでは必ず `.claude/skills/ocr-finetune/SKILL.md` を参照すること。

実況動画の制作・合成（--render-out / generate_gap_commentary / render_commentary_video / biim / フィラー / 戦況パネル 等）に関するタスクでは必ず `.claude/skills/video-commentary/SKILL.md` を参照すること。手順と地雷リストが揃っており、実装の内部理解なしでも安全に量産できる。

「VRAM予算を確認して」と言うだけで関連Skillが自動ロードされます。

---

### Subagents（専門エージェント）

Subagentsは独自のコンテキストウィンドウ・ツール制限・モデルを持つ専門エージェントで、複雑なタスクの委譲に使います。 
```
.claude/agents/
├── aws-deployer.md    # EC2・S3・Bedrockの操作専門
├── cost-checker.md    # API料金の試算・監視専門
└── ocr-trainer.md     # EasyOCR・YOLOの学習データ作成専門