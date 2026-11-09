# RVC推論ぶんを入れたVRAM予算の試算（2026-11-10）

small-tasks「VRAM予算表にRVC推論ぶんを入れた試算（ADRとの整合も確認）」の成果物。
前提は `docs/design/rvc-integration-proposal.md`（推奨=案A パス1.8新設）。
**数値はファイルサイズとRVC本体の設定からの机上見積もり。実測は未実施（要実機確認・手順は5章）**。

---

## 1. RVC推論のVRAM内訳（見積もり）

RVC WebUI（`C:\Users\rotat\Retrieval-based-Voice-Conversion-WebUI-main\...`）の
`configs/config.py` を確認した結果:

- RTX 3080 は fp32 強制リスト（16xx/P40/P10/1060/1070/1080）に入らない → **`is_half=True`（fp16）で動く**
- 10GB GPUなので `gpu_mem <= 4` の省メモリ設定は効かない → 分割設定は `x_pad=3, x_query=10, x_center=60, x_max=65`
  （最大65秒ぶんを1チャンクで処理）。実況クリップは数秒なので、ほぼ常に1チャンクで終わる
- 検索インデックスは `faiss-cpu`（`requirements.txt`）→ **index（`added_IVF1717_..._kurepi_v2.index`・約211MB）はRAMに載り、VRAMは使わない**
- RMVPE は `infer/modules/vc/pipeline.py` で初回ロード後、同じプロセス内で使い回される

| 要素 | ファイルサイズ | VRAM見積もり |
|---|---|---|
| CUDAコンテキスト＋PyTorch本体 | — | 0.3〜0.5 GB |
| HuBERT（`assets/hubert/hubert_base.pt`） | 約190MB | 0.1〜0.2 GB（fp16） |
| RMVPE（`assets/rmvpe/rmvpe.pt`・f0method=rmvpe時） | 約181MB | 0.1〜0.2 GB（fp16） |
| 生成器（`assets/weights/kurepi.pth`・v2/40k） | 約55MB | 約0.06 GB |
| 推論中の中間テンソル＋PyTorchキャッシュアロケータの確保ぶん | — | 0.3〜1.0 GB（クリップ長しだい） |
| faiss index | 約211MB | **0（CPU/RAM）** |
| **合計** | | **約1〜2 GB（安全側の上限 2.5 GB で計算する）** |

補足:
- PyTorchは解放したメモリをすぐ返さず確保したまま持つので、nvidia-smiの表示は実使用より大きめに出る。
  予算は nvidia-smi 基準（=他プロセスから見て空いていない量）で考える
- `f0method` を `harvest`/`pm`（CPU）にすれば RMVPE ぶんは消えるが、品質が落ちるので試算は rmvpe 前提
- 最終手段として RVC は CPU でも動く（VRAM 0・ただし大幅に遅い）

## 2. フェーズ別のVRAM予算（RVC追加後）

既存の値は ADR-003 追記（2026-08-04）・`docs/design/system-architecture.md` 3章・ADR-006 から引用。

| フェーズ | 稼働コンポーネント | VRAM合計（上限側） | 判定 |
|---|---|---|---|
| パス1（現行） | EasyOCR(0〜0.5) + YOLO(0.5〜1) | 1〜1.5 GB | ✅ |
| パス1＋gemma2:9bフォールバック発火時（現行） | 上＋gemma2:9b(5.4) | 6.4〜6.9 GB | ✅ 余裕3GB超 |
| **パス1.8（案A・RVC単独）** | RVC(1〜2.5) | **1〜2.5 GB** | ✅ 余裕7.5GB超 |
| パス2（render_commentary_video） | ffmpeg（CPU想定） | ほぼ0 | ✅ |
| v2c（アバター録画・現行） | VMC(4〜5) | 4〜5 GB | ✅ |
| 案B: パス1の中でRVCを呼ぶ | EasyOCR/YOLO(1.5) + RVC(2.5) | 2.5〜4 GB | ✅ ただし↓ |
| 案B＋gemma2:9bフォールバック発火時 | 1.5 + 2.5 + 5.4 | **7.9〜9.4 GB** | ⚠️ 余裕0.6GB・OOM圏内 |
| 将来: v2c中にリアルタイムRVC | VMC(5) + RVC(2.5) | 5〜7.5 GB | ✅（ライブ凍結中・当面不要） |
| 将来: ライブ全部入り | OCR/YOLO(1.5) + VMC(5) + RVC(2.5) + gemma2:9b(5.4) | **約14.4 GB** | ❌ 10GB超過 |

### 読み取れること

1. **案A（パス1.8を独立させる）なら、どの組み合わせでも10GBに対して7GB以上余る**。
   RVC推論のためにVRAM配分を見直す必要はない
2. **案B（VoicevoxClientにフック）は、Bedrock失敗→gemma2:9bフォールバックが重なった瞬間に9.4GBまで行く**。
   gemma2:9bは普段アイドル自動アンロードで載っていないので「たまにだけOOMする」タイプの
   再現しにくい障害になりやすい。rvc-integration-proposal.md で案Aを推奨した理由（制約1）が数字でも裏付けられた
3. ライブ配信を再開して全部を同時に動かす場合は、RVCを入れなくても gemma2:9b＋VMC で
   既に10.4〜11.9GBと超過している（既存の問題）。ライブ再開時はフォールバックLLMの見直しが先

## 3. 運用上の注意（実装時にSKILL.mdへ入れる候補）

- **パス1.8は、パス1（`pipeline.py --input`）とOllamaが終わってから流す**。
  同時に別ターミナルで走らせると案Bと同じ状況（最大9.4GB）になる
- パス1.8中にVMC/OBSを起動しない（v2cはパス2のあと。現行手順どおりなら自然に守られる）
- `convert_voice_rvc.py` は RVC を1プロセスで全WAVまとめて処理する（`infer_batch_rvc.py` 方式）。
  1ファイルごとにサブプロセスを立てるとCUDAコンテキスト確保が毎回走って遅いだけでなく、
  前プロセスの解放待ちが重なると一時的に二重確保になる

## 4. ADRとの整合チェック

| ADR / 文書 | 現在の記述 | RVC導入との関係 | 対応 |
|---|---|---|---|
| ADR-004（VOICEVOX） | 「CPU動作のためGPUリソースを消費しない（VRAM節約）」 | RVCを後段に足すと**音声経路がGPUを使うようになる**。VOICEVOX自体はCPUのままなので記述は誤りではないが、「音声はVRAMゼロ」という前提は崩れる | **実装前にADR-004へ「VOICEVOX後段のRVC（パス1.8・VRAM 1〜2.5GB・他フェーズと時間分離）」を追記**（rvc-integration-proposal.md のTODOに既出） |
| ADR-003（ローカルLLM）追記 | パス1は約1〜1.5GB・gemma2:9b込みで6.4〜6.9GB | 案Aなら影響なし。案Bを選ぶ場合は「フォールバック発火時9.4GB」の注記が必要 | 案A採用なら変更不要 |
| ADR-006（3Dモデル） | VMC 4〜5GB | v2cとRVCは時間分離するので影響なし。外注VRM（2027年1〜2月納品）でVMCの消費が変わる可能性はRVCと無関係に要確認 | 変更不要（VRM納品時に `vrm-model-swap-checklist.md` 側で実測） |
| ADR-009（動画ファースト） | VRAM 10GB制約は据え置き・パス分離 | 案Aはパス分離の考え方そのまま | 変更不要 |
| `docs/design/system-architecture.md` 3章 | フェーズ別表にRVCが無い | 実装したら「パス1.8」行を追加する | 実装時に更新 |
| `docs/manual/ollama-setup.md` VRAM表 | Phi-3 mini/VTube Studio前提の古い表（合計9.5GB） | RVCとは無関係だが現行（gemma2:9b・VMC）と食い違っている | 別件として記録（下記） |

**結論: 案Aで実装するかぎり、ADR違反になる設計変更はない。ADR-004への追記だけ実装前に必要。**

### ついでに見つけた古い記述（今回は直していない）

- `docs/manual/ollama-setup.md` の「VRAM配分」表が Phi-3 mini 2〜3GB・VTube Studio 4〜5GB のまま。
  現行は gemma2:9b（5.4GB）・VMC。small-tasks のドキュメント棚卸しで拾う候補
- `docs/design/system-architecture.md` 3章の「フォールバック時のみ」行が「Phi-3 mini 4bit +2〜3GB」のまま。
  ADR-003追記（2026-08-04）で gemma2:9b（5.4GB）に変わっている

## 5. 実測手順（要実機確認・ユーザー復帰後）

RVC聴き比べのついでに、Windows側で1回測れば見積もりを実測値に置き換えられる。

```powershell
# ターミナル1: 500msごとにVRAM使用量を記録
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -lms 500 > rvc_vram.csv

# ターミナル2: RVCのvenvで1ファイル変換（パラメータは聴き比べで使う値）
cd C:\Users\rotat\Retrieval-based-Voice-Conversion-WebUI-main\Retrieval-based-Voice-Conversion-WebUI-main
venv\Scripts\python.exe tools\infer_cli.py --model_name kurepi.pth --input_path <VOICEVOXのWAV> --opt_path out.wav --f0method rmvpe --index_path logs\kurepi\added_IVF1717_Flat_nprobe_1_kurepi_v2.index --index_rate 0.75 --resample_sr 24000
```

- 変換前のベースライン（Windowsデスクトップ自体が使っているぶん）との差がRVCの消費量
- 長めのWAV（パス1の `commentary_track.wav` を10秒程度に切ったもの）でも1回測ると、中間テンソルの上限が分かる
- 測ったら本メモ1章の表と、system-architecture.md 3章に実測値を書き込む
- 引数名は `tools/infer_cli.py` で確認済み（`--f0method` の既定は `harvest` なので rmvpe は明示する）。
  ⚠️ `--is_half` は `type=bool` なので `--is_half False` と書いても文字列が真になり fp16 のまま。
  fp32 で比べたい時は CLI ではなく `configs/config.py` 側で切り替える必要がある
