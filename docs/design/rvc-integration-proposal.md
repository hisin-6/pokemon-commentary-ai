# RVC声質変換をパイプラインに差し込む位置の調査メモ（2026-11-09）

改善ロードマップ①声質変更の続き。RVCモデル（`kurepi.pth`・v2/40k・250epoch）は
2026-08-16に学習済みだが、今は RVC WebUI の推論タブで手作業変換しかできない。
本メモは「VOICEVOX出力→RVC変換」をどこに差し込むかの調査と設計案。**実装はユーザー復帰後**。

前提資料: `docs/manual/rvc-voice-setup.md`（学習手順・推論パラメータ・地雷）、ADR-004（VOICEVOX）、ADR-009（実況動画ファースト）

---

## 1. 現状のWAV生成経路（調査結果）

WAVを作る箇所はすべて `src/output/voicevox_client.py` の `VoicevoxClient.generate_wav(text) -> bytes` を通る。

| 呼び出し元 | 生成物 | 保存先（render_dir相対） | duration算出 |
|---|---|---|---|
| `src/pipeline.py:4945`（動画モード・試合後の後付け生成） | 本編実況 | `wav/NNNN_<event>.wav` ＋ `manifest.jsonl` | `RenderSink.wav_duration()` |
| `src/pipeline.py:5306`（ライブ・`_speak`スレッド） | 本編実況（即再生） | 同上（render_sink有効時） | 同上 |
| `scripts/generate_gap_commentary.py:224` `synthesize_fillers()` | フィラー | `wav/fNNNN_filler.wav` ＋ `fillers.jsonl` | 同上 |
| `scripts/generate_predictions.py:306` `synthesize_predictions()` | 予測・回収 | `wav/pNNNN_<event>.wav` ＋ `predictions.jsonl` | 同上 |
| `scripts/resynthesize_fillers.py:90` | フィラー差し替え | 既存 `wav/fNNNN_filler.wav` を上書き | 同上 |

WAVを使う側:

- `scripts/render_commentary_video.py:207` `build_commentary_track()` が全WAVを1本の
  `commentary_track.wav` に並べる。**全WAVのレート・ch数・ビット幅が同一でないと
  `ValueError("WAVフォーマット不一致")` で落ちる**（VOICEVOX出力は 24kHz/mono/16bit）
- `scripts/play_and_animate_avatar.py` は `commentary_track.wav` を再生するだけ。
  口パクはVMC側の音声入力で自動連動なので、**RVC後の音声を流せばそのまま口が合う**
- 字幕・`check_spoilers.py` はテキストしか見ないので影響なし

RVC側（`C:\Users\rotat\Retrieval-based-Voice-Conversion-WebUI-main\...`・別リポジトリ・別venv）:

- ファイル変換に使えるのは `tools/infer_cli.py`（1ファイル）と `tools/infer_batch_rvc.py`（フォルダ一括）。
  中身は `infer/modules/vc/modules.py` の `VC.vc_single()` / `vc_multi()`
- `api_240604.py` は sounddevice を使うリアルタイム変換用。ファイル変換には使えない
- 出力レートはモデルの `tgt_sr`（kurepiは40kHz）。**`--resample_sr 24000` を指定すれば
  VOICEVOXと同じ24kHzで書き出せる**（`tgt_sr != resample_sr >= 16000` のとき再サンプリング）

## 2. 設計上の制約

1. **VRAM 10GB**: パス1（`pipeline.py --input`）は EasyOCR＋YOLO を GPU で回している。
   同じプロセス・同じ時間帯に RVC（HuBERT＋生成器）を載せるのは避けたい。
   RVC推論ぶんのVRAM試算は別タスク（small-tasks「VRAM予算表にRVC推論ぶん」）で詰める
2. **venv分離**: RVCは独自のtorch/fairseq依存を持つ。AITuberProjectのvenvに入れると
   依存が衝突しやすいので、**RVCのvenvのpythonをサブプロセスで呼ぶ**のが安全
3. **声の混在禁止**: 一部だけ変換されたWAVが混ざると、フォーマットが同じでも
   声色がバラバラな動画になる。`build_commentary_track()` のフォーマット検証では検知できない
4. **再実行性**: フィラーは `resynthesize_fillers.py` で何度も作り直す運用
   （SKILL.md・パス1.7）。作り直したWAVもRVCを通し直せる必要がある
5. **duration**: RVC変換で長さが数ms〜数十msずれる可能性がある。変換後WAVから
   duration を測り直してjsonlを更新しないと、配置・字幕の終端がずれる

## 3. 差し込み位置の案

### 案A（推奨）: 独立した一括変換パス「パス1.8」を新設

```
パス1（manifest）→ 1.5（fillers）→ 1.6（predictions）→ 1.7（重複確認・resynthesize）
  → ★パス1.8: scripts/convert_voice_rvc.py renders/<動画名>
  → check_spoilers → パス2（render_commentary_video）
```

- 新規スクリプト `scripts/convert_voice_rvc.py` が `render_dir/wav/*.wav` を全部RVCに通す
  - 元WAVは `wav_voicevox/` に退避（初回のみ。2回目以降は `wav_voicevox/` を入力にする＝何度でもやり直せる）
  - RVCのvenv pythonで `tools/infer_batch_rvc.py`（またはvc_multiを呼ぶ薄いラッパー）をサブプロセス実行。
    `--resample_sr 24000` でVOICEVOXと同じフォーマットにそろえる
  - 変換後に manifest / fillers / predictions の `duration` を測り直して書き戻す
  - `render_info.json` に `"voice": {"rvc_model": "kurepi.pth", "index_rate": ..., "protect": ..., "f0up_key": ..., "converted_at": ...}` を記録
- `render_commentary_video.py` 側に小さなガードを足す: `render_info.json` に `voice` があるのに
  `wav_voicevox/` より新しいVOICEVOX生WAV（=変換漏れ）が `wav/` にあれば警告して止める
  （制約3・4対策。resynthesize後に1.8を忘れたケースを拾う）

良い点:
- **既存の生成コード（pipeline.py・generate_*）を一切触らない**。挙動変更がパス1.8に閉じる
- パス1が終わってからGPUを使うので、OCR/YOLOとVRAMを取り合わない
- パラメータ（index_rate/protect/ピッチ）を変えて**パス1.8だけ何度でもやり直せる**。
  聴き比べ（`rvc-voice-setup.md`の未決事項）にもそのまま使える
- `--persona neutral` 運用中は1.8をスキップするだけで今まで通り（くれぴ切り替えと連動させやすい）

弱い点:
- 手順が1つ増える（SKILL.mdの手順と地雷リストに追記が必要）
- ライブ配信には効かない（ADR-009でライブは凍結中なので当面問題なし）

### 案B: `VoicevoxClient` に変換フックを入れる

`generate_wav()` の戻り値をRVCに通す `VoiceConverter` を注入できるようにする。

- 良い点: 全呼び出し元に一発で効く。ライブにも効く
- 弱い点: パス1の最中にRVCが走る→**OCR/YOLOとVRAMを同時に使う**（制約1）。
  1クリップごとにサブプロセス起動 or 常駐RVCサーバーが必要で遅い。
  パラメータを変えたら**パス1からやり直し**（Bedrock再生成も走る）になり、聴き比べに向かない

### 案C: `commentary_track.wav` を丸ごと1回変換

パス2の `build_commentary_track()` 後、完成トラック1本だけをRVCに通す。

- 良い点: 変換1回・実装が最小
- 弱い点: 動画尺ぶん（大半が無音）の長尺推論でメモリ・時間が無駄。
  パス2をやり直すたびに変換も走る。無音区間でのノイズ・F0抽出の誤動作リスク。
  `play_and_animate_avatar.py` もこのトラックを使うので、変換忘れの検知が難しい

### 比較

| | 既存コード変更 | VRAM競合 | パラメータ再調整 | ライブ対応 |
|---|---|---|---|---|
| 案A パス1.8 | ほぼ無し（ガードのみ） | 無し | パス1.8だけ再実行 | × |
| 案B Clientフック | voicevox_client＋DI | **あり** | パス1から | ○ |
| 案C トラック丸ごと | render_commentary_video | 無し | パス2ごと | × |

**推奨: 案A**。ライブが必要になったら、その時に案Bを「常駐RVCサーバー＋Clientフック」として追加する。

## 4. 実装時のTODO（ユーザー復帰後）

- [ ] 最適エポック・推論パラメータ（index_rate/protect/f0up_key/f0method）を聴き比べで決める
      （案Aのスクリプトを先に作れば、それ自体を聴き比べツールにできる）
- [ ] `scripts/convert_voice_rvc.py` 作成（RVC venvパス・モデル名・indexパスはCLI引数か設定で受け取る。
      パスは環境依存なのでハードコードしない）
- [ ] 純粋関数部分（対象WAV列挙・退避先決定・duration書き戻し・変換漏れ検知）にテスト
- [ ] `render_commentary_video.py` の変換漏れガード
- [ ] SKILL.md（video-commentary）の手順にパス1.8、地雷リストに「resynthesize後の1.8忘れ」を追記
- [ ] ADR-004に「VOICEVOX後段でRVC声質変換（パス1.8）」を追記（ADRに反する設計変更はADR先行のルール）
- [ ] VRAM試算（別タスク）の結果を確認
- [ ] **要確認（権利表記）**: RVCで声色を変えても元はVOICEVOX四国めたんの合成音声。
      VOICEVOX/四国めたんの利用規約上のクレジット表記（現在 `generate_thumbnail.py` の
      `_CHARACTER_NAME_NEUTRAL = "VOICEVOX：四国めたん"`・概要欄）をくれぴ名義に切り替えた後も
      どう残すかを規約で確認する
- [ ] **要実機確認**: RVC出力の音量（rms_mix_rate）が `render_commentary_video.py` の
      ダッキング・volume補償と合うか
