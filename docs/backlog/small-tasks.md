# 小さな改善ネタ帳（2026-10-28〜2027-02 試験期間用）

ユーザーが「今日の1件やって」と声をかけたら、Claudeは上から未着手の1件を選んで実施する。

## 運用ルール
- 1回1件・30分程度で終わる粒度。終わらなければ途中で止めて状況を記録する
- **低リスクのみ**: テスト追加・ドキュメント・新規スクリプト・小規模リファクタまで。
  `src/pipeline.py` 等の挙動を変える修正は「提案メモ」を下の提案欄に書くだけで止める
- GPU/VMC/OBS/EC2/録画動画が必要な検証はしない（必要なら「要実機確認」と書いて残す）
- git操作はしない。完了時に変更ファイル一覧とコミットメッセージ案を提示する
- 完了したら `[x]` と日付・一言を付ける。WSL pytestが全パスすることを確認してから完了扱い

## 2月（VRM納品）に効くもの
- [x] VRM 0.x の BlendShapeProxy 名一覧を出す確認スクリプト（`scripts/vrm_bone_reader.py` の流用可否から調べる）。
      納品モデルで `Joy`/`Sorrow`/`Fun`/`A`〜`O` が揃っているかを一発チェックできるように
      → 2026-10-29 `scripts/check_vrm_model.py` 作成（表情・回すボーン・メタ情報・VRM 1.0対応、テスト13件）。
      納品されたら `python scripts/check_vrm_model.py <model>.vrm` を流すだけ
- [x] 外注モデル差し替えチェックリスト `docs/manual/` に作成（クロップ再計測・idle pose角度再検証・
      表情名・クロマキー色・`--avatar-crop` 等。SKILL.md の地雷リストから拾う）
      → 2026-11-03 `docs/manual/vrm-model-swap-checklist.md` 作成（現行値の控え表＋Ⅰ納品物確認〜Ⅵ値反映・persona切り戻しまで6段階）
- [x] `play_and_animate_avatar.py` の表情名・ボーン角度をモデルごとに切り替えられるよう設定ファイル化する案を提案メモに
      → 2026-11-06 `docs/design/avatar-model-profile-proposal.md` 作成（推奨=案A モデルプロファイルJSON＋`--avatar-profile`）。
      腕下げ角度80°が `_IDLE_POSE_DEG` と `explore_avatar_poses._IDLE_ARM_DEG` に重複していると判明

## RVC（声質変更）の下ごしらえ
- [x] VOICEVOX出力→RVC変換を差し込む位置の調査メモ（`voicevox_client.py`・`generate_gap_commentary.py`・
      `resynthesize_fillers.py`）。設計案のみ、実装はユーザー復帰後
      → 2026-11-09 `docs/design/rvc-integration-proposal.md` 作成（推奨=案A 一括変換「パス1.8」新設）。
      WAVは全経路`VoicevoxClient.generate_wav()`経由・`build_commentary_track()`がフォーマット混在で落ちるため
      RVC出力は`--resample_sr 24000`でそろえる必要ありと判明
- [x] VRAM予算表にRVC推論ぶんを入れた試算（ADRとの整合も確認）
      → 2026-11-10 `docs/design/rvc-vram-budget.md` 作成。RVC推論は机上見積もり約1〜2GB（上限2.5GB・fp16・faiss indexはCPU）。
      案A（パス1.8）ならどのフェーズでも7GB以上余る。案Bはgemma2:9bフォールバックと重なると9.4GBでOOM圏内。
      ADR違反は無し・実装前にADR-004追記だけ必要。実測手順は同メモ5章（要実機確認）。
      ついでに `ollama-setup.md`・`system-architecture.md` 3章のフォールバックLLM行が Phi-3 のままと判明（未修正）

## テスト・品質
- [x] テストが薄いスクリプトを洗い出して一覧化（`scripts/` で tests に対応が無いもの）
      → 2026-11-11 `docs/backlog/scripts-test-coverage.md` 作成。60本中テストあり15本＋間接1本。
      次の候補（優先A）= generate_review_checklist → resynthesize_fillers → vrm_bone_reader → pose_tuner_gui
- [x] 上の一覧から純粋関数だけのものにテスト追加（1回1スクリプト・優先A 4本完了）
      - [x] 2026-11-13 `generate_review_checklist` → `tests/test_generate_review_checklist.py`（19件・main含む）。次=resynthesize_fillers
      - [x] 2026-11-16 `resynthesize_fillers` → `tests/test_resynthesize_fillers.py`（17件・VoicevoxClientはフェイク差し替え）。
            存在しないseqが混ざると1件も書き換えずに止まる（中途半端な更新にならない）ことも確認。次=vrm_bone_reader
      - [x] 2026-11-18 `vrm_bone_reader` → `tests/test_vrm_bone_reader.py`（14件・最小GLBをバイト列で組み立て）。
            VRM 0.x/1.0両方のhumanBones・パディング/BINチャンク付きGLB・親子の座標積算を確認。次=pose_tuner_gui
      - [x] 2026-11-20 `pose_tuner_gui` → `tests/test_pose_tuner_gui.py`（37件・GUI本体は対象外）。
            tkinter/python-oscが無い環境だけMagicMockで差し替え。カメラが常に注視点を向くこと・
            「Pythonコードを出力」の結果がそのままPOSESとして読み戻せることを確認。優先Aはこれで完了
- [x] `scripts/test_*.py`（実機検証用の使い捨て）の整理案（残す/消すの提案だけ）
      → 2026-11-23 `docs/backlog/scripts-test-cleanup-proposal.md` 作成。7本中 残す4本（ocr_areas・vmc系3本）／消す3本。
      `test_ocr_fix*.py` は `fix_annotations.py` の古いコピー（末尾助詞に「と」が無い）でassertも無いので、
      ケースを `tests/test_fix_annotations.py` に移してから削除。`test_manga_ocr.py` は2026-04-19に見送り済み。
      `test_vmc_pose_reaction.py` は差し替えチェックリストへの追記を提案。削除・追記はユーザーOK待ち
      → 2026-11-26 実施: `tests/test_fix_annotations.py`（39件）新設・`fix_annotations.py` の本体処理を `main()` 化（importで
        `annotations_fixed.json` を書き出さないように）・3本削除・`vrm-model-swap-checklist.md` Ⅲ節に `test_vmc_pose_reaction.py` を追記

## ドキュメント
- [x] `docs/pipeline-logic-overview.md` を現行コードと突き合わせて古い記述を洗い出す
      → 2026-11-23 `docs/backlog/pipeline-logic-overview-audit.md` 作成。「誤り」13件・動画モード後付け生成など新機能11項目が未記載で、
      全面改訂が妥当と判断（本体はまだ直していない）。ボールモデル無効化で「ボール数減少による気絶確定」が死に経路の疑い（要確認）
- [x] `docs/daily/` に無い期間（09-07以降）の要点を MEMORY から補う必要があるか確認
      → 2026-11-23 確認。09-07以降の作業日は 09-12（レギュレーション反映208→231種）・09-14（モデラー概算見積もり受領）・
      10-02（ADR-006改訂 `87ddf8c`・待ち期間の棚卸し）・10-29〜11-20（このネタ帳の各項目）。
      10-29以降は本ファイルとコミットに日付つきで残っているので補完不要。
      09-12・09-14・10-02 の3日はリポジトリ内に記録が無い（MEMORYのみ）ので、短い日報を補う価値あり（ユーザー判断待ち）。
      ついでに 09-03 の日報は一度も作られていない（09-01・09-02のみ）と判明
      → 2026-11-26 実施: `docs/daily/2026-09-12.md`・`2026-09-14.md`・`2026-10-02.md` をメモリとコミットから補完（冒頭に「後日補完」と明記）。09-03 は記録の材料が無いので作らない

## 提案メモ（挙動変更が必要なもの・実装はユーザー判断）
- server.py のプロンプトキャッシュ（cache_control）導入: 2026-08-10 にユーザー判断で保留中
- 残り体数を気絶数の積み上げから推定して実況に活かす案（2026-07-30メモ）
- アバターのモデル依存値をプロファイルJSON化（2026-11-06・`docs/design/avatar-model-profile-proposal.md`・案A/B/C選択待ち）
- RVC声質変換の差し込み位置（2026-11-09・`docs/design/rvc-integration-proposal.md`・案A/B/C選択待ち）
- 【2026-11-26 pipeline-logic-overview全面改訂で発見・コード読解ベース】
  - `faint_repeat_hint` がBedrockに届いていない: `_generate_posthoc_commentary` が `game_state` に入れるが、`_build_bedrock_context` の出力に無いので `server.py` 側は常に空（manifestには記録されるので効いているように見える）。→ ✅2026-11-26 修正（context に追加・Phi-3 側も配線・回帰テスト3件）
  - バトル開始前の「ゆけっ」キャッシュ（`run()` 内の `_YUKE_RE`）が旧パターンのまま＝2026-09-07の二つ名修正が入っていない → ✅2026-11-26 修正（`_extract_yuke_candidates` 新設・テスト9件）
  - 動画末尾の battle_end が `frame=None` で `_process_event` に入り、`cv2.imwrite(debug/pipeline_turn_*.png, None)` で例外になる可能性（後付け生成は finally で走る） → ✅2026-11-26 修正（例外になることを確認のうえ `frame is None` ガード）
  - `pipeline.py` の古いコメント11件・`hpbar_analyzer.py`/`screen_capture.py` のdocstring 2件（一覧は `docs/pipeline-logic-overview.md` 10章）。コメントのみなので挙動は変わらない
