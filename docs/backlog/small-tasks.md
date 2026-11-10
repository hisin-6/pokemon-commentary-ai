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
- [ ] 上の一覧から純粋関数だけのものにテスト追加（1回1スクリプト）
- [ ] `scripts/test_*.py`（実機検証用の使い捨て）の整理案（残す/消すの提案だけ）

## ドキュメント
- [ ] `docs/pipeline-logic-overview.md` を現行コードと突き合わせて古い記述を洗い出す
- [ ] `docs/daily/` に無い期間（09-07以降）の要点を MEMORY から補う必要があるか確認

## 提案メモ（挙動変更が必要なもの・実装はユーザー判断）
- server.py のプロンプトキャッシュ（cache_control）導入: 2026-08-10 にユーザー判断で保留中
- 残り体数を気絶数の積み上げから推定して実況に活かす案（2026-07-30メモ）
- アバターのモデル依存値をプロファイルJSON化（2026-11-06・`docs/design/avatar-model-profile-proposal.md`・案A/B/C選択待ち）
- RVC声質変換の差し込み位置（2026-11-09・`docs/design/rvc-integration-proposal.md`・案A/B/C選択待ち）
