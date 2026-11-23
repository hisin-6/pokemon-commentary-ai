# `scripts/test_*.py`（実機検証用スクリプト）の整理案

2026-11-23 作成。提案のみで、削除・リネームはまだしていない（実施はユーザー判断）。

## 前提
- `pytest.ini` が `testpaths = tests` なので、これらは pytest に収集されない。CIや全件テストへの影響は無い
- ただ名前が `test_` で始まるため、pytest のテストと紛らわしい

## 一覧と提案

| スクリプト | 最終更新 | 中身 | 参照元 | 提案 |
|---|---|---|---|---|
| `test_ocr_areas.py` | 2026-04-19 | 動画のフレームごとにROI別のOCR結果を表示する診断ツール | `ocr-finetune/SKILL.md`・`docs/manual/ocr-finetune-cycle.md`(2箇所)・`docs/manual/screen_capture.md`・`docs/easyocr-finetuning-guide.md` | **残す**（現役の診断ツール） |
| `test_vmc_expression.py` | 2026-07-30 | VMCへ表情ブレンドシェイプを1つ送って目視確認 | `vrm-model-swap-checklist.md`・`play_and_animate_avatar.py` docstring | **残す**（2月のVRM差し替えで使う） |
| `test_vmc_pose.py` | 2026-07-30 | 腕ボーンに指定角度を送ってT-pose解消角度を探す | `vrm-model-swap-checklist.md`・`video-commentary/SKILL.md`・`play_and_animate_avatar.py`・`explore_avatar_poses.py` | **残す**（同上。モデルが変わると角度の再検証が要る） |
| `test_vmc_pose_reaction.py` | 2026-08-03 | `play_and_animate_avatar.py` の本番関数でポーズ遷移・スウェイを段階確認 | なし | **残す**＋差し替えチェックリストに追記（後述） |
| `test_ocr_fix.py` | 2026-05-24 | `chars_differ_only_by_ocr_error` の動作確認（printで目視） | なし | **消す**（テストを `tests/` へ移してから） |
| `test_ocr_fix2.py` | 2026-05-24 | 上の追加ケース＋`find_correction` の動作確認 | なし | **消す**（同上） |
| `test_manga_ocr.py` | 2026-05-04 | EasyOCR と MangaOCR の比較 | `docs/daily/2026-04-19.md` のみ（作業記録） | **消す** |

## 消す候補の理由

### `test_ocr_fix.py` / `test_ocr_fix2.py`
- 中身は `scripts/fix_annotations.py` の `normalize` / `chars_differ_only_by_ocr_error` / `find_correction` を**コピーした古い版**。
  本体はその後、末尾助詞の判定に「と」が加わっているので、コピーはもう本体と一致しない
- 結果を print するだけで assert が無く、回帰テストとしては機能しない
- 入っているテストケース（`オオニユーラの→オオニューラの`、`リザードンを→リザードンは` は不可、`イダイトウ」→イダイトウ` など）は価値があるので、
  **`tests/test_fix_annotations.py` を新設して本体を import する形で移植**してから2本を消すのがよい。
  `fix_annotations` は `scripts-test-coverage.md` でもテスト候補（純粋関数4つ）に挙がっている

### `test_manga_ocr.py`
- 2026-04-19 の調査で「MangaOCR は見送り」と結論済み（`docs/daily/2026-04-19.md` 6節）
- 以降どこからも参照されていない。`manga-ocr` パッケージ前提なので、今の venv で動く保証も無い
- 比較結果は日報に表で残っているので、スクリプトを消しても経緯は追える。git履歴からも戻せる

## 残すものへの追加提案
1. **`vrm-model-swap-checklist.md` に `test_vmc_pose_reaction.py` を追記**: 本番の `send_reaction` / `play_pose_reaction` /
   `run_idle_sway` / `run_idle_gestures` をそのまま呼ぶので、新モデルでポーズ遷移が破綻しないかの確認にちょうどよい。
   今のチェックリストは表情単体（`test_vmc_expression.py`）と腕下げ（`test_vmc_pose.py`）しか載っていない
2. **リネームは VRM 差し替え後に検討**: `test_` → `check_`（例: `check_ocr_areas.py`・`check_vmc_pose.py`）にすると pytest と混同しなくなる。
   ただし参照元がドキュメント・SKILL.md・docstring に計10箇所以上あり、2月の差し替え作業中に名前が変わると手順書とずれる。
   差し替えが終わって落ち着いてから、参照元の一括更新とセットでやるのが安全

## 実施する場合の手順（ユーザーOK後）
1. `tests/test_fix_annotations.py` を作り、`test_ocr_fix*.py` のケースを assert 付きで移植 → WSL pytest 全パス確認
2. `scripts/test_ocr_fix.py`・`scripts/test_ocr_fix2.py`・`scripts/test_manga_ocr.py` を削除（3ファイル削除なので要確認）
3. `docs/backlog/scripts-test-coverage.md` 72行目の扱いを更新
4. `vrm-model-swap-checklist.md` に `test_vmc_pose_reaction.py` の確認手順を追記
