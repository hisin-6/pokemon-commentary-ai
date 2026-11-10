# scripts/ のテスト対応状況（2026-11-11 棚卸し）

`scripts/*.py`（60本）のうち、`tests/` から参照されているかを機械的に調べた一覧。
判定方法: モジュール名が `tests/*.py` に出現するか（`grep -lE "\b<module>\b" tests/*.py`）。
次タスク「純粋関数だけのものにテスト追加（1回1スクリプト）」の選定に使う。

## テストあり（直接15本＋間接1本）

| スクリプト | 対応テスト |
|---|---|
| build_pokedb | test_build_pokedb.py |
| check_spoilers | test_check_spoilers.py |
| check_vrm_model | test_check_vrm_model.py |
| compare_local_llm_commentary | test_compare_local_llm_commentary.py |
| explore_avatar_poses | test_check_vrm_model.py（POSES 読み込みのみ）/ test_play_and_animate_avatar.py |
| generate_chapters | test_generate_chapters.py |
| generate_gap_commentary | test_gap_commentary.py / test_server.py |
| generate_predictions | test_generate_predictions.py |
| generate_thumbnail | test_generate_thumbnail.py |
| play_and_animate_avatar | test_play_and_animate_avatar.py |
| render_bubble_overlay | test_render_bubble_overlay.py |
| render_commentary_video | test_render_video.py / test_gap_commentary.py |
| screen_pass1 | test_screen_pass1.py |
| team_preview_gui | test_pipeline.py（`_load_team_preview_hint` 経由の出力形式のみ） |
| update_champions_roster | test_update_champions_roster.py |
| vrm_bone_reader | 直接テストなし。`check_vrm_model` が `_read_glb_json` を使うので間接的に通過 |

※ `vrm_bone_reader` は間接カバーなので下の表にも載せている。

## テストなし・優先度順（テスト追加候補）

優先度は「現役の制作フローで使うか」×「純粋関数（ファイル/GPU/VMC/OCR不要）が切り出せるか」で付けた。

### A: 現役フローで使う・純粋関数あり（次にやる候補）

| スクリプト | 行数 | 使われ方 | テストしやすい関数 |
|---|---|---|---|
| resynthesize_fillers | 105 | 実況動画パス1.7（SKILL.md記載） | `parse_edit`（`"<idx>:<text>"`パース）、`load_jsonl`/`write_jsonl` 往復 ※VOICEVOX呼び出し部はモック必須 |
| generate_review_checklist | 239 | パス1検証（pass1-verification-checklist.md） | `_judgement`、`_build_segments`、`_find_segment`、`_turn_state_summary`、`build_markdown`（tmp_pathにJSONLを置けば完結） |
| vrm_bone_reader | 100 | 外注VRM納品時の check_vrm_model の土台 | `_read_glb_json`（最小GLBをバイト列で組める）、`node_world_position`（親子ノードの平行移動の合成） |
| pose_tuner_gui | 478 | アバターのポーズ調整（SKILL.md・VRM差し替えチェックリスト記載） | `look_rotation_quat`、`orbit_to_camera`、`pose_to_axis_values`⇄`axis_values_to_steps`、`format_pose_dict` ※`import tkinter` がトップレベル（46行目）なので、テスト時はtkinter無し環境でのimport可否に注意 |

→ **おすすめ順: ① generate_review_checklist ② resynthesize_fillers ③ vrm_bone_reader ④ pose_tuner_gui**
（①は関数がほぼ全部純粋で1回で収まりやすい。③④は2月のVRM納品直前に効く）

### B: 純粋関数はあるが、現役度が低い／保留中機能

| スクリプト | 行数 | 状況 | 純粋関数 |
|---|---|---|---|
| fix_annotations | 388 | OCRファインチューニング（保留中） | `normalize`、`chars_differ_only_by_ocr_error`、`find_correction`、`is_noise` |
| generate_battle_template | 410 | ログ→テンプレ変換。参照ドキュメントなし | `_is_ability_noise`、`_clean_*`、`format_template` |
| extract_ocr_changes | 158 | OCRログ差分抽出。参照ドキュメントなし | `fields_changed`、`format_entry` |
| replay_phase_events | 172 | 診断JSONLのリプレイ検証 | `_frame_time` |
| detect_command_bbox | 127 | 座標調査用 | `_is_command_like`、`_bbox_center`、`_bbox_topleft` |
| export_to_yolo / prepare_end_screen_dataset | 108 / 172 | YOLO学習データ作成（再学習しない方針・2026-07-30） | `_extract_stem`（2本で重複実装） |
| compute_vmd_retarget_correction / compute_bone_local_frames | 140 / 106 | MMDモーション変換（2026-08-03クローズ・3Dモデル完成まで保留） | `vec_*`、`quat_from_vectors`、`mat_to_quat` |
| explore_vmd_motion / explore_bvh_motion / visualize_vmd_stick_figure | 376 / 421 / 346 | 同上 | `quat_*`、`parse_vmd`/`parse_bvh`、`bone_state_at_frame` |

メモ: クォータニオン演算（`quat_mul`/`quat_conjugate`/`quat_slerp`）が `explore_vmd_motion`・`explore_bvh_motion`・
`visualize_vmd_stick_figure`・`pose_tuner_gui` にそれぞれ別実装されている。MMDモーション作業を再開するなら共通化候補。

### C: テスト対象外でよい（GUI・実機・使い捨て・薄いラッパー）

| スクリプト | 理由 |
|---|---|
| annotation_reviewer | Flask製のアノテーション確認GUI |
| ocr_logger | ライブキャプチャ＋OCRのロガー（`_format_ocr_raw`/`_area_label` だけは純粋。必要になったら） |
| calibrate_hpbar / debug_hp_bbox / debug_hp_preprocess / hpbar_grid / hpbar_measure / hpbar_zoom / inspect_hpbar_roi / visualize_coords | HPバー・座標の目視デバッグ用（画像を出すだけ） |
| capture_end_screen / collect_ocr_crops / crop_ball_rois | 動画・OCR・ファイル操作が主体 |
| balance_background / create_lmdb / prepare_lmdb_data / setup_finetune_model / finetune_ocr | 学習データ準備の薄いラッパー |
| pmx_bone_reader | 調査用ダンプ（関数1つ） |
| test_manga_ocr / test_ocr_areas / test_ocr_fix / test_ocr_fix2 / test_vmc_expression / test_vmc_pose / test_vmc_pose_reaction | 実機検証用の使い捨て。整理案は small-tasks.md の別項目で扱う |

## 気づいたこと（未対応）
- `scripts/test_*.py` という名前は pytest の収集パターン（`test_*.py`）に一致する。`pytest.ini` が `testpaths = tests`
  なので普段は収集されないが、`pytest scripts/` のように明示すると実機スクリプトが走る恐れがある。
  整理案の項目で一緒に検討する
- `_extract_stem` が `export_to_yolo` と `prepare_end_screen_dataset` に重複
