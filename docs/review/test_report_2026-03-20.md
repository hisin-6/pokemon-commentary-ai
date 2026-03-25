# テストレポート（2026-03-20）

## 概要

`review_2026-03-20.md` の課題 **「4. テストコードが存在しない」** に対応して自動テストを導入した。

| 項目 | 内容 |
|------|------|
| 実施日 | 2026-03-20 |
| 実行環境 | Python 3.8.10 / pytest 8.3.5 |
| テスト数 | **162 件** |
| 結果 | **全件 PASSED（0 FAILED）** |
| 実行時間 | 約 2.1 秒 |

```
============================= test session starts ==============================
162 passed in 2.07s
```

---

## テストファイル構成

```
tests/
├── __init__.py
├── conftest.py          # モック設定（GPU/AWS/音声デバイス依存を排除）
├── test_pipeline.py     # 85 テスト
├── test_server.py       # 35 テスト
└── test_classifier.py   # 42 テスト
```

### `tests/conftest.py` — モック設定

テスト実行時に以下の重い依存を排除している。実際の GPU・AWS 認証・音声デバイスなしに CI/ローカルどちらでも動く。

| モック対象 | 理由 |
|-----------|------|
| `easyocr` | GPU 初期化（数十秒）を防ぐ |
| `boto3` / `botocore` | AWS 認証不要にする |
| `sounddevice` / `soundfile` | 音声デバイス不要にする |
| `src.commentary.phi3_client` | `str \| None` 記法（Python 3.10+構文）が Python 3.8 でパースエラーになるため |
| `src.output.voicevox_client` | 同上 |
| `src.output.audio_player` | 同上 |

---

## テスト詳細

### `test_pipeline.py` — 85 テスト

`src/pipeline.py` の純粋ロジック（外部依存なし）を網羅している。

#### `TestCleanCommentary` — 10 テスト

LLM 出力のクリーンアップロジック `_clean_commentary()` を検証。

| テスト | 確認内容 |
|--------|---------|
| `test_removes_after_triple_dash` | `---` 以降を切り捨てる |
| `test_removes_after_bracket` | `【` 以降を切り捨てる |
| `test_removes_lines_with_shiji` | 「指示」を含む行以降を削除 |
| `test_removes_lines_with_shitsumon` | 「質問」を含む行以降を削除 |
| `test_removes_leading_bullet` | 行頭の `- ` を除去 |
| `test_removes_leading_dot_bullet` | 行頭の `・ ` を除去 |
| `test_removes_kakko` | 鉤括弧「」を除去 |
| `test_keeps_only_first_two_sentences` | 最大 2 文（。！？）に制限 |
| `test_empty_string` | 空文字列で例外が出ない |
| `test_plain_text_unchanged` | クリーンな文はそのまま通る |

#### `TestIsBattleScreen` — 6 テスト

バトル外画面を OCR テキストから判定する `_is_battle_screen()` を検証。

| テスト | 確認内容 |
|--------|---------|
| `test_empty_results_returns_true` | OCR 結果なし → True（通過させる） |
| `test_battle_keyword_present_returns_true` | 「HP」→ True |
| `test_non_battle_keyword_returns_false` | 「バトルスタジアム」→ False |
| `test_offline_keyword_returns_false` | 「オフライン」→ False |
| `test_normal_battle_text_returns_true` | 「たたかう」→ True |
| `test_union_circle_returns_false` | 「ユニオンサークル」→ False |

#### `TestOcrResultsToText` — 8 テスト

OCR 結果をプロンプト用テキストに変換する `_ocr_results_to_text()` を検証。

| テスト | 確認内容 |
|--------|---------|
| `test_empty_results` | 空リスト → 「（テキスト未検出）」 |
| `test_low_confidence_filtered` | confidence < 0.4 は除外 |
| `test_high_confidence_included` | confidence ≥ 0.4 は含む |
| `test_multiple_texts_joined_with_slash` | 複数テキストを ` / ` で結合 |
| `test_max_chars_truncated` | 120 文字（OCR_MAX_CHARS）で切る |
| `test_threshold_boundary_exactly_04` | confidence = 0.4 は含む（境界値） |
| `test_with_classifier_excludes_move` | classifier 付き: 技名を除外 |
| `test_with_classifier_keeps_battle_message` | 助詞「を」含む文はバトルメッセージとして残す |

#### `TestExtractStructuredInfo` — 15 テスト

HP 値・ポケモン名候補を OCR 結果から抽出する `_extract_structured_info()` を検証。
Sprint 7 で実装した y 座標ベースの自分/相手分類・ダブルバトル制約も確認。

| テスト | 確認内容 |
|--------|---------|
| `test_hp_extracted` | `176/176` 形式が hp_values に入る |
| `test_hp_low_denom_excluded_as_pp` | 分母 < 50 は PP 値として除外 |
| `test_hp_assigned_to_player_side_by_y` | y ≥ 500 の HP → hp_values_player |
| `test_hp_assigned_to_opponent_side_by_y` | y < 500 の HP → hp_values_opponent |
| `test_command_menu_items_excluded` | y ≥ 700（コマンドエリア）のテキストは除外 |
| `test_ui_words_excluded` | 「たたかう」等 UI ワードは除外 |
| `test_lv_prefix_excluded` | 「Lv50」は除外 |
| `test_text_ending_no_excluded` | 「ピカチュウの」（「の」で終わる）は除外 |
| `test_status_panel_skips_name_collection` | 「戦闘中」テキストで名前候補収集をスキップ |
| `test_with_classifier_pokemon_included` | PokeClassifier がポケモン名と判定 → 候補に追加 |
| `test_with_classifier_non_pokemon_excluded` | 技名と判定 → 除外 |
| `test_max_5_candidates_per_side` | 各サイド最大 5 件に制限 |
| `test_max_2_hp_per_side` | ダブルバトル: HP は各サイド最大 2 件 |
| `test_chinese_name_added_to_opponent` | 相手エリアの CJK 文字 → 中国語ポケモン名候補として登録 |

#### `TestBattlePhaseClassifier` — 20 テスト

OCR テキストからバトルフェーズを分類し、イベントを検知する `BattlePhaseClassifier` を検証。

**`classify()` — フェーズ判定（11 テスト）**

| テスト | 確認内容 |
|--------|---------|
| `test_empty_returns_unknown` | OCR なし → unknown |
| `test_tatakau_is_command_select` | 「たたかう」→ command_select |
| `test_dousuru_is_command_select` | 「どうする」→ command_select |
| `test_batsugun_is_animation` | 「バツグンだ」→ animation |
| `test_faint_phase_hp_zero` | 「0/100」→ faint |
| `test_faint_not_triggered_for_low_denom` | 「0/8」（PP値）→ faint にならない |
| `test_switch_select` | 「こうたい」→ switch_select |
| `test_battle_end` | 「勝負に勝った」→ battle_end |
| `test_battle_end_loss` | 「勝負に負けた」→ battle_end |
| `test_selection_screen` | 「ポケモンを選んで」→ selection_screen |
| `test_priority_battle_end_over_command` | battle_end が command_select より優先 |

**`detect()` — イベント検知（9 テスト）**

| テスト | 確認内容 |
|--------|---------|
| `test_detect_battle_start_on_first_command` | 初回 command_select → battle_start |
| `test_detect_battle_start_only_once` | 同フェーズ継続で battle_start は一度だけ |
| `test_detect_move_used_on_command_to_animation` | command_select → animation → move_used |
| `test_detect_switch_on_command_to_switch_select` | command_select → switch_select → switch |
| `test_detect_faint_on_hp_zero` | HP=0 出現 → faint |
| `test_detect_battle_end` | animation → battle_end 遷移 → battle_end |
| `test_battle_started_resets_on_selection_screen` | 選出画面通過後 → 次の command_select で battle_start が再発火 |
| `test_debounce_suppresses_duplicate_events` | 60 秒デバウンス内の同イベントを抑制 |
| `test_processing_flag_suppresses_events` | 処理中フラグ ON 時は battle_end 以外を抑制 |
| `test_processing_flag_allows_battle_end` | 処理中でも battle_end は割り込み検知 |

#### `TestBattleStateTracker` — 26 テスト

試合全体の戦況を蓄積する `BattleStateTracker` を検証。Sprint 7 のリファクタリング内容（ダブルバトル制約・HP y 座標分類・faint 誤判定バグ修正）を重点的に確認。

| テスト | 確認内容 |
|--------|---------|
| `test_initial_state_empty` | 初期状態で「情報収集中」と返す |
| `test_update_creates_player_slot` / `opponent_slot` | update() でスロットが生成される |
| `test_pokemon_marked_on_field_when_seen` | 検出されたポケモンは on_field=True |
| `test_turn_increments_on_update` | update() ごとにターン数が増える |
| `test_cap_on_field_max_2` | **ダブルバトル制約**: 場は最大 2 匹 |
| `test_opponent_not_added_to_player_side` | 相手登録済みのポケモンは自分側に混入しない |
| `test_hp_assigned_to_on_field_player` | HP 値が場のポケモンに紐付く |
| `test_faint_event_marks_pokemon_fainted` | **faint イベント + HP=0 → 気絶フラグ ON** |
| `test_non_faint_event_does_not_faint_hp_zero` | **Sprint 7 バグ修正確認**: faint 以外では HP=0 でも気絶しない（ゴリランダー誤気絶防止） |
| `test_max_slots_4_per_side` | 最大 4 スロット制限 |
| `test_update_move_records_to_slot` | update_move() で技リストに追加 |
| `test_update_move_no_duplicate` | 同技名は重複しない |
| `test_update_move_max_4_moves` | 最大 4 技まで記録 |
| `test_set_not_on_field_exact_match` | set_not_on_field() で完全一致 → 場から降ろす |
| `test_set_not_on_field_partial_match` | OCR 誤読の部分一致も対応（「ゴリランダ」→「ゴリランダー」） |
| `test_set_not_on_field_returns_false_if_not_found` | 見つからない場合は False |
| `test_pokemon_removed_from_field_after_miss_threshold` | _ON_FIELD_MISS_THRESHOLD ターン不検出で場から降ろす |
| `test_to_context_shows_field_and_bench` | 場と控えが分離して出力される |
| `test_to_context_fainted_shown_as_hinshi` | 気絶ポケモンに「(ひんし)」が付く |
| `test_to_context_event_log_appended` | イベントログにターン・イベント種別が記録される |
| `test_to_context_hp_pinch_marker` | HP ≤ 25% に「★ピンチ」が付く |
| `test_status_updated_from_game_state` | YOLO 由来の状態異常がスロットに記録される |
| `test_hp_fallback_uses_all_hp_values` | hp_values_player が空の場合 hp_values からフォールバック |

#### `TestFieldPokemon` — 2 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_default_values` | 各フィールドのデフォルト値 |
| `test_moves_used_not_shared_between_instances` | `field(default_factory=list)` でインスタンス間共有なし |

---

### `test_server.py` — 35 テスト

`src/api/server.py`（EC2 Flask API）を検証。Bedrock・S3 はモックで代替。

#### `TestParseCommentary` — 6 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_extracts_jikkyou_section` | 【実況】セクションのみを抽出 |
| `test_extracts_jokyo_section` | 【状況】セクションを抽出 |
| `test_fallback_when_no_markers` | マーカーなし → 全文をフォールバック |
| `test_jikkyou_trims_whitespace` | 前後の空白をトリム |
| `test_multiple_brackets_stops_at_next` | 次の【セクション】で切る |
| `test_empty_string` | 空文字列で例外が出ない |

#### `TestBuildVisionPrompt` — 11 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_prompt_contains_event_type` | イベント種別がプロンプトに含まれる |
| `test_prompt_contains_battle_start_hint` | battle_start 向けの実況指示が含まれる |
| `test_prompt_contains_faint_hint` | faint 向けの実況指示が含まれる |
| `test_prompt_contains_ocr_text` | OCR テキストがプロンプトに渡される |
| `test_prompt_contains_turn_info` | ターン数が含まれる |
| `test_prompt_contains_history` | 直前の実況履歴が含まれる |
| `test_prompt_contains_rag_info` | RAG 情報（ポケモン図鑑）が含まれる |
| `test_prompt_no_rag_info_when_empty` | RAG 情報なし → 図鑑セクションが出力されない |
| `test_prompt_contains_jikkyou_marker` | 【実況】マーカーが含まれる |
| `test_prompt_contains_jokyo_marker` | 【状況】マーカーが含まれる |
| `test_prompt_player_field_and_bench_shown` | 自分の場・控えポケモンが含まれる |

#### `TestHealthEndpoint` — 3 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_health_returns_200` | ステータス 200 |
| `test_health_returns_ok_status` | `{"status": "ok"}` |
| `test_health_returns_timestamp` | `timestamp` フィールドが含まれる |

#### `TestVisionEndpoint` — 8 テスト（POST /api/vision）

| テスト | 確認内容 |
|--------|---------|
| `test_missing_json_returns_400` | JSON でないリクエスト → 400 |
| `test_missing_image_returns_400` | image_base64 なし → 400 |
| `test_missing_context_returns_400` | context なし → 400 |
| `test_invalid_event_type_returns_400` | 不正 event_type → 400 |
| `test_invalid_base64_returns_400` | Base64 デコード失敗 → 400 |
| `test_successful_vision_call` | Bedrock モック → 200 + commentary 返却 |
| `test_bedrock_timeout_returns_504` | ReadTimeoutError → 504 |
| `test_all_valid_event_types_accepted` | 5 種の event_type が 400 にならない |

#### `TestLogEndpoint` — 7 テスト（POST /api/log）

| テスト | 確認内容 |
|--------|---------|
| `test_missing_json_returns_400` | JSON でないリクエスト → 400 |
| `test_missing_session_id_returns_400` | session_id なし → 400 |
| `test_missing_commentary_returns_400` | commentary なし → 400 |
| `test_s3_not_configured_returns_500` | S3_BUCKET 未設定 → 500 |
| `test_successful_log_save` | S3 モック → 200 + s3_log_path 返却 |
| `test_log_path_contains_session_id_and_turn` | S3 パスに session_id・ターン番号（ゼロ埋め）が含まれる |
| `test_image_saved_when_provided` | image_base64 付き → S3 put_object が 2 回呼ばれる（JSON + 画像） |

---

### `test_classifier.py` — 42 テスト

`src/pokedb/classifier.py` を検証。本番 DB 不要：テスト用の一時 SQLite DB（4 ポケモン / 4 技 / 3 特性 / 2 アイテム）を使用。

#### `TestPokeClassifierInit` — 2 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_raises_if_db_not_found` | DB なし → FileNotFoundError |
| `test_loads_successfully_with_valid_db` | DB あり → 各テーブルのレコード数が正しい |

#### `TestClassify` — 15 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_exact_pokemon_name_returns_confident` | 「ピカチュウ」→ pokemon / confident=True |
| `test_exact_move_name_returns_move` | 「まもる」→ move |
| `test_exact_ability_name_returns_ability` | 「せいでんき」→ ability |
| `test_exact_item_name_returns_item` | 「きあいのタスキ」→ item |
| `test_empty_string_returns_unknown` | 空文字 → unknown |
| `test_single_char_returns_unknown` | 1 文字 → unknown（len < 2 条件） |
| `test_completely_unrelated_text_returns_unknown` | 無関係テキスト → unknown |
| `test_ocr_partial_read_fuzzy_match` | 「エルフー」→ スコア次第で pokemon |
| `test_confident_flag_above_threshold` | スコア ≥ 90 → confident=True |
| `test_best_score_wins_across_categories` | 複数カテゴリで最高スコアが勝つ |
| `test_canonical_ja_matches_db_value` | 正規化された日本語名が返る |
| `test_canonical_en_is_populated` | 英語名も返る |
| `test_score_is_float` | score が float 型 |

#### `TestClassifyBatch` — 3 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_returns_list_of_same_length` | 入力と同じ長さのリストを返す |
| `test_each_result_is_classify_result` | 各要素が ClassifyResult 型 |
| `test_empty_list_returns_empty_list` | 空リスト → 空リスト |

#### `TestFilterPokemonNames` — 5 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_returns_only_pokemon_names` | ポケモン名のみ抽出（技名・アイテム名は除外） |
| `test_normalizes_to_canonical_name` | DB の正規名に変換される |
| `test_empty_list_returns_empty` | 空リスト → 空リスト |
| `test_all_non_pokemon_returns_empty` | ポケモン名なし → 空リスト |
| `test_multiple_pokemon_all_included` | 複数ポケモン名がすべて含まれる |

#### `TestGetPokemonInfo` — 12 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_returns_dict_for_known_pokemon` | 既知ポケモン → dict を返す |
| `test_returns_none_for_unknown_pokemon` | 不明ポケモン → None |
| `test_info_contains_required_keys` | name_ja / name_en / type / abilities / moves キーが存在 |
| `test_type_string_format` | 単タイプ → 「でんき」（スラッシュなし） |
| `test_dual_type_has_slash` | 複合タイプ → 「くさ / フェアリー」 |
| `test_abilities_list_populated` | 特性リストに要素がある |
| `test_hidden_ability_marked_with_dream` | 夢特性に「（夢）」が付く |
| `test_moves_list_populated` | 技リストに要素がある |
| `test_moves_limited_to_max_rag` | MAX_MOVES_FOR_RAG（12）以内 |
| `test_fuzzy_correction_for_slightly_wrong_name` | 誤読名でもクラッシュしない |
| `test_name_en_matches_expected` | 英語名が DB と一致 |

#### `TestUtilityMethods` — 7 テスト

| テスト | 確認内容 |
|--------|---------|
| `test_is_pokemon_true_for_pokemon` | is_pokemon("ピカチュウ") → True |
| `test_is_pokemon_false_for_move` | is_pokemon("まもる") → False |
| `test_is_move_true_for_move` | is_move("まもる") → True |
| `test_is_move_false_for_pokemon` | is_move("ピカチュウ") → False |
| `test_is_ability_true_for_ability` | is_ability("せいでんき") → True |
| `test_is_ability_false_for_pokemon` | is_ability("ピカチュウ") → False |
| `test_is_pokemon_false_for_empty_string` | is_pokemon("") → False |

---

## 副産物：本番コードへの修正

テスト導入にあたり `src/api/server.py` に以下を追加した。

```python
from __future__ import annotations  # Python 3.8 互換化（list[str] 等の型ヒント）
```

動作への影響なし。EC2 の Python バージョンが 3.9+ であれば不要だが、害もない。

---

## テスト対象外のモジュール（今後の課題）

| モジュール | テスト未整備の理由 | 将来の対応案 |
|-----------|-------------------|-------------|
| `src/capture/screen_capture.py` | EasyOCR 初期化・カメラデバイス依存 | サンプル画像を用いた統合テスト |
| `src/capture/yolo_detector.py` | YOLOv8 モデルファイル・GPU 依存 | 検出結果の後処理関数のみ切り出してテスト |
| `src/commentary/phi3_client.py` | Ollama ローカル起動が必要 | モック + プロンプト構築ロジックのみテスト |
| `src/output/voicevox_client.py` | VOICEVOX サービス起動が必要 | `requests.post` をモックして HTTP 呼び出しのみテスト |
| `src/output/audio_player.py` | sounddevice・オーディオデバイス依存 | sounddevice モックで再生ロジックのみテスト |
| `scripts/build_pokedb.py` | PokeAPI への実際の HTTP アクセスが必要 | キャッシュ機構のテスト・モックでの API 呼び出し検証 |

---

## テスト実行方法

```bash
# pytest インストール（未インストールの場合）
venv/bin/pip install pytest flask requests

# 全テスト実行
venv/bin/python -m pytest tests/ -v

# ファイル別実行
venv/bin/python -m pytest tests/test_pipeline.py -v
venv/bin/python -m pytest tests/test_server.py -v
venv/bin/python -m pytest tests/test_classifier.py -v
```

---

*テスト作成・実施: 2026-03-20*
