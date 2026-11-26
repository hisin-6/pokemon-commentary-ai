# `docs/pipeline-logic-overview.md` 現行コード突き合わせ結果

> **2026-11-26 改訂済み**: この結果をもとに `docs/pipeline-logic-overview.md` を全面改訂した（対象 commit `7fcd18c`）。下の指摘は改訂前の記録として残す。訂正が1点ある。「13. ピクセルHP」の「command_cy ゲートは `assign_slots_from_ocr` 側だけ」は誤りで、`BattleStateTracker.update()` 内の OCR HP 割当にも 300〜450 のゲートが残っている（ゲートが無いのは定期の数値HP割当 `assign_hp_from_ocr` だけ）。「要確認」のボール数ロジックは、改訂版の 4.10節で「既定構成では死に経路（コード読解ベース）」と結論づけた。`pipeline.py` の古いコメントは直していない。

2026-11-23 実施。ドキュメントの最終更新は 2026-07-09。`src/pipeline.py`（6632行）と関連モジュールを読んで突き合わせた（読み取りのみ・ドキュメント本体はまだ直していない）。
行番号は 2026-11-23 時点（`86fac78`）のもの。

## 結論
**全面改訂が妥当。** 今も正しいのは1〜3節（OCR・HP正規化・HP分類）と10節（特性エリア）、13節の command_cy 制御の一部くらい。
4〜9節と12〜14節は閾値・識別子・発火条件がずれていて「誤り」が10件以上ある。
2026-07-29 以降の動画ファースト化で、ドキュメントの軸だった「ライブで1イベントずつ Vision 呼び出し」が主経路ではなくなったのに、後付け生成・move_single・メガシンカ・team_preview・戦況推論ヒントが丸ごと載っていない。

## 主な「誤り」（優先して直すもの）
| ドキュメント | 現行コード |
|---|---|
| 自分の名前候補は `cy >= 500`（L92, 99） | `cy > _PLAYER_NAME_Y_MIN (930)`。500〜930はどちらにも入れない（`pipeline.py:133, 488-492`） |
| フェーズ優先順に communication が無い（L111-120） | 2番目に `communication`（`_COMM_RE`・入場0.7秒/退出3秒の平滑化）（`pipeline.py:637-650, 737-808`） |
| command_select の KW に「ゆけつ」（L117） | `{"たたかう","どうする"}` のみ（`pipeline.py:629-633`） |
| move_used は `command_select`→他 で発火（L135） | `communication`→他 で発火・`_battle_started` ガード付き（`pipeline.py:861-875`） |
| `MSG_X_MAX` 900（L144, 155, L270表#3） | 1150（`pipeline.py:1043`・2026-07-30拡張） |
| 相手の繰り出しも `switch_in`（L150） | `opponent_switch_in`（Dual RE・「繰」切れ対応）（`pipeline.py:1083-1091, 1199-1207`） |
| `PipelineRunner._update_move_log()`（L162） | クラス名は `Pipeline`（`pipeline.py:3001`）。`PipelineRunner` はコメントに残るだけ |
| dense scan は最低90フレーム（L173） | move_used で `_dense_scan_remaining = 9999`・turn_start/battle_start/battle_end で停止（`pipeline.py:3607-3627, 3688-3739`） |
| `_player` は場の2匹（L220-221） | 各側最大4匹（`MAX_SLOTS=4`）・場は最大2（`MAX_ON_FIELD=2`）（`pipeline.py:1268-1269`） |
| `per_pokemon_moves` / `is_fainted`（L224-225） | 実体は `FieldPokemon.moves_used` / `.fainted`（`pipeline.py:1250, 1252`） |
| HPpx を OCR HP より優先（L235-242, 表#12） | 鮮度比較方式（`hp_px_turn > hp_turn` のときだけ px）（`pipeline.py:2296-2331`） |
| Bedrock トリガーは3種（L260-264） | `BEDROCK_EVENTS` は7種（battle_start・move_used・move_single・switch・faint・battle_end・mega_evolution）（`pipeline.py:96-97`） |
| faint 保留は12秒（L266） | `_FAINT_PENDING_TIMEOUT = 75.0`＋各種即フラッシュ条件（`pipeline.py:3148` ほか） |

## 「古い」「不足」（節ごと）
- **全体の流れ**: 実際の1ループは YOLO終了画面→降参時WIN/LOSE待ち→HPバーピクセル解析（毎フレーム）→差分or定期OCR（バトル中1.5秒/外3秒）→技ログ・メッセージ→フェーズ→dense scan（`pipeline.py:3239-3740`）。動画モードは `_posthoc_mode` でイベントを溜め、スキャン後に `_generate_posthoc_commentary` が `_call_bedrock_text` で一括生成（`3123, 4166-4175, 4682-`）。フォールバックLLMは `Phi3Client` だが実体は ollama `gemma2:9b`（`phi3_client.py:28`）
- **1. OCR**: `run_ocr` の定義場所（`screen_capture.py:99`）と `preprocess_dense`（メッセージROI 2倍拡大）が未記載。conf閾値はROIで違う（メッセージ0.35・特性0.3・通信判定0.0）
- **2. HP正規化**: 分母50未満（PP値）除外 `_HP_MIN_DENOM = 50` が未記載
- **3. HP分類**: 状態/技詳細パネル表示中のスキップ、定期OCRの位置ゲート割当 `assign_hp_from_ocr`（2回一致で確定・1桁%保留）が未記載（`1519-1583`）
- **4. 名前分類**: 「相手を見る」/技詳細パネルのスキップ・中国語名フォールバック・`_CJK_UI_WORDS`・`--game-mode champions` が未記載
- **5. フェーズ**: switch_select に「交代する」、battle_end/selection に「切断されました」「更新されるまで」「リーダー」「選出順」が追加済み
- **6. イベント**: battle_start はデバウンス無し。turn_start に `_allow_turn_start` と脱出弁（`_CMD_SESSION_GAP_SEC=10`・`_TURN_START_ESCAPE_SEC=15`）。faint はエッジトリガ＋`_FAINT_REARM_SEC=20`。battle_end の発行経路（YOLO 3連続＋OCR AND・降参時の遅延発行）が未記載
- **7. メッセージパース**: 気絶は `opponent_faint`/`faint` に分割・ミラー戦の重複抑止。自分の繰り出しは二つ名対応（2026-09-07）。交代に「戻れ」「引っこめた」（`opponent_switch_out`）。状態異常は漢字形式とROI外検索に対応
- **8. 技ログ**: スキャン1はメッセージROI内に限定。スキャン2フォールバック（「の」欠け）と `move_single` 実況（`_dispatch_move_commentary`）が未記載
- **9. switch-in**: 末尾は `_KURIDASHITA_SFX`。バトル開始前キャッシュ未記載。L189「くりだした未検証」とL278の✅が自己矛盾
- **11. 状態異常同期**: x座標推定ではなく側×スロット4矩形。`update_status_by_name`・`apply_status_from_move` が未記載
- **12. BattleStateTracker**: `slot_index`・`mega_evolved`・`hp_turn`/`hp_px_turn`・`game_turn`・`turn_history`・場のコンディション・新規登録ヒステリシスが未記載
- **13. ピクセルHP**: 毎フレーム `HpBarAnalyzer.analyze()`→`update_pixel_hp()`・交代時 `slot_reset_cb` リセットが未記載。command_cy ゲートは `assign_slots_from_ocr` 側だけ
- **14. Bedrock**: ライブ=Vision、動画=`_call_bedrock_text`（どちらも `POST {ec2_url}/api/vision`）。battle_start 保留・合成faint・`_announced_faints` 陣営タプル化が未記載
- **チャンピオンズ表**: #8 ボールモデル「再学習待ち」・#9 状態異常アイコン「再学習必要」→ どちらも既定 None でパイプライン未使用、再学習しない方針（2026-07-30決定）

## 丸ごと抜けている機能（新節候補）
| 機能 | 根拠 | 追記先 |
|---|---|---|
| 動画モードの後付け実況生成 | `pipeline.py:3123, 4682-4965` | 新節「後付け生成」 |
| RenderSink 素材出力（manifest/timeline/states.jsonl/render_info.json）・`_record_panel_state` | `render_sink.py`・`pipeline.py:5010` | 新節「レンダ出力」 |
| move_single と技の対象ヒント（HP減少・状態異常・まもる・外れ・ひるみ） | `pipeline.py:5949-, 4328-4680, 5575-5658` | 8・14節 |
| メガシンカ検出と専用実況 | `pipeline.py:5050-5074, 6018-`・`mega_forms.py` | 新節 |
| team_preview（パス0）の `team_preview_hint` 注入 | `pipeline.py:3116-3121, 3192-3204` | 14節 |
| 戦況推論ヒント（タイプ相性・技範囲・天候/壁/トリル/おいかぜ・素早さランク・溜め技） | `pipeline.py:1328-1342, 5076-5245, 5692-5825` | 12節拡張 or 新節 |
| 戦況ウェアハウス（`battle_situations.sqlite`・勝敗バックフィル） | `situation_warehouse.py`・`pipeline.py:4700-4712` | 後付け生成の節 |
| 降参時WIN/LOSE待ち・相手トレーナー名による左右判定 | `pipeline.py:3354-3420, 2686-2800` | 6節 |
| 合成faint実況（60秒抑制窓） | `pipeline.py:3157-3167, 5827-5947` | 14節 |
| `_clean_commentary`・`_replace_glitch_commentary` | `pipeline.py:2592, 2811` | 14節 |
| CLI（`--persona neutral`・`--debug-prompts`・`--video-fps`・`--render-out`・`--game-mode`） | `pipeline.py:6578-6607` | 新節「CLI」 |

## 改訂の進め方（提案）
1. 構成を ①ライブ/動画の2本立て全体図 ②フェーズとイベント ③戦況トラッカー ④メッセージと技ログ ⑤実況生成（コンテキスト・ヒント） ⑥レンダ出力 ⑦CLI に再編
2. 閾値は値を直書きせず定数名で参照する（次に古くなるのを遅らせる）
3. ついでに `pipeline.py` 内の古いコメントも直す: `BattleMessageParser` の「x < 520」（`1024`付近）、`_update_move_log` の「x < 520」「全OCRトークン」（`6072-6076`）、`PipelineRunner` コメント（`1318`）。コメントのみなので挙動は変わらない

## 要確認（コードからの推論・実機未確認）
- ボールモデル無しだと `balls_remaining` は常に空のはず（`_build_game_state`・`yolo_detector.py:351-358`）。
  その場合 `BattleStateTracker.update()` の「ボール数減少による気絶確定」（`pipeline.py:1858-1870`）は死に経路になっている可能性が高い
