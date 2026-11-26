# パイプラインロジック概要

**対象ファイル**: `src/pipeline.py`（約6600行）と、そこから呼ばれる `src/capture/`・`src/output/render_sink.py`・`src/analytics/situation_warehouse.py`・`src/commentary/phi3_client.py`・`src/pokedb/` の各モジュール
**作成日**: 2026-04-19
**全面改訂**: 2026-11-26（対象 commit: `7fcd18c`）。改訂前の突き合わせ結果は `docs/backlog/pipeline-logic-overview-audit.md`
**目的**: パイプラインが「1フレームで何をして、どの条件でイベントを出し、何を根拠に実況文を作るか」を、コードを開く前に机上で追えるようにする

## この文書の読み方

- 閾値は定数名で書き、現在値を括弧に添える（例: `MSG_X_MAX`（1150））。値を変えたら定数側だけ直せば済むようにするため
- 場所は `ファイル:クラス.メソッド` で示す。行番号はすぐずれるので、シンボルが無い箇所だけに使う
- 座標はすべて 1920x1080 基準
- 実況動画の制作手順（パス0〜2・ネタバレ検査・アバター合成など）はここでは扱わない。手順と地雷リストは `.claude/skills/video-commentary/SKILL.md` と `docs/manual/video-commentary-quickstart.md` を見ること。動画モードを主経路にした理由は `docs/adr/ADR-009-video-first-commentary.md`
- 「コード読解ベース・実機未確認」と書いた箇所は、コードを読んで推論しただけで実機ログでは確かめていない

---

## 1. 全体像

### 1.1 動作モードは実質3通り

`Pipeline.__init__` の引数の組み合わせで決まる。判定は `self._posthoc_mode = render_sink is not None and video_path is not None`。

| モード | 起動オプション | 実況文の生成タイミング | 画像 | 音声 |
|---|---|---|---|---|
| 動画・後付け生成（主経路・ADR-009） | `--input` ＋ `--render-out` | 動画を最後までスキャンした後に一括（`_generate_posthoc_commentary`） | 送らない（`_call_bedrock_text`） | WAV保存＋`manifest.jsonl` |
| ライブ | `--camera`（`--input` なし） | イベント検知のたびに即時（`_dispatch_commentary`） | 800x450 に縮小して送る（`_call_bedrock_vision`） | その場で再生 |
| 動画・即時生成 | `--input` のみ | ライブと同じ即時経路 | 送る | その場で再生 |

`--render-out` をライブで使うと即時経路のまま WAV を保存するが、`event_time` が実時間になるため警告が出る。ライブ特有の改修は ADR-009 で凍結中。

### 1.2 1フレームの処理順（`Pipeline.run`）

```
フレーム取得（動画: _video_frame_skip ごとにシーク / ライブ: OBS仮想カメラ）
 ├ YOLO detect（毎フレーム。既定ではモデル無しで空振り → 4.10節）
 ├ 終了画面 YOLO（バトル中・_MIN_BATTLE_DURATION(25.0)秒経過後・_END_SCREEN_CONFIRM(3)連続で確定）
 ├ 降参時の WIN/LOSE 画面待ち（保留中のみ毎フレーム OCR）
 ├ HPバーのピクセル解析（バトル中は毎フレーム: HpBarAnalyzer.analyze → update_pixel_hp）
 ├ 差分検出 or 定期OCR のときだけ以下
 │   ├ run_ocr（フルフレーム）
 │   ├ バトル中: メインOCRの構造化（名前蓄積・スロット割当・数値HP割当・パネル記録・battle_start保留の確定）
 │   ├ バトル中: 技ログ / 効果 / まもる / 外れ / ひるみ / 場のコンディション / メガシンカ / 交代退場 /
 │   │           状態異常アイコン / 特性エリア / BattleMessageParser
 │   ├ バトル前: 繰り出しメッセージから自分・相手の名前を事前キャッシュ
 │   └ BattlePhaseClassifier.detect → イベントがあれば品質チェック → _process_event
 ├ dense scan（move_used 後、メッセージROIだけ 2倍拡大OCR）
 └ 次フレームへ（dense scan 中は動画のスキップ幅を 1/5、ライブの待ちを 0.1秒に短縮）
```

- 差分検出は `screen_capture.py:DiffDetector`（`DIFF_THRESHOLD`（30）かつ `MIN_CHANGE_AREA`（1000））
- 差分が無くても定期OCRは走る。間隔は `_PERIODIC_OCR_INTERVAL_BATTLE`（1.5秒）／`_PERIODIC_OCR_INTERVAL_IDLE`（3.0秒）。この間隔だけは `time.perf_counter()`（実時間）で測っており、動画モードでも動画内時間ではない
- 以下、差分検出・定期のどちらで走ったフルフレームOCRも「メインOCR」と呼ぶ。コード中のコメントではこれを「定期OCR」と書いていることが多い
- 動画のサンプリングは `--video-fps`（既定2.0）から `_video_frame_skip = int(fps // video_fps)` を決める

### 1.3 時計とカウンタ

- 時計は `Pipeline._now()`。動画モードは動画内時刻（`frame_pos / fps`）、ライブは `time.time()`。`BattlePhaseClassifier` と `BattleMessageParser` にも同じ時計を注入しているので、デバウンス等の秒数は動画でも本番と同じ尺度で効く
- カウンタは3つあり、混同しやすい

| 名前 | 増えるタイミング | 用途 |
|---|---|---|
| `run()` 内のローカル `turn` | `_process_event` を呼ぶたび | デバッグ画像の連番だけ |
| `BattleStateTracker.turn` | `update()` 1回ごと（内部イベントカウンタ） | 新規登録ヒステリシス・HP の鮮度スタンプ（`hp_turn`/`hp_px_turn`） |
| `BattleStateTracker.game_turn` | `turn_start` 処理時（faint 統合で繰り上げもある） | 実ゲームのターン数。技ログの `T{n}:`、場のコンディション残りターン、場からの降ろし判定 |

### 1.4 使っている外部コンポーネント

| 役割 | 実体 | 備考 |
|---|---|---|
| OCR | EasyOCR（`init_reader`） | `~/.EasyOCR/model/pokemon_g2.pth` があればそれを `["ja"]` で使う。無ければ `japanese_g2` を `["ja","en"]` で使う |
| 終了画面検出 | YOLO `--end-model` | 現役は `train_end_screen2` |
| 状態異常・ボール検出 | YOLO `--model` / `--ball-model` | 既定 None。パイプライン未使用（4.10節） |
| 名前・技の分類 | `PokeClassifier(game_mode=...)` | DB が無ければ None になり手動フィルターで動く |
| 実況文（主） | Bedrock（EC2 の `POST {ec2_url}/api/vision`） | 現行モデルは `server.py` の `BEDROCK_MODEL_ID`。`--ec2-url` 未指定なら呼ばない |
| 実況文（フォールバック） | `Phi3Client` | クラス名は Phi-3 のままだが、実体は Ollama の `gemma2:9b`（`phi3_client.py:MODEL_NAME`） |
| 音声 | VOICEVOX（`--speaker` 既定2） | |

EC2 側の `/api/log` と S3 連携は 2026-11-26 に廃止した（ADR-005）。パイプラインから EC2 に送るのは `/api/vision` だけ。

---

## 2. OCR と構造化

### 2.1 OCR 実行（`screen_capture.py:run_ocr`）

- 出力は `[{"text", "confidence", "bbox"}, ...]`
- フルフレームOCRが基本。`preprocess_hp`（相手HPのROIを黒塗りして個別OCR）は残っているが、パイプラインからは使っていない
- `preprocess_dense=True` は dense scan 用で、2倍に拡大（CUBIC）してから読み、bbox を元のスケールに戻す。docstring には「白テキスト抽出」とあるが、実装は拡大だけ（`_DENSE_THRESH` は未使用）
- 信頼度の足切りは処理ごとに違う

| 処理 | 閾値 |
|---|---|
| 構造化（HP・名前）・フェーズ分類・ログ表示 | 0.4 |
| メッセージROI（`BattleMessageParser`・技ログのスキャン） | 0.35 |
| 特性エリア・交代退場・場のコンディション | 0.3 |
| 技ログの使い手探索・後付け修正 | 0.25 |
| 通信待機の判定（`_COMM_CONF_MIN`） | 0.0（正規表現の一致を信頼する） |

### 2.2 HP テキストの正規化（`_extract_structured_info` 内の `_normalize_hp_text`）

| 補正 | 内容 |
|---|---|
| O→0 | `O`・`o` を `0` に置換 |
| {→8 | `{` を `8` に置換 |
| スラッシュ誤読 | `^(\d{1,3})[17](\d{2,3})$` にマッチし、かつ分子 ≤ 分母のときだけ `XX/YY` に直す |

正規化の後、`X/Y` 形式は次の2条件を満たすものだけ HP として採用する。

- 分母が `_HP_MIN_DENOM`（50）以上。PP（最大40）を HP と誤認しないため
- 分子 ≤ 分母。`787/201` のような誤読を捨てる

```
"1557155" → "155/155"（スラッシュが7に化けた）
"787/201" → 不採用
"{0%"     → "80%"
"8O%"     → "80%"
"12/40"   → 不採用（PP とみなす）
```

### 2.3 HP の自分・相手振り分け

| 形式 | 条件 | 振り分け先 |
|---|---|---|
| `X/Y` | `cy < _PLAYER_Y_THRESHOLD`（500） | 相手 |
| `X/Y` | それ以外（bbox 無しも含む） | 自分 |
| `XX%`（チャンピオンズの相手HP） | `cy < _PLAYER_Y_THRESHOLD` かつ 100 以下 | 相手。自分側の `%` は捨てる |

- 「相手を見る」パネル（「戦闘中」「たたかえる」「たたかえない」を含む）と技の詳細パネル（「能力」と「ステータス」が両方ある）の表示中は、`%` 形式と名前候補の収集を止める。`X/Y` 形式だけはパネル中も集める
- 出力は y 順の `hp_values_player/opponent`（各2件）、x 順の `hp_player_by_slot`/`hp_opponent_by_slot`（0=左, 1=右）、位置ゲート用の `hp_*_with_xy`（各4件）
- `COMMAND` という UI 文字列（OCR 揺れ対応の正規表現）の y 座標を `command_cy` として返す。行動選択・技選択画面では約377、対象選択・交代選択では約129。4.5節の HP 割当ゲートに使う

### 2.4 ポケモン名候補の振り分け

| 条件 | 振り分け先 |
|---|---|
| `cy < _PLAYER_Y_THRESHOLD`（500） | 相手の名前候補 |
| `cy > _PLAYER_NAME_Y_MIN`（930） | 自分の名前候補 |
| 500 ≤ cy ≤ 930 | どちらにも入れない（メッセージボックス・選出画面・パネルの帯） |
| PokeClassifier 無しで `cy > _COMMAND_Y_MIN`（700） | 捨てる（コマンドメニュー対策）。つまり分類器無しでは自分側の名前が取れない |

この前に次のものを落とす。

- `Lv` 始まり、数字だけ、`_UI_WORDS`・`_UI_OVERLAY_WORDS`・`_BATTLE_RESULT_WORDS` に一致または部分一致するもの、`_UI_OVERLAY_SUBSTRINGS`（通信待機・待機中・通信中）を含むもの
- 4文字未満の英数字、末尾が「の」「「」「!」のもの、「タイプ」を含むもの
- 「ガブリアス」のような鉤括弧は外してから判定する

PokeClassifier があるときは、カテゴリが pokemon のものだけを採り、正規化名（`canonical_ja`）で登録する。例外として、相手エリアで分類不能・2〜8文字・漢字を含みかなを含まない文字列は「中国語設定の相手」の名前とみなして素通しする。ただし `_CJK_UI_WORDS`（能力・特性・持ち物 など）は除く。分類器が無いときは `_MOVE_ABILITY_WORDS` で手動除外する。

`--game-mode champions` を付けると、PokeClassifier の照合対象が `champions_pokemon` テーブルの許可リストに絞られる。

名前と HP のペアは `_proximity_pair`（ユークリッド距離の近傍マッチ）で作る。

### 2.5 `game_state` の組み立て（`_build_game_state`）

イベント処理時に、上の構造化結果に次を足した辞書を作る。

- `status`: YOLO の状態異常（既定ではモデル無しなので常に「なし」）
- `balls_remaining`: YOLO のボール数（既定では常に空リスト）
- `ocr_text`: `_ocr_results_to_text`。conf 0.4 以上を ` / ` 区切りで結合し、最大 `OCR_MAX_CHARS`（120）文字。分類器があれば、助詞を含まない単独の技名・特性名・道具名を除く
- `ability_msg_player` / `ability_msg_opp`: 5.3節の特性エリアの直近テキスト
- `command_cy`

---

## 3. フェーズとイベント

### 3.1 フェーズ分類（`BattlePhaseClassifier.classify`）

conf 0.4 以上のテキストを対象に、上から順に判定する。`battle_end` だけはトークンを連結した文字列でも照合し、「勝負に」「勝った！」の分割を吸収する。

| 順 | フェーズ | 条件 |
|---|---|---|
| 1 | `battle_end` | `_END_KW`: 勝負に勝 / 勝負に負 / 降参が選ばれ / 通信エラー / 切断されました / 更新されるまで |
| 2 | `communication` | `_COMM_RE`（`通.?[待侍].中` / `^待機中$` / `通信中`）。「マッチング」を含むものは除く |
| 3 | `selection_screen` | `_SELECTION_KW`: ポケモンを選んで / 選出 / きめる / リーダー / 選出順 |
| 4 | `faint` | `_HP_ZERO_RE`: `0/(50以上)` または `0%` |
| 5 | `switch_select` | `_SWITCH_KW`: こうたい / ポケモンをえらんで / 交代する |
| 6 | `command_select` | `_COMMAND_KW`: たたかう / どうする |
| 6' | `command_select` | `_TECH_SELECT_KW`: こうかあり（技選択UIが開いている） |
| 7 | `animation` | `_ANIM_KW`: バツグンだ / きゅうしょ / 急所 / ひんし |
| 8 | `unknown` | どれにも当たらない |

`communication` はチャンピオンズの「全員のコマンド確定待ち」画面で、これが終わった瞬間が行動開始の合図になる。`_smooth_communication` で平滑化している。

- 入場: 連続 `_COMM_ENTRY_SEC`（0.7秒）検出で確定。確定前は `unknown` を返す
- 退場: 最後に見てから `_COMM_EXIT_GRACE_SEC`（3.0秒）は `communication` を維持する
- 退場確定のフレームは必ず `unknown` を返す。`communication → command_select` と直接遷移すると turn_start の条件に引っかかるため

### 3.2 イベント発火（`BattlePhaseClassifier.detect`）

`prev`（前フレームのフェーズ）と `curr` の遷移で決める。

| イベント | 条件 |
|---|---|
| `battle_start` | `curr == command_select` かつ未開始（`_battle_started` が False）。同時に `_allow_turn_start` を立てる |
| `turn_start` | `curr == command_select`・開始済み・`prev` が command_select / selection_screen / communication 以外・`_allow_turn_start` が真 |
| `move_used` | `prev == communication` かつ `curr` が communication / battle_end 以外、かつ開始済み |
| `faint` | `curr == faint` かつ `prev != faint`、かつ前回 faint 表示から `_FAINT_REARM_SEC`（20.0秒）以上途切れていた |
| `switch` | `curr == switch_select` かつ `prev` が switch_select / communication 以外 |
| `battle_end` | `curr == battle_end` かつ `prev != battle_end`。`_battle_started` と `_allow_turn_start` を落とす |

補足:

- `selection_screen` に入ると `_battle_started` と `_allow_turn_start` を落とす
- `_allow_turn_start` は move_used（と battle_start）で立ち、turn_start で落ちる。ダブルバトルで1匹目→2匹目のコマンド選択に移るときの余計な turn_start をこれで止める
- 脱出弁: move_used を取りこぼすと `_allow_turn_start` が落ちたまま固まる。そこで、新しいコマンドセッション（`_CMD_SESSION_GAP_SEC`（10.0秒）以上の空白の後に出たコマンド画面）の開始から10秒以内（ここは定数化されていない）で、前回 turn_start から `_TURN_START_ESCAPE_SEC`（15.0秒）以上経っていれば、遷移条件とデバウンスを無視して turn_start を出す
- `_is_processing` 中（`_process_event` 実行中）は battle_end の割り込みだけを受け付ける

デバウンス（同じイベントの再発火を止める秒数）:

| イベント | 秒数 |
|---|---|
| `turn_start` | `_DEBOUNCE_OVERRIDES["turn_start"]`（15.0） |
| `move_used` | `_DEBOUNCE_OVERRIDES["move_used"]`（10.0） |
| `faint` | `_DEBOUNCE_OVERRIDES["faint"]`（25.0）。上のエッジトリガ・再アームと併用 |
| `switch` | 既定の `debounce_seconds`（10.0） |
| `battle_start` / `battle_end` | なし |

`_process_event` の後（turn_start 以外）は `reset_after_processing` が `_prev_phase` を `unknown` に戻し、move_used のデバウンス時刻を今に更新する。

### 3.3 Pipeline 側のゲート（`Pipeline.run`）

`detect` が返したイベントは、さらに次の条件でふるいにかける。

1. `battle_end` はバトル中（`_battle_active`）だけ有効
2. 降参の WIN/LOSE 待ち中はフェーズ由来の `battle_end` を捨てる（二重発行防止）
3. `battle_end` の後 `_BATTLE_START_COOLDOWN`（10.0秒）は `battle_start` を捨てる
4. `move_used` なら、品質チェックより前に dense scan を始める（5.5節）
5. 品質チェック: `battle_end` 以外は OCR 2件以上かつ `_is_battle_screen`（`_NON_BATTLE_KEYWORDS` を含まない）でなければ捨てる。このとき保留中の faint があり、捨てたのが move_used なら faint を単独送信する

品質チェックで捨てたイベントもデバウンス時刻と `_allow_turn_start` は `detect` の時点で更新済みになっている。

### 3.4 battle_end の経路

| 経路 | 条件 | 備考 |
|---|---|---|
| YOLO 終了画面 | `detect_end_screen` が `_END_SCREEN_CONFIRM`（3）回連続、かつ各フレームの OCR を連結して `_END_SCREEN_OCR_KEYWORDS`（勝った / 負けた / 選ばれました）が一致 | キーワード不一致なら誤検出としてカウントを戻す |
| フェーズ遷移 | 3.2節の `battle_end` | |
| 動画末尾 | 動画を読み終えた時点でバトル中なら合成発行 | `frame=None` で `_process_event` を呼ぶ |

降参の扱い: 「降参が選ばれました」には勝敗が書かれていない。そこで YOLO 経路でもフェーズ経路でも即発行せず、`_END_SCREEN_WIN_LOSE_TIMEOUT`（15.0秒）まで毎フレーム OCR して次を待つ。

1. 「〜との勝負に勝った/負けた」（`_detect_battle_result`）。相手が降参した場合はこちらが先に出ることがある
2. WIN/LOSE ロゴ画面（`_detect_result_from_win_lose_ocr`）。左右どちらが自分かは、「〜との勝負に」から抜いた相手トレーナー名（`_extract_opponent_trainer_name`）との照合で決める

勝敗が分かればそれを持たせて、タイムアウトなら勝敗無しで battle_end を出す。降参で終わったことは `battle_surrendered` として実況とマニフェストに渡し、「全滅した」と言わせないようにしている。

### 3.5 battle_start 前後の扱い

- バトル開始前（`_battle_active` が False）はメッセージ解析をしない。代わりに全OCRから「くりだした / 繰り出した」（Dual 形式含む）と「ゆけ」を探し、PokeClassifier スコア 80 以上の名前を `_pre_battle_opponent` / `_pre_battle_player` にためておく
- `battle_start` を処理するとき（`_process_event`）は、`_reset_battle_state` でトラッカー・HPバー解析・技ログ・勝敗などを全部捨ててから、事前キャッシュと、`_SENDOUT_CARRYOVER_SEC`（25.0秒）以内のメッセージ由来の繰り出し履歴を登録し直す
- 遅延起動: battle_start を取りこぼしたまま `move_used`・`faint`・`switch` が来たら、その時点で同じリセットをしてバトルを開始扱いにする
- battle_start の実況保留: ダブルで片側だけ2匹揃い、もう片側が1匹以下（`_battle_start_roster_incomplete`）なら実況を保留する。確定のタイミングは次のどれか早いもの。`event_time` は検知時刻のまま変えない
  - メインOCRの時点で登録が揃ったとき
  - 次の `_process_event`
  - move_single / mega_evolution のディスパッチ
  - `run()` の終了時

---

## 4. 戦況トラッカー（`BattleStateTracker`）

### 4.1 データ構造

各陣営 `MAX_SLOTS`（4）匹までを `FieldPokemon` で持つ。場に出ているのは各陣営 `MAX_ON_FIELD`（2）匹まで（`_cap_on_field` で confidence の低い方を降ろす）。

| フィールド | 内容 |
|---|---|
| `name` | 正規化済みの名前 |
| `hp` / `hp_turn` | OCR の数値HP（`"62/135"` または `"45%"`）と、それを更新した `turn` |
| `hp_pct_pixel` / `hp_px_turn` | HPバーのピクセル解析値（0.0〜1.0）と、値が変わったときの `turn` |
| `status` | まひ / やけど / どく / もうどく / ねむり / こおり など |
| `moves_used` | 使った技（最大4件） |
| `on_field` / `fainted` | 場にいるか / 気絶済みか |
| `confidence` | 目撃回数（満杯時の追い出し判定に使う） |
| `last_seen_turn` | 最後に見た `game_turn` |
| `slot_index` | 画面上の位置（0=左, 1=右, None=未割当） |
| `mega_evolved` | メガシンカ済みか |

トラッカー全体では、ほかに `_event_log`（`MAX_EVENTS`（8））、`_turn_history`（`MAX_TURN_HISTORY`（8）・turn_start ごとのスナップショット）、場のコンディション（4.8節）を持つ。

### 4.2 登録経路と信頼度（`_get_or_create`）

| 経路 | 信頼度 | 例 |
|---|---|---|
| メッセージ由来（繰り出し・ゆけっ・気絶・技ログからの校正） | 高（`low_trust=False`） | `register_opponent_on_field`・`mark_on_field_by_name`・`register_opponent_fainted` |
| メインOCR・イベント時の名前候補 | 低（`low_trust=True`） | `update()` の名前蓄積・`accumulate_player_name` |

`_get_or_create` は次の順で同一個体を探し、無ければ作る。

1. 完全一致
2. 前方一致の吸収（`_ABSORB_MIN_LEN`（3）文字以上。「リザード」と「リザードン」は同一とみなし、長い方の名前に揃える）
3. 濁点ゆらぎの吸収（`_normalize_dakuten` で清音化して一致すれば同一。既存の表記を残す）
4. 低信頼経路なら新規登録ヒステリシス: `_NEW_NAME_CONFIRM_COUNT`（2）回目撃するまで保留。`_NEW_NAME_PENDING_TTL`（8）`turn` 空いたら目撃数をリセット
5. 空きがあれば新規登録
6. 満杯なら、高信頼経路に限り、場にいない・未気絶・confidence 最少のスロットを幽霊とみなして追い出す

### 4.3 場の出入り

- `update()`（イベントごと）: その時点の名前候補に見えたものは場に出す（気絶済みは除く）。相手側に登録済みで相手エリアに見えていない名前が自分エリアに出たら、y 座標の誤分類とみなして無視する（逆も同様）
- 降ろす条件:
  - 自分側は `game_turn - last_seen_turn > _ON_FIELD_MISS_THRESHOLD`（3）
  - 相手側は上の条件に加えて、「今フレームで相手名は見えているのに自分は見えず、かつ未知の新しい名前が出ている」場合に即座に降ろす
- HPバーが読めている間は `update_pixel_hp` が `last_seen_turn` を更新するので、名前OCRが少ない個体でも場から落ちにくい
- メッセージ由来の退場は 5.2節・5.3節を参照

### 4.4 スロット番号

- `_assign_slot_indices`: 未割当で場にいるポケモンを、名前の cx の相対順で左右に振る。候補1匹で両スロットが空いているときは `_SLOT_BAR_CENTERS`（`hpbar_analyzer.slot_bar_centers()` から算出。自分 291.5/687.5、相手 1336/1732）に近い方へ入れ、`_SLOT_CX_TOLERANCE`（200）より離れていれば保留する
- 呼ばれるのは `update()`（battle_start 以外）と、メインOCRごとの `assign_slots_from_ocr`。後者は `command_cy` が 300〜450（行動選択画面）のときだけ割り当てる
- 場を離れたポケモンの `slot_index` は割当の前に `_release_stale_slot_indices` で外す
- `update_pixel_hp` も、未割当が1匹だけで曖昧さが無い場合に限り、HPバーの位置からスロットを早期に割り当てる

### 4.5 HP

ピクセル解析（`src/capture/hpbar_analyzer.py:HpBarAnalyzer`）:

- 毎フレーム `analyze()` を呼び、各スロットの結果を `update_pixel_hp()` に渡す
- バー直上のネームプレート帯の色が `_ANCHOR_MIN`（0.4）以上見えないときは HUD 非表示とみなす
- 左端から最初の着色セグメントの幅を満タン幅で割る。幅が `_MIN_SEG_WIDTH`（15px）未満、または着色密度が `_SEG_DENSITY_MIN`（0.85）未満なら捨てる。15px 未満は読めないので、およそ HP 6.6% 未満はピクセルでは取れない
- 安定化: 直近 `_STABILITY_WINDOW`（6）サンプルの幅が `_STABILITY_DELTA`（0.02）以内に収まったら確定する。読めないフレームでは最後の確定値を返し続ける
- スロットの占有者が替わったら `_claim_slot` が `slot_reset_cb`（= `HpBarAnalyzer.reset_slot`）を呼んで安定化状態を捨て、そのサイクルの値は使わない。バトル開始時は全スロットをリセットする

OCR の数値HPは2経路ある。

| 経路 | ゲート | 確定条件 |
|---|---|---|
| `update()`（イベント時） | `command_cy` が 300〜450（関数内ローカルの `_CMD_CY_MIN`/`_CMD_CY_MAX`） | なし（その場で代入）。faint イベント以外は `0/X`・`0%` を代入しない |
| `assign_hp_from_ocr`（メインOCRごと） | 表示位置。自分は `cy ≥ _NUM_HP_PLAYER_Y_MIN`（950）、相手は `cy ≤ _NUM_HP_OPP_Y_MAX`（200）、かつバー中心から `_SLOT_CX_TOLERANCE` 以内 | 同じ読み値が `X/Y` は `_OCR_HP_CONFIRM_XY`（2）回、`%` は `_OCR_HP_CONFIRM_PCT`（3）回連続。未読のサイクルでは保留を消さない。1桁%は既知HPが25%以下のときだけ受け付ける |

表示（`_format_pokemon`・`_display_hp_pct`）は鮮度比較で決める。`hp_pct_pixel` があり、かつ `hp` が無いか `hp_px_turn > hp_turn` ならピクセル値（`HP:45%(px)`）、そうでなければ OCR 値。25%以下には `★ピンチ` を付ける。ピクセルの値は変わったときにしか `hp_px_turn` を更新しないので、止まった古いピクセル値が新しい数値HPを隠し続けることはない。

### 4.6 状態異常

| 経路 | 内容 |
|---|---|
| アイコン文字の OCR（`Pipeline._sync_status_from_ocr_bbox`） | 状態語（まひ・やけど等）単体のトークンが `_STATUS_ICON_AREAS` の4矩形（陣営×スロット）に入れば、そのスロットに付与（`update_status_from_yolo`。名前は YOLO だが OCR 経路でも使う） |
| メッセージ（`BattleMessageParser` の status → `update_status_by_name`） | 「〇〇は まひ…」や漢字形式（眠って・凍りつ・麻痺 など） |
| 技からの推定（`apply_status_from_move`） | `_STATUS_MOVE_TABLE`（おにび・どくどく・どくのこな・キノコのほうし・さいみんじゅつ・うたう）の技が技ログに載ったら、相手陣営の場にいる未状態異常の全員に付与（ダブルでは対象が特定できないため） |
| YOLO アイコン | `--model` 未指定の既定ではモデルが無く、何も出ない |

### 4.7 気絶

気絶フラグが立つ経路は次の3つ。

1. 「たおれた」メッセージ（`confirm_player_faint_by_name` / `confirm_opponent_faint_by_name`）。HP を `0/最大` または `0%` に書き換える。相手側でロスターに居ない場合は、PokeClassifier スコア 90 以上のときだけ `register_opponent_fainted` で登録と同時に気絶にする
2. faint イベント時の `update()`。HP が `0/` か `0%` のスロットを気絶にする
3. ボール数の減少（既定では動かない。4.10節）

相手側で気絶済みのスロットが OCR で再検出されたら、誤判定とみなして気絶を解除する。

### 4.8 場のコンディション

開始ターン（`game_turn`）だけを記録し、参照するたびに残りターンを逆算する（`_turns_left`）。

| 種類 | 検出（`Pipeline._update_battle_conditions`・メッセージROI内のみ） | 持続 |
|---|---|---|
| 天候 | `_WEATHER_KEYWORDS`（技名・特性名の両方） | `_WEATHER_DURATION`（5）。特性由来（`_WEATHER_ABILITY_KEYWORDS`）は終了ターン無しで継続扱い |
| 壁 | `_SCREEN_KEYWORDS`（リフレクター・ひかりのかべ・オーロラベール） | `_SCREEN_DURATION`（5） |
| トリックルーム | `_TRICK_ROOM_KEYWORD` | `_TRICK_ROOM_DURATION`（5） |
| おいかぜ | `_TAILWIND_KEYWORD` | `_TAILWIND_DURATION`（4） |

壁とおいかぜの陣営は、メッセージROI内に「あいて」「相手」があるかで決める。道具による延長（ひかりのねんど等）は画面から分からないので考慮していない。

### 4.9 出力（`to_context` / `to_panel_state`）

`to_context()` は LLM に渡す戦況サマリー（battle_context の土台）を返す。

- `player_field` / `opponent_field`: 場の各ポケモンを `_format_pokemon` で整形したもの（無ければ「情報収集中」）
- `player_bench` / `opponent_bench`: 控え。気絶は `名前(ひんし)`
- `player_pokemon` / `opponent_pokemon`: `場: … / 控え: …`（server.py 互換）
- `player_names` / `opponent_names`: confidence 順の全登録名（RAG 用）
- `turn`、`event_log`、`turn_history`、場のコンディション（`weather`・`weather_turns_left` または `weather_is_ability`・`screens`・`trick_room_turns_left`・`tailwind`）

`to_panel_state()` は戦況パネル用のスナップショットを返す。場のポケモン（スロット順・各最大2）の `name`/`hp_pct`/`hp_text`/`status` と、`turn`・`alive_player`・`alive_opponent` を持つ。後者はボール数が取れないので、既定では「把握済みの未気絶数」になる。

### 4.10 ボール数ロジック（死に経路の確認）

**結論（コード読解ベース・実機未確認）: 既定構成（`--ball-model` も `--model` も未指定）では、ボール数に関わる処理はすべて動いていない。**

根拠:

1. `Pipeline.__init__` は `YoloDetector(..., enable_pretrained_fallback=False)` で作るので、`--model` 未指定なら `_model` は None、`--ball-model` 未指定なら `_ball_model` も None
2. `YoloDetector.detect()` はどちらのモデルも無いと検出が空になり、`player_balls`/`opponent_balls` は `BallCount()`（alive=0）のまま。`detect_balls()` もボールモデルが無ければ即座に空の `BattleState` を返す
3. そのため `_last_ball_yolo` は一度もセットされず、`_build_game_state` の `balls_remaining` は常に `[]` になる
4. `BattleStateTracker.update()` では `cur_o_alive` が常に None になり、次の処理が全部素通りする
   - 「ボール数減少＋今回降ろした相手 → 気絶確定」
   - 「把握数 > ボール数 → 超過分を気絶確定」
   - `_player_alive_count` / `_opponent_alive_count` の更新と、faint 時の減算
5. 連鎖して `to_context()` の控えの「不明×N」補完も出ない。`to_panel_state()` の残り頭数は把握済み数にフォールバックし、Bedrock に送る `balls_remaining_player/opponent` は常に `"?"` になる
6. YOLO の状態異常（`game_state["status"]`）も常に「なし」なので、`update()` 内の YOLO 由来の状態異常更新も動かない

一方で、合成 faint（6.4節）は死んでいない。`_track_new_faints` は `fainted_names()` と実況済み集合の差分を見ているだけで、気絶の根拠はメッセージ（4.7節の1）と faint イベント時の HP=0（同2）で足りるからだ。`_dispatch_faint_inferred`・`faint_focus`・`update()` のコメントには「ボール数推定」と書かれた箇所が残っているが、既定構成で実際に効いている根拠はメッセージ由来の方になる。

ボール検出・状態異常アイコンの YOLO は再学習しない方針（2026-07-30決定）なので、このロジックを生かすことは当面ない。消すかどうかは別途判断すること。

---

## 5. メッセージと技ログ

### 5.1 `BattleMessageParser.parse`

メッセージボックスROI（`MSG_X_MIN`（120）≤ cx < `MSG_X_MAX`（1150）、`MSG_Y_MIN`（740）< cy < `MSG_Y_MAX`（930）、conf 0.35 以上）のトークンを、y を40px単位に丸めた行順→x順で結合し、`_normalize_ocr_message`（「バッグンだ」→「バツグンだ」）をかけてから解析する。`MSG_X_MAX` は「(トレーナー名)は (称号) (ポケモン名)を 繰り出した!」が右へ伸びる分を見込んで 900 から広げた値。

| イベント | パターン | 補足 |
|---|---|---|
| `opponent_faint` / `faint` | `_FAINT_RE`（〜は/が たおれた） | 「相手の」系プレフィックスがあれば相手。崩れた「あい 手の」も名前の直前10文字で相手と判定する。自分側の候補が直近の `opponent_faint` と部分一致したら、ミラー戦の再読とみなして捨てる |
| `switch_in` | `_SWITCH_IN_RE`（〇〇、ゆけ / ゆけ… 〇〇！ / 〇〇が とびだした） | 「ゆけ」の後ろは文末記号まで丸ごと取り、空白区切りの最後のトークンを種族名にする（二つ名対策・2026-09-07） |
| `opponent_switch_in` | `_DUAL_OPPONENT_SWITCH_IN_RE`（AとBを〜）を先に試し、だめなら `_OPPONENT_SWITCH_IN_RE` | 末尾は `_KURIDASHITA_SFX`: くりだした / くゆだした / 繰り出した / 繰ゆ出した / 「繰」＋最大3文字で文末（下端見切れ対策） |
| `switch_out` | もどれ、〇〇 / 〇〇と こうたいした / 〇〇 戻れ | 自分側の交代として扱う |
| `opponent_switch_out` | 〇〇を 引っこめた / 引込めた | 相手側の交代 |
| `status` | `_STATUS_RE` | ROI内のテキストに加えて全OCRも検索する。漢字形式は `_STATUS_KANJI_MAP` でひらがなに直す |

同じ（種別, 名前）は `DEDUP_TTL`（8.0秒）以内なら出さない。

### 5.2 メッセージイベントの反映（`Pipeline._handle_message_event`）

| イベント | 処理 |
|---|---|
| `faint` | `confirm_player_faint_by_name` |
| `opponent_faint` | 分類器で正規化してから `confirm_opponent_faint_by_name`。未登録ならスコア90以上のときだけ `register_opponent_fainted` |
| `switch_in` | カテゴリ pokemon のときだけ正規化して `mark_on_field_by_name`（自分側だけ探す）。`_note_sendout("player", …)` |
| `opponent_switch_in` | 同様に正規化して `register_opponent_on_field`。`_note_sendout("opponent", …)` |
| `switch_out` / `opponent_switch_out` | `mark_bench_by_name(side="player" / "opponent")`。陣営を限定してミラー戦の誤ベンチ化を防ぐ |
| `status` | `update_status_by_name` |

`_note_sendout` は2つの履歴に書く。`_recent_sendouts`（60秒で刈り込む・battle_start 時の引き継ぎ用）と、`_sendout_history`（動画全体を保持・後付け生成の交代ヒント用。同じ繰り出しの15秒以内の重複は1件にまとめる）。

### 5.3 その他のメッセージ検出（特記が無いものはメインOCRと dense scan の両方で動く）

| メソッド | 検出対象 | 記録先・用途 |
|---|---|---|
| `_update_switch_out` | 「戻って」を含む＋末尾「は」のトークン（とんぼがえり等）／「戻れ」の直前4トークンを分類（コマンド交代） | 前者は `set_not_on_field`（場に同名が2匹以上なら何もしない）、後者は自分側の `mark_bench_by_name`。メインOCRのみ |
| `_update_move_effectiveness` | 「バツグンだ」（`_EFFECTIVENESS_TAGS`） | 直近の技ログ1件に「（バツグン）」タグ |
| `_update_protect_history` | `_PROTECT_MSG_RE`（〇〇は…身を守った） | `_protect_history`。`_PROTECT_DEDUP_SEC`（3.0）で重複除去 |
| `_update_miss_history` | `_MISS_MSG_RE`（外れた / あたらなかった） | `_miss_history`（時刻のみ）。`_MISS_DEDUP_SEC`（3.0） |
| `_update_flinch_history` | `_FLINCH_MSG_RE`（〇〇は…ひるんで） | `_flinch_history`。`_FLINCH_DEDUP_SEC`（3.0） |
| `_update_battle_conditions` | 4.8節 | トラッカーの場のコンディション |
| `_update_mega_evolution` | `_MEGA_EVOLUTION_RE`（〇〇は…メガシンカ） | `mega_evolved` を立て、初回だけ `mega_evolution` 実況を出す。タイプ変化は `src/pokedb/mega_forms.py` に載っている種（現状リザードン・ミュウツー）だけ反映する |
| `_scan_ability_msg` | `_ABILITY_MSG_AREAS`（自分 x0〜555 / 相手 x1365〜1920、y450〜570）、conf 0.3 以上 | 直近値を `game_state` の `ability_msg_*` に載せる。メインOCRのみ |

まもる・ひるみの陣営は、名前の直前8文字に「相手」「あいて」があるかで決める。

### 5.4 技ログ（`Pipeline._update_move_log`）

技ログ `_move_log` は `T{ターン}:{使い手}の{技名}` 形式の文字列を最大 `_MAX_MOVE_LOG`（8）件持つ。ターンは dense scan の起点ターン（`_dense_scan_start_turn`）があればそれを、無ければ `game_turn` を使う。

スキャン対象はメッセージROI内（`cx < MSG_X_MAX`、`MSG_Y_MIN < cy < MSG_Y_MAX`、conf 0.35 以上）のトークンだけ。

1. **スキャン1（隣接トークン）**: 「Xの」なら次のトークン、「Xは」なら後続4トークン（`_HA_SCAN_WINDOW`）を技名候補にする。「は」の場合は `_MOVE_ALIAS_MAP`（手助けする→てだすけ、攻撃から身を守った→まもる 等）で変換し、末尾の助詞を落とす。直前トークンに「相手」「あいて」があれば相手側として扱う
2. **スキャン2（結合テキスト）**: ROIの結合文字列に `_MOVE_IN_MSG_RE` をかける
3. **スキャン2のフォールバック（「の」欠け）**: 技と判定できるトークンの直前トークンを使い手とみなす。直前が「〜を」なら対象なので捨てる。2文字の使い手は正規名の先頭一致を要求する

「相手の[技]」のように使い手が書かれていないときは `_try_register_opponent_attack` で使い手を探す。

- まず「イエてだすけ」のような略称＋技名の連結誤読を分割して試す
- 次に、全OCR（dense scan 時はメインOCRのキャッシュも含む）の cy > 600 にある「Xの / Xは」から使い手を探す
- 見つからなければ、場の相手が1匹だけならその1匹で仮確定（tentative）する。2匹いれば登録しない

登録の検証（`_try_register`）は次の順に進む。

- 使い手は分類器でスコア80以上のポケモンに正規化する。だめならロスター名との前方一致で救済し、そのときは仮確定扱いにする。どちらも失敗したら捨てる
- 相手側とされたが自分ロスターにしか居ない名前は自分側に直す。相手側でロスター未登録なら、その場で相手ロスターに登録する（繰り出し取りこぼしの救済）
- 技名は `_normalize_ocr_kana`（チエ→チェ 等）の後に分類し、カテゴリ move かつスコア80以上、かつ候補の1.5倍より長くない名前だけ採る
- 次のどれかに当たれば仮確定にする: ロスター救済経由、`is_move_learnable` で覚えない技、`confident` でない（僅差の候補あり）
- 同ターン・同技は重複として捨てる。ただし場にいる別々の実在ポケモンの同時使用は残す

登録できたら次を行う。

- `render_sink.add_moment`（timeline.jsonl・陣営は `move_user_side`）
- `_dispatch_move_commentary`（move_single 実況。仮確定でも出す）
- `update_move`・`apply_status_from_move`
- dense scan の残りを最低90フレームに引き上げる

仮確定のエントリは `_tentative_opponent_moves` で追う。後のメインOCRで同じターン・同じ技の正しい使い手が見つかれば書き換え、表示（`_move_log_display`）では「（推定）」を付ける。

### 5.5 dense scan

- 開始: `move_used` を検知した時点で `_dense_scan_remaining = 9999` にし、`_dense_scan_start_turn` に現在の `game_turn` を記録する
- 停止: `turn_start`・`battle_start`・`battle_end` で0に戻す。技を検出したときは残りを最低90にする（move_used 以外の契機で技が出たときの延長）
- 内容: 毎フレーム `frame[MSG_Y_MIN:MSG_Y_MAX, 0:MSG_X_MAX]` を `preprocess_dense=True` で読み、bbox を元の座標に戻してから、技ログ・効果・まもる・外れ・ひるみ・場のコンディション・メガシンカの検出に通す。`BattleMessageParser` と交代退場・状態異常アイコン・特性エリアは通さない
- 「通信中」を含むフレームでは残りカウントを減らさない（OCR自体は実行する）
- dense scan 中は動画のスキップ幅を `max(1, _video_frame_skip // 5)`、ライブの待ち時間を0.1秒にする

---

## 6. 実況生成

### 6.1 イベント種別と発生元

`BEDROCK_EVENTS` は `battle_start`・`move_used`・`move_single`・`switch`・`faint`・`battle_end`・`mega_evolution` の7種。`turn_start` は `game_turn` を進めるだけで実況しない。

| 種別 | 発生元 | 通る処理 |
|---|---|---|
| `battle_start` / `move_used` / `switch` / `faint` / `battle_end` | フェーズ遷移（3.2節）・YOLO終了画面・動画末尾 | `_process_event` → `_dispatch_commentary` |
| `faint`（合成） | `_process_event` 内の `_track_new_faints` | `_dispatch_faint_inferred` → `_dispatch_commentary` |
| `move_single` | 技ログへの登録（5.4節） | `_dispatch_move_commentary` → `_dispatch_commentary` |
| `mega_evolution` | メガシンカの初回検出 | `_dispatch_mega_evolution_commentary` → `_dispatch_commentary` |

`move_single` と `mega_evolution` は `_process_event` を通らない。そのため、それぞれのディスパッチ関数が自前で battle_context を作り、保留中の faint と battle_start を先に確定させてから実況を出す（時系列を守るため）。

### 6.2 ディスパッチ（`_dispatch_commentary`）

- 後付けモード: イベントを `_pending_render_events` に積むだけ（`event_time`・`game_state`・`battle_context`・`move_log`・`render_context`）。生成は6.6節
- 即時モード:
  1. `attempt_bedrock`（`--ec2-url` あり・`BEDROCK_EVENTS` に含まれる・バトル中）なら `_call_bedrock_vision`
  2. 実況文が返らなければ `Phi3Client.generate_commentary`（`samples=1`）
  3. `_clean_commentary` → `_replace_glitch_commentary` → `_speak_async`
  4. 直近5件を履歴に残し、Bedrock には3件送る

### 6.3 保留と統合

**faint 保留**（`attempt_bedrock` が真のときだけ）: faint イベントはその場で実況せず `_pending_faint_*` に保留する。

| 契機 | 動作 |
|---|---|
| 次の `move_used`（経過 < `_FAINT_PENDING_TIMEOUT`（75.0秒）） | 統合する。保留時の場の状態を `faint_context` として move_used に足し、`faint_side` も引き継ぐ。保留時点から `game_turn` が進んでいなければ1つ繰り上げ、直後の turn_start を1回だけ飛ばす（`_skip_next_turn_start`）。繰り上げたら `_refresh_condition_hint` で残りターンを計算し直す |
| 次の `move_used`（経過 ≥ 75秒） | 先に faint を単独送信してから move_used を処理する |
| `turn_start`・新しい `faint`・`battle_end`・`move_single`・`mega_evolution`・品質不足で捨てた `move_used` | 保留 faint を単独送信する |

タイムアウトはタイマーで見ているわけではなく、move_used が来たときに経過時間を比べるだけ。単独送信（`_flush_pending_faint`）の時点で、気絶の対象（`faint_focus`）と `player_field`/`player_bench` 等を最新のトラッカー状態から作り直す。`event_time` は faint 検知時点のまま使う。

**battle_start 保留**: 3.5節。

### 6.4 気絶実況の重複対策と合成 faint

- `_announced_faints` は（陣営, 名前）のタプル集合。同名ミラー戦で片側の実況が反対側を握りつぶさないよう、2026-09-07 にタプル化した。試合ごとにリセットする
- faint イベントのときは、そのイベントの `update()` で新たに気絶した分だけを実況済みに登録する（差分が空なら現在の全気絶を登録）
- faint 以外のイベントのときは、「現在の気絶 − 実況済み」を陣営ごとの合成 faint として、現在のイベントより先にディスパッチする（`faint_focus` に対象名、`faint_side` に陣営を入れる）
- ただし直前の通常 faint イベントから `_FAINT_CATCHUP_SUPPRESS_SEC`（60.0秒）以内なら合成しない。`battle_end` はこの後に拾い直す機会が無いので、例外として抑制しない
- `faint_focus` は通常の faint イベントにも付ける。faint イベントのときは「現在の気絶 − イベント処理前の実況済み集合」、それ以外は `update()` 前後の差分で求める

### 6.5 LLM に渡す情報

Bedrock への payload は `context`（`_build_bedrock_context` が作る）・`history`（直近3件）・`battle_state`（= battle_context）・ライブなら `image_base64`。

`context` の主なキー:

- `event_type`、`ocr_text`、`hp_values`、`name_candidates_*`、`status_*`、`balls_remaining_*`（既定では常に "?"）、`detected_moves`（`_move_log_display(5)`）、`persona`
- `rag_pokemon_info`: 蓄積済みの名前ごとに図鑑DBのタイプ・特性。代表技は創作の元になるので渡さない
- `game_state` から写すもの: `faint_context`、`faint_focus`、`battle_result`、`battle_surrendered`、`move_focus`、`switch_focus`、`no_new_switch_hint`、`mega_evolution_focus`

`battle_context`（= `to_context()` ＋ 次のヒント）:

| キー | 作るところ | 中身 |
|---|---|---|
| `team_preview_hint` | `_load_team_preview_hint` | パス0で手入力した `<render_out>/team_preview.json` の両陣営の種族名。起動時に1回だけ読む |
| `type_hint` | `_compute_type_hint` | 場の攻守のタイプ相性（等倍は省略・最大4行）。直近の技ログの実際の技タイプでの相性（`_latest_move_type_hint`・ウェザーボールは天候で読み替え）を先頭に置く。防御側が2匹以上なら断定せず、候補を `_type_hint_candidates` に回す |
| `move_effect_hint` | `_latest_move_effect_hint` | 直近の技の効果テキスト（DB）。DBに無い溜め技は `_CHARGE_MOVE_NOTES` |
| `move_range_hint` | `_latest_move_target_type` | 直近の技の対象範囲（DB） |
| `condition_hint` | `_compute_condition_hint` | 天候・壁・トリックルーム・おいかぜの残りターンと、技ログ内の素早さ低下技（`_SPEED_STAGE_MOVES`）から計算した素早さランク変化 |
| `faint_side` | `diff_fainted_side` | faint で新たに気絶した陣営（"player"/"opponent"/None）。表情連動に使う |
| `battle_result` / `battle_surrendered` | battle_end のみ | 勝敗・降参 |

勝敗（`battle_result`）は battle_end のときだけ `game_state` に入れる。それより前のイベントに混ぜるとネタバレになるため。

### 6.6 後付け生成（`_generate_posthoc_commentary`）

`run()` の `finally` で、スキャンが終わってから呼ぶ。各イベントの context はスキャン中のその時点の値なので、未来の情報は混ざらない。例外は「そのイベント直後の観測」から作る次のヒントで、実況音声は事象の数秒後に流れるのでネタバレにはならないという整理になっている。

| 対象 | 追加するもの | 観測窓 |
|---|---|---|
| `move_single` | `move_target_hint`（`_compute_move_target_hint`）: 技の範囲（`_MOVE_RANGE_HINT_TEXT`）、まもる成功、外れ、ひるみ、HP減少（`_TARGET_HINT_MIN_HP_DROP`（3.0）ポイント以上）、状態異常付与。何も観測されなければ「断定しないこと」という否定ヒント。単体対象技で複数体に観測がまたがったら最も確かな1件だけ残す | 技の時刻から、次の move_single / move_used / switch / battle_end まで（最大 `_TARGET_HINT_MAX_WINDOW_SEC`（20.0秒）） |
| `move_single` | 技が防がれた・外れたなら `move_effect_hint` に「追加効果・反動は発生していない」と注記（`_voided_move_effect_hint`） | 同上 |
| `move_single` | `_type_hint_candidates` があれば、観測から推定した実際の対象（`_infer_primary_target_name`）の1件に `type_hint` を差し替える。決まらなければ `type_hint` を消す | 同上 |
| `switch` / `move_used` | `switch_focus`（窓内の繰り出しメッセージ）。その最後の繰り出し時刻まで `event_time` を繰り下げ、音声の出が早すぎるのを直す（次のイベント時刻は越えない） | 遡り `_SENDOUT_HINT_LOOKBACK`（switch 3.0 / move_used 5.0）、前方は `_SENDOUT_HINT_FORWARD_CAP`（switch 10.0 / move_used 20.0） |
| `switch` / `move_used` | `no_new_switch_hint`: `switch_focus` に出てこない陣営について「新しい交代は確認されていない」 | 同上 |
| `faint` / `move_used` | 控え欄の `(ひんし)` のうち、move_single の HP 0% 観測（`_HP_ZERO_HINT_RE`）や過去の faint で既に伝えた名前について `faint_repeat_hint`。faint イベントで全員が既報告ならそのイベントの生成自体を飛ばす | — |

その後、イベントごとに次を行う。

1. `_record_situation_snapshot`（6.8節）
2. `_call_bedrock_text`（画像無し）
3. 実況文が返らなければ Phi-3 フォールバック（`samples=3`）
4. `_clean_commentary`・`_replace_glitch_commentary`
5. VOICEVOX で同期合成して `render_sink.add`

最後に勝敗が分かっていれば戦況ウェアハウスに書き戻す。

**既知の不具合（コード読解ベース・実機未確認）**: `faint_repeat_hint` は `game_state` に入るが、`_build_bedrock_context` の出力キーに含まれていない。そのため Bedrock 側の `server.py` が `context.get("faint_repeat_hint")` で読もうとしても常に空になり、プロンプトに届かない。Phi-3 側も読んでいない。マニフェスト（`render_context`）には記録されるので、ログ上はヒントが付いているように見える。テスト（`tests/test_render_sink.py`）は `_call_bedrock_text` に渡す `game_state` までしか見ていないので検出できていない。全員既報告のときに生成を飛ばす処理は pipeline 側で完結しているので、こちらは効いている。

### 6.7 Bedrock 呼び出しと後処理

- `_call_bedrock_vision` / `_call_bedrock_text`: どちらも `POST {ec2_url}/api/vision`、タイムアウト15秒。違いは画像の有無だけ。失敗しても例外を出さず `(None, None)` を返し、Phi-3 に回る
- `_clean_commentary`: `---`・コードフェンス・【】ラベル・HTMLタグ・「指示/質問/注:」以降・行頭記号・鉤括弧を除き、最初の2文に切り、絵文字（U+1F300〜1FAFF）を除く
- `_replace_glitch_commentary`: 保留や困惑の応答（`_GLITCH_CAUSE_KEYWORDS`）を、キャラ口調の定型文（`--persona neutral` なら中立版）に差し替える。VOICEVOX 合成の前に行う

### 6.8 戦況ウェアハウス（`src/analytics/situation_warehouse.py`）

後付け生成のときだけ、`data/battle_situations.sqlite` の `situations` テーブルにイベントごとの状況（場・控え、天候、壁、トリル、おいかぜ、type_hint、HP）を記録する。`match_id` は元動画のファイル名（拡張子なし）。生成の前に同じ `match_id` の行を消し（`clear_match`）、最後に勝敗を `outcome` へ書き戻す（`backfill_outcome`）。いまは記録するだけで、読み出して判断に使う処理は無い。

---

## 7. レンダ出力（`src/output/render_sink.py:RenderSink`）

`--render-out <dir>` を指定すると作られる。同じ出力先で再実行すると、前回の `manifest.jsonl`・`timeline.jsonl`・`states.jsonl`・`fillers.jsonl`・`wav/*.wav` を先に消す。`team_preview.json` は消さない。

| ファイル | 書くところ | 中身 |
|---|---|---|
| `render_info.json` | `Pipeline.__init__` | 元動画パス・サンプリングfps・話者・作成日時 |
| `manifest.jsonl` | `RenderSink.add`（即時経路は `_speak_async`、後付けは `_generate_posthoc_commentary`） | 1行1実況: `seq`・`event_time`・`event_type`・`commentary`・`wav`・`duration`・`context` |
| `wav/NNNN_<event_type>.wav` | 同上 | VOICEVOX の音声 |
| `timeline.jsonl` | `RenderSink.add_moment`（技ログ登録時） | 実況にならなかった出来事も含む瞬間ログ: `time`・`kind`（現状 `move` のみ）・`text`・`side` |
| `states.jsonl` | `RenderSink.add_state`（`_record_panel_state`） | 戦況パネル用スナップショット（`to_panel_state()`＋`time`）。直前と同じ内容なら書かない |

`manifest.jsonl` の `context` は `_render_context` が作る。`move_log`・`turn`・`player`・`opponent` に加え、battle_context にあれば `faint_side`・`battle_result`・`battle_surrendered`・`type_hint`・`move_effect_hint`・`move_range_hint`・`condition_hint`・`team_preview_hint` を写す。後付け生成で足したヒント（`move_target_hint`・`switch_focus`・`no_new_switch_hint`・`faint_repeat_hint`）もここに書き込まれる。

`_record_panel_state` はメインOCRごととイベント処理後に呼ばれ、同じ内容をメモリ上の `_panel_state_history` にも積む。後付け生成の HP 減少・状態異常の観測はここから引く。

パス1.5以降（フィラー生成・予測→回収・合成）はこれらのファイルを読む別スクリプトなので、手順は SKILL.md を参照。

---

## 8. CLI（`src/pipeline.py:main`）

| オプション | 既定 | 内容 |
|---|---|---|
| `--camera` | 3 | OBS仮想カメラの番号（ライブ） |
| `--input` | なし | 動画ファイル。指定すると動画モード |
| `--render-out` | なし | 素材の出力先。`--input` と併用すると後付け生成モード |
| `--video-fps` | 2.0 | 動画のサンプリングレート |
| `--interval` | 1.0 | ライブのキャプチャ間隔（秒） |
| `--ec2-url` | なし | Bedrock を使う EC2 API の URL。無ければ常に Phi-3 フォールバック |
| `--end-model` | なし | 終了画面 YOLO（現役は `train_end_screen2`） |
| `--model` / `--ball-model` | なし | 状態異常・ボールの YOLO。既定で未使用（4.10節） |
| `--conf` | 0.5 | YOLO の信頼度閾値 |
| `--game-mode` | sv | `champions` で PokeClassifier をチャンピオンズの許可リストに絞る |
| `--persona` | kurepi | `neutral` で花圓くれぴの名乗り・口調を外す。当面は neutral 運用。Bedrock 経路に効かせるには EC2 側 server.py も同じ版である必要がある |
| `--debug-prompts` | off | Phi-3 フォールバックに送るプロンプト全文をログに出す。Bedrock 側は EC2 の環境変数 `DEBUG_PROMPTS=1` |
| `--speaker` | 2 | VOICEVOX 話者 |
| `--audio-device` | なし | 再生デバイス |
| `--cpu` | off | EasyOCR を CPU で動かす |

実況動画のパス1の実際のコマンド例は SKILL.md の「パス1」を使うこと（VOICEVOX の事前起動など注意点がまとまっている）。

---

## 9. チャンピオンズ対応の確認ポイント（2026-11-26 時点）

| # | ロジック | 状態 | 備考 |
|---|---|---|---|
| 1 | HP正規化（`[17]` スラッシュ誤読・`{`→8・分母50未満除外） | 実装済み | 2.2節 |
| 2 | 自分HPの y 分類（`cy ≥ 500` を自分） | 実装済み | 2.3節 |
| 3 | 自分の名前は `cy > _PLAYER_NAME_Y_MIN`（930） | 実装済み（2026-07-08） | 旧版の「cy ≥ 500」は誤り |
| 4 | メッセージROI拡張（`MSG_X_MAX`=1150） | 実装済み（2026-07-30） | 称号付きの繰り出し対策 |
| 5 | faint の `0%` 対応 | 実装済み | `_HP_ZERO_RE` |
| 6 | 繰り出しの漢字形式（繰り出した）と末尾見切れ | 実装済み | `_KURIDASHITA_SFX` |
| 7 | 通信待機中フェーズで move_used を判定 | 実装済み | 3.1節 |
| 8 | `command_select` キーワード（たたかう・どうする）／交代（交代する） | 実機で継続検出 | |
| 9 | 特性・道具エリア | 実装済み | 5.3節 |
| 10 | 状態異常アイコン（OCR 文字の4矩形判定） | 実装済み | 4.6節 |
| 11 | ボール検出 YOLO（`train7`） | 既定で未使用・再学習しない（2026-07-30決定） | 関連ロジックは死に経路（4.10節） |
| 12 | 状態異常アイコン YOLO（`train4`） | 既定で未使用・再学習しない（2026-07-30決定） | OCR経路で代替 |
| 13 | 終了画面 YOLO（`train_end_screen2`） | 稼働中 | P=0.989 / R=1.0 |
| 14 | HP の異常値フィルター（分子 > 分母を捨てる） | 実装済み | |
| 15 | HP 表示の鮮度比較（px と OCR の新しい方） | 実装済み | 旧版の「px 無条件優先」は廃止 |
| 16 | `command_cy` による OCR HP 割当ゲート | 実装済み | イベント経路とメインOCRのスロット割当だけ。メインOCRの数値HPは位置ゲート |
| 17 | `--game-mode champions`（許可リスト） | 実装済み（2026-07-30） | レギュレーション更新は `scripts/update_champions_roster.py` |
| 18 | 新バトルギミック（Zワザ・ダイマックス・テラスタル） | 未対応 | 画面文言が未確認。メガシンカだけ専用イベントあり |

---

## 10. 既知の制約・未確認事項

- ボール数・YOLO 状態異常のロジックは既定構成で動いていない（4.10節・コード読解ベース）
- `faint_repeat_hint` が Bedrock のプロンプトに届いていない疑い（6.6節・コード読解ベース）
- バトル開始前の「ゆけっ」事前キャッシュ（`run()` 内の `_YUKE_RE`）は旧パターン `(?:\S+の\s+)?(\S{2,12})` のまま。2026-09-07 の二つ名対応は試合中の `BattleMessageParser._SWITCH_IN_RE` にしか入っていないので、「の」で終わらない二つ名が付いた自分の先発は、開始前キャッシュでは二つ名側を拾う可能性がある。PokeClassifier のスコア80以上という条件で大半は弾かれるはず（コード読解ベース・実機未確認）
- 動画末尾で合成する battle_end は `frame=None` で `_process_event` を呼ぶ。最後の `cv2.imwrite(debug/pipeline_turn_*.png, frame)` は None を渡すと例外になるはずで、そうなると後付け生成は `finally` で走るものの、終了時にトレースバックが出る可能性がある（コード読解ベース・実機未確認）
- 名前ベースの陣営判定（`_find_slot` 等）は同名ミラー戦で取り違えうる。根本対応はフェーズ2候補として見送り中
- 場のコンディションの持続は道具の延長を考えない近似。素早さランクは技ログの直近8件だけで数える近似
