# バトルメッセージ検出の改修メモ

作成日: 2026-04-02（2026-07-09に一部の数値が後日変更されている旨を追記）

## 背景と目的

`scripts/ocr_logger.py` で対戦中の生OCRデータを複数戦分ロギングし、  
実際に画面に流れるメッセージパターンを確認した結果、以下の問題が判明した。

---

## 改修内容と理由

### 1. `turn_start` デバウンス 12s → 6s

**場所**: `BattlePhaseClassifier._DEBOUNCE_OVERRIDES["turn_start"]`

**問題**: ログを分析すると、1ターンが12秒未満で完結するケースが頻繁にあった。  
とくにとんぼがえり・Uターン・素早い交代が絡む展開では、コマンド選択が  
12秒以内に連続して現れ、2ターン目以降が取りこぼされていた。

**対応**: 6秒に短縮。6秒未満のターンは実務上ほぼ存在しないため誤発火リスクは低い。

> **⚠️2026-07-09追記**: この値はその後のターン検出の根本改修（通信フェーズ検出・脱出弁・faint再アーム等の導入）で完全に置き換わっている。現在の`_DEBOUNCE_OVERRIDES["turn_start"]`は**15.0秒**（`src/pipeline.py`）で、根拠も「ターン長分布」ではなく「動画内時間で実測したターン間ギャップ・turn_start間隔」に基づく別の設計になっている。最新のロジックは`docs/pipeline-logic-overview.md`を参照。

---

### 2. 繰り出しメッセージの正規表現修正（`_SWITCH_IN_RE`）

**場所**: `BattleMessageParser._SWITCH_IN_RE`

**問題**: 旧パターン `(.{2,12})、?\s*ゆけ` は「〇〇、ゆけ！」形式（名前が前）を想定していたが、  
SVの実際のメッセージボックスには「ゆけつ！ (げんきいっぱいの) トドロクツキ！」のように  
**名前がゆけの後ろに来る**形式で表示される。

ログ実例:
```
だっしゆつボタンで 戻つていく  ← 脱出ボタン
ゆけつ！ げんきいっぱいの トドロクツキ！
```

**対応**: 以下の3パターンに拡張。

```python
_SWITCH_IN_RE = re.compile(
    r'(.{2,12})、?\s*ゆけ'                  # 旧パターン（保持）
    r'|ゆけ\S*\s+(?:\S+の\s+)?(\S{2,12})'  # SVの実際の形式（名前が後）
    r'|(.{2,12})が\s*とびだした'             # とびだした形式
)
```

`(?:\S+の\s+)?` で「げんきいっぱいの」等の性格プレフィックスを読み飛ばす。

---

### 3. 相手繰り出し検出の追加（`_OPPONENT_SWITCH_IN_RE`）

**場所**: `BattleMessageParser._OPPONENT_SWITCH_IN_RE`（新規追加）

**問題**: 相手がポケモンを繰り出す際の「〇〇をくりだした！」メッセージが  
これまで一切捕捉されていなかった。相手の場のポケモンがメッセージベースで  
追跡できず、「情報収集中」のままになるケースが多かった。

**対応**:

```python
_OPPONENT_SWITCH_IN_RE = re.compile(r'(?:.{2,12}と\s*)?(.{2,12})をくりだした')
```

`switch_in` イベントとして emit し、`BattleStateTracker.mark_on_field_by_name`  
が自分・相手両チームを検索するため既存の処理で流用できる。

**ダブルバトルの制限**: 「AとBをくりだした」形式では B のみ捕捉。  
A 側の名前には「と」が混入するため、ダブル1体目は捕捉を諦めている。

---

### 4. `_emit` のポケモン名クリーニング強化

**場所**: `BattleMessageParser.parse()` 内の `_emit` ヘルパー

**問題**: OCRは「トドロクツキ！」「ゴリランダー、」のように末尾に  
感嘆符・読点が混入することがある。これが `_find_slot` の部分一致に  
影響する場合があった（完全一致優先パスで外れるケース）。

**対応**: `rstrip('！!」、')` で末尾ノイズを除去。

---

---

### 5. 技検出スキャン構造の刷新（汎用化）✅ 2026-04-05

**場所**: `PipelineRunner._update_move_log()` のスキャン1〜3

**問題**: てだすけ・まもる等「の」を使わない技は専用パターン（`_TEDASUKE_RE`・`_MAMORU_RE`）を個別追加するしかなく、技が増えるたびにコード追加が必要だった。根本原因はスキャン1が `[X]の[技]` ペアのみ対応で、`[X]は[技]` 形式に未対応だったこと。

**対応**:

1. **`_MOVE_ALIAS_MAP` 新設**（OCR変形表記 → 正規技名）
   ```python
   _MOVE_ALIAS_MAP: dict[str, str] = {
       "手助けする":         "てだすけ",   # 「相手のXはYを手助けする体勢に入った！」
       "手助けした":         "てだすけ",
       "手助け":             "てだすけ",
       "攻撃から身を守った": "まもる",
   }
   ```
   PokeClassifier が認識できない変形表記をここで補正する。新しい表記が出たら1行追加するだけで対応可能。

2. **スキャン1を「は」エンディングに対応拡張**
   - 「の」: 従来通り直後トークン1個を技名候補として試みる
   - 「は」: 後続4トークンまでウィンドウスキャン。`_MOVE_ALIAS_MAP` 変換を適用してから `_try_register` に渡す
   - PokeClassifier（score≥80・category=move）が誤登録を防ぐ

3. **削除したもの**: `_TEDASUKE_RE`・`_MAMORU_RE`・スキャン1.5・スキャン3・スキャン2内のまもる/てだすけ専用パターン

**検証結果**: T2:イエッサンのてだすけ・T3:リキキリンのでんじは（相手バドレックス）が安定して検出されることを複数ログで確認（221436.log にて T1〜T3 全技が参照と完全一致）。

---

### 6. 技検出後の dense scan 最低フレーム維持 ✅ 2026-04-05

**場所**: `_try_register()` 内の dense scan 再起動ロジック

**問題**: `まもる` 検出時に dense_scan=60 が起動するが、「まもるを」テキスト表示中（約12秒）に消費し尽くされ、ワイドフォース検出後のでんじは（約3秒後）がdense_scan=0で取りこぼされていた。

**対応**: 技登録成功時に dense scan を最低30フレーム（約3秒）に底上げ。まもるの60フレームは上書きしない（`elif ... < 30` の条件）。

```python
elif self._dense_scan_remaining < 30:
    self._dense_scan_remaining = 30
```

これにより「技A → 技B → 技C」の連鎖でも各技検出後に次の技メッセージ捕捉バッファが保証される。

> **⚠️2026-07-09追記**: 底上げ値はその後 **90フレーム** に引き上げ済み（`src/pipeline.py`のdense scan該当箇所）。30では不足するケースが後日見つかったための変更。

---

### 7. `_try_register` でのポケモン名正規化 ✅ 2026-04-05

**場所**: `_try_register()` の技名分類処理の前段

**問題**: OCR揺らぎで同一ポケモンが「イエッサン」と「イエツサン」に読み分けられ、entry 文字列が異なるため重複チェックをすり抜けて同ターン同技が2件登録されていた。

**対応**: 技名を classify する前にポケモン名も PokeClassifier で正規化する。

```python
p_result = self._classifier.classify(_normalize_ocr_kana(pokemon_name))
if p_result and p_result.category == CATEGORY_POKEMON and p_result.score >= 80:
    pokemon_name = p_result.canonical_ja or pokemon_name
```

副作用として `update_move(pokemon_name, ...)` にも正規化済み名前が渡されトラッカー精度も向上する。

---

## 今後の課題

### 脱出ボタン（だっしゅつボタン）の交代検出

ログに以下のパターンが確認された:
```
相手の(1.00,コマンド) / 愛管侍は(0.56,コマンド) / だっしゆつボタンで(0.60,コマンド) / 戻つていく」(0.95,コマンド)
```

「〇〇は だっしゅつボタンで 戻っていく」は `_update_switch_out` の  
「戻っていく」パターンで `set_not_on_field` が呼ばれる流れだが、  
OCRが「愛管侍は」と「だっしゆつボタンで」を別テキストに分けるため  
ポケモン名の抽出が難しい。`_update_switch_out` の `text.endswith("は")` 判定で  
拾える可能性はあるが、精度要確認。

### `_MIN_BATTLE_DURATION` と各種debounceの最適化

OCRログを追加収集してターン長の分布を確認し、  
`faint` の 25秒制限やその他の秒数制御を見直す余地がある。

### battle_end 誤発火の根本解決 ✅（2026-04-02 実装済み）

OCR + YOLO の AND 条件化を実装した。

**対応内容**:
- `detect_end_screen()` の conf 閾値を `0.5 → 0.9` に引き上げ（`src/capture/yolo_detector.py`）
- YOLO が3回連続検出した時点で OCR を1回実行し、以下のキーワードが含まれる場合のみ `battle_end` を発火:

```python
_END_SCREEN_OCR_KEYWORDS = ("勝った", "負けた", "選ばれました")
# 勝ち: 「〇〇との勝負に勝った！」
# 負け: 「〇〇との勝負に負けた！」
# 降参: 「降参が選ばれました」
```

- キーワード不一致の場合はカウントリセット・ログに `OCRキーワード不一致のため誤発火と判定` を出力

**検証結果**:
- conf 0.5 時: 1試合で6回誤発火
- conf 0.9 時: 2回に減少
- OCR AND条件追加後: 動画検証で確認予定（2026-04-03）

---

## T4 ターン分割問題（2026-04-06）

### 現象

ダブルバトルで味方が気絶（faint）→ 即座にポケモン繰り出し（ゆけつ!）→ 次ターン技使用
という流れのとき、参照では「1ターン」だが、ログでは2回 Bedrock が送信される。

**具体例（pipeline_20260406_210158.log）**:
- 参照 T4: リキキリン倒れる → ミライドン登場 → イナズマドライブ → ブリザードランス（全部1ターン）
- ログ: faint イベント → Bedrock①「リキキリン倒れた」
       → turn_start(T4) → move_used → Bedrock②「ミライドン登場後の状況」

### 根本原因

`faint` は `BEDROCK_EVENTS` に含まれており、HP=0 を検出した時点で即 Bedrock 送信する。
その後、繰り出し選択（ゆけつ!）が終わって次のコマンド選択画面になると
`turn_start → move_used` が発火して再度 Bedrock が送信される。

なお、「ゆけつ!」画面では `unknown → command_select` による偽の `turn_start` が発火するが、
`reset_after_processing("faint")` で設定されるデバウンス（15 秒）によってスキップされる。
本物の `turn_start` はデバウンス明けの15秒後に発火する。

### 解決案

#### 案A: faint を BEDROCK_EVENTS から除外

`BEDROCK_EVENTS = {"battle_start", "move_used", "switch", "faint", "battle_end"}` から
`"faint"` を削除し、faint 情報は次の `move_used` の戦況テキストに引き継ぐ。

- **メリット**: 実装がシンプル（1行削除のみ）
- **デメリット**: 気絶した瞬間の実況がなくなる。次のターンの実況で初めて言及される。

#### 案B: faint Bedrock 送信を遅延させて次イベントと統合（採用）

faint イベント発生時に Bedrock を即送信せず、`_pending_faint_state` に保存する。
次の `turn_start` または `move_used` が来たとき（またはタイムアウト後）にフラッシュして送信する。

```
faint 発生
  ↓ Bedrock は送信しない。_pending_faint_state に保存、タイマー起動
  ↓ （タイムアウト前に turn_start or move_used が来たら）
  ↓ pending の faint 情報を _last_faint_summary として保存
  ↓ move_used の Bedrock プロンプトに「直前の気絶情報」を追加して送信
  ↓ _pending_faint_state をクリア

タイムアウト（例: 12秒）経過しても次イベントが来なければ、
保存した faint 情報で単独 Bedrock 送信する。
```

- **メリット**: 気絶→繰り出し→技使用を1まとまりの実況にできる
- **デメリット**: 実装が複雑。気絶の実況が最大12秒遅延する。

### 実装箇所（案B）

- `PipelineRunner` に `_pending_faint_state` / `_pending_faint_time` フィールドを追加
- `_process_event` の `event_type == "faint"` 分岐で Bedrock 送信を抑制し pending に保存
- `_process_event` の `event_type in ("turn_start", "move_used")` 分岐で pending をチェックし、
  あれば `game_state["faint_context"]` に追加してから Bedrock 送信
- メインループの毎フレーム処理でタイムアウトチェック（pending があり 12 秒超過したら単独送信）
