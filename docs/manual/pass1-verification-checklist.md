# パス1 無課金検証チェックリスト

未処理の録画（`D:\ゲーム録画\`）を1本ずつ使って、パス1（状態抽出＋実況テキスト）の
精度をBedrock課金なしで確認し、不具合があれば直す作業のためのチェックリスト。

## 目的・スコープ

- 対象は **パス1のみ**（`renders/<動画名>/manifest.jsonl` / `timeline.jsonl` / `states.jsonl`）。
  パス1.5（フィラー）・パス2（合成動画）はこの検証には含めない。
- 見るのは「試合内容（状態抽出の正しさ）」と「実況内容（生成テキストの妥当性）」の2軸。
  動画としての完成度（字幕はみ出し等）は対象外 → その確認は
  `.claude/skills/video-commentary/SKILL.md` の合成後チェックリストを使う。

## 実行コマンド（Windows・無課金）

```
# 事前に必ずVOICEVOXを起動（音声合成に使う・Bedrockは呼ばないがVOICEVOXは要る）
venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\<動画ファイル名>.mp4" ^
  --end-model runs/detect/train_end_screen2/weights/best.pt ^
  --conf 0.3 --render-out renders\<動画名>
```

- **`--ec2-url` を指定しない**のがポイント。`attempt_bedrock` が false になり、
  自動でローカルのPhi-3 miniにフォールバックする（`src/commentary/phi3_client.py`）。
  Bedrock課金は一切発生しない。
- チャンピオンズ動画なら `--game-mode champions` を追加（対象UIか確認してから）。
- ログ末尾に `[レンダ] 素材出力完了: N 件` が出ればOK。0件ならVOICEVOX未起動を疑う。

## A. 試合内容（状態抽出）のチェック

| # | 確認項目 | 見る場所 | 判定基準 |
|---|---------|---------|---------|
| A1 | ターン進行 | `states.jsonl` の `turn` | 動画を早送り確認したターン数と一致（抜け・重複なし） |
| A2 | 場のポケモン | `states.jsonl` の `player`/`opponent` | 幽霊ポケモン（存在しないのに残る）・消失（まだいるのに消える）がないか。※ダブルバトルで相手2匹目が1フレーム不検出で消える既知バグ（`quick_threshold=1`）は許容範囲として扱う |
| A3 | 交代検出 | `timeline.jsonl`（switch系）/`manifest.jsonl`の`context` | 交代の見落とし・誤検出がないか |
| A4 | HP | `states.jsonl` の `hp_pct`/`hp_text` | 画面のHP%表示と一致するか。交代直後の1フレームだけの誤帰属は既知の軽微ノイズとして許容（[[project_sprint7_progress]]参照） |
| A5 | 状態異常 | `states.jsonl` の `status` | 画面表示（まひ/やけど/どく等）と一致するか。特に漢字表記のすれ違いに注意 |
| A6 | 気絶検出 | `timeline.jsonl`（faint）/ボール数変化 | 気絶した瞬間が漏れなく記録されているか |
| A7 | 技帰属 | `timeline.jsonl` の `text`/`side` | 使い手ポケモンが正しいか。隣接誤帰属・断片一致の幽霊技（似た技名への誤登録）に注意 |
| A8 | 場の効果 | `manifest.jsonl` の `context.move_log` | 天候（にほんばれ/あまごい等）・壁（リフレクター/ひかりのかべ）・トリックルーム・おいかぜが、技名/特性名の直接マッチで正しく検出されているか。壁の残りターン数表示も確認 |
| A9 | 試合終了 | `manifest.jsonl`（battle_end） | 検出タイミングが実際の終了画面と一致するか |

## B. 実況内容（テキスト）のチェック

| # | 確認項目 | 見る場所 | 判定基準 |
|---|---------|---------|---------|
| B1 | 内容の整合性 | `manifest.jsonl` の `commentary` | 場にいないポケモンへの言及・間違った技名など、Aで確認した実際の試合内容と矛盾していないか |
| B2 | AIグリッチ応答 | `manifest.jsonl` の `commentary` | 「データが矛盾していて〜」等の保留・困惑文がそのまま残っていないか。残っていたら`_GLITCH_CAUSE_KEYWORDS`未収録の新パターンの合図 → `src/pipeline.py`の`_GLITCH_CAUSE_KEYWORDS`に追記 |
| B3 | 絵文字混入 | `manifest.jsonl` の `commentary` | `U+1F300`〜`U+1FAFF`の絵文字ブロックが混じっていないか（VOICEVOXは発話しないが字幕合成時に豆腐化する） |
| B4 | 書き出しパターン | `manifest.jsonl` の `commentary`（時系列で通し読み） | 「あ、あ」等の相槌書き出しが連発していないか |
| B5 | Phi-3固有の癖 | 同上 | ローカルLLM（Phi-3 mini）特有の破綻（意味不明・文が切れる等）がないか。Bedrock版と違う課題が出る可能性があるので新規に見る |

※B1〜B4はBedrock/Phi-3どちらでも共通の観点。B5は今回ローカル生成にしたことで新たに出てくる可能性がある観点。

## C. 手順

1. `--ec2-url` なしでパス1を実行（上記コマンド）
2. `renders/<動画名>/manifest.jsonl` を時系列で通し読みして B1〜B5 をチェック
3. `states.jsonl` / `timeline.jsonl` を見て A1〜A9 をチェック
   - 気になる時刻があれば元動画のフレームを確認:
     `ffmpeg -y -ss <秒> -i "D:\ゲーム録画\<動画>.mp4" -frames:v 1 frame.png`
4. 誤りを見つけたら:
   - 単純な検出漏れ・誤検出 → 原因コード（`BattleStateTracker`/`BattlePhaseClassifier`/
     `BattleMessageParser`等）を特定してから修正方針を相談
   - Phi-3特有のテキスト破綻 → プロンプト（`Phi3Client`側）の調整を検討
5. 直したら同じ動画で再実行して回帰確認（同じ`--render-out`への再実行は前回素材を
   自動クリアするので上書きでOK）

## 関連

- 合成・地雷リストは `.claude/skills/video-commentary/SKILL.md`
- 過去の状態抽出バグの傾向は `[[project_sprint7_progress]]`（memory）
- 診断JSONLを使ったより精密な検証（フレーム単位のOCRリプレイ）が要る場合は
  `scripts/replay_phase_events.py`（今回のチェックリストとは別系統・診断ログの別収集が必要）
