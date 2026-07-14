# 実況動画づくり クイックスタート（人間向け・コピペ手順）

録画1本を実況付きBiim風動画にする5ステップ。
詳しい仕組み・地雷・調整方法は `.claude/skills/video-commentary/SKILL.md` を参照。

> `<動画名>` は `16-14-39` のような録画の識別名。以下すべて同じ名前で揃えること。

## ① パス1: 解析＋実況素材（Windows PowerShell・約20分）

**先にVOICEVOXを起動する**（忘れると素材ゼロでBedrock代だけ消える）。

```powershell
venv\Scripts\python.exe src/pipeline.py --input "D:\ゲーム録画\<録画ファイル名>.mp4" --model runs/detect/train4/weights/best.pt --ball-model runs/detect/train7/weights/best.pt --end-model runs/detect/train_end_screen2/weights/best.pt --ec2-url http://<EC2のIP>:5000 --conf 0.3 --render-out renders/<動画名>
```

✅ ログ末尾が `[レンダ] 素材出力完了: N 件`（N≧5くらい）ならOK。0件ならVOICEVOX未起動。

## ② パス1.5: フィラー生成（Windows・VOICEVOXそのまま）

```powershell
venv\Scripts\python.exe scripts\generate_gap_commentary.py renders\<動画名> --ec2-url http://<EC2のIP>:5000
```

✅ `フィラー素材出力完了: N 件 → renders\<動画名>\fillers.jsonl` が出ればOK。
※ `--dry-run` を付けると文面確認だけできる（**ファイルは書かれない**ので、最後は必ず付けずに実行）。

## ③ ネタバレ検査（WSL・省略禁止！）

```bash
python3 scripts/check_spoilers.py renders/<動画名>
```

- **「技名の先読みなし✅」→ ④へ**
- **⚠️が出たら**: 該当フィラーが「まだ画面に映っていない技」を喋っている。
  `renders/<動画名>/fillers.jsonl` を開き、該当行（event_timeで探す）を
  **消す**か、**event_timeをその技の📺時刻より後の空き時間に書き換える**
  （テキストの書き換えは音声と合わなくなるので不可）→ もう一度③を実行
- 表示されるタイムラインもざっと見て、気絶・勝敗の先取りがないか確認
  （★Fの内容が「それより上の行の情報だけ」で書けているかを見る）

## ④ パス2: 合成（WSL・数分）

```bash
python3 scripts/render_commentary_video.py renders/<動画名> --layout biim
```

✅ `合成完了: renders/<動画名>/<動画名>_commentary_biim.mp4`

## ⑤ 仕上げ確認

- 出来上がったmp4を通しで視聴（字幕・パネル・実況タイミング）
- 気になる箇所は③のfillers.jsonl修正→④のやり直しだけで直せる（課金なし）

## よくあるトラブル

| 症状 | 原因 | 対処 |
|------|------|------|
| ①や②で素材/フィラーが0件 | VOICEVOX未起動 | 起動して再実行 |
| ②で 404 | EC2のserver.pyが古い | WinSCPで転送→`sudo systemctl restart pokemon-api` |
| ②で 504 | Bedrock応答待ちタイムアウト | 再実行。続くならSKILL.mdの地雷リスト参照 |
| ④でパネルが出ない | states.jsonlが無い（古い素材） | ①からやり直し |
| フィラーを作り直した | — | **必ず③からやり直す**（検査済みが無効になる） |
