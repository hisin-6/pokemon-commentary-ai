# 実況動画合成（Biim風）スキル

録画したポケモン対戦動画に、AI実況音声＋Biim風レイアウト（枠・字幕・戦況パネル）を
合成するワークフローの完全手順。正式設計: `docs/adr/ADR-009-video-first-commentary.md`。
**このスキルの手順・地雷リストに従えば、実装の内部理解なしでも安全に動画を量産できる。**

## 全体像（3パス構成）

```
パス1   [Windows]  src/pipeline.py --render-out
        動画を解析し実況素材を出力（Bedrock代 約10〜15円/本・約20分/6分動画）
  ↓ renders/<名前>/ … manifest.jsonl（イベント実況+WAV）・timeline.jsonl（技の瞬間ログ）
                     ・states.jsonl（戦況スナップショット）・render_info.json
パス1.5 [Windows]  scripts/generate_gap_commentary.py
        無言区間を埋めるライブ風フィラーを生成（無言区間ごとに個別にBedrock呼び出し・
        1試合あたり数円程度。2026-07-22〜: ネタバレ防止のため区間より未来の情報は
        プロンプトに含めない設計に変更・区間数だけ呼び出し回数が増える）
  ↓ fillers.jsonl ＋ wav/fNNNN_filler.wav
パス2   [WSL]      scripts/render_commentary_video.py --layout biim
        枠・字幕・戦況パネル・音声を一発合成（課金なし・何度でもやり直し可）
  ↓ renders/<名前>/<名前>_commentary_biim.mp4
```

- 合成のやり直し（パス2）にBedrock/VOICEVOXは不要。レイアウト変更はパス2の再実行だけ
- フィラーの作り直しはパス1.5の再実行だけ（約1円）。パス1のやり直しが必要なのは
  「解析自体の改善」か「states/timeline形式の変更」のときだけ

## 工程の分担（どこが機械的で、どこに判断が要るか）

| # | 工程 | 実行場所 | 性質 |
|---|------|---------|------|
| ① | パス1（解析＋素材） | Windows（要VOICEVOX） | **機械的**（コマンド1つ） |
| ② | パス1.5（フィラー生成） | Windows（要VOICEVOX・EC2） | **機械的**（コマンド1つ） |
| ③ | ネタバレ検査 | WSL | **半自動**: `check_spoilers.py`が技名先読みを自動検出（実測されたネタバレの全パターン）。＋タイムラインの目視で勝敗・気絶の先取りを確認 |
| ④ | パス2（合成） | WSL | **機械的**（コマンド1つ） |
| ⑤ | 仕上げ確認 | どちらでも | **目視**: フレーム抽出2〜3枚＋通し視聴 |

⚠️ **③は省略禁止**。フィラーのネタバレ発生率は実測で1〜2割あり、プロンプト側の
抑制だけでは防ぎきれない（2026-07-14に検査スキップした生成分から1件が動画まで
到達した実例あり）。**フィラーを再生成したら必ず③からやり直す**こと。

## 実行手順

### パス1（Windows PowerShell）

```
# ⚠️ 事前に必ず VOICEVOX を起動する（忘れると素材ゼロ・Bedrock代だけ消える）
venv\Scripts\python.exe src/pipeline.py --input "D:\ゲーム録画\<動画>.mp4" ^
  --end-model runs/detect/train_end_screen2/weights/best.pt ^
  --ec2-url http://<EC2のIP>:5000 --conf 0.3 --render-out renders/<動画名>
```

（`--model`＝状態異常YOLO・`--ball-model`＝ボール数YOLOは2026-07-15より未指定がデフォルト。
状態異常はテキストOCRで代替済み、ボール数は現在パイプライン未使用。学習済みモデル自体・
指定方法は残っているので、必要になれば`--model runs/detect/train4/weights/best.pt`等を
追加するだけで復活できる）

成功確認: ログ末尾に `[レンダ] 素材出力完了: N 件`（0件なら失敗＝ほぼVOICEVOX未起動）。
`renders/<動画名>/` に manifest.jsonl・timeline.jsonl・states.jsonl・wav/ ができる。

⚠️ **同じ--render-outへの再実行は前回素材（manifest/wav/timeline/states/fillers）を
自動クリアして作り直す**（RenderSinkの仕様・追記による新旧混在事故の再発防止）。

### パス1.5（Windows・VOICEVOX起動必須）

```
# まず --dry-run でフィラー文面を確認（VOICEVOX不要・fillers.jsonlは書かれない）
venv\Scripts\python.exe scripts\generate_gap_commentary.py renders\<動画名> --ec2-url http://<EC2のIP>:5000 --dry-run
# 文面OKなら --dry-run を外して本実行（これでfillers.jsonlが書かれる）
venv\Scripts\python.exe scripts\generate_gap_commentary.py renders\<動画名> --ec2-url http://<EC2のIP>:5000
```

⚠️ dry-runは**ファイルに書き込まない**。「実行した」と思ってもfillers.jsonlの
タイムスタンプが古いままなら、それはdry-runだった可能性が高い。
⚠️ Bedrock生成は毎回ガチャ。dry-runで見た文面と本実行の文面は別物になる。

### ネタバレ検査（WSL・パス2の前に必ず実行）

```
python3 scripts/check_spoilers.py renders/<動画名>
```

- **exit 0＝技名先読みなし** / **exit 1＝要確認フラグあり**（⚠️付きで該当フィラーと
  「画面に映る時刻」が表示される）
- フラグが出たら目視確認のうえ fillers.jsonl を修正（下記「ネタバレの直し方」）→再度実行
- あわせて表示されるタイムラインを流し読みし、技名以外の先取り
  （気絶・勝敗・登場していないポケモンへの言及）がないか目視する

### パス2（WSL）

```
python3 scripts/render_commentary_video.py renders/<動画名> --dry-run   # スケジュール確認のみ
python3 scripts/render_commentary_video.py renders/<動画名> --layout biim  # Biim風合成（推奨）
python3 scripts/render_commentary_video.py renders/<動画名>              # 音声のみ合成（plain・映像再エンコなし）
```

### アバター合成（v2c・方式A=VMC録画ワイプ・任意）

0. **【初回のみ】音声ルーティングを一度だけ設定**: wav再生専用のアプリを1つ決めて、
   Windowsの音量ミキサー（アプリごとの出力先）でそのアプリの出力を「CABLE Input」に
   設定する。**この設定はアプリ単位で記憶される**ため、動画ごとにフォルダが変わっても
   毎回の切り替え・戻し作業は不要（普段使いの他アプリは既定デバイスのまま触らない）
1. パス2を一度実行して `renders/<動画名>/commentary_track.wav` を得る
2. Windows: VMC＋3Dモデルを起動（背景グリーン単色）。リップシンク入力はCABLE Output
   （手順0で設定した仮想ケーブルの出口）を選択
3. **OBS録画を先に開始→`scripts\play_commentary_track.bat <動画名>` でWAVを頭から再生**
   （録画開始から再生開始までの秒数をメモ=offset。バッチはrenders配下の同名ファイルを
   毎回自動解決するので手でフォルダを探さなくてよい）
4. 合成: `python3 scripts/render_commentary_video.py renders/<動画名> --layout biim --avatar-video <録画mp4> --avatar-offset <秒> --avatar-crop <w:h:x:y>`
   - 右下344px幅にクロマキー合成（`--avatar-width`/`--avatar-chroma`で調整可）
   - `--avatar-crop`（任意）: 全身録画から上半身だけを切り出してから拡大する
     （例 `"300:480:810:230"`＝顔・肩まわり）。全身のまま縮小すると人物が極小になるため、
     バストアップで見せたい場合は指定推奨
   - クロマキーは類似度0.25＋`despill`（緑かぶり除去）がデフォルト。それでも縁が残る場合は
     `--avatar-chroma`と合わせて類似度をさらに上げる
   - アバター録画が動画より短い場合は最終フレームで静止（正常動作）
   - **offsetは0以上のみ**（録画を先に始める運用で統一）。詳細: `docs/design/v2c-avatar-design.md`

## 合成後の検証チェックリスト（必ずやる）

1. **スケジュール確認**: `renders/<動画名>/schedule.json` の `scheduled` を時刻順に見る。
   `fillers_dropped` は正常動作（収まらないフィラーの安全破棄）なので気にしない
2. **フレーム目視**: `ffmpeg -y -ss <秒> -i <出力mp4> -frames:v 1 frame.png` で
   字幕表示中・パネル更新後のフレームを2〜3枚抽出し、Readで目視
   （字幕のはみ出し・パネルのHP/技がゲーム画面内の表示と一致するか）
3. **音声配置の確認（任意）**: `ffmpeg -ss <秒> -t 4 -i <mp4> -map 0:a -af volumedetect -f null -`
   で実況区間（-25dB前後）と無言区間（元動画が無音なら-91dB）を比較

品質目安: 発話カバレッジ40〜50%・最長無言60秒未満・ネタバレゼロ・字幕はみ出しゼロ。

## ネタバレが見つかったときの直し方

fillers.jsonl を直接編集してパス2を再実行する（音声再合成は不要）:
- **時刻をずらす**: 該当行の `event_time` を、言及している出来事より後のギャップ内へ移動
- **消す**: 行ごと削除（内容が事実誤りの場合はこれ。テキスト編集はWAVと不一致になるので不可）
- ずらした先でイベント実況と物理的にかぶる場合、fit_fillersが自動破棄する（それでOK）

多層防御が入っているので手修正が必要なのは稀:
①ギャップを📺時刻で分割（区間内先読みの構造対策）②プロンプトの時系列交互形式
③fit_fillersの12秒超シフト破棄（先読みフィラーはイベントと衝突しやすく自動で落ちる）

## 地雷リスト（全部実際に踏んだもの・回避必須）

| 地雷 | 症状 | 回避/対処 |
|------|------|-----------|
| VOICEVOX未起動でパス1/1.5 | 素材0件（Bedrock代だけ消える） | 実行前に必ず起動。パス1はログ末尾のレンダサマリーで検知 |
| server.py変更後の未デプロイ/未再起動 | `/api/script` 404 | WinSCPで転送→`sudo systemctl restart pokemon-api`（gunicornは自動リロードしない） |
| Bedrock read_timeout | `/api/script` 504 | vision=5秒/script=60秒に分離済み（`bedrock_script`）。scriptを重くしたら`BEDROCK_SCRIPT_TIMEOUT_SEC`を見直す。2026-07-22〜: `/api/script`は無言区間1つ分だけを処理する設計になり1リクエストあたりの生成量が減ったため、gunicorn worker timeout（30秒）に抵触するリスクはむしろ下がった |
| ffmpeg 4.2（WSL）のamix | `Option 'normalize' not found` | normalize不使用の互換実装済み（1/2スケールをvolume=2補償＋トラック同尺パディング）。**この構造を変えないこと** |
| ASSの日本語折り返し | 字幕が帯からはみ出す | libassはスペース基準で日本語に効かない→`_wrap_jp`が手動\N挿入（禁則処理付き）。フォントサイズ変更時は`_SUBTITLE_WRAP_CHARS`も連動して変えること |
| 日本語フォント | 字幕が豆腐/英字 | WSLにフォント不要。`fontsdir=/mnt/c/Windows/Fonts`のmeiryoを参照している |
| renders/のcommit | 数GBのmp4がリポジトリに入る | .gitignore済み。**外さないこと** |
| states.jsonlが無い旧素材 | 戦況パネルが出ない | パス1の再実行が必要（timeline/statesは新パス1からのみ生成） |
| モックアップ/日報の実フレーム | 対戦相手のトレーナー名が映る | publicリポジトリに載せてよいかはユーザー判断（2026-07-14に確認済み・現方針は許容） |
| アバター全身録画をそのまま縮小 | 344px枠内で人物が極小・Tポーズの腕が窮屈 | `--avatar-crop`で上半身だけ切り出してから拡大する |
| クロマキーのデフォルト設定（類似度0.15） | 輪郭に緑フリンジが残る | 類似度0.25＋`despill`をデフォルト化済み（2026-07-15）。まだ残るなら類似度をさらに上げる |
| 音声ルーティングを毎回手動切替 | 録画前後の切替が面倒・戻し忘れ | wav再生専用アプリを1つ決めて出力先を一度だけCABLE Inputに設定（アプリ単位で記憶される）。`scripts/play_commentary_track.bat <動画名>`でフォルダ解決も自動化 |
| フィラー文中の絵文字（💦等の絵文字ブロック文字） | 字幕が豆腐（□）化。ffmpegログに`Glyph 0x1F4A6 not found`等が出る | Meiryoに絵文字グリフが無いのが原因。♪♡等の記号（Miscellaneous Symbolsブロック）は問題なし。fillers.jsonlから該当絵文字だけ削除して④を再実行（WAVは元々発声してないので音声とズレない）。仕上げ確認のフレーム目視で毎回チェックすること |

## レイアウト調整ポイント（scripts/render_commentary_video.py の定数）

| 定数 | 現在値 | 意味 |
|------|--------|------|
| `_SUBTITLE_FONT_SIZE` / `_SUBTITLE_WRAP_CHARS` | 48 / 37 | 字幕サイズと折り返し文字数（**必ずセットで変更**: wrap≒1824÷フォントpx） |
| `_BIIM_GAME_W/H/X/Y` | 1440×810 (16,12) | ゲーム画面の縮小配置 |
| `_PANEL_TEXT_X` / `_PANEL_BAR_W` | 1496 / 300 | パネル左端・HPバー幅 |
| `_hp_bar_color` | 緑>50/黄>20/赤 | HPバー3色（ゲーム準拠） |
| `_FILLER_MAX_SHIFT_SEC` | 12 | フィラー配置の最大ずれ（大きくするとネタバレ防御が弱まる） |
| `_DEFAULT_GAIN` | 1.4 | 実況音量。ゲーム音とのバランスは`--gain`で上書き可 |
| server.py `_gap_filler_count` | 18秒/件・上限5 | フィラー密度（変更はEC2再デプロイ必要） |

パネル下部（y660以降）と画面右下はv2c（アバター）用に空けてある。

## 横展開（新しい動画への適用）

パス1→1.5→2を上から順に実行するだけ。動画ごとに `renders/<動画名>` を分ける。
検証チェックリストは毎回実施（特にネタバレ検査）。書き起こしがある動画なら
パネルのHP値と書き起こしの照合も行うと確実。

## 関連ファイル

- 実装: `src/output/render_sink.py`（パス1素材出力）・`src/pipeline.py`（`_record_panel_state`/`_render_context`/瞬間ログフック）・`src/api/server.py`（`/api/script`・プロンプト）・`scripts/generate_gap_commentary.py`・`scripts/render_commentary_video.py`・`scripts/play_commentary_track.bat`（アバター録画用wav再生・Windows）
- テスト: `tests/test_render_sink.py`・`tests/test_render_video.py`・`tests/test_gap_commentary.py`・`tests/test_server.py`（TestScript系）
- 設計: `docs/adr/ADR-009-video-first-commentary.md`・レイアウト原案 `docs/design/frame-mockups/mockup_A_biim.png`
- 経緯・実測値: `docs/daily/2026-07-14.md`
