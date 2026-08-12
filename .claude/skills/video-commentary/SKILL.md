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
   - **【任意・実験的】表情連動させる場合**（改善ロードマップ③）: 上記バッチの代わりに
     `venv\Scripts\python.exe scripts\play_and_animate_avatar.py <動画名> --osc-port 39540`
     を使う（`pip install python-osc`が事前に必要）。WAV再生に合わせてVMCへOSCで表情
     ブレンドシェイプを送る（faint=自分が倒れたら哀しい/相手を倒したら嬉しい・
     battle_end=勝敗で喜び/哀しみ・battle_start=楽しそう。⚠️2026-07-30に発見した
     faint統合バグ（faintイベントが次のmove_usedに統合される際にfaint_sideが
     manifest.jsonlへ引き継がれず表情が発火しなかった）は同日中に修正済み）。
     **2026-08-01追加**: move_single/move_used/switchも実況テキストのキーワードから
     Fun/Sorrowを推定して反応するようになった（技1つ1つに表情がつく）。
     **2026-08-01モーション拡張**（実機未検証・角度は次回録画で調整予定）: 待機モーションは
     Spine単発のスウェイから、Spine+Chestの2ボーン×主周期/副周期を重ねたレイヤードサイン
     （呼吸っぽい揺れ・`--no-sway`で無効化可）に拡張。加えてHeadへ数秒〜十数秒おきに
     ランダムな軽い仕草（首かしげ等）を挟み、長い無反応区間の単調さを崩す
     （`--no-idle-gestures`で無効化可）。
     **2026-08-03ポーズ統合**: 表情が変わった瞬間のリアクションは、`explore_avatar_poses.py`
     （＋`pose_tuner_gui.py`）で実機検証済みのポーズへSLERP遷移→保持→idle_downへ戻す
     動作に統合（Joy=`victory_arms_up`・Sorrow=`bow_apologetic`・Fun=`head_tilt_curious`。
     マッピング外の表情はNeckの標準うなずきにフォールバック）。ポーズ再生中は該当ボーンを
     常時スウェイ/ランダム仕草から一時除外（`_suspend_bones`/`_resume_bones`）して競合を
     防ぐ。`fist_pump_right`/`thinking_chin`/`lean_back_confident`は実機検証済みだが
     まだイベント未マッピング（`explore_avatar_poses.POSES`には残っているので今後拡張可）。
     **事前にVMCの設定画面でReceiver（39540 or 39541）の「有効化」チェックボックスを
     ONにしておくこと**（デフォルトOFF・OFFのままだと表情が一切変わらない＝実機で
     確認済みの地雷）。schedule.jsonが必要なので先に`render_commentary_video.py <動画名>
     --dry-run`を実行しておく。表情マッピングは`scripts/play_and_animate_avatar.py`の
     `_EVENT_EXPRESSION`/`_FAINT_EXPRESSION`/`_BATTLE_RESULT_EXPRESSION`/
     `_POSITIVE_KEYWORDS`/`_NEGATIVE_KEYWORDS`/`_EXPRESSION_POSE`を、モーションは
     `_IDLE_BONES`/`_NOD_*`/`_GESTURE_*`/`play_pose_reaction`を参照。
     - **手順は「スクリプト起動→OBS録画→Enter」の順**（2026-07-30変更・T-pose対策後）:
       ①スクリプトを起動すると即座に腕を下ろした初期姿勢をOSC送信→
       ②「録画が始まったらEnterを押してください」の表示を待ってここでOBS録画を開始→
       ③録画開始後にEnterを押す（この瞬間からWAV再生・表情連動が始まる）。
       録画開始前にTポーズが解消されるため、**録画にTポーズが一切映り込まない**
       （旧手順「録画→スクリプト実行」だと録画開始直後の一瞬Tポーズが映っていた）。
       この順序なら録画開始がWAV再生開始より確実に先になるので、下記
       `--avatar-offset`は0固定でよい（目視で秒数を計る必要がなくなった）
4. 合成: `python3 scripts/render_commentary_video.py renders/<動画名> --layout biim --avatar-video <録画mp4> --avatar-offset <秒> --avatar-crop <w:h:x:y>`
   - 右下344px幅にクロマキー合成（`--avatar-width`/`--avatar-chroma`で調整可）
   - `--avatar-crop`（任意）: 全身録画から上半身だけを切り出してから拡大する
     （例 `"300:480:810:230"`＝顔・肩まわり）。全身のまま縮小すると人物が極小になるため、
     バストアップで見せたい場合は指定推奨
   - クロマキーは類似度0.25＋`despill`（緑かぶり除去）がデフォルト。それでも縁が残る場合は
     `--avatar-chroma`と合わせて類似度をさらに上げる
   - アバター録画が動画より短い場合は最終フレームで静止（正常動作）
   - **offsetは0以上のみ**（録画を先に始める運用で統一）。詳細: `docs/design/v2c-avatar-design.md`

### サムネイル自動生成（改善ロードマップ⑥・任意・WSL）

パス1の素材（manifest.jsonl・states.jsonl）から「盛り上がった瞬間」
（battle_end > faint(KO) > HP急変の優先度）を機械的に選び、元動画の該当フレームに
その瞬間の実況テキストを焼き込んだサムネイルPNGを出力する。パス2とは独立（動画本体
の合成をやり直さなくてもサムネイルだけ再生成できる）。

```
python3 scripts/generate_thumbnail.py renders/<動画名>                    # 自動選択
python3 scripts/generate_thumbnail.py renders/<動画名> --time 230.5 --label "きめ台詞！"  # 手動指定
```

- 出力: `renders/<動画名>/thumbnail.png`（`--out`で変更可）
- 元動画パスは`render_info.json`から自動解決（D:\...→/mnt/d/...変換込み）。`--video`で上書き可
- HP急変の検出閾値（既定30pt）は`--hp-swing-threshold`で調整可
- 自動選択が微妙な場合は`--time`/`--label`で手動指定するのが手っ取り早い

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
| アバター全身録画をそのまま縮小 | 344px枠内で人物が極小・Tポーズの腕が窮屈 | `--avatar-crop`で上半身だけ切り出してから拡大する。**クロップ範囲の決め方**: 収録した生の録画（グリーンバック）から1フレーム抽出し、Pythonで緑以外のピクセルのbounding boxを検出すると人物の実座標が分かる（`PIL`+`numpy`で`is_green = (g>150)&(r<100)&(b<100)`の否定領域を`np.where`）。2026-07-30収録（`HairSample_Female.vrm`・1920x1080・全身がx760-1161/y273-1079に収まっていた）では`--avatar-crop "300:480:810:230"`（過去動画の実績値）でバストアップになり、これがそのまま流用できることを確認済み。モデルが変わったら同じ手順で再計測すること |
| `--device`に数値インデックスを渡してもsounddeviceが認識しない | `play_and_animate_avatar.py --device 7`のように番号を渡すとエラー、または無関係のデバイスが選ばれる | sounddeviceは`device`引数が文字列だと**名前の部分一致検索**になり、数字文字列を渡しても数値インデックスとしては解釈されない。2026-07-30に対策済み: 数字文字列なら自動でintにキャストするよう`play_and_animate_avatar.py`を修正。デバイス一覧は`python -c "import sounddevice as sd; print(sd.query_devices())"`で確認し、CABLE Input側のMME/WASAPI版インデックスを指定する |
| `play_and_animate_avatar.py`実行中にCtrl+Cで中断できない | 再生を途中で止めたくても効かない | `sd.wait()`のブロッキング待ちがWindowsだとシグナルを拾えないことがある。2026-07-30に対策済み: `sd.get_stream().active`をポーリングするループ＋`KeyboardInterrupt`ハンドリングに変更（`sd.stop()`で即座に再生停止） |
| VMCがリップシンク中にフリーズする | 表情もリップシンクも反応しなくなる | 2026-07-30に実機で1回発生・VMC再起動で解消。再現条件は不明のため、フリーズしたらまず再起動を試す |
| ダブルバトルで相手の2匹目が戦況パネルから消える | 一度検出された後、1フレームOCR不検出で即座に場から降ろされ二度と戻らない（`BattleStateTracker`の`quick_threshold=1`が原因） | **2026-07-30時点で未修正**（緩めると「交代直後に旧ポケモンが残り続ける」という別の既知バグが再発するリスクがあり慎重な調整が必要）。当面は戦況パネルの2匹目表示が抜けることがあると認識し、実況（manifest.jsonl）自体はボール数ロジックで気絶を把握できているので実況内容への影響は限定的 |
| VMCがトラッキングデバイス無しでT-poseのまま | 腕を横に伸ばした初期姿勢で録画されてしまう | **2026-07-30に対策済み**: `/VMC/Ext/Bone/Pos`でLeftUpperArm/RightUpperArmにZ軸回転（80°・右は符号反転）を送ると腕が下がることを実機確認（`scripts/test_vmc_pose.py`で角度検証）。`play_and_animate_avatar.py`が再生開始前に自動送信する（`send_idle_pose`）。指先が服にわずかに埋まるが上半身のみ表示のため実害なしと判断。モデルが変わったら角度の再検証が必要な可能性あり（`_IDLE_POSE_DEG`/`_IDLE_POSE_AXIS`） |
| クロマキーのデフォルト設定（類似度0.15） | 輪郭に緑フリンジが残る | 類似度0.25＋`despill`をデフォルト化済み（2026-07-15）。まだ残るなら類似度をさらに上げる |
| 音声ルーティングを毎回手動切替 | 録画前後の切替が面倒・戻し忘れ | wav再生専用アプリを1つ決めて出力先を一度だけCABLE Inputに設定（アプリ単位で記憶される）。`scripts/play_commentary_track.bat <動画名>`でフォルダ解決も自動化 |
| 実況文中の絵文字（💦💕等の絵文字ブロック文字） | 字幕が豆腐（□）化。ffmpegログに`Glyph 0x1F4A6 not found`等が出る | Meiryoに絵文字グリフが無いのが原因。♪♡等の記号（Miscellaneous Symbolsブロック・U+2600-27BF）は問題なし。**fillers.jsonlだけでなくmanifest.jsonl（通常のイベント実況）にも出る**（2026-07-29確認）。パス1無課金検証（サンプリング25本）で83件と想定より頻発すると判明したため、**2026-08-12に生成時点での自動除去に対策済み**: `pipeline.py`の`_clean_commentary`が正規表現`[\U0001F300-\U0001FAFF]`で全commentary（Bedrock/Phi-3どちらの経路も）から自動除去する。以降は基本的に手動除去は不要のはずだが、仕上げ確認のフレーム目視は念のため継続すること（`_clean_commentary`を通らない経路や新しい絵文字レンジが漏れる可能性に備える） |
| 実況文中の生コード片・HTMLタグ漏れ（\`\`\`python...\`\`\`・`</span>`等） | 字幕・音声にコードやマークアップがそのまま混入。VOICEVOXが記号ごと読み上げる形になる | Phi-3が稀にプロンプト内の説明例（コードブロック）やHTML風タグをそのまま出力することがある（パス1無課金検証で発見・2026-08-12確認）。**2026-08-12に生成時点での自動除去に対策済み**: `_clean_commentary`が`\`\`\``以降を切り捨て＋正規表現`<[^>]*>`でHTMLタグ風の文字列を除去する。絵文字対策と同じ関数内の処理なので経路も同じ |
| 末尾実況の映像はみ出し（battle_endが動画終端間際に発火） | 締めの実況が映像なし（黒画面/プレイヤー依存）で流れる（20-14-17で6.6秒・08-15-22で2.4秒実発生） | **biimは2026-07-30から自動対応**: 実況トラックが動画長を超えると最終フレーム静止で映像を延長（`tpad=stop_mode=clone`・字幕描画の前段なので延長区間でも字幕は時刻通り）。plainは映像コピーのため非対応（警告のみ） |
| Bedrockの保留・困惑応答（「データが矛盾していて実況できません」等） | 言い訳文がそのまま実況としてmanifest.jsonlに入り合成される（07-00-19で2件・08-15-22で4件実発生。原因はロスターバグ・同名ミラー混乱・試合間空データ等ケースごとにバラバラ） | **2026-07-30に恒久対策済み**: `pipeline.py`の`_replace_glitch_commentary`がVOICEVOX合成前にキーワード検出（矛盾・ちぐはぐ・モヤモヤ等=`_GLITCH_CAUSE_KEYWORDS`）し、くれぴ口調の「AIグリッチ」定型文（`_GLITCH_TEMPLATES`3種×原因4種）に差し替える。manifest.jsonlに問題テキストは書き込まれない。**新しい言い回しの保留応答を見つけたら`_GLITCH_CAUSE_KEYWORDS`にキーワードを追加すること**（仕上げ確認でmanifest/fillersのcommentaryを目視） |
| VMCのOSC表情操作（`play_and_animate_avatar.py`）が無反応 | OSCメッセージを送っても表情が一切変わらない | VMCの設定画面でReceiver（39540/39541）のポート番号は入っていても、**「有効化」チェックボックスがデフォルトOFFだと外部OSC入力が反映されない**（2026-07-30に実機で確認・チェックを入れたら即動作した）。必ずONにしてから実行すること |

## レイアウト調整ポイント（scripts/render_commentary_video.py の定数）

| 定数 | 現在値 | 意味 |
|------|--------|------|
| `_SUBTITLE_FONT_SIZE` / `_SUBTITLE_WRAP_CHARS` | 48 / 37 | 字幕サイズと折り返し文字数（**必ずセットで変更**: wrap≒1824÷フォントpx） |
| `_BIIM_GAME_W/H/X/Y` | 1440×810 (16,12) | ゲーム画面の縮小配置 |
| `_PANEL_TEXT_X` / `_PANEL_BAR_W` | 1496 / 300 | パネル左端・HPバー幅 |
| `_hp_bar_color` | 緑>50/黄>20/赤 | HPバー3色（ゲーム準拠） |
| `_FILLER_MAX_SHIFT_SEC` | 12 | フィラー配置の最大ずれ（大きくするとネタバレ防御が弱まる） |
| `_DEFAULT_GAIN` | 1.4 | 実況音量。ゲーム音とのバランスは`--gain`で上書き可 |
| server.py `_gap_filler_count` | 40秒/件・上限3 | フィラー密度（変更はEC2再デプロイ必要）。18秒/件・上限5→2026-07-30視聴fb「多い」で30秒/件・上限3に減量→「もう少し増やしたい」で25秒/件・上限4に再調整→さらに「あ、あが耳につく・フィラー減らして実況を活かしたい」で40秒/件・上限3に再々調整 |
| generate_gap_commentary.py `_DEFAULT_MIN_GAP_SEC` | 40秒 | フィラー対象とする最小無言秒数（`--min-gap`で上書き可）。同上の再調整で25秒→40秒 |
| プロンプトの書き出しバリエーション指示 | （新規） | 2026-07-30〜: 「あ、あ」等の相槌の書き出し連発を抑制する指示を`_build_script_prompt`（フィラー用）・`_build_vision_prompt`（実況本編用）の両方に追加済み。片方だけ直しても同じキャラ設定を共有しているためもう片方で再発するので注意 |

パネル下部（y660以降）と画面右下はv2c（アバター）用に空けてある。

## 横展開（新しい動画への適用）

パス1→1.5→2を上から順に実行するだけ。動画ごとに `renders/<動画名>` を分ける。
検証チェックリストは毎回実施（特にネタバレ検査）。書き起こしがある動画なら
パネルのHP値と書き起こしの照合も行うと確実。

## 関連ファイル

- 実装: `src/output/render_sink.py`（パス1素材出力）・`src/pipeline.py`（`_record_panel_state`/`_render_context`/瞬間ログフック/`BattleStateTracker.fainted_names`・`diff_fainted_side`）・`src/api/server.py`（`/api/script`・プロンプト）・`scripts/generate_gap_commentary.py`・`scripts/render_commentary_video.py`・`scripts/generate_thumbnail.py`（サムネイル自動生成）・`scripts/play_commentary_track.bat`（アバター録画用wav再生・Windows）・`scripts/play_and_animate_avatar.py`（表情連動版wav再生・Windows・改善ロードマップ③）
- テスト: `tests/test_render_sink.py`・`tests/test_render_video.py`・`tests/test_gap_commentary.py`・`tests/test_server.py`（TestScript系）・`tests/test_play_and_animate_avatar.py`
- 設計: `docs/adr/ADR-009-video-first-commentary.md`・レイアウト原案 `docs/design/frame-mockups/mockup_A_biim.png`
- 経緯・実測値: `docs/daily/2026-07-14.md`
