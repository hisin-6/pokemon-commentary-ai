# 実況動画合成（Biim風）スキル

録画したポケモン対戦動画に、AI実況音声＋Biim風レイアウト（枠・字幕・戦況パネル）を
合成するワークフローの完全手順。正式設計: `docs/adr/ADR-009-video-first-commentary.md`。
**このスキルの手順・地雷リストに従えば、実装の内部理解なしでも安全に動画を量産できる。**

## 全体像（3パス構成）

```
パス0   [Windows]  scripts/team_preview_gui.py（任意・2026-08-24新設・パス1より前！）
        「対戦準備中」画面（両陣営6匹・スプライトのみでテキスト名は無い）を
        ユーザーが目視して手入力し、選出前チームプレビューを保存する
  ↓ renders/<名前>/team_preview.json
パス1   [Windows]  src/pipeline.py --render-out
        動画を解析し実況素材を出力（Bedrock代 約10〜15円/本・約20分/6分動画）
  ↓ renders/<名前>/ … manifest.jsonl（イベント実況+WAV）・timeline.jsonl（技の瞬間ログ）
                     ・states.jsonl（戦況スナップショット）・render_info.json
パス1.5 [Windows]  scripts/generate_gap_commentary.py
        無言区間を埋めるライブ風フィラーを生成（無言区間ごとに個別にBedrock呼び出し・
        2026-07-22〜: ネタバレ防止のため区間より未来の情報は
        プロンプトに含めない設計に変更・区間数だけ呼び出し回数が増える。
        **2026-08-21: 対象閾値を20秒→6秒に下げたため区間数が増え、費用も
        「1試合あたり数円程度」から目安1本あたり数倍〜10円程度に増加**）
  ↓ fillers.jsonl ＋ wav/fNNNN_filler.wav
パス1.6 [Windows]  scripts/generate_predictions.py（2026-08-21新設・任意）
        「予測→回収」実況を生成。2系統が独立して動く:
        ①条件系（自分側/相手側それぞれ最大1件・最大2組。場のコンディションを
          確立した技＝おいかぜ/壁/天候等を検出時刻とし、試合最後のfaint
          （無ければbattle_end）で回収）
        ②選出予想（2026-08-24新設・パス0でteam_preview.jsonがある試合のみ。
          相手のリード[先頭2匹]を試合開始直前に予想し、battle_startで回収）
        的中/外れの判定はどちらもPython側が機械的に確定・LLMには演技だけさせる
  ↓ predictions.jsonl ＋ wav/pNNNN_*.wav（対象なしなら空ファイル）
パス2   [WSL]      scripts/render_commentary_video.py --layout biim
        枠・字幕・戦況パネル・音声を一発合成（課金なし・何度でもやり直し可）。
        fillers.jsonl/predictions.jsonlは両方とも「あれば自動マージ」方式
  ↓ renders/<名前>/<名前>_commentary_biim.mp4
```

- 合成のやり直し（パス2）にBedrock/VOICEVOXは不要。レイアウト変更はパス2の再実行だけ
- フィラーの作り直しはパス1.5の再実行だけ（約1円）。パス1のやり直しが必要なのは
  「解析自体の改善」か「states/timeline形式の変更」のときだけ
- パス1.6（予測→回収）は任意ステップ。`predictions.jsonl`が無ければパス2は
  何も変わらず動く（fillers.jsonlと同じ「無ければ何も起きない」自動検出方式）
- パス0（選出前チームプレビュー）も任意ステップだが、**他の任意ステップと違い
  パス1より必ず先に実行する必要がある**（`team_preview.json`はパス1起動直後の
  `Pipeline.__init__`で1回だけ読み込まれる。パス1実行後に置いても・パス1実行中に
  置いても反映されない）

## 工程の分担（どこが機械的で、どこに判断が要るか）

| # | 工程 | 実行場所 | 性質 |
|---|------|---------|------|
| ① | パス1（解析＋素材） | Windows（要VOICEVOX） | **機械的**（コマンド1つ） |
| ② | パス1.5（フィラー生成） | Windows（要VOICEVOX・EC2） | **機械的**（コマンド1つ） |
| ②.5 | パス1.6（予測→回収・任意） | Windows（要VOICEVOX・EC2） | **機械的**（コマンド1つ・材料が無ければ0件で正常終了） |
| ③ | ネタバレ検査 | WSL | **半自動**: `check_spoilers.py`が技名先読みを自動検出（実測されたネタバレの全パターン）。＋タイムラインの目視で勝敗・気絶の先取りを確認。predictions.jsonlも②.5実行時は③の前に生成しておき、回収文が未来情報を含んでいないか目視確認に含めること（当落判定自体はPython側の確定事実だが、③のスクリプト対応状況は実装時に要確認） |
| ④ | パス2（合成） | WSL | **機械的**（コマンド1つ） |
| ⑤ | 仕上げ確認 | どちらでも | **目視**: フレーム抽出2〜3枚＋通し視聴 |

⚠️ **③は省略禁止**。フィラーのネタバレ発生率は実測で1〜2割あり、プロンプト側の
抑制だけでは防ぎきれない（2026-07-14に検査スキップした生成分から1件が動画まで
到達した実例あり）。**フィラーを再生成したら必ず③からやり直す**こと。

## persona（キャラクター設定）

- `--persona kurepi`（デフォルト）＝花圓くれぴの名前・口調（語尾♪・タメ口・テンション高め）。
  `--persona neutral`＝名乗り・一人称のキャラ付けを外した中立実況（3Dモデル一時差し替え
  検証用・2026-08-14新設）
- **⚠️2026-08-15〜当面は`--persona neutral`で運用**（3Dモデル外注完成までのつなぎで
  四国めたん公式VRMを試用中のため。VRM自体はVMC側で手動差し替え・コード変更不要）
- **パス1・パス1.5の両方に必ず同じ値を指定すること**（パス1で決まった口調にパス1.5の
  フィラーが合わせる設計のため。片方だけ付け忘れてkurepi口調のまま混在した実例あり
  ＝`manifest.jsonl`の実況文に「♪」等が残っていないか要確認）

## 実況の口調（2026-08-21更新: 事実＋意見）

「事実の説明だけの実況」から「事実＋意見」への改善（出力ルールへの追加のみ・コード
アーキテクチャは無変更）。`kurepi_persona.OUTPUT_RULES_LINES`／`server.py`側の対応箇所に
「驚き・期待・不安等の感情、次の一手への短い予想を一言添えてよい」というルールを追加した。
既存の断定禁止ルール（技の対象・タイプ相性・場のコンディション等の事実に関するもの）は
無変更で、感情・予想はそれとは別枠として明示的に許可している。試合をまたぐ長期の
「予測→回収」演出は別パス（上記パス1.6・`scripts/generate_predictions.py`）が担当する。
詳細な設計判断の経緯: `docs/design/prediction-payoff-commentary-idea.md`。

## プロンプトのデバッグログ（2026-08-21新設）

RAGヒント（`move_effect_hint`/`move_target_hint`/`condition_hint`等）が実際にLLMへ
どう組み込まれたか確認したい時のためのオプション。**当面は付けたままでOK**（不要になったら
外してよい）。

- パス1（`pipeline.py`）: `--debug-prompts` を付けると、Phi-3ローカルフォールバックに
  送った実際のプロンプト全文が `logs/pipeline_*.log` に出る（`src.commentary.phi3_client`
  ロガーだけDEBUGに上げる仕組み・他モジュールのdebugログは混ざらない）
- パス1.5・パス1本体のBedrock経路（実況の主経路）: `pipeline.py`側のフラグでは出ない。
  EC2の`server.py`起動環境で環境変数 `DEBUG_PROMPTS=1` を設定して`sudo systemctl restart
  pokemon-api`（EC2側のログ確認が必要。恒常化するならsystemdユニットに
  `Environment=DEBUG_PROMPTS=1`を追加）

## 実行手順

### パス0（Windows・任意・2026-08-24新設・必ずパス1より前に実行）

```
venv\Scripts\python.exe scripts\team_preview_gui.py --render-dir renders\<動画名>
```

- 「対戦準備中」画面（自分・相手それぞれ6匹の構築がスプライトで表示される画面。
  テキスト名は無いのでOCRでは自動取得できない）を録画から目視し、自分・相手それぞれ
  6匹（種族名のみ・持ち物や特性はこの画面には出ないので入力不要）をGUIに入力して保存する
- 自分の構築は同じ構築を繰り返し使うことが多いので、「プリセット」欄で名前を付けて保存し
  次回以降はドロップダウンから一括読込できる
- 保存すると `renders/<動画名>/team_preview.json` ができる。**この後にパス1を実行**すると
  実況プロンプトに「選出前チームプレビュー」として自動で注入される
  （`_build_vision_prompt`/`_build_script_prompt`両方・持ち物/特性/技構成は不明なので
  断定しないよう指示付き）
- 相手の構築を入力しておくと、パス1.6（`generate_predictions.py`）が
  「選出予想→battle_startで回収」も自動生成するようになる（2026-08-24新設。
  下記パス1.6を参照）
- ⚠️ WSLには`tkinter`が入っていないため起動確認はWindows側でのみ可能（`pose_tuner_gui.py`と同じ制約）
- パス1の「同じ--render-outへの再実行で前回素材をクリア」対象に`team_preview.json`は
  **含まれない**ため、パス1を何度再実行してもteam_preview.jsonは消えない
  （試行錯誤しながらパス1をやり直しても入力し直す必要はない）

### パス1（Windows PowerShell）

```
# ⚠️ 事前に必ず VOICEVOX を起動する（忘れると素材ゼロ・Bedrock代だけ消える）
# ⚠️ 当面 --persona neutral を付ける（上記「persona」参照）
# ⚠️ 当面 --debug-prompts も付ける（下記「プロンプトのデバッグログ」参照）
venv\Scripts\python.exe src/pipeline.py --input "D:\ゲーム録画\<動画>.mp4" ^
  --end-model runs/detect/train_end_screen2/weights/best.pt ^
  --ec2-url http://<EC2のIP>:5000 --conf 0.3 --render-out renders/<動画名> --persona neutral --debug-prompts
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
# ⚠️ --persona はパス1と必ず同じ値にする（当面 neutral）
# まず --dry-run でフィラー文面を確認（VOICEVOX不要・fillers.jsonlは書かれない）
venv\Scripts\python.exe scripts\generate_gap_commentary.py renders\<動画名> --ec2-url http://<EC2のIP>:5000 --persona neutral --dry-run
# 文面OKなら --dry-run を外して本実行（これでfillers.jsonlが書かれる）
venv\Scripts\python.exe scripts\generate_gap_commentary.py renders\<動画名> --ec2-url http://<EC2のIP>:5000 --persona neutral
```

⚠️ dry-runは**ファイルに書き込まない**。「実行した」と思ってもfillers.jsonlの
タイムスタンプが古いままなら、それはdry-runだった可能性が高い。
⚠️ Bedrock生成は毎回ガチャ。dry-runで見た文面と本実行の文面は別物になる。

### パス1.6（予測→回収・Windows・VOICEVOX起動必須・任意・2026-08-21新設）

```
# まず --dry-run で予測・回収の文面を確認（VOICEVOX不要・predictions.jsonlは書かれない）
venv\Scripts\python.exe scripts\generate_predictions.py renders\<動画名> --ec2-url http://<EC2のIP>:5000 --persona neutral --dry-run
# 文面OKなら --dry-run を外して本実行
venv\Scripts\python.exe scripts\generate_predictions.py renders\<動画名> --ec2-url http://<EC2のIP>:5000 --persona neutral
```

- 材料（場のコンディションを確立した技）が無い試合は「予測ポイントなし」で0件のまま
  正常終了する（空の`predictions.jsonl`が書かれる）。毎回何かが生成されるとは限らない
- **選出予想（2026-08-24新設）**: パス0で`team_preview.json`（相手の構築）を保存済みなら、
  条件系予測とは無関係に「相手のリード予想→battle_startで回収」も自動生成される
  （相手の構築が無ければこちらは0件のまま・条件系予測には影響しない）
- `--persona`はパス1・パス1.5と同じ値にすること
- 未実行でも`predictions.jsonl`が存在しなければパス2は従来通り動く（省略可能な機能）

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
     **2026-08-03ポーズ統合 → 場面別ポーズ分岐 2026-08-15**: 表情が変わった瞬間の
     リアクションは、`explore_avatar_poses.py`（＋`pose_tuner_gui.py`）で実機検証済みの
     ポーズへSLERP遷移→保持→idle_downへ戻す動作に統合。同じ表情でも発生場面
     （`_pose_variant_key`）でポーズを変える設計に拡張済み: Joy=`victory_arms_up`
     （battle_end＝試合の勝敗が決まった瞬間のみ）/`fist_pump_right`（faint＝1匹倒す度の
     軽いガッツポーズ）・Sorrow=`bow_apologetic`（自分が倒された時）/`thinking_chin`
     （sentiment＝ピンチ等の実況キーワード反応）・Fun=`head_tilt_curious`（sentiment）/
     `lean_back_confident`（battle_start＝対戦開始の意気込み）。これで元々未使用だった
     `fist_pump_right`/`thinking_chin`/`lean_back_confident`の3ポーズも全て稼働＋
     腕を大きく振り上げる`victory_arms_up`の発火頻度も下がる（`--avatar-crop`から
     はみ出しやすい問題の緩和にもなる。上記地雷リスト参照）。マッピング外の表情は
     Neckの標準うなずきにフォールバック。ポーズ再生中は該当ボーンを常時スウェイ/
     ランダム仕草から一時除外（`_suspend_bones`/`_resume_bones`）して競合を防ぐ。
     **事前にVMCの設定画面でReceiver（39540 or 39541）の「有効化」チェックボックスを
     ONにしておくこと**（デフォルトOFF・OFFのままだと表情が一切変わらない＝実機で
     確認済みの地雷）。schedule.jsonが必要なので先に`render_commentary_video.py <動画名>
     --dry-run`を実行しておく。表情マッピングは`scripts/play_and_animate_avatar.py`の
     `_EVENT_EXPRESSION`/`_FAINT_EXPRESSION`/`_BATTLE_RESULT_EXPRESSION`/
     `_POSITIVE_KEYWORDS`/`_NEGATIVE_KEYWORDS`/`_POSE_VARIANTS`/`_pose_variant_key`を、
     モーションは`_IDLE_BONES`/`_NOD_*`/`_GESTURE_*`/`play_pose_reaction`を参照。
     - **手順は「スクリプト起動→OBS録画→Enter」の順**（2026-07-30変更・T-pose対策後）:
       ①スクリプトを起動すると即座に腕を下ろした初期姿勢をOSC送信→
       ②「録画が始まったらEnterを押してください」の表示を待ってここでOBS録画を開始→
       ③録画開始後にEnterを押す（この瞬間からWAV再生・表情連動が始まる）。
       録画開始前にTポーズが解消されるため、**録画にTポーズが一切映り込まない**
       （旧手順「録画→スクリプト実行」だと録画開始直後の一瞬Tポーズが映っていた）。
       この順序なら録画開始がWAV再生開始より確実に先になるので、下記
       `--avatar-offset`は0固定でよい（目視で秒数を計る必要がなくなった）
4. 合成: `python3 scripts/render_commentary_video.py renders/<動画名> --layout biim --avatar-video <録画mp4> --avatar-offset <秒> --avatar-crop <w:h:x:y> --avatar-width <px> --avatar-x-shift <px> --avatar-y-shift <px>`
   - 右下344px幅にクロマキー合成（`--avatar-width`/`--avatar-chroma`で調整可）
   - `--avatar-crop`（任意）: 全身録画から上半身だけを切り出してから拡大する
     （例 `"300:480:810:230"`＝顔・肩まわり）。全身のまま縮小すると人物が極小になるため、
     バストアップで見せたい場合は指定推奨
   - `--avatar-x-shift`/`--avatar-y-shift`（2026-08-15新設・既定0）: 右下固定位置からの
     追加オフセット（x正=右へ・y正=下へ、画面外にはみ出た分は見えなくなる）。
     `--avatar-width`を大きくすると右下固定のままでは戦況パネルの大部分が隠れるため、
     人物の下半身側を画面外に追い出してパネルを見せる範囲を調整する用途
   - **2026-08-15実績値（四国めたん公式VRM・1920x1080）**: `--avatar-crop "663:765:631:314"
     --avatar-width 700 --avatar-x-shift 80 --avatar-y-shift 250`。相手の場は完全に見え、
     自分の場の2匹目（下側）のHP/技だけ隠れる状態で確定（ユーザー承認・支障が出たら再調整）
   - クロマキーは類似度0.25＋`despill`（緑かぶり除去）がデフォルト。それでも縁が残る場合は
     `--avatar-chroma`と合わせて類似度をさらに上げる
   - アバター録画が動画より短い場合は最終フレームで静止（正常動作）
   - **offsetは0以上のみ**（録画を先に始める運用で統一）。詳細: `docs/design/v2c-avatar-design.md`

### サムネイル自動生成（改善ロードマップ⑥・任意・WSL）

パス1の素材（manifest.jsonl・states.jsonl）から「盛り上がった瞬間」
（battle_end > faint(KO) > HP急変の優先度・battle_endは既定で結果ネタバレ防止のため
除外）を機械的に選び、元動画の該当フレームにテキストを焼き込んだサムネイルPNGを
出力する。パス2とは独立（動画本体の合成をやり直さなくてもサムネイルだけ再生成できる）。

**2026-08-15確定運用**: オプション無指定（`--avatar-video`だけ指定）で
構築アイコン無し・固定タイトルロゴ「ポケモンダブルバトル／AI自動実況」・
アバターは上半身まで大きく（scale 0.62）・persona=neutralがデフォルトになる
（`main()`のargparse既定値として組み込み済み。ユーザー承認済みの標準仕様）。

```
# 標準運用（2026-08-15〜）: --avatar-videoだけ指定すればOK
python3 scripts/generate_thumbnail.py renders/<動画名> --avatar-video "<録画mp4>"

# 自動選択された実況テキストを使いたい場合（旧来の挙動）は--labelを未指定にせず
# 明示的にNone相当を指定できないため、選ばれたテキストを目視で確認して--labelに渡す
python3 scripts/generate_thumbnail.py renders/<動画名> --time 230.5 --label "きめ台詞！"  # 手動指定
```

- 出力: `renders/<動画名>/thumbnail.png`（`--out`で変更可）
- 元動画パスは`render_info.json`から自動解決（D:\...→/mnt/d/...変換込み）。`--video`で上書き可
- HP急変の検出閾値（既定30pt）は`--hp-swing-threshold`で調整可
- 自動選択が微妙な場合は`--time`/`--label`で手動指定するのが手っ取り早い
- `--big-logo-text`に改行(`\n`)を含めると、その位置で必ず折り返される
  （英数字混じりの短いテキストだと自動折り返しが"AI"のような単語の途中で
  割れることがあるため。既定値`$'ポケモンダブルバトル\nAI自動実況'`もこの対策込み）
- `--avatar-crop`/`--avatar-face-scale`（既定`"663:765:631:314"`/`0.62`）はアバター顔用の
  クロップ・拡大率。動画合成（`render_commentary_video.py --avatar-crop`）と同じ実績値を
  流用しているが、こちらは体まで大きく見せる用途なのでscaleは合成側より大きめ。
  モデルが変わったら合成側と同様に再計測すること

## YouTubeチャプター自動生成（2026-08-16新設・任意・WSL）

パス1の素材（manifest.jsonl）からターンの切り替わり・試合開始・決着のタイミングを拾い、
YouTube概要欄にそのまま貼れるチャプターリストをテキスト出力する。動画本体には無関係
（読み取り専用）なので、パス2より後ならいつでも実行できる。

```
python3 scripts/generate_chapters.py renders/<動画名>
```

- 出力: `renders/<動画名>/chapters.txt`（`--out`で変更可）。標準出力にも表示
- ラベルは「オープニング／試合開始／ターンN／決着」のみで**内容（誰が倒れた・勝敗）は
  含めない**（チャプターは視聴前の概要欄で見えてしまうため、これ以上書くと本編以上の
  ネタバレになる）
- チャプター間隔はYouTube仕様の最低10秒を既定にしている（`--min-gap`で調整可）。
  最低3チャプター無いとYouTube側でチャプター機能が有効化されない点に注意

## プレイヤーの吹き出し（v2d・2026-08-16新設・任意・WSL）

AIの実況ミス（NG）をそのまま茶番のネタにする演出。プレイヤー本人（文字のみ出演）が
「それ違うよ〜」のようにツッコむ吹き出しを画面に焼き込む。ボイロ実況的な「掛け合い」を
台本無しで実現する方向性の一部（[[project_video_first_policy]]参照）。LLMには**書かせない**
（本人の生の言葉としての強みを残すため）。

1. NGチェック（`review_checklist.md`）で見つけた間違いのうち、面白い/分かりやすいものを
   厳選し、`renders/<動画名>/bubbles.jsonl` に1行1件で書く:
   ```
   {"time": 275.8, "text": "あれ、それペリッパーじゃなくてブリジュラス狙ってたんじゃ…？"}
   ```
   `duration`（省略可・既定4.5秒）で表示秒数を調整できる。全部のNGに付けると煩いので
   **厳選する**こと
2. パス2（`render_commentary_video.py`）を通常どおり実行するだけ。`bubbles.jsonl`が
   あれば**自動で検出・合成**される（fillers.jsonlと同じ「無ければ何も起きない」方式・
   専用CLIフラグは無い）
3. 見た目は`scripts/render_bubble_overlay.py`の`draw_speech_bubble`が描画（暖色クリーム地＋
   オレンジ枠の吹き出し・「トレーナー」ラベル・左上配置）。ゲーム画面の寒色（シアン枠）・
   戦況パネルとはっきり区別し「AIとは別の声が割り込んでいる」印象を狙った配色。
   絵文字は字幕と同じフォント（meiryo太字）で豆腐化するので本文に使わないこと
4. 単体で見た目だけ試したい場合:
   ```
   python3 scripts/render_bubble_overlay.py --frame <1920x1080のPNG> --text "好きな文言" --out preview.png
   ```

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
| モックアップ/日報の実フレーム | 対戦相手のトレーナー名が映る | publicリポジトリに載せてよいかはユーザー判断（2026-07-14に確認済み・現方針は許容）。**YouTube公開動画も同じ方針で画面表示は許容**（2026-08-15確認） |
| 実況テキストが相手トレーナー名をそのまま読み上げる懸念 | 動画は公開されるため、音声で個人特定できる名前を読み上げると画面表示より踏み込んだ形になる | **2026-08-15対策済み**: vision/script両プロンプト（`server.py`）に「トレーナー名（画面のIDやハンドルネーム）をそのまま呼ばない・「相手」「お相手」等でぼかす」指示を追加。画面表示自体は上記行の方針どおり許容、実況の読み上げだけを避ける |
| アバター全身録画をそのまま縮小 | 344px枠内で人物が極小・Tポーズの腕が窮屈 | `--avatar-crop`で上半身だけ切り出してから拡大する。**クロップ範囲の決め方**: 収録した生の録画（グリーンバック）から1フレーム抽出し、Pythonで緑以外のピクセルのbounding boxを検出すると人物の実座標が分かる（`PIL`+`numpy`で`is_green = (g>150)&(r<100)&(b<100)`の否定領域を`np.where`）。2026-07-30収録（`HairSample_Female.vrm`・1920x1080・全身がx760-1161/y273-1079に収まっていた）では`--avatar-crop "300:480:810:230"`（過去動画の実績値）でバストアップになり、これがそのまま流用できることを確認済み。**モデルが変わったら同じ手順で再計測すること**（2026-08-15に四国めたん公式VRMへ切り替え済み・新しい実績値は`--avatar-crop "663:765:631:314"`。[[domain_shikoku_metan_appearance]]参照＝ピンク髪メイド服で合っている） |
| `--avatar-width`を大きくすると戦況パネルが隠れる | 700px等に拡大すると右下固定位置のままでは戦況パネル（HP/技ログ）の大部分がアバターの下に隠れる（2026-08-15実機発見） | `--avatar-x-shift`/`--avatar-y-shift`（2026-08-15新設）で右下固定位置から追加オフセットし、人物の下半身側を画面外に追い出してパネルの見える範囲を確保する。実績値は上記「アバター合成」節を参照 |
| バストアップ用クロップだと`victory_arms_up`（バンザイ）で腕が枠外に見切れる | 静止時のbounding boxだけでクロップを決めると、腕を斜め上55°まで振り上げるポーズ（Joy表情の一部・`play_and_animate_avatar.py`の`_POSE_VARIANTS`参照）で肩から上・左右にはみ出す（2026-08-15発見） | **クロップ計測はidle姿勢の1フレームだけでなく、`explore_avatar_poses.py --pose victory_arms_up`を実機で再生させた状態のフレームでも行い、両方のbounding boxを包含する範囲に広げること**（UIや情報表示に被ってもよい前提で許容）。2026-08-15対応: `victory_arms_up`の発火頻度自体もJoy表情の"default"（battle_end＝試合の勝敗が決まった瞬間のみ）に絞り、1匹倒す度の"faint"variantは腕をあまり広げない`fist_pump_right`に変更済み（クロップを広げる対応との合わせ技） |
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
| バトル開始前（「対戦準備中」画面）でmove_usedが誤発火し技/ポケモン名を捏造 | ロスター未確定（情報収集中）の空contextでLLMが辻褄合わせの技名・ポケモン名を作文する（パス1無課金検証で3件実発生・2026-08-12発見） | **2026-08-12に恒久対策済み**: 「対戦準備中」画面の各プレイヤー準備完了ステータス「待機中」が`BattlePhaseClassifier._COMM_RE`の`^待機中$`枝（本来「通信待機中」のOCR誤読対策）に誤一致→communicationフェーズ誤判定→入場演出への遷移でmove_usedが誤発火していたと実OCRで特定。`elif prev == "communication" and curr not in (...)`に`self._battle_started`ガードを追加し、battle_start（初回command_select）より前の誤発火のみをピンポイントで抑制（試合中の正当な発火には無影響） |
| VMCのOSC表情操作（`play_and_animate_avatar.py`）が無反応 | OSCメッセージを送っても表情が一切変わらない | VMCの設定画面でReceiver（39540/39541）のポート番号は入っていても、**「有効化」チェックボックスがデフォルトOFFだと外部OSC入力が反映されない**（2026-07-30に実機で確認・チェックを入れたら即動作した）。必ずONにしてから実行すること |
| switch/move_used実況のタイミングずれ（1テンポ前の交代を今起きたかのように実況） | 「ブリジュラスへの交代だ！」（実際はペリッパー再登場）等、直前の別の交代・技を現在形で実況する（実機2026-08-14_20-46-44の#11/#14/#18） | **2026-08-15に恒久対策済み（後付け生成モードのみ＋プロンプト改善は両モード）**: 原因は2つ。①switchイベントは faint→switch_select 遷移＝**交代選択画面の時点**で発火し、「ゆけっ!」メッセージ（実際の交代先）は数秒後に出るため、LLMが直前の交代で穴埋めしていた。②move_usedは通信終了＝**ターン冒頭（技がまだ出ていない時点）**で発火するのに、event_hintが「今ターンで使われた技を実況」だったため前ターンの技ログを現在形で語っていた。対策=(a)後付け生成時に繰り出しメッセージ履歴（`_sendout_history`・`_note_sendout`で全動画分蓄積）から実際の交代先を逆引きし`switch_focus`として注入（switchはevent_hint差し替え・move_usedは状況行）(b)move_usedのevent_hintを「新しいターンの攻防が始まる場面・戦況全体を実況・過去の技や交代を実況し直さない」に変更。**server.py変更を含むためEC2再デプロイが必要**。manifest.jsonlのcontextに`switch_focus`が出ていれば動作している |
| 気絶イベントの実況未検出（トラッカーは把握しているのに実況されない） | 「たおれた」メッセージやHP0を正しく検出しているのに、faint実況が生成されない（実機2026-08-14_20-46-44で4件: ライチュウ/メタグロス/コノヨザル/ペリッパー。ミロカロスだけ実況された） | **2026-08-15に恒久対策済み**。原因は4系統の複合: ①faintイベント時に`_track_new_faints`が現在の全気絶を「実況済み」登録→未実況の気絶が封印（ライチュウ）②保留faint（move_used統合待ち）が75秒タイムアウト前のturn_startで持ち越され、次ターンのmove_usedに統合されて47秒遅れ＋交代と混同（メタグロス=#14の混乱実況）③連続faintで保留が上書き消滅（コノヨザル）④battle_end時に保留が未フラッシュのまま消滅（ペリッパー）。対策: 実況済み登録をdiff限定化＋turn_start/battle_end/新faint検知時に保留を必ずフラッシュ（event_time=検知時刻なので配置は正確）＋合成faintを自分側にも拡張（従来は相手側のみ）。pipeline.py側のみ=EC2デプロイ不要（faint_focus文言の微修正のみserver.py） |
| move_singleの技の対象（方向）誤認 | 「インファイトがオオニューラへ！」等、実際と違うポケモンを対象として実況する（実機2026-08-14_20-46-44で7件・最頻NG） | **2026-08-15に恒久対策済み（後付け生成モードのみ）**: move_singleは技テキスト検出の瞬間に発火するが、対象の証拠（HP減少・状態異常メッセージ・まもる成功）は数秒〜20秒後に出るため、LLMが場のポケモンから対象を推測して外していた。後付け生成時に「技の直後〜次イベントまでの観測」（`_panel_state_history`のHP/状態異常差分＋`_protect_history`のまもる成功メッセージ）から対象を逆引きし、`move_target_hint`としてプロンプト注入する（観測に厳密に従う指示付き・観測ゼロなら従来の「断定しない」安全策に委ねる）。**server.py変更を含むためEC2再デプロイが必要**。manifest.jsonlのcontextに`move_target_hint`が出ていれば動作している。実データドライランでNG7件中3件を確実修正・残りは断定回避へ倒れることを確認済み。**2026-08-16追加**: 上記は「技の直後の事後観測」のみに頼っており、つるぎのまい等の自分対象の変化技・範囲技（相手全体対象）は観測が原理的に薄い/存在しないため誤爆が残っていた。`moves`テーブルに`target`列（PokeAPI由来・`fetch_move_targets`でbackfill）を追加し、「技そのものの対象範囲」（自分自身/相手全体/自分の場等）という事後観測と独立した事実を`move_range_hint`として`_dispatch_move_commentary`/`_process_event`の両方でbattle_contextに配線→`_compute_move_target_hint`が観測より最優先で合流させる。単体対象（相手単体等）は既存の断定回避指示と同義のためあえて文言を足さない。manifest.jsonlのcontextに`move_range_hint`が出ていれば配線されている（プロンプト自体は`move_target_hint`に統合済みなのでserver.py/phi3_client.py側の新規キーは無し・文言修正のみでEC2再デプロイ必要） |
| 降参で終わった試合のbattle_end実況が気絶・全滅を捏造 | 「〜が落ちた、これで全滅です」等、実際は降参なのに倒れた描写で締める（実機2026-08-14_20-46-44で実発生）。battle_resultも未検出になりがち | **2026-08-15に恒久対策済み**: battle_endの発火経路は「YOLO終了画面」「フェーズ遷移（`_END_KW`の「降参が選ばれ」）」の2つあり、降参のWIN/LOSE待機（2026-08-14実装）はYOLO経路のみ対応でフェーズ遷移経路が先に即発行していた。フェーズ遷移経路にも同じ待機を追加し、`_battle_surrendered`フラグを新設してプロンプト（server.py/phi3両方）に「降参による決着・倒れた描写をしない」を注入する。**server.py変更を含むためEC2再デプロイが必要**。manifest.jsonlのcontextに`battle_surrendered: true`が出ていれば動作している |
| パス0（team_preview_gui.py）をパス1の後に実行してしまう | `team_preview.json`を保存してもプロンプトに一切反映されない | `_team_preview_hint`はパス1起動直後（`Pipeline.__init__`）で1回だけ読み込む設計のため、**パス1より必ず先に**保存する必要がある。既にパス1を実行済みでも、パス0→パス1の順で撮り直せばよい（team_preview.jsonは再実行でクリアされないので、パス0だけ先にやってからパス1を再実行すれば反映される） |

## レイアウト調整ポイント（scripts/render_commentary_video.py の定数）

| 定数 | 現在値 | 意味 |
|------|--------|------|
| `_SUBTITLE_FONT_SIZE` / `_SUBTITLE_WRAP_CHARS` | 48 / 37 | 字幕サイズと折り返し文字数（**必ずセットで変更**: wrap≒1824÷フォントpx） |
| `_BIIM_GAME_W/H/X/Y` | 1440×810 (16,12) | ゲーム画面の縮小配置 |
| `_PANEL_TEXT_X` / `_PANEL_BAR_W` | 1496 / 300 | パネル左端・HPバー幅 |
| `_hp_bar_color` | 緑>50/黄>20/赤 | HPバー3色（ゲーム準拠） |
| `_FILLER_MAX_SHIFT_SEC` | 12 | フィラー配置の最大ずれ（大きくするとネタバレ防御が弱まる） |
| `_DEFAULT_GAIN` | 1.4 | 実況音量。ゲーム音とのバランスは`--gain`で上書き可 |
| server.py `_gap_filler_count` | 20秒/件・上限5 | フィラー密度（変更はEC2再デプロイ必要）。18秒/件・上限5→2026-07-30視聴fb「多い」で30秒/件・上限3に減量→「もう少し増やしたい」で25秒/件・上限4に再調整→さらに「あ、あが耳につく・フィラー減らして実況を活かしたい」で40秒/件・上限3に再々調整→**2026-08-15訂正**: 上記「あ、あ」fbは実際には言葉遣い（相槌の書き出し）への指摘で、無言埋め自体の生成頻度を絞る話ではなかったとユーザーから訂正。相槌対策は下記「書き出しバリエーション指示」で別途対応済みのため、無言埋めは20秒/件・上限5へ積極的に戻した |
| generate_gap_commentary.py `_DEFAULT_MIN_GAP_SEC` | **6秒**（`--min-gap`で上書き可） | フィラー対象とする最小無言秒数。25秒→40秒→2026-08-15に20秒へ再訂正→**2026-08-21: 実測で動画の39%が無言（5秒以上＝ラジオの放送事故基準）と判明し6秒へ大幅引き下げ**。下記の可変フィラー長とセットで機能する |
| generate_gap_commentary.py `_INTRO_MIN_GAP_SEC`（新規） | **3秒**（2026-08-21・旧8秒） | 動画冒頭だけに使う特別な閾値（`compute_gaps`）。開始時挨拶を確実に入れるため、通常の`_DEFAULT_MIN_GAP_SEC`より短い基準で対象化し`is_intro: True`を付ける。`_DEFAULT_MIN_GAP_SEC`を6秒へ下げたのに合わせ、冒頭はそれより更に短くした。server.pyの`_build_script_prompt`が`gap.get("is_intro")`を見て「最初のフィラーは視聴者への挨拶から始めること」を必須指示する（2026-08-15新設・EC2再デプロイ必要） |
| server.py `_build_script_prompt`のフィラー目標文字数（新規・2026-08-21） | 無言区間の長さで可変（<10秒=15〜25文字／10〜20秒=40〜60文字／20秒以上=60〜100文字） | 従来60〜100文字（読み上げ10〜18秒）固定で、5〜19秒の無言区間（実測の最多帯）に収まらず`fit_fillers()`で破棄されていた対策。predictモード（予測→回収）は単発発火なので対象外・常に60〜100文字のまま（EC2再デプロイ必要） |
| generate_gap_commentary.py の0件区間再試行（新規・2026-08-21） | 1回だけ再試行 | Bedrockが区間を丸ごと0件で返した場合（ガチャ外れ）、`find_empty_gaps()`で検出しその区間だけ`request_fillers()`を再実行する。無言区間の取りこぼしは実害が大きいため、二重生成のコストより取りこぼし防止を優先 |
| プロンプトの書き出しバリエーション指示 | （新規） | 2026-07-30〜: 「あ、あ」等の相槌の書き出し連発を抑制する指示を`_build_script_prompt`（フィラー用）・`_build_vision_prompt`（実況本編用）の両方に追加済み。片方だけ直しても同じキャラ設定を共有しているためもう片方で再発するので注意 |

パネル下部（y660以降）と画面右下はv2c（アバター）用に空けてある。

## 横展開（新しい動画への適用）

パス1→1.5→2を上から順に実行するだけ。動画ごとに `renders/<動画名>` を分ける。
検証チェックリストは毎回実施（特にネタバレ検査）。書き起こしがある動画なら
パネルのHP値と書き起こしの照合も行うと確実。

**投稿準備の一環として`scripts/generate_chapters.py`も毎回実行する**（2026-08-16〜）。
パス2完了後ならいつでも実行できる（動画本体とは無関係な読み取り専用処理）。
出力される`chapters.txt`をYouTubeの概要欄にそのまま貼り付ける。

**プレイヤーの吹き出し（v2d）も毎回検討する**（2026-08-16〜）。手順:
1. 一度パス2まで通して合成し、通し視聴（検証チェックリスト）でNGを見つける
2. 面白い/分かりやすいNGを厳選して`renders/<動画名>/bubbles.jsonl`に手書きする
   （詳細は「プレイヤーの吹き出し」節参照）
3. パス2を**再実行するだけ**（`bubbles.jsonl`を自動検出・音声もBedrockもVOICEVOXも
   再実行不要・パス2のやり直しと同じ扱い）

## 関連ファイル

- 実装: `src/output/render_sink.py`（パス1素材出力）・`src/pipeline.py`（`_record_panel_state`/`_render_context`/瞬間ログフック/`BattleStateTracker.fainted_names`・`diff_fainted_side`/`_load_team_preview_hint`）・`src/api/server.py`（`/api/script`・プロンプト・`_build_script_prompt`のmode="predict"/`_build_payoff_prompt`）・`scripts/generate_gap_commentary.py`・`scripts/generate_predictions.py`（予測→回収・2026-08-21新設）・`scripts/render_commentary_video.py`（`load_predictions`）・`scripts/generate_thumbnail.py`（サムネイル自動生成）・`scripts/generate_chapters.py`
（YouTubeチャプター自動生成）・`scripts/render_bubble_overlay.py`（プレイヤーの吹き出し・v2d）
・`scripts/play_commentary_track.bat`（アバター録画用wav再生・Windows）・`scripts/play_and_animate_avatar.py`（表情連動版wav再生・Windows・改善ロードマップ③）
・`scripts/team_preview_gui.py`（選出前チームプレビュー入力GUI・パス0・2026-08-24新設）
・`src/pokedb/team_preview.py`（保存/読込/整形ロジック・自分の構築プリセット）
- テスト: `tests/test_render_sink.py`・`tests/test_render_video.py`・`tests/test_gap_commentary.py`・`tests/test_generate_predictions.py`・`tests/test_server.py`（TestScript系・TestBuildScriptPromptPredictMode・TestBuildPayoffPrompt）・`tests/test_play_and_animate_avatar.py`・`tests/test_render_bubble_overlay.py`・`tests/test_generate_chapters.py`・`tests/test_team_preview.py`
- 設計: `docs/adr/ADR-009-video-first-commentary.md`・レイアウト原案 `docs/design/frame-mockups/mockup_A_biim.png`
- 経緯・実測値: `docs/daily/2026-07-14.md`
