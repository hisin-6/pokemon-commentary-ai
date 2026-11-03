# 外注VRMモデル差し替えチェックリスト（人間向け・上から順に）

外注の正式モデル（花圓くれぴ・VRM 0.x・ADR-006）が納品されたら、四国めたん公式VRMから
差し替えるときにこの順で確認する。根拠になった地雷は `.claude/skills/video-commentary/SKILL.md`
の地雷リストにまとまっている。

> 目安: Ⅰ〜Ⅱ（WSLだけ）10分、Ⅲ〜Ⅴ（VMC＋OBS実機）1〜2時間。
> 変更する値は全部「モデル依存の実測値」なので、**四国めたん用の値は消さずにメモしておく**こと
> （外注モデルに不具合があったら戻すため）。

## 0. 現行値（四国めたん公式VRM・2026-08-15時点の実績）

差し替え前の控え。再計測した新しい値は右の欄に書き込む。

| 項目 | 四国めたん実績値 | 新モデル |
|------|------------------|----------|
| `render_commentary_video.py --avatar-crop` | `"663:765:631:314"` | |
| `--avatar-width` / `--avatar-x-shift` / `--avatar-y-shift` | `700` / `80` / `250` | |
| `--avatar-chroma`（クロマキー色） | `0x00FF00`（既定値） | |
| `generate_thumbnail.py` の `_CLI_DEFAULT_AVATAR_CROP` | `"663:765:631:314"` | |
| `generate_thumbnail.py` の `_CLI_DEFAULT_AVATAR_FACE_SCALE` | `0.62` | |
| `play_and_animate_avatar.py` の `_IDLE_POSE_DEG` / `_IDLE_POSE_AXIS` | `80.0` / `"z"` | |
| `--persona`（パス1・1.5・1.6・サムネ） | `neutral` | |

## Ⅰ. 納品物の確認（WSL・VMCに読み込む前）

- [ ] `.vrm` と UniVRM 0.x のセットアップ済み `.unitypackage` が両方届いている（ADR-006で依頼済み）
- [ ] 一発チェックを流す:
      ```bash
      python3 scripts/check_vrm_model.py "/mnt/c/path/to/kurepi.vrm"
      ```
  - [ ] 終了コード0（必須の表情 `Neutral/Joy/Sorrow/Fun/Angry` と、回す予定のボーンが全部揃っている）
  - [ ] **VRMバージョンが0.x**（1.0の警告が出たら表情名が `happy` 等に変わるので、パイプライン改修が要る＝モデラーさんに0.xで再出力を相談）
  - [ ] 「バインドが1つも無い表情」の警告が出ていない（名前だけあって顔が動かない状態）
  - [ ] 推奨の `A/I/U/E/O/Blink` も揃っている（リップシンク・まばたき用）
  - [ ] メタ情報（作者・ライセンス・商用利用/改変の可否）が依頼内容と一致している（YouTube公開に使うため）
- [ ] `.vrm` を `renders/` と同じくgit管理外の場所に置く（`docs/charactordesign/` と同様、リポジトリに入れない）

## Ⅱ. VMCに読み込んで基本動作（Windows）

- [ ] VMCでモデルを読み込み、背景をグリーン単色にする
- [ ] リップシンク入力が **CABLE Output** になっている（モデルを変えても設定は残るはずだが念のため）
- [ ] 設定画面のReceiver（39540/39541）の**「有効化」チェックがON**（デフォルトOFF・OFFだと表情もポーズも一切反応しない）
- [ ] 表情の単体テスト: `venv\Scripts\python.exe scripts\test_vmc_expression.py --port 39540 --name Joy` を `Sorrow`/`Fun` でも流し、目で見て変わる
- [ ] VMCがリップシンク中に固まったら、まずVMC再起動（原因不明の既知症状）

## Ⅲ. ポーズ角度の再検証（Windows・VMC起動中）

ボーン角度はモデルの体型・初期姿勢で見え方が変わる。四国めたん基準で調整した値なので全部見直す。

- [ ] **Tポーズ対策（腕下げ）**: `venv\Scripts\python.exe scripts\test_vmc_pose.py --port 39540 --deg -80` で `LeftUpperArm`/`RightUpperArm` のZ軸80°で腕が自然に下がるか確認
  - 指が服に深く埋まる・腕が浮く場合は `play_and_animate_avatar.py` の `_IDLE_POSE_DEG`/`_IDLE_POSE_AXIS` を調整
- [ ] **6ポーズを1つずつ再生して目視**:
      ```powershell
      venv\Scripts\python.exe scripts\explore_avatar_poses.py --port 39540 --list
      venv\Scripts\python.exe scripts\explore_avatar_poses.py --port 39540 --pose victory_arms_up
      ```
      対象: `victory_arms_up` / `fist_pump_right` / `bow_apologetic` / `thinking_chin` / `head_tilt_curious` / `lean_back_confident`
  - [ ] 腕が体や髪・服にめり込んでいない
  - [ ] 指の曲がりが不自然でない（手の大きさが違うと特に崩れやすい）
  - 崩れたポーズは `scripts\pose_tuner_gui.py --port 39540` で読み込み→調整→「Pythonコードを出力」→`explore_avatar_poses.py` の `POSES` に貼り替える
- [ ] **待機モーション**: `play_and_animate_avatar.py` で数十秒流し、Spine/Chestの揺れ（`_IDLE_BONES`）と首かしげ等の仕草が大きすぎ/小さすぎないか
- [ ] `fist_pump_right` への遷移が不自然な件（見送り中の既知課題）が新モデルでも出るか、ついでに見ておく

## Ⅳ. 収録→クロップ再計測（Windows録画＋WSL計測）

- [ ] 手順は「スクリプト起動→OBS録画→Enter」の順（録画にTポーズが映らない・`--avatar-offset 0` でよい）
- [ ] OBSは「ウィンドウキャプチャ」でVMCを撮る
- [ ] **クロップは2フレームで計測する**: idle姿勢の1フレームだけで決めると、`victory_arms_up` で腕が枠外に見切れる
  1. idle姿勢のフレーム
  2. `explore_avatar_poses.py --pose victory_arms_up` を再生中のフレーム
- [ ] 各フレームを抜き出して人物のbounding boxを出す（WSL）:
      ```bash
      ffmpeg -y -ss <秒> -i <アバター録画.mp4> -frames:v 1 /tmp/avatar_frame.png
      python3 - <<'EOF'
      import numpy as np
      from PIL import Image
      a = np.asarray(Image.open("/tmp/avatar_frame.png").convert("RGB")).astype(int)
      r, g, b = a[..., 0], a[..., 1], a[..., 2]
      is_green = (g > 150) & (r < 100) & (b < 100)
      ys, xs = np.where(~is_green)
      print(f"x {xs.min()}-{xs.max()} / y {ys.min()}-{ys.max()}")
      EOF
      ```
      ※髪や服に緑系の色が入っているモデルだとこの判定が人物を背景扱いする。その場合は下記Ⅴのクロマキー色変更を先に決める
- [ ] 2つのbounding boxを両方含む範囲で `--avatar-crop "w:h:x:y"` を決める（UIに多少かぶるのは許容）
- [ ] 試し合成して `--avatar-width`/`--avatar-x-shift`/`--avatar-y-shift` を詰める。相手の場が全部見えること（自分の場の2匹目が隠れる程度は四国めたん時に許容済み）:
      ```bash
      python3 scripts/render_commentary_video.py renders/<動画名> --layout biim --avatar-video <録画.mp4> --avatar-offset 0 --avatar-crop "<新しい値>" --avatar-width 700 --avatar-x-shift 80 --avatar-y-shift 250
      ```

## Ⅴ. クロマキー・見た目の仕上げ

- [ ] 合成後のフレームを2〜3枚抜き出して目視（SKILL.md「合成後の検証チェックリスト」と同じ手順）
  - [ ] 輪郭に緑のフリンジが残っていない（残るなら `render_commentary_video.py` の `_AVATAR_SIMILARITY`（既定0.25・CLIフラグ無し）を上げる）
  - [ ] **髪・服・目に緑系の色があって透けていない**（キャラデザに緑が入っていたら、背景をブルー等に変えて `--avatar-chroma 0x0000FF` のように指定する）
  - [ ] 揺れ物（髪・スカート等）が激しく動いたフレームでもクロップ枠に収まっている
- [ ] サムネイルを1枚作って確認（`--avatar-crop`/`--avatar-face-scale` は合成用とは別の値・体まで大きく見せる用途）:
      ```bash
      python3 scripts/generate_thumbnail.py renders/<動画名> --avatar-video "<録画.mp4>" --avatar-crop "<新しい値>" --avatar-face-scale 0.62
      ```

## Ⅵ. 値の反映とキャラ設定の切り戻し（ここだけコード・ドキュメント変更）

実機で値が固まってから、まとめて反映する。

- [ ] `scripts/generate_thumbnail.py` の `_CLI_DEFAULT_AVATAR_CROP` / `_CLI_DEFAULT_AVATAR_FACE_SCALE` を新しい値に
- [ ] `play_and_animate_avatar.py` の `_IDLE_POSE_*`、`explore_avatar_poses.py` の `POSES`（調整したものだけ）
- [ ] **personaを `kurepi` に戻すか決める**（`neutral` は「外注モデル完成までのつなぎ」の方針だった）。戻すなら:
  - パス1・1.5・1.6で `--persona` を付けない（既定が `kurepi`）、またはすべてに `--persona kurepi` を付けて揃える（片方だけだと口調が混在する地雷あり）
  - サムネイルのpersona既定（`generate_thumbnail.py` 2026-08-15確定運用＝neutral）も合わせて見直す
- [ ] SKILL.md の更新:「アバター合成」節の実績値・「persona」節の当面neutral運用・地雷リストの「モデルが変わったら再計測」行
- [ ] `docs/manual/video-commentary-quickstart.md` のアバター手順（必要なら）
- [ ] WSLで `pytest` 全パス（定数を変えたのでテストの期待値に引っかかるものがないか）
- [ ] 新モデルで1本、パス1〜アバター合成〜サムネイルまで通しで作って視聴確認

## 関連

- 判断の経緯: `docs/adr/ADR-006-3d-model-rendering.md`（外注VRM 0.x方針）・`docs/design/v2c-avatar-design.md`（アバター合成設計）
- 一発チェック: `scripts/check_vrm_model.py`（2026-10-29作成）
- 地雷と実績値の出典: `.claude/skills/video-commentary/SKILL.md`
