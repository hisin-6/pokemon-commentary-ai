# 提案メモ: アバターのモデル依存値を設定ファイル（モデルプロファイル）に切り出す

- 作成: 2026-11-06（試験期間の小タスク・提案のみ。実装はユーザー判断）
- 対象: `scripts/play_and_animate_avatar.py` / `scripts/explore_avatar_poses.py`（＋参照側の `scripts/check_vrm_model.py`）
- 関連: ADR-006（外注VRM 0.x方針）・`docs/manual/vrm-model-swap-checklist.md`

## 1. 背景

2027年1〜2月に外注VRM（花圓くれぴ）が納品されたら、四国めたん公式VRMから差し替える。
今はモデル依存の値がコード内の定数に直書きされているので、差し替えるたびにPythonを書き換えることになる。
外注モデルに不具合があって四国めたんに戻す場合も、同じ書き換えをもう一度やる必要がある。

## 2. 現状の棚卸し（モデル依存の値がどこにあるか）

| 種類 | 場所 | 現在値 | モデル依存度 |
|------|------|--------|--------------|
| 腕下げ角度 | `play_and_animate_avatar.py` `_IDLE_POSE_DEG` / `_IDLE_POSE_AXIS` / `_IDLE_POSE_BONES` | `80.0` / `"z"` / 両UpperArm | 高（腕の初期姿勢で変わる） |
| 腕下げ角度（**重複**） | `explore_avatar_poses.py` `_IDLE_ARM_DEG` | `80.0` | 高 |
| ポーズ7種のボーン角度 | `explore_avatar_poses.py` `POSES`（約66ボーン指定） | 実機調整済みの値 | 高（体型・腕の長さ・指の構造で頭貫通などが起きる） |
| 表情名 | `play_and_animate_avatar.py` `send_expression` の `all_known`、`_EVENT_EXPRESSION` 等 | `Neutral/Fun/Joy/Sorrow/Angry` | 低（VRM 0.xのプリセット名なら共通） |
| 待機揺れ | `_IDLE_BONES`（Spine/Chest）・`_NOD_*`（Neck）・`_GESTURE_*`（Head） | 振幅数度 | 中（大きく変わる可能性は低い） |
| 表情→ポーズの対応 | `_POSE_VARIANTS` | 演出上の選択 | なし（モデルではなく演出の設定） |
| チェック対象ボーン | `check_vrm_model.py` `_ANIMATE_BONES` | 上記の手動コピー | 参照側（テストで一致を検証済み） |

### 気づいた点

**腕下げ角度 `80.0` が2か所に重複している。** `play_and_animate_avatar.py` の `_IDLE_POSE_DEG`（起動時の腕下げ）と
`explore_avatar_poses.py` の `_IDLE_ARM_DEG`（ポーズ遷移の戻り先 `idle_down` と、各ポーズの腕の基準）は別の定数になっている。
片方だけ直すと、起動直後の腕とリアクション後に戻る腕の角度がずれる。
差し替えチェックリスト（`vrm-model-swap-checklist.md` の0章）には `_IDLE_POSE_DEG` しか書かれていないので、
設定ファイル化を見送る場合でも、チェックリストに `_IDLE_ARM_DEG` を1行追記するとよい。

## 3. 提案

### 案A: モデルプロファイルJSONに切り出す（推奨）

`config/avatar_profiles/<モデル名>.json` を置き、CLIで選ぶ。

```jsonc
// config/avatar_profiles/shikoku_metan.json
{
  "name": "shikoku_metan",
  "idle_pose": {"deg": 80.0, "axis": "z", "bones": ["LeftUpperArm", "RightUpperArm"]},
  "expressions": ["Neutral", "Fun", "Joy", "Sorrow", "Angry"],
  "poses": {
    "victory_arms_up": {"LeftUpperArm": [["z", -55.0]], "RightUpperArm": [["z", 55.0]]}
    // 省略したポーズはコード内の既定値（=四国めたん実績値）を使う
  }
}
```

```
python scripts/play_and_animate_avatar.py <動画名> --osc-port 39540 --avatar-profile kurepi
```

- `--avatar-profile` 未指定なら今の定数をそのまま使う。既存の手順書・コマンドは変わらない
- 腕下げ角度はプロファイルの `idle_pose.deg` 1か所だけにし、`explore_avatar_poses.py` 側もそこを参照する（重複を解消）
- `poses` は「上書きしたいポーズだけ書く」差分方式にする。頭貫通が起きたポーズだけ直せばよい
- `pose_tuner_gui.py` に「今の値をプロファイルJSONに書き出す」ボタンを付ければ、GUIで追い込んだ値をそのまま保存できる（任意）
- `check_vrm_model.py --profile kurepi` で、プロファイルが使うボーンと表情がVRMに揃っているかも一緒に検査できる
- 演出の設定（`_POSE_VARIANTS`・表情の保持秒数）はモデルと関係ないのでコードに残す

### 案B: 定数をモジュール1つ（`avatar_model_config.py`）に集めるだけ

- JSONは使わず、モデル依存の定数をPythonの辞書1つに集める。重複は解消できる
- 切り替えは辞書のキーを書き換える。実装は小さいが、戻すときにコードを編集する点は今と同じ

### 案C: 何もしない（チェックリストへの追記だけ）

- 差し替えは1回きりと割り切る。`_IDLE_ARM_DEG` をチェックリストに追記するだけにする

## 4. 推奨と進め方

**案A**を推す。理由は、納品後に「外注モデルで不具合 → 四国めたんに戻して動画を作る」場面が想定でき、
そのときCLI引数1つで戻せるのが一番安全だから。今は定数を変えずに読み込み口だけ足すので、挙動は変わらない。

実装する場合の段取り（ユーザー復帰後・1〜2時間程度の見込み）:

1. `config/avatar_profiles/shikoku_metan.json` に現行値を書き出す（値は変えない）
2. `play_and_animate_avatar.py` に `--avatar-profile` を追加。未指定時は現状どおり
3. `_IDLE_ARM_DEG` と `_IDLE_POSE_DEG` をプロファイル経由に統一
4. テスト: プロファイル未指定時の送信値が現状と完全一致すること、差分上書きが効くこと、不正JSONで分かりやすいエラーになること
5. `check_vrm_model.py` に `--profile` を追加（任意）
6. SKILL.md・クイックスタート・差し替えチェックリストに `--avatar-profile` を追記

要実機確認: 手順2〜3の後、VMCで四国めたんの見た目が変わっていないことを確認する（テストでは送信値の一致までしか見られない）。

## 5. 判断してほしいこと

- 案A/B/Cのどれにするか
- 案Aの場合、`config/` ディレクトリを新設してよいか（現状リポジトリ直下に `config/` は無い）
