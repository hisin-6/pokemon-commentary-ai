# パス1 無課金検証ワークフロー 対象動画リスト

`D:\ゲーム録画\` 内の全録画を対象に、パス1（Bedrock無課金・Phi-3フォールバック）の
精度検証を進めるための管理リスト。

- 判定基準・手順は `docs/manual/pass1-verification-checklist.md` を参照
- NGは `docs/manual/pass1-verification-ng-findings.md` に集約（この時点では
  修正しない・収集フェーズ。2026-08-10〜運用）
- **この作業をする際は必ずこのファイルを見て、次にどの動画をやるか・
  何本終わったかを確認すること**
- **2026-08-11〜: 各コマンドブロックの3個目に `scripts/screen_pass1.py` を追加**
  （自動一次スクリーニング。既知NGパターンを機械検出し`screening_report.md`を出力・
  フラグの有無で目視の濃淡を判断する。詳細は`pass1-verification-checklist.md`参照）
- **2026-08-11〜: サンプリング方式に切替**（121本は多すぎるためユーザー判断で縮小。
  詳細は下記「サンプリング方針」参照）
- 生成日: 2026-08-10（`ffprobe`で全132本の尺を実測）
- 各動画のコマンドは全て `--game-mode champions`（[[project_champions_v1]]の
  メモで「2026-04-12以降の録画は全部チャンピオンズ」と確認済み）
- コマンドは全て**Windows側**（`venv\Scripts\python.exe`）でそのまま実行可能

## 現在の状況

- 全録画: **132本**
- 実況動画として完成済み（対象外）: **6本**
- 無課金パス1検証 判定済み: **14本**（サンプリング25本中14本完了・残り11本）
- 無課金パス1検証 対象（サンプリング後）: **25本**
- サンプリング対象外（15分超）: **9本**
- サンプリング対象外（15分以内・非選出）: **87本**

## 完成済み実況動画（無課金検証の対象外）

実況動画ファースト方針（ADR-009）で既にパス1〜2まで完成済みの6本。品質は
視聴チェック済みのため、この検証ワークフローの対象からは除外する。

| 尺 | ファイル | renders |
|---|---|---|
| 5:40 | `2026-04-14 20-14-17.mp4` | 20-14-17（完成） |
| 5:56 | `2026-04-13 07-00-19.mp4` | 07-00-19（完成） |
| 6:12 | `2026-04-12 18-12-45.mp4` | 18-12-45（完成） |
| 6:38 | `2026-04-12 16-14-39.mp4` | 16-14-39（完成） |
| 7:42 | `2026-04-13 06-25-46.mp4` | 06-25-46（完成） |
| 19:59 | `2026-04-14 08-15-22.mp4` | 08-15-22（完成） |

## 判定済み（無課金パス1検証ワークフロー）

| # | 尺 | ファイル | レビューMarkdown |
|---|---|---|---|
| 1 | 2:02 | `2026-06-03 22-57-11.mp4` | `renders/2026-06-03_22-57-11/review_checklist.md` |
| 2 | 2:58 | `2026-06-07 12-48-22.mp4` | `renders/2026-06-07_12-48-22/review_checklist.md` |
| 3 | 2:59 | `2026-04-13 06-22-25.mp4` | `renders/2026-04-13_06-22-25/review_checklist.md`（NG2件） |
| 4 | 3:03 | `2026-04-14 21-44-43.mp4` | `renders/2026-04-14_21-44-43/review_checklist.md`（NG3件） |
| 5 | 3:04 | `2026-06-03 21-35-48.mp4` | `renders/2026-06-03_21-35-48/review_checklist.md`（NG5件・自動スクリーニング初実戦投入） |
| 6 | 3:20 | `2026-04-14 08-39-39.mp4` | `renders/2026-04-14_08-39-39/review_checklist.md`（NG3件・ユーザー目視） |
| 7 | 3:56 | `2026-04-12 17-12-38.mp4` | `renders/2026-04-12_17-12-38/review_checklist.md`（NG7件・ユーザー目視） |
| 8 | 4:41 | `2026-04-14 21-40-01.mp4` | `renders/2026-04-14_21-40-01/review_checklist.md`（NG7件・ユーザー目視） |
| 9 | 4:58 | `2026-06-07 12-07-11.mp4` | `renders/2026-06-07_12-07-11/review_checklist.md`（NG9件・ユーザー目視） |
| 10 | 5:06 | `2026-06-06 17-30-38.mp4` | `renders/2026-06-06_17-30-38/review_checklist.md`（NG7件・ユーザー目視） |
| 11 | 9:25 | `2026-06-27 12-13-37.mp4` | `renders/2026-06-27_12-13-37/review_checklist.md`（NG20件・ユーザー目視） |
| 12 | 10:25 | `2026-04-13 06-34-11.mp4` | `renders/2026-04-13_06-34-11/review_checklist.md`（NG15件・ユーザー目視） |
| 13 | 11:32 | `2026-07-03 21-06-34.mp4` | `renders/2026-07-03_21-06-34/review_checklist.md`（NG29件・ユーザー目視） |
| 14 | 14:42 | `2026-07-03 21-54-06.mp4` | `renders/2026-07-03_21-54-06/review_checklist.md`（NG22件・ユーザー目視） |

## サンプリング方針（2026-08-11決定）

残り121本のうち3本を検証したところ、既知パターンの再発が続く一方で新パターンも
継続的に見つかっており（`ng-findings.md`参照）、全数を律儀にこなす負荷対効果が
見合わないとユーザーが判断。以下の方針で対象を絞り込んだ:

1. **15分以内の動画のみを対象**（長尺は情報量が多く1本あたりの検証コストが
   跳ね上がるため。15分超の9本は今回は対象外）
2. **15分以内の121-9=112本から、尺で均等間隔サンプリングして25本を選出**
   （最短3:20〜最長14:42の範囲を満遍なくカバー。単純に短い順に25本取ると
   短尺の試合ばかりに偏るため、尺のバリエーションを確保する狙い）
3. 選出結果は日付も4月/6月/7月にまたがっており、特定日への偏りもない
   （4月9本・6月13本・7月3本）
4. 非選出の87本＋15分超の9本は下記「サンプリング対象外」に一覧を残す
   （今後NGパターンが増え続けるようなら追加サンプリングの候補として使える）

## 未処理（サンプリング後・25本・尺の短い順）

チェック済みになったら見出しの `- [ ]` を `- [x]` に変更し、判定済みテーブルにも
追記すること。各項目のコマンドはそのままコピペで実行可能（1個目=パス1実行・
2個目=レビューMarkdown生成・3個目=自動スクリーニング。いずれもWindows側で続けて実行）。

- [x ] 1. 3:20 — `2026-04-14 08-39-39.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 08-39-39.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_08-39-39
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_08-39-39
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-14_08-39-39
  ```
- [x ] 2. 3:56 — `2026-04-12 17-12-38.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-12-38.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-12-38
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-12-38
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-12_17-12-38
  ```
- [x ] 3. 4:41 — `2026-04-14 21-40-01.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 21-40-01.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_21-40-01
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_21-40-01
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-14_21-40-01
  ```
- [ x] 4. 4:58 — `2026-06-07 12-07-11.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 12-07-11.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_12-07-11
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_12-07-11
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-07_12-07-11
  ```
- [x ] 5. 5:06 — `2026-06-06 17-30-38.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-30-38.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-30-38
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-30-38
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-06_17-30-38
  ```
- [ ] 6. 5:34 — `2026-04-14 23-22-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 23-22-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_23-22-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_23-22-26
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-14_23-22-26
  ```
- [ ] 7. 5:56 — `2026-06-03 21-48-45.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-48-45.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-48-45
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-48-45
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-03_21-48-45
  ```
- [ ] 8. 6:14 — `2026-06-21 17-17-00.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 17-17-00.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_17-17-00
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_17-17-00
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-21_17-17-00
  ```
- [ ] 9. 6:28 — `2026-06-04 21-44-45.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-44-45.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-44-45
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-44-45
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-04_21-44-45
  ```
- [ ] 10. 6:51 — `2026-04-14 20-32-31.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 20-32-31.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_20-32-31
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_20-32-31
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-14_20-32-31
  ```
- [ ] 11. 7:03 — `2026-06-06 19-34-04.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 19-34-04.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_19-34-04
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_19-34-04
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-06_19-34-04
  ```
- [ ] 12. 7:23 — `2026-04-12 17-04-52.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-04-52.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-04-52
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-04-52
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-12_17-04-52
  ```
- [ ] 13. 7:33 — `2026-06-27 12-34-29.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-27 12-34-29.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-27_12-34-29
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-27_12-34-29
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-27_12-34-29
  ```
- [ ] 14. 7:42 — `2026-04-13 21-46-08.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 21-46-08.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_21-46-08
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_21-46-08
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-13_21-46-08
  ```
- [ ] 15. 7:52 — `2026-07-03 23-48-45.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-48-45.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-48-45
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-48-45
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-07-03_23-48-45
  ```
- [ ] 16. 7:58 — `2026-06-03 20-55-44.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 20-55-44.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_20-55-44
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_20-55-44
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-03_20-55-44
  ```
- [ ] 17. 8:13 — `2026-06-06 17-12-07.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-12-07.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-12-07
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-12-07
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-06_17-12-07
  ```
- [ ] 18. 8:25 — `2026-06-02 22-13-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 22-13-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_22-13-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_22-13-26
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-02_22-13-26
  ```
- [ ] 19. 8:32 — `2026-04-14 19-40-30.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 19-40-30.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_19-40-30
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_19-40-30
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-14_19-40-30
  ```
- [ ] 20. 8:51 — `2026-06-07 13-17-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 13-17-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_13-17-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_13-17-10
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-07_13-17-10
  ```
- [ ] 21. 9:07 — `2026-06-06 20-22-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 20-22-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_20-22-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_20-22-10
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-06_20-22-10
  ```
- [ x] 22. 9:25 — `2026-06-27 12-13-37.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-27 12-13-37.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-27_12-13-37
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-27_12-13-37
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-27_12-13-37
  ```
- [x ] 23. 10:25 — `2026-04-13 06-34-11.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 06-34-11.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_06-34-11
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_06-34-11
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-13_06-34-11
  ```
- [x ] 24. 11:32 — `2026-07-03 21-06-34.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-06-34.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-06-34
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-06-34
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-07-03_21-06-34
  ```
- [ x] 25. 14:42 — `2026-07-03 21-54-06.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-54-06.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-54-06
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-54-06
  venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-07-03_21-54-06
  ```

## サンプリング対象外（15分超・9本）

今回は対象外。将来的に長尺も検証する場合はここから選ぶ。

| 尺 | ファイル |
|---|---|
| 15:20 | `2026-04-14 22-47-23.mp4` |
| 18:35 | `2026-07-10 21-28-36.mp4` |
| 23:42 | `2026-06-07 12-52-10.mp4` |
| 35:56 | `2026-06-27 14-42-58.mp4` |
| 45:17 | `2026-06-11 08-05-00.mp4` |
| 56:27 | `2026-06-10 21-52-25.mp4` |
| 64:48 | `2026-06-22 21-23-04.mp4` |
| 76:21 | `2026-06-09 22-34-02.mp4` |
| 172:30 | `2026-06-21 21-02-21.mp4` |

## サンプリング対象外（15分以内・非選出・87本）

15分以内だが均等間隔サンプリングで選ばれなかった動画。今後NGパターンが増え続ける
ようなら、ここから追加でサンプリングする候補になる。

| 尺 | ファイル |
|---|---|
| 3:23 | `2026-04-14 09-08-14.mp4` |
| 3:28 | `2026-04-13 22-03-37.mp4` |
| 3:40 | `2026-06-06 19-54-01.mp4` |
| 3:41 | `2026-04-14 21-07-44.mp4` |
| 4:27 | `2026-06-06 19-48-51.mp4` |
| 4:27 | `2026-06-06 17-45-09.mp4` |
| 4:36 | `2026-07-03 23-26-22.mp4` |
| 4:46 | `2026-04-12 17-24-26.mp4` |
| 4:48 | `2026-06-04 21-39-15.mp4` |
| 4:51 | `2026-07-03 21-30-49.mp4` |
| 4:54 | `2026-04-13 07-18-18.mp4` |
| 5:02 | `2026-04-13 20-47-00.mp4` |
| 5:04 | `2026-06-02 21-28-43.mp4` |
| 5:05 | `2026-04-12 17-18-57.mp4` |
| 5:10 | `2026-07-03 23-34-29.mp4` |
| 5:12 | `2026-04-14 09-02-43.mp4` |
| 5:21 | `2026-06-04 21-32-54.mp4` |
| 5:32 | `2026-04-12 18-45-51.mp4` |
| 5:42 | `2026-04-12 16-29-27.mp4` |
| 5:46 | `2026-06-04 21-26-10.mp4` |
| 5:47 | `2026-07-03 23-19-55.mp4` |
| 5:55 | `2026-06-11 09-04-08.mp4` |
| 6:00 | `2026-04-12 16-51-41.mp4` |
| 6:03 | `2026-06-06 20-10-04.mp4` |
| 6:13 | `2026-06-03 21-04-26.mp4` |
| 6:16 | `2026-07-03 23-12-35.mp4` |
| 6:17 | `2026-06-03 22-32-34.mp4` |
| 6:23 | `2026-07-03 22-52-49.mp4` |
| 6:28 | `2026-06-03 21-23-40.mp4` |
| 6:29 | `2026-04-12 16-58-03.mp4` |
| 6:33 | `2026-06-06 18-11-24.mp4` |
| 6:36 | `2026-06-04 21-18-52.mp4` |
| 6:42 | `2026-04-12 19-28-37.mp4` |
| 6:52 | `2026-06-03 21-11-46.mp4` |
| 6:53 | `2026-04-13 07-07-31.mp4` |
| 7:00 | `2026-04-13 07-38-30.mp4` |
| 7:04 | `2026-06-02 22-00-00.mp4` |
| 7:05 | `2026-04-12 19-09-29.mp4` |
| 7:17 | `2026-04-13 21-54-07.mp4` |
| 7:23 | `2026-04-12 17-40-16.mp4` |
| 7:26 | `2026-06-03 22-22-07.mp4` |
| 7:28 | `2026-06-02 21-43-53.mp4` |
| 7:31 | `2026-04-14 08-43-53.mp4` |
| 7:33 | `2026-04-13 21-38-13.mp4` |
| 7:33 | `2026-04-13 07-23-41.mp4` |
| 7:36 | `2026-06-07 11-48-50.mp4` |
| 7:36 | `2026-04-12 19-01-37.mp4` |
| 7:45 | `2026-06-07 11-58-38.mp4` |
| 7:46 | `2026-06-06 17-36-31.mp4` |
| 7:46 | `2026-06-06 17-21-33.mp4` |
| 7:49 | `2026-07-10 22-27-25.mp4` |
| 7:54 | `2026-04-12 16-43-26.mp4` |
| 7:57 | `2026-06-03 21-39-48.mp4` |
| 7:57 | `2026-06-02 21-34-07.mp4` |
| 8:02 | `2026-04-12 18-37-29.mp4` |
| 8:05 | `2026-06-07 12-39-49.mp4` |
| 8:12 | `2026-04-12 16-06-11.mp4` |
| 8:13 | `2026-04-12 18-53-04.mp4` |
| 8:16 | `2026-07-10 21-49-15.mp4` |
| 8:19 | `2026-07-10 21-08-14.mp4` |
| 8:21 | `2026-07-03 23-40-01.mp4` |
| 8:22 | `2026-06-06 17-50-23.mp4` |
| 8:28 | `2026-04-12 17-47-57.mp4` |
| 8:30 | `2026-06-21 17-53-10.mp4` |
| 8:31 | `2026-06-03 21-55-23.mp4` |
| 8:34 | `2026-06-21 20-43-36.mp4` |
| 8:38 | `2026-06-21 17-28-36.mp4` |
| 8:41 | `2026-06-03 22-47-14.mp4` |
| 8:48 | `2026-06-27 13-22-59.mp4` |
| 8:52 | `2026-06-02 23-00-18.mp4` |
| 8:54 | `2026-04-14 08-51-35.mp4` |
| 9:06 | `2026-04-13 21-22-29.mp4` |
| 9:09 | `2026-04-12 17-30-42.mp4` |
| 9:15 | `2026-06-06 17-59-33.mp4` |
| 9:22 | `2026-06-21 17-43-17.mp4` |
| 9:23 | `2026-06-21 16-41-03.mp4` |
| 9:35 | `2026-06-06 19-59-01.mp4` |
| 9:47 | `2026-06-07 12-27-14.mp4` |
| 9:47 | `2026-06-03 22-11-55.mp4` |
| 10:23 | `2026-07-03 22-30-26.mp4` |
| 10:52 | `2026-04-14 20-49-38.mp4` |
| 10:55 | `2026-04-12 19-16-58.mp4` |
| 11:14 | `2026-06-21 18-17-23.mp4` |
| 12:08 | `2026-07-03 23-00-04.mp4` |
| 12:23 | `2026-06-04 21-53-05.mp4` |
| 13:00 | `2026-04-12 17-57-51.mp4` |
| 13:12 | `2026-04-14 22-34-08.mp4` |

