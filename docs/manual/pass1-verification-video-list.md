# パス1 無課金検証ワークフロー 対象動画リスト

`D:\ゲーム録画\` 内の全録画を対象に、パス1（Bedrock無課金・Phi-3フォールバック）の
精度検証を1本ずつ進めるための管理リスト。**動画の尺が短い順**に並べてあるので、
上から順に処理していく運用を想定。

- 判定基準・手順は `docs/manual/pass1-verification-checklist.md` を参照
- NGは `docs/manual/pass1-verification-ng-findings.md` に集約（この時点では
  修正しない・収集フェーズ。2026-08-10〜運用）
- **この作業をする際は必ずこのファイルを見て、次にどの動画をやるか・
  何本終わったかを確認すること**
- 生成日: 2026-08-10（`ffprobe`で全132本の尺を実測）
- 各動画のコマンドは全て `--game-mode champions`（[[project_champions_v1]]の
  メモで「2026-04-12以降の録画は全部チャンピオンズ」と確認済み）
- コマンドは全て**Windows側**（`venv\Scripts\python.exe`）でそのまま実行可能

## 現在の状況

- 全録画: **132本**
- 実況動画として完成済み（対象外）: **6本**
- 無課金パス1検証 判定済み: **4本**
- 無課金パス1検証 未処理: **122本**

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

## 未処理（尺の短い順・ここから消化していく）

チェック済みになったら見出しの `- [ ]` を `- [x]` に変更し、判定済みテーブルにも
追記すること。各項目のコマンドはそのままコピペで実行可能（1個目=パス1実行・
2個目=レビューMarkdown生成。どちらもWindows側で続けて実行）。

- [x] 1. 2:59 — `2026-04-13 06-22-25.mp4`（判定済み・NG2件→ng-findings.md）
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 06-22-25.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_06-22-25
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_06-22-25
  ```
- [x] 2. 3:03 — `2026-04-14 21-44-43.mp4`（判定済み・NG3件→ng-findings.md）
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 21-44-43.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_21-44-43
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_21-44-43
  ```
- [ ] 3. 3:04 — `2026-06-03 21-35-48.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-35-48.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-35-48
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-35-48
  ```
- [ ] 4. 3:20 — `2026-04-14 08-39-39.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 08-39-39.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_08-39-39
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_08-39-39
  ```
- [ ] 5. 3:23 — `2026-04-14 09-08-14.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 09-08-14.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_09-08-14
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_09-08-14
  ```
- [ ] 6. 3:28 — `2026-04-13 22-03-37.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 22-03-37.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_22-03-37
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_22-03-37
  ```
- [ ] 7. 3:40 — `2026-06-06 19-54-01.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 19-54-01.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_19-54-01
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_19-54-01
  ```
- [ ] 8. 3:41 — `2026-04-14 21-07-44.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 21-07-44.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_21-07-44
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_21-07-44
  ```
- [ ] 9. 3:56 — `2026-04-12 17-12-38.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-12-38.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-12-38
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-12-38
  ```
- [ ] 10. 4:27 — `2026-06-06 19-48-51.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 19-48-51.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_19-48-51
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_19-48-51
  ```
- [ ] 11. 4:27 — `2026-06-06 17-45-09.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-45-09.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-45-09
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-45-09
  ```
- [ ] 12. 4:36 — `2026-07-03 23-26-22.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-26-22.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-26-22
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-26-22
  ```
- [ ] 13. 4:41 — `2026-04-14 21-40-01.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 21-40-01.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_21-40-01
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_21-40-01
  ```
- [ ] 14. 4:46 — `2026-04-12 17-24-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-24-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-24-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-24-26
  ```
- [ ] 15. 4:48 — `2026-06-04 21-39-15.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-39-15.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-39-15
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-39-15
  ```
- [ ] 16. 4:51 — `2026-07-03 21-30-49.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-30-49.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-30-49
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-30-49
  ```
- [ ] 17. 4:54 — `2026-04-13 07-18-18.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 07-18-18.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_07-18-18
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_07-18-18
  ```
- [ ] 18. 4:58 — `2026-06-07 12-07-11.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 12-07-11.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_12-07-11
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_12-07-11
  ```
- [ ] 19. 5:02 — `2026-04-13 20-47-00.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 20-47-00.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_20-47-00
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_20-47-00
  ```
- [ ] 20. 5:04 — `2026-06-02 21-28-43.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 21-28-43.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_21-28-43
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_21-28-43
  ```
- [ ] 21. 5:05 — `2026-04-12 17-18-57.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-18-57.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-18-57
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-18-57
  ```
- [ ] 22. 5:06 — `2026-06-06 17-30-38.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-30-38.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-30-38
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-30-38
  ```
- [ ] 23. 5:10 — `2026-07-03 23-34-29.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-34-29.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-34-29
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-34-29
  ```
- [ ] 24. 5:12 — `2026-04-14 09-02-43.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 09-02-43.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_09-02-43
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_09-02-43
  ```
- [ ] 25. 5:21 — `2026-06-04 21-32-54.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-32-54.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-32-54
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-32-54
  ```
- [ ] 26. 5:32 — `2026-04-12 18-45-51.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 18-45-51.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_18-45-51
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_18-45-51
  ```
- [ ] 27. 5:34 — `2026-04-14 23-22-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 23-22-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_23-22-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_23-22-26
  ```
- [ ] 28. 5:42 — `2026-04-12 16-29-27.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 16-29-27.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_16-29-27
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_16-29-27
  ```
- [ ] 29. 5:46 — `2026-06-04 21-26-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-26-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-26-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-26-10
  ```
- [ ] 30. 5:47 — `2026-07-03 23-19-55.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-19-55.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-19-55
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-19-55
  ```
- [ ] 31. 5:55 — `2026-06-11 09-04-08.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-11 09-04-08.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-11_09-04-08
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-11_09-04-08
  ```
- [ ] 32. 5:56 — `2026-06-03 21-48-45.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-48-45.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-48-45
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-48-45
  ```
- [ ] 33. 6:00 — `2026-04-12 16-51-41.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 16-51-41.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_16-51-41
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_16-51-41
  ```
- [ ] 34. 6:03 — `2026-06-06 20-10-04.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 20-10-04.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_20-10-04
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_20-10-04
  ```
- [ ] 35. 6:13 — `2026-06-03 21-04-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-04-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-04-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-04-26
  ```
- [ ] 36. 6:14 — `2026-06-21 17-17-00.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 17-17-00.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_17-17-00
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_17-17-00
  ```
- [ ] 37. 6:16 — `2026-07-03 23-12-35.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-12-35.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-12-35
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-12-35
  ```
- [ ] 38. 6:17 — `2026-06-03 22-32-34.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 22-32-34.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_22-32-34
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_22-32-34
  ```
- [ ] 39. 6:23 — `2026-07-03 22-52-49.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 22-52-49.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_22-52-49
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_22-52-49
  ```
- [ ] 40. 6:28 — `2026-06-03 21-23-40.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-23-40.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-23-40
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-23-40
  ```
- [ ] 41. 6:28 — `2026-06-04 21-44-45.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-44-45.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-44-45
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-44-45
  ```
- [ ] 42. 6:29 — `2026-04-12 16-58-03.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 16-58-03.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_16-58-03
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_16-58-03
  ```
- [ ] 43. 6:33 — `2026-06-06 18-11-24.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 18-11-24.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_18-11-24
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_18-11-24
  ```
- [ ] 44. 6:36 — `2026-06-04 21-18-52.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-18-52.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-18-52
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-18-52
  ```
- [ ] 45. 6:42 — `2026-04-12 19-28-37.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 19-28-37.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_19-28-37
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_19-28-37
  ```
- [ ] 46. 6:51 — `2026-04-14 20-32-31.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 20-32-31.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_20-32-31
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_20-32-31
  ```
- [ ] 47. 6:52 — `2026-06-03 21-11-46.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-11-46.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-11-46
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-11-46
  ```
- [ ] 48. 6:53 — `2026-04-13 07-07-31.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 07-07-31.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_07-07-31
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_07-07-31
  ```
- [ ] 49. 7:00 — `2026-04-13 07-38-30.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 07-38-30.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_07-38-30
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_07-38-30
  ```
- [ ] 50. 7:03 — `2026-06-06 19-34-04.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 19-34-04.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_19-34-04
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_19-34-04
  ```
- [ ] 51. 7:04 — `2026-06-02 22-00-00.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 22-00-00.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_22-00-00
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_22-00-00
  ```
- [ ] 52. 7:05 — `2026-04-12 19-09-29.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 19-09-29.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_19-09-29
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_19-09-29
  ```
- [ ] 53. 7:17 — `2026-04-13 21-54-07.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 21-54-07.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_21-54-07
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_21-54-07
  ```
- [ ] 54. 7:23 — `2026-04-12 17-40-16.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-40-16.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-40-16
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-40-16
  ```
- [ ] 55. 7:23 — `2026-04-12 17-04-52.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-04-52.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-04-52
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-04-52
  ```
- [ ] 56. 7:26 — `2026-06-03 22-22-07.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 22-22-07.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_22-22-07
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_22-22-07
  ```
- [ ] 57. 7:28 — `2026-06-02 21-43-53.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 21-43-53.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_21-43-53
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_21-43-53
  ```
- [ ] 58. 7:31 — `2026-04-14 08-43-53.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 08-43-53.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_08-43-53
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_08-43-53
  ```
- [ ] 59. 7:33 — `2026-04-13 21-38-13.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 21-38-13.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_21-38-13
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_21-38-13
  ```
- [ ] 60. 7:33 — `2026-06-27 12-34-29.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-27 12-34-29.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-27_12-34-29
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-27_12-34-29
  ```
- [ ] 61. 7:33 — `2026-04-13 07-23-41.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 07-23-41.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_07-23-41
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_07-23-41
  ```
- [ ] 62. 7:36 — `2026-06-07 11-48-50.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 11-48-50.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_11-48-50
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_11-48-50
  ```
- [ ] 63. 7:36 — `2026-04-12 19-01-37.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 19-01-37.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_19-01-37
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_19-01-37
  ```
- [ ] 64. 7:42 — `2026-04-13 21-46-08.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 21-46-08.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_21-46-08
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_21-46-08
  ```
- [ ] 65. 7:45 — `2026-06-07 11-58-38.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 11-58-38.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_11-58-38
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_11-58-38
  ```
- [ ] 66. 7:46 — `2026-06-06 17-36-31.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-36-31.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-36-31
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-36-31
  ```
- [ ] 67. 7:46 — `2026-06-06 17-21-33.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-21-33.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-21-33
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-21-33
  ```
- [ ] 68. 7:49 — `2026-07-10 22-27-25.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-10 22-27-25.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-10_22-27-25
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-10_22-27-25
  ```
- [ ] 69. 7:52 — `2026-07-03 23-48-45.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-48-45.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-48-45
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-48-45
  ```
- [ ] 70. 7:54 — `2026-04-12 16-43-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 16-43-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_16-43-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_16-43-26
  ```
- [ ] 71. 7:57 — `2026-06-03 21-39-48.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-39-48.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-39-48
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-39-48
  ```
- [ ] 72. 7:57 — `2026-06-02 21-34-07.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 21-34-07.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_21-34-07
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_21-34-07
  ```
- [ ] 73. 7:58 — `2026-06-03 20-55-44.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 20-55-44.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_20-55-44
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_20-55-44
  ```
- [ ] 74. 8:02 — `2026-04-12 18-37-29.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 18-37-29.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_18-37-29
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_18-37-29
  ```
- [ ] 75. 8:05 — `2026-06-07 12-39-49.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 12-39-49.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_12-39-49
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_12-39-49
  ```
- [ ] 76. 8:12 — `2026-04-12 16-06-11.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 16-06-11.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_16-06-11
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_16-06-11
  ```
- [ ] 77. 8:13 — `2026-04-12 18-53-04.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 18-53-04.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_18-53-04
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_18-53-04
  ```
- [ ] 78. 8:13 — `2026-06-06 17-12-07.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-12-07.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-12-07
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-12-07
  ```
- [ ] 79. 8:16 — `2026-07-10 21-49-15.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-10 21-49-15.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-10_21-49-15
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-10_21-49-15
  ```
- [ ] 80. 8:19 — `2026-07-10 21-08-14.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-10 21-08-14.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-10_21-08-14
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-10_21-08-14
  ```
- [ ] 81. 8:21 — `2026-07-03 23-40-01.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-40-01.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-40-01
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-40-01
  ```
- [ ] 82. 8:22 — `2026-06-06 17-50-23.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-50-23.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-50-23
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-50-23
  ```
- [ ] 83. 8:25 — `2026-06-02 22-13-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 22-13-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_22-13-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_22-13-26
  ```
- [ ] 84. 8:28 — `2026-04-12 17-47-57.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-47-57.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-47-57
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-47-57
  ```
- [ ] 85. 8:30 — `2026-06-21 17-53-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 17-53-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_17-53-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_17-53-10
  ```
- [ ] 86. 8:31 — `2026-06-03 21-55-23.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 21-55-23.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_21-55-23
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_21-55-23
  ```
- [ ] 87. 8:32 — `2026-04-14 19-40-30.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 19-40-30.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_19-40-30
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_19-40-30
  ```
- [ ] 88. 8:34 — `2026-06-21 20-43-36.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 20-43-36.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_20-43-36
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_20-43-36
  ```
- [ ] 89. 8:38 — `2026-06-21 17-28-36.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 17-28-36.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_17-28-36
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_17-28-36
  ```
- [ ] 90. 8:41 — `2026-06-03 22-47-14.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 22-47-14.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_22-47-14
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_22-47-14
  ```
- [ ] 91. 8:48 — `2026-06-27 13-22-59.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-27 13-22-59.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-27_13-22-59
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-27_13-22-59
  ```
- [ ] 92. 8:51 — `2026-06-07 13-17-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 13-17-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_13-17-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_13-17-10
  ```
- [ ] 93. 8:52 — `2026-06-02 23-00-18.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-02 23-00-18.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-02_23-00-18
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-02_23-00-18
  ```
- [ ] 94. 8:54 — `2026-04-14 08-51-35.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 08-51-35.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_08-51-35
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_08-51-35
  ```
- [ ] 95. 9:06 — `2026-04-13 21-22-29.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 21-22-29.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_21-22-29
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_21-22-29
  ```
- [ ] 96. 9:07 — `2026-06-06 20-22-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 20-22-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_20-22-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_20-22-10
  ```
- [ ] 97. 9:09 — `2026-04-12 17-30-42.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-30-42.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-30-42
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-30-42
  ```
- [ ] 98. 9:15 — `2026-06-06 17-59-33.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 17-59-33.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_17-59-33
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_17-59-33
  ```
- [ ] 99. 9:22 — `2026-06-21 17-43-17.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 17-43-17.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_17-43-17
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_17-43-17
  ```
- [ ] 100. 9:23 — `2026-06-21 16-41-03.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 16-41-03.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_16-41-03
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_16-41-03
  ```
- [ ] 101. 9:25 — `2026-06-27 12-13-37.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-27 12-13-37.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-27_12-13-37
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-27_12-13-37
  ```
- [ ] 102. 9:35 — `2026-06-06 19-59-01.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 19-59-01.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_19-59-01
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_19-59-01
  ```
- [ ] 103. 9:47 — `2026-06-07 12-27-14.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 12-27-14.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_12-27-14
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_12-27-14
  ```
- [ ] 104. 9:47 — `2026-06-03 22-11-55.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-03 22-11-55.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-03_22-11-55
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-03_22-11-55
  ```
- [ ] 105. 10:23 — `2026-07-03 22-30-26.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 22-30-26.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_22-30-26
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_22-30-26
  ```
- [ ] 106. 10:25 — `2026-04-13 06-34-11.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 06-34-11.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_06-34-11
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_06-34-11
  ```
- [ ] 107. 10:52 — `2026-04-14 20-49-38.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 20-49-38.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_20-49-38
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_20-49-38
  ```
- [ ] 108. 10:55 — `2026-04-12 19-16-58.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 19-16-58.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_19-16-58
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_19-16-58
  ```
- [ ] 109. 11:14 — `2026-06-21 18-17-23.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 18-17-23.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_18-17-23
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_18-17-23
  ```
- [ ] 110. 11:32 — `2026-07-03 21-06-34.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-06-34.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-06-34
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-06-34
  ```
- [ ] 111. 12:08 — `2026-07-03 23-00-04.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 23-00-04.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_23-00-04
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_23-00-04
  ```
- [ ] 112. 12:23 — `2026-06-04 21-53-05.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-04 21-53-05.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-04_21-53-05
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-04_21-53-05
  ```
- [ ] 113. 13:00 — `2026-04-12 17-57-51.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-12 17-57-51.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-12_17-57-51
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-12_17-57-51
  ```
- [ ] 114. 13:12 — `2026-04-14 22-34-08.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 22-34-08.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_22-34-08
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_22-34-08
  ```
- [ ] 115. 14:42 — `2026-07-03 21-54-06.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-54-06.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-54-06
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-54-06
  ```
- [ ] 116. 15:20 — `2026-04-14 22-47-23.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 22-47-23.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_22-47-23
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_22-47-23
  ```
- [ ] 117. 18:35 — `2026-07-10 21-28-36.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-10 21-28-36.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-10_21-28-36
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-10_21-28-36
  ```
- [ ] 118. 23:42 — `2026-06-07 12-52-10.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 12-52-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_12-52-10
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_12-52-10
  ```
- [ ] 119. 35:56 — `2026-06-27 14-42-58.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-27 14-42-58.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-27_14-42-58
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-27_14-42-58
  ```
- [ ] 120. 45:17 — `2026-06-11 08-05-00.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-11 08-05-00.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-11_08-05-00
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-11_08-05-00
  ```
- [ ] 121. 56:27 — `2026-06-10 21-52-25.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-10 21-52-25.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-10_21-52-25
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-10_21-52-25
  ```
- [ ] 122. 64:48 — `2026-06-22 21-23-04.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-22 21-23-04.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-22_21-23-04
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-22_21-23-04
  ```
- [ ] 123. 76:21 — `2026-06-09 22-34-02.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-09 22-34-02.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-09_22-34-02
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-09_22-34-02
  ```
- [ ] 124. 172:30 — `2026-06-21 21-02-21.mp4`
  ```
  venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-21 21-02-21.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-21_21-02-21
  venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-21_21-02-21
  ```
