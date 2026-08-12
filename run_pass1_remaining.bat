@echo off
chcp 65001 >nul
cd /d "C:\Users\rotat\AITuberProject"

venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-14 19-40-30.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-14_19-40-30
venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-14_19-40-30
venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-14_19-40-30

venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-07 13-17-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-07_13-17-10
venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-07_13-17-10
venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-07_13-17-10

venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-06-06 20-22-10.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-06-06_20-22-10
venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-06-06_20-22-10
venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-06-06_20-22-10

venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-04-13 06-34-11.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-04-13_06-34-11
venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-04-13_06-34-11
venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-04-13_06-34-11

venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-06-34.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-06-34
venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-06-34
venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-07-03_21-06-34

venv\Scripts\python.exe src\pipeline.py --input "D:\ゲーム録画\2026-07-03 21-54-06.mp4" --end-model runs/detect/train_end_screen2/weights/best.pt --game-mode champions --conf 0.3 --render-out renders\2026-07-03_21-54-06
venv\Scripts\python.exe scripts\generate_review_checklist.py renders\2026-07-03_21-54-06
venv\Scripts\python.exe scripts\screen_pass1.py renders\2026-07-03_21-54-06

echo.
echo ===== 残り6本すべて完了 =====
pause
