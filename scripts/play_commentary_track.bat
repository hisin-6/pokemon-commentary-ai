@echo off
rem VMCアバター録画用: renders\<動画名>\commentary_track.wav を再生する。
rem 再生アプリの出力先を一度だけ「CABLE Input」に設定しておけば、
rem 動画名（フォルダ）が変わっても毎回の切り替え・戻し作業は不要（Windowsの
rem 音量ミキサーはアプリ単位で出力先を記憶するため）。
rem
rem 使い方: scripts\play_commentary_track.bat <動画名>
rem 例:     scripts\play_commentary_track.bat 16-14-39

setlocal
if "%~1"=="" (
    echo 使い方: play_commentary_track.bat ^<動画名（renders配下のフォルダ名）^>
    exit /b 1
)

set "WAV=%~dp0..\renders\%~1\commentary_track.wav"
if not exist "%WAV%" (
    echo commentary_track.wav が見つかりません: %WAV%
    echo パス2（render_commentary_video.py）を先に実行してください
    exit /b 1
)

echo 再生: %WAV%
echo ※OBS録画を先に開始してから、このバッチを実行してください（offset秒をメモ）
start "" "%WAV%"
