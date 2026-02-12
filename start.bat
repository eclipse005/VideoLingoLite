@echo off
chcp 65001 >nul
title VideoLingoLite

echo.
echo ==================================================
echo   VideoLingoLite - 音视频转写翻译工具
echo ==================================================
echo.
echo   Web UI:    http://localhost:8000
echo.
echo ==================================================
echo.

REM 创建必要目录
if not exist "api\uploads" mkdir "api\uploads"
if not exist "output" mkdir "output"
if not exist "api\logs" mkdir "api\logs"

REM 启动服务
echo 启动服务...
echo.
REM 设置环境变量
set NUMEXPR_MAX_THREADS=8
uv run python start.py

pause
