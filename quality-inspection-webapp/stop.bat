@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在停止智能质检台服务...
"%~dp0智能质检台.exe" --stop
echo 已停止（浏览器页面刷新会显示无法连接，属于正常现象）。
timeout /t 2 >nul
