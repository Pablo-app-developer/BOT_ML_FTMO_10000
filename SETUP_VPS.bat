@echo off
title FTMO VPS Setup
cd /d "%~dp0"
echo ============================================
echo   FTMO Bot - Instalacion en VPS
echo ============================================
echo.
echo Instalando dependencias Python...
pip install -r requirements_vps.txt
echo.
echo Instalacion completada.
echo.
echo SIGUIENTE PASO:
echo 1. Abre MetaTrader 5
echo 2. Conecta tu cuenta FTMO
echo 3. Ejecuta ARRANCAR_BOT.bat
echo.
pause
