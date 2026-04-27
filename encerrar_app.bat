@echo off
title HOUER - Encerrar Simulador
echo ==========================================================
echo   ENCERRANDO SIMULADOR DE ILUMINACAO...
echo ==========================================================
echo.

taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM streamlit.exe /T >nul 2>&1

echo.
echo [SUCESSO] Processos encerrados e memoria liberada.
echo.
timeout /t 3
exit
