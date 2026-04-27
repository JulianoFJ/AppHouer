@echo off
setlocal
title HOUER - Simulador de Iluminacao Publica v2.0

echo ==========================================================
echo   HOUER - SISTEMA DE PREVISAO E SIMULACAO (ML)
echo ==========================================================
echo.

:: 1. Verificacao de Dependencias
echo [PASSO 1/3] Sincronizando bibliotecas (PDF, Mapas, Dash)...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar dependencias. Verifique sua conexao.
    pause
    exit /b
)

:: 2. Verificacao de Modelos Treinados
echo [PASSO 2/3] Validando inteligencia artificial...
if not exist "modelo_emed_limpo.pkl" (
    echo [AVISO] Modelos nao encontrados. Iniciando treinamento rapido...
    python 02_treinar_modelo.py
)

:: 3. Inicializacao do Streamlit
echo [PASSO 3/3] Iniciando o Servidor...
echo.
echo ----------------------------------------------------------
echo   APP RODANDO! 
echo   Acesse: http://localhost:8501
echo   (Nao feche esta janela enquanto estiver usando o App)
echo ----------------------------------------------------------
echo.

python -m streamlit run 03_app_previsao.py --server.port 8501 --theme.base "dark"

pause
