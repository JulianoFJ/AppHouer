@echo off
setlocal
title Plataforma IP - Iluminacao Publica

echo ==========================================================
echo   PLATAFORMA IP - ILUMINACAO PUBLICA
echo   Simulacao NBR 5101  +  Analise de Cadastro
echo ==========================================================
echo.

:: 1. Verificacao de Dependencias
echo [PASSO 1/4] Sincronizando bibliotecas...
py -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar dependencias. Verifique sua conexao.
    pause
    exit /b
)

:: 2. Verificacao de Modelos Treinados
echo [PASSO 2/4] Validando inteligencia artificial...
if not exist "modelo_w_limpo.pkl" (
    echo [AVISO] Modelos nao encontrados. Iniciando treinamento...
    py 02_treinar_modelo.py
    if %errorlevel% neq 0 (
        echo [ERRO] Falha no treinamento dos modelos.
        pause
        exit /b
    )
)

:: 3. Suite de Testes Automatizados
echo [PASSO 3/4] Executando testes do modelo...
py scratch\test_full.py > nul 2>&1
if %errorlevel% neq 0 (
    echo [AVISO] Alguns testes falharam. Rodando diagnostico...
    py scratch\test_full.py
    echo.
    echo Pressione qualquer tecla para iniciar mesmo assim, ou feche para cancelar.
    pause
) else (
    echo [OK] Todos os testes passaram.
)

:: 4. Inicializacao do Streamlit
echo [PASSO 4/4] Iniciando o Servidor...
echo.
echo ----------------------------------------------------------
echo   APP RODANDO!
echo   Acesse: http://localhost:8501
echo   (Nao feche esta janela enquanto estiver usando o App)
echo ----------------------------------------------------------
echo.

py -m streamlit run app.py --server.port 8501 --theme.base "dark"

pause
