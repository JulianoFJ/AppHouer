#!/bin/sh
# Migra o schema antes de servir tráfego. Rodar isto a cada start (não só na primeira
# vez) é o que permite trocar de host sem lembrar de um passo manual de deploy: a
# imagem se auto-atualiza contra o banco que encontrar, aqui ou na AWS depois.
set -e

alembic upgrade head
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
