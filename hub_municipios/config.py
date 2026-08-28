"""
Caminhos e parâmetros do Hub de Municípios.

Layout de dados (definido em conjunto com o usuário):

    Plataforma_IP/
      dados/                          <- fora do repositório git (o repo é app/)
        bdgd/brutos/                  <- os .gdb entram aqui (dezenas de GB)
        bdgd/processados/             <- parquet intermediário por distribuidora
        siconfi/cache/                <- cadastro de entes + consultas COSIP
      app/
        hub_municipios/data/          <- derivado LEVE, versionável e publicável
          bdgd_municipios.parquet     <- agregado por município (poucos MB)

A separação é proposital: o bruto é grande demais para versionar ou subir em deploy;
o derivado agregado é pequeno e acompanha o app.
"""

from __future__ import annotations

import os
from pathlib import Path

# app/hub_municipios/config.py -> app/hub_municipios -> app -> Plataforma_IP
PACOTE = Path(__file__).resolve().parent
APP = PACOTE.parent
RAIZ = APP.parent

# ── Dados brutos e intermediários (fora do repositório) ──────────────────────
# Sobrescreva com a variável de ambiente HOUER_DADOS para apontar outro disco.
DADOS = Path(os.environ.get("HOUER_DADOS", RAIZ / "dados"))
BDGD_BRUTOS = DADOS / "bdgd" / "brutos"
BDGD_PROCESSADOS = DADOS / "bdgd" / "processados"
SICONFI_CACHE = DADOS / "siconfi" / "cache"

# Repositórios adicionais de .gdb, varridos junto com BDGD_BRUTOS. O acervo nacional
# 2024 vive no Drive compartilhado do time — ler de lá funciona, mas cada base é baixada
# sob demanda pelo Drive File Stream: medido em 28/08/2026, ~7 min para uma base de
# 5,65 GB contra ~55 s para uma base local de 15,9 GB. O gargalo é rede, não CPU.
# Sobrescreva com HOUER_BDGD_EXTRA (caminhos separados por ';').
BDGD_PASTAS_EXTRA = [
    Path(r"G:\Drives compartilhados\Head de Energia\06. Projetos"
         r"\PVSC - Assistente de Pré-Viabilidade\CLP\Dados BDGD"),
]
if os.environ.get("HOUER_BDGD_EXTRA"):
    BDGD_PASTAS_EXTRA = [Path(p) for p in os.environ["HOUER_BDGD_EXTRA"].split(";") if p.strip()]


def pastas_bdgd() -> list[Path]:
    """Pasta local de brutos + repositórios extras que existirem no momento."""
    pastas = [BDGD_BRUTOS]
    pastas += [p for p in BDGD_PASTAS_EXTRA if p.exists()]
    return pastas

# ── Derivado leve, que acompanha o app ──────────────────────────────────────
DATA_PACOTE = PACOTE / "data"
BDGD_MUNICIPIOS = DATA_PACOTE / "bdgd_municipios.parquet"
BDGD_TECNOLOGIA = DATA_PACOTE / "bdgd_tecnologia.parquet"
ENTES_CACHE = DATA_PACOTE / "entes_siconfi.parquet"


def garantir_pastas() -> None:
    """Cria as pastas de dados se ainda não existirem. Idempotente."""
    for p in (BDGD_BRUTOS, BDGD_PROCESSADOS, SICONFI_CACHE, DATA_PACOTE):
        p.mkdir(parents=True, exist_ok=True)


# ── Parâmetros de negócio ───────────────────────────────────────────────────
# Custo de referência da PPP de iluminação pública, em R$ por ponto por MÊS. É o
# único parâmetro financeiro da triagem, e substituiu a combinação anterior
# "tarifa R$/kWh x consumo da BDGD". A troca resolve dois problemas:
#   1. é o número que o time já usa e que o mercado cota — R$ 38 na planilha CLP,
#      R$ 32 no estudo da RMBH, mediana de R$ 34,23 nos 181 contratos assinados;
#   2. o consumo em kWh da BDGD é inutilizável em 1.820 dos 5.417 municípios
#      (33,6%), e amarrar o indicador a ele condenava um terço do país a número
#      errado. Contagem de pontos e carga instalada seguem confiáveis.
CUSTO_PPP_PONTO_MES_PADRAO = 38.00

POTENCIA_FUTURA_PADRAO_W = 60.0

# Horas de operação anuais observadas na BDGD da Cemig-D (2024): 4.160 h/ano,
# ou 11,4 h/dia — coerente com acionamento por relé fotoelétrico. Usada apenas nas
# checagens de consistência do dado da distribuidora, não nos indicadores.
HORAS_OPERACAO_ANO = 4160.0
