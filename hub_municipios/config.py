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
# Tarifa B4a (rede de distribuição) média com tributos, R$/kWh. É um DEFAULT de
# triagem, não premissa de estudo: a tarifa real vem da resolução homologatória da
# distribuidora e do regime tributário do município. Editável na interface.
TARIFA_B4A_PADRAO = 0.72

# Consumo de referência para estimar o potencial de eficientização (W médio por
# ponto após retrofit integral em LED). Faixa de mercado 45–75 W; 60 W é conservador
# para via urbana típica. Também editável na interface.
POTENCIA_LED_REFERENCIA_W = 60.0

# Horas de operação anuais. O valor observado na BDGD da Cemig-D (2024) é 4.160 h/ano
# (11,4 h/dia), coerente com acionamento por relé fotoelétrico. Usado apenas quando o
# município não tem consumo declarado na BDGD.
HORAS_OPERACAO_ANO = 4160.0
