"""
Caminhos de dados brutos/intermediários, fora do repositório git — compartilhados entre
pacotes de domínio que lidam com BDGD/SICONFI/ANEEL (`cadastro_bdgd`, `cadastro_ip`,
`hub_municipios`).

Extraído de `hub_municipios/config.py` (que tinha virado o "config compartilhado" de
fato, por acidente: `cadastro_bdgd`/`cadastro_ip` importavam direto de um pacote de
domínio que não é o deles). `hub_municipios/config.py` continua existindo e reexporta
tudo daqui — ele guarda, além disso, os caminhos do SEU PRÓPRIO derivado publicável
(`hub_municipios/data/`) e os parâmetros de negócio do Hub, que não são genéricos o
bastante para morar num módulo compartilhado.

Layout de dados:

    Plataforma_IP/
      dados/                          <- fora do repositório git (o repo é AppHouer/)
        bdgd/brutos/                  <- os .gdb entram aqui (dezenas de GB)
        bdgd/processados/             <- parquet intermediário por distribuidora
        siconfi/cache/                <- cadastro de entes + consultas COSIP
        aneel/cache/                  <- bruto do ETL de tarifas (CSV do CKAN)
      AppHouer/
        hub_municipios/data/          <- derivado LEVE do Hub, versionável e publicável
"""

from __future__ import annotations

import os
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent


# Sobrescreva com a variável de ambiente PLATAFORMA_IP_DADOS para apontar outro disco.
#
# `HOUER_DADOS` continua sendo aceita como nome antigo: ela está configurada nas
# máquinas que já processam a BDGD, e quebrar essas configurações silenciosamente
# faria o ETL cair de volta para `RAIZ/dados` sem avisar — que existe, está vazia, e
# produziria um agregado nacional truncado em vez de um erro.
def _env(nome_novo: str, nome_antigo: str) -> str | None:
    return os.environ.get(nome_novo) or os.environ.get(nome_antigo)


DADOS = Path(_env("PLATAFORMA_IP_DADOS", "HOUER_DADOS") or (RAIZ / "dados"))
BDGD_BRUTOS = DADOS / "bdgd" / "brutos"
BDGD_PROCESSADOS = DADOS / "bdgd" / "processados"
SICONFI_CACHE = DADOS / "siconfi" / "cache"

# Bruto do ETL de tarifas (CSV baixado do CKAN da ANEEL), fora do repositório.
ANEEL_CACHE = DADOS / "aneel" / "cache"

# Repositórios adicionais de .gdb, varridos junto com BDGD_BRUTOS. O acervo nacional
# 2024 vive no Drive compartilhado do time — ler de lá funciona, mas cada base é baixada
# sob demanda pelo Drive File Stream: medido em 28/08/2026, ~7 min para uma base de
# 5,65 GB contra ~55 s para uma base local de 15,9 GB. O gargalo é rede, não CPU.
# Sobrescreva com PLATAFORMA_IP_BDGD_EXTRA (caminhos separados por ';');
# `HOUER_BDGD_EXTRA` segue aceita como nome antigo, pela mesma razão de DADOS.
BDGD_PASTAS_EXTRA = [
    Path(r"G:\Drives compartilhados\Head de Energia\06. Projetos"
         r"\PVSC - Assistente de Pré-Viabilidade\CLP\Dados BDGD"),
]
_extra = _env("PLATAFORMA_IP_BDGD_EXTRA", "HOUER_BDGD_EXTRA")
if _extra:
    BDGD_PASTAS_EXTRA = [Path(p) for p in _extra.split(";") if p.strip()]


def pastas_bdgd() -> list[Path]:
    """Pasta local de brutos + repositórios extras que existirem no momento."""
    pastas = [BDGD_BRUTOS]
    pastas += [p for p in BDGD_PASTAS_EXTRA if p.exists()]
    return pastas


def garantir_pastas() -> None:
    """Cria as pastas de dados genéricas se ainda não existirem. Idempotente."""
    for p in (BDGD_BRUTOS, BDGD_PROCESSADOS, SICONFI_CACHE, ANEEL_CACHE):
        p.mkdir(parents=True, exist_ok=True)
