"""
Estimativa do parque de IP quando o município não tem BDGD processada.

Por que existe
--------------
A BDGD cobre 5.417 dos 5.570 municípios (97,3%), mas o que falta são justamente
distribuidoras pequenas e áreas sem base publicada. Sem uma estimativa, esses entes
sumiriam da triagem — e é melhor um número marcado como estimado do que a ausência
do município na análise.

O que muda em relação à regra manual
------------------------------------
A planilha CLP usa **população ÷ 10**, isto é, 100 pontos por mil habitantes, constante.
Medido contra os 5.416 municípios que TÊM BDGD e população conhecida (28/08/2026):

    densidade real (pontos por mil hab):  p25=95,2 · mediana=126,5 · p75=157,5

Ou seja, a regra constante **subestima o parque em cerca de 21%** na mediana. E
subestimar pontos **infla** a arrecadação por ponto (COSIP ÷ pontos), fazendo o
município parecer mais viável do que é — o erro é otimista, não conservador.

A densidade também não é constante: cai de forma monotônica com o porte do município,
de 148 pontos/mil habitantes nos menores a 71 nos acima de 500 mil. Município pequeno
tem malha viária longa para pouca gente; capital adensa moradores por via iluminada.
Por isso a estimativa aqui é por FAIXA DE POPULAÇÃO, não por constante única.

REGRA DE OURO: todo município estimado sai marcado em `origem_pontos = "Estimado"`.
Dado medido e dado inferido nunca se misturam sem etiqueta.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Densidade mediana de pontos de IP por mil habitantes, por faixa de população.
# Calibrado em 28/08/2026 sobre 5.416 municípios com BDGD (V11/2024) e população do
# cadastro do SICONFI. Reproduzir com `py -m hub_municipios._calibrar_densidade`.
#
#   faixa de população      n     p25   mediana   p75
#   até 5 mil            1.211   116,7   148,2   180,0
#   5–10 mil             1.144   100,7   131,0   162,5
#   10–20 mil            1.327    89,7   122,2   152,4
#   20–50 mil            1.059    87,9   118,7   148,1
#   50–100 mil             339    82,0   114,5   141,1
#   100–500 mil            289    75,7   101,6   122,9
#   acima de 500 mil        47    48,5    71,5    93,6
DENSIDADE_POR_FAIXA = [
    (5_000, 148.2),
    (10_000, 131.0),
    (20_000, 122.2),
    (50_000, 118.7),
    (100_000, 114.5),
    (500_000, 101.6),
    (float("inf"), 71.5),
]

# A regra manual da planilha CLP, mantida para comparação e para quem quiser reproduzir
# o número antigo exatamente.
DENSIDADE_CONSTANTE_CLP = 100.0

ORIGEM_MEDIDA = "BDGD"
ORIGEM_ESTIMADA = "Estimado"


def densidade_por_populacao(populacao: float) -> float:
    """Pontos de IP por mil habitantes esperados para um município desse porte."""
    for limite, densidade in DENSIDADE_POR_FAIXA:
        if populacao <= limite:
            return densidade
    return DENSIDADE_POR_FAIXA[-1][1]


def estimar_pontos(populacao, usar_regra_clp: bool = False) -> Optional[float]:
    """
    Pontos de IP estimados a partir da população. Devolve None sem população — é
    preferível não ter o município na análise a colocá-lo com um número inventado.
    """
    try:
        pop = float(populacao)
    except (TypeError, ValueError):
        return None
    if not pop or pop <= 0 or pd.isna(pop):
        return None
    dens = DENSIDADE_CONSTANTE_CLP if usar_regra_clp else densidade_por_populacao(pop)
    return round(pop / 1000.0 * dens)


def completar_parque(df: pd.DataFrame, usar_regra_clp: bool = False) -> pd.DataFrame:
    """
    Preenche `pontos_ip` a partir da população onde a BDGD não cobre, e cria a coluna
    `origem_pontos` ("BDGD" ou "Estimado") em TODAS as linhas.

    Espera as colunas `pontos_ip` e `populacao`. Não toca em nenhum município que já
    tenha parque medido.
    """
    out = df.copy()
    if "pontos_ip" not in out.columns:
        out["pontos_ip"] = pd.NA

    medidos = pd.to_numeric(out["pontos_ip"], errors="coerce")
    out["origem_pontos"] = medidos.notna().map({True: ORIGEM_MEDIDA, False: ORIGEM_ESTIMADA})

    faltando = medidos.isna()
    if faltando.any() and "populacao" in out.columns:
        out.loc[faltando, "pontos_ip"] = [
            estimar_pontos(p, usar_regra_clp) for p in out.loc[faltando, "populacao"]
        ]
        # sem população não há estimativa possível: volta a ficar sem origem definida
        ainda_sem = pd.to_numeric(out["pontos_ip"], errors="coerce").isna()
        out.loc[ainda_sem, "origem_pontos"] = None

    return out
