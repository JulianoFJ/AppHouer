"""
Motor de dimensionamento — relações de engenharia expressas como **fórmulas Excel**
que referenciam as células de premissa da aba `Inputs_IP`.

A parametrização é "viva": nada é congelado em Python. Cada saída referencia a célula
de input correspondente, de modo que alterar uma premissa no Excel recalcula expansão,
equipes, veículos e a distribuição de custos no tempo. As fórmulas ficam visíveis na
planilha (auditáveis).

Convenção: o valor de cada parâmetro mora em `Inputs_IP!F{linha_modelo}`.
"""

from __future__ import annotations

from functools import lru_cache

from . import schema

ABA_INPUTS = "Inputs_IP"
COL_VALOR = "F"  # coluna onde o valor de cada parâmetro é escrito


@lru_cache(maxsize=1)
def _linha_por_id() -> dict[str, int]:
    s = schema.carregar()
    return {p.id: p.linha_modelo for sec in s.secoes for p in sec.parametros}


def cel(param_id: str) -> str:
    """Referência absoluta à célula de valor de um parâmetro (ex.: Inputs_IP!$F$19)."""
    linha = _linha_por_id().get(param_id)
    if linha is None:
        raise KeyError(f"parâmetro desconhecido: {param_id}")
    return f"{ABA_INPUTS}!${COL_VALOR}${linha}"


def existe(param_id: str) -> bool:
    return param_id in _linha_por_id()


# ── IDs de parâmetros usados nas fórmulas (derivados do modelo de referência) ──
# Prazos
PRAZO_CONCESSAO_ANOS = "p19"
MESES_ANO = "p8"
# Quantitativo de parque
PARQUE_TOTAL = "p37"
# Expansão / Demanda Reprimida
EXP_ANUAL_PONTOS = "p349"
EXP_INICIO_MES = "p350"
CUSTO_MEDIO_LUM_EXP = "p296"
# Equipes operacionais (nº por marco)
EQ_MARCO = {"setup": "p575", "m1": "p577", "m2": "p578", "m3": "p579", "pos": "p580"}
# Composição da equipe operacional × salário correspondente (mesma ordem do modelo)
COMPOSICAO_SALARIO = [
    ("p583", "p665"),  # Motociclista noturno
    ("p584", "p666"),  # Eletricista diurno
    ("p585", "p667"),  # Ajudante diurno
    ("p586", "p668"),  # Eletricista noturno
    ("p587", "p669"),  # Ajudante noturno
    ("p588", "p670"),  # Eletricista folguista
]
ENCARGOS = "p641"
BENEFICIOS = "p639"
# Veículos
VEIC_QTD_CESTO, VEIC_QTD_MOTO = "p686", "p687"
VEIC_VAL_CESTO, VEIC_VAL_MOTO = "p689", "p690"
FROTA_MARCO = {"setup": "p694", "m1": "p696", "m2": "p697", "m3": "p698", "pos": "p699"}
VEIC_LOCACAO_CESTO = "p706"
VEIC_COMBUSTIVEL_CESTO = "p712"
VEIC_DESPESAS_CESTO = "p715"
# Telegestão
TELEGESTAO_CAPEX_PONTO = "p241"
TELEGESTAO_OPEX_PONTO = "p242"
# Reinvestimento (vida útil, anos)
VIDA_UTIL_LUMINARIA = "p200"
VIDA_UTIL_TELEGESTAO = "p202"


# ── Construtores de fórmula ───────────────────────────────────────────────────
def f_custo_equipe_operacional() -> str:
    """Custo mensal de uma equipe operacional = Σ(comp×salário)×(1+encargos) + Σ(comp)×benefícios."""
    sumprod = "+".join(f"{cel(c)}*{cel(s)}" for c, s in COMPOSICAO_SALARIO)
    soma_comp = "+".join(cel(c) for c, _ in COMPOSICAO_SALARIO)
    return f"=({sumprod})*(1+{cel(ENCARGOS)})+({soma_comp})*{cel(BENEFICIOS)}"


def f_custo_operacional_marco(marco: str, cel_custo_equipe: str) -> str:
    """Custo operacional do marco = nº de equipes do marco × custo de uma equipe."""
    return f"={cel(EQ_MARCO[marco])}*{cel_custo_equipe}"


def f_capex_veiculos() -> str:
    """CAPEX de aquisição de veículos."""
    return (f"={cel(VEIC_QTD_CESTO)}*{cel(VEIC_VAL_CESTO)}"
            f"+{cel(VEIC_QTD_MOTO)}*{cel(VEIC_VAL_MOTO)}")


def f_opex_veiculos_marco(marco: str) -> str:
    """OPEX mensal de veículos do marco = frota × (locação + combustível + despesas)."""
    return (f"={cel(FROTA_MARCO[marco])}*({cel(VEIC_LOCACAO_CESTO)}"
            f"+{cel(VEIC_COMBUSTIVEL_CESTO)}+{cel(VEIC_DESPESAS_CESTO)})")


def f_expansao_pontos_ano(ano_1based: int) -> str:
    """Pontos expandidos no ano (0 antes do início da expansão, expansão anual depois)."""
    inicio_ano = f"CEILING({cel(EXP_INICIO_MES)}/{cel(MESES_ANO)},1)"
    return f"=IF({ano_1based}>={inicio_ano},{cel(EXP_ANUAL_PONTOS)},0)"


def f_capex_expansao_ano(cel_pontos_ano: str) -> str:
    """CAPEX de expansão no ano = pontos do ano × custo médio de luminária de expansão."""
    return f"={cel_pontos_ano}*{cel(CUSTO_MEDIO_LUM_EXP)}"


def f_reinvestimento_ano(cel_base_capex: str, cel_vida_util: str, ano_1based: int) -> str:
    """
    Distribuição temporal por vida útil/reinvestimento: reinveste o CAPEX base a cada
    `vida_util` anos (no ano 1 e nos múltiplos seguintes), senão 0.
    """
    return (f"=IF(AND({ano_1based}>1,MOD({ano_1based}-1,{cel_vida_util})=0),"
            f"{cel_base_capex},0)")


def f_capex_telegestao() -> str:
    """CAPEX de telegestão = parque total × custo de telegestão por ponto."""
    return f"={cel(PARQUE_TOTAL)}*{cel(TELEGESTAO_CAPEX_PONTO)}"


def anos_concessao_default() -> int:
    """Prazo de concessão default (do catálogo) para dimensionar a linha do tempo."""
    p = schema.carregar().parametro(PRAZO_CONCESSAO_ANOS)
    try:
        return int(float(p.default)) if p and p.default else 25
    except (TypeError, ValueError):
        return 25


__all__ = [
    "ABA_INPUTS", "cel", "existe",
    "f_custo_equipe_operacional", "f_custo_operacional_marco",
    "f_capex_veiculos", "f_opex_veiculos_marco",
    "f_expansao_pontos_ano", "f_capex_expansao_ano",
    "f_reinvestimento_ano", "f_capex_telegestao", "anos_concessao_default",
    "EQ_MARCO", "FROTA_MARCO",
]
