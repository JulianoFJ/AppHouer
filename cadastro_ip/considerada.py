"""
Regra da coluna "Considerada" (seção 8).

A tecnologia/potência/quantidade Considerada é o valor que prevalece após o
tratamento e alimenta o inventário final.

Regra geral: prevalece a INSPEÇÃO (campo é mais confiável que cadastro).
Exceção única: Cadastro = LED E Inspeção = Convencional → prevalece o CADASTRO
(presume-se erro de coleta em campo).

Aplica-se simultaneamente às três colunas correlatas:
Tecnologia / Potência / Quantidade Considerada.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValorConsiderado:
    tecnologia: str | None
    potencia: float | None
    quantidade: float | None
    fonte: str  # 'INSPECAO' ou 'CADASTRO'


def _eh_led(codigo: str | None) -> bool:
    return codigo is not None and str(codigo).strip().upper() == "LED"


def resolver_ponto(
    cad_tec: str | None,
    cad_pot: float | None,
    cad_qtd: float | None,
    insp_tec: str | None,
    insp_pot: float | None,
    insp_qtd: float | None,
) -> ValorConsiderado:
    """
    Resolve qual fonte prevalece (cadastro vs inspeção) para um único ponto.

    Args:
        cad_* — valores do cadastro (códigos já padronizados — LED, VS, VM, ...)
        insp_* — valores da inspeção (códigos já padronizados)

    Returns:
        ValorConsiderado com os 3 atributos e a fonte aplicada.
    """
    # Exceção: cadastro LED + inspeção Convencional → mantém cadastro (LED)
    if _eh_led(cad_tec) and insp_tec is not None and not _eh_led(insp_tec):
        return ValorConsiderado(
            tecnologia=cad_tec,
            potencia=cad_pot,
            quantidade=cad_qtd,
            fonte="CADASTRO",
        )

    # Quando há inspeção válida, ela prevalece (regra geral)
    if insp_tec is not None and str(insp_tec).strip() != "":
        return ValorConsiderado(
            tecnologia=insp_tec,
            potencia=insp_pot,
            quantidade=insp_qtd,
            fonte="INSPECAO",
        )

    # Sem inspeção: fica com o cadastro
    return ValorConsiderado(
        tecnologia=cad_tec,
        potencia=cad_pot,
        quantidade=cad_qtd,
        fonte="CADASTRO",
    )


def aplicar(
    df_cruzado: pd.DataFrame,
    col_cad_tec: str = "cad_codigo_tecnologia",
    col_cad_pot: str = "cad_potencia",
    col_cad_qtd: str = "cad_quantidade",
    col_insp_tec: str = "insp_codigo_tecnologia",
    col_insp_pot: str = "insp_potencia",
    col_insp_qtd: str = "insp_quantidade",
) -> pd.DataFrame:
    """
    Aplica a regra Considerada a um DataFrame que já tem o cruzamento
    cadastro × inspeção feito (uma linha por ponto, com colunas
    `cad_*` e `insp_*` preenchidas — `NaN` onde não houve match).

    Adiciona as colunas:
      - 'tecnologia_considerada'
      - 'potencia_considerada'
      - 'quantidade_considerada'
      - 'fonte_considerada'
    """
    df = df_cruzado.copy()
    resultados = [
        resolver_ponto(
            row.get(col_cad_tec), row.get(col_cad_pot), row.get(col_cad_qtd),
            row.get(col_insp_tec), row.get(col_insp_pot), row.get(col_insp_qtd),
        )
        for _, row in df.iterrows()
    ]
    df["tecnologia_considerada"] = [r.tecnologia for r in resultados]
    df["potencia_considerada"]   = [r.potencia for r in resultados]
    df["quantidade_considerada"] = [r.quantidade for r in resultados]
    df["fonte_considerada"]      = [r.fonte for r in resultados]
    return df


__all__ = ["ValorConsiderado", "resolver_ponto", "aplicar"]
