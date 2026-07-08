"""
Fator de extrapolação (seção 9).

A inspeção é amostral. Para que os tratamentos reflitam o universo do cadastro,
aplica-se um fator único por município = FLOOR(total_cadastro / total_amostra).

Arredondamento sempre para baixo. O fator é inteiro (não decimal).
Não se aplica em IAE/ID — essas bases são completas, não amostrais.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Extrapolacao:
    total_cadastro: int
    total_amostra: int
    fator: int

    @property
    def cobertura(self) -> float:
        if self.total_cadastro == 0:
            return 0.0
        return self.total_amostra / self.total_cadastro


def calcular_fator(total_cadastro: int, total_amostra: int) -> Extrapolacao:
    """
    Calcula o fator de extrapolação.

    Exemplos:
      - 15000 / 300 = 50.0 → fator = 50
      - 11272 / 331 ≈ 34.05 → fator = 34 (sempre arredonda para baixo)

    Edge cases:
      - amostra = 0 → fator = 0 (não há base para extrapolar)
      - cadastro < amostra → fator = 0 (situação anômala; reportar no relatório)
    """
    if total_amostra <= 0 or total_cadastro <= 0:
        return Extrapolacao(total_cadastro, total_amostra, 0)
    fator = math.floor(total_cadastro / total_amostra)
    return Extrapolacao(total_cadastro, total_amostra, int(fator))


# ── Fórmula viva para gravar nas células do .xlsx (seção 9.3) ─────────────────
def formula_fator_excel(ref_cadastro: str = "Cadastro!A:A", ref_amostra: str = "Amostra!A:A") -> str:
    """
    Gera a fórmula recomendada para a célula do fator de extrapolação.

    O fator deve estar em UMA célula nomeada e referenciada (ex: 'Fator_Extrapolacao'
    em $B$2), não chumbado em cada linha. Quando o usuário editar a amostra ou o
    cadastro, todo o modelo recalcula sozinho.
    """
    return f"ROUNDDOWN(COUNTA({ref_cadastro})/COUNTA({ref_amostra}),0)"


__all__ = ["Extrapolacao", "calcular_fator", "formula_fator_excel"]
