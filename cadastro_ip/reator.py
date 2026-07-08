"""
Perdas de reator (seção 6 — Tabela 4 ABNT).

Para cada ponto convencional, encontra na tabela a potência da lâmpada mais
próxima da potência cadastrada e usa a perda correspondente. LED não tem
reator (perda = 0).

Empates de distância: usa o de maior perda (regra conservadora — seção 6.2).
Lâmpadas de 1000 W: default é a linha 250 V (perda 110 W); registrar no relatório.
"""

from __future__ import annotations

import pandas as pd


# ── Tabela 4 ABNT (seção 6.1) ─────────────────────────────────────────────────
# Ordem: (potência lâmpada [W], perda reator [W], rótulo descritivo)
TABELA_REATOR: list[tuple[float, float, str]] = [
    (50,   12,  "50 W / 85 V"),
    (70,   14,  "70 W / 90 V"),
    (100,  17,  "100 W / 100 V"),
    (150,  22,  "150 W / 100 V"),
    (250,  30,  "250 W / 100 V"),
    (400,  38,  "400 W / 100 V"),
    (1000, 90,  "1000 W / 100 V"),
    (1000, 110, "1000 W / 250 V"),
]

# Default para 1000 W (seção 6.3) → usar 250 V (perda 110 W)
PERDA_1000W_DEFAULT = 110

# Mensagem que vai no relatório quando algum 1000 W for tratado
ALERTA_1000W = (
    "Pontos de 1000 W tratados com perda de 110 W (tensão de arco 250 V) por default. "
    "Revise se o município opera com configuração de 100 V."
)


def perda_reator(potencia_lampada: float | None, is_led: bool = False) -> float:
    """
    Retorna a perda do reator (em watts) para uma dada potência de lâmpada.

    Regras:
      - LED → 0 W (não tem reator)
      - Potência ausente → 0 W (sem dado para olhar)
      - Empate de distância → escolhe o de maior perda (regra conservadora)
      - Lâmpada 1000 W → default 250 V (110 W)
    """
    if is_led:
        return 0.0
    if potencia_lampada is None:
        return 0.0
    try:
        p = float(potencia_lampada)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(p) or p <= 0:
        return 0.0

    # Caso especial: 1000 W
    if int(p) == 1000:
        return float(PERDA_1000W_DEFAULT)

    # Como há duas linhas de 1000W, removemos a duplicata para o cálculo de distância
    # geral (a regra 1000W default já foi tratada acima).
    candidatos = [(pot, perda) for pot, perda, _ in TABELA_REATOR if pot != 1000]
    candidatos.append((1000, PERDA_1000W_DEFAULT))

    melhor = None
    menor_dist = float("inf")
    for pot, perda in candidatos:
        dist = abs(p - pot)
        if dist < menor_dist - 1e-9:
            melhor = perda
            menor_dist = dist
        elif abs(dist - menor_dist) < 1e-9:
            # Empate de distância: prevalece o de maior perda (conservador)
            if perda > (melhor or 0):
                melhor = perda
    return float(melhor or 0)


def aplicar_reator(
    df: pd.DataFrame,
    coluna_potencia: str = "potencia",
    coluna_familia: str = "familia_tecnologia",
    coluna_saida: str = "reator_w",
) -> pd.DataFrame:
    """
    Adiciona coluna de perda de reator ao DataFrame.

    Espera que `aplicar_classificacao()` (módulo tecnologia) já tenha rodado,
    para que a coluna `familia_tecnologia` exista (LED vs Convencional).
    Se ela não existir, assume Convencional para tudo (mais conservador).
    """
    df = df.copy()
    if coluna_familia in df.columns:
        df[coluna_saida] = [
            perda_reator(pot, is_led=(fam == "LED"))
            for pot, fam in zip(df[coluna_potencia], df[coluna_familia])
        ]
    else:
        df[coluna_saida] = df[coluna_potencia].map(lambda p: perda_reator(p, is_led=False))
    return df


def teve_1000w(df: pd.DataFrame, coluna_potencia: str = "potencia", coluna_familia: str = "familia_tecnologia") -> bool:
    """Indica se algum ponto convencional de 1000 W foi tratado — alimenta o relatório."""
    if coluna_potencia not in df.columns:
        return False
    pot = pd.to_numeric(df[coluna_potencia], errors="coerce")
    if coluna_familia in df.columns:
        return bool(((pot == 1000) & (df[coluna_familia] != "LED")).any())
    return bool((pot == 1000).any())


# ── Fórmula viva para gravar nas células do .xlsx (seção 6.5 + 12.2.1) ────────
def formula_reator_excel(celula_potencia: str, celula_familia: str) -> str:
    """
    Gera uma fórmula Excel que aplica a regra de associação à célula informada.

    Args:
        celula_potencia: referência da célula com a potência (ex: 'C5')
        celula_familia: referência da célula com a família (ex: 'D5') —
            valor esperado é 'LED' ou 'Convencional'

    Retorna a fórmula como string (sem '=' inicial — o caller adiciona se quiser).

    Implementação: usa SE aninhado em vez de PROCV para evitar dependência de
    tabela auxiliar e manter o arquivo autocontido. Quando família = LED → 0.
    Caso contrário, busca a potência mais próxima na tabela ABNT.
    """
    p = celula_potencia
    f = celula_familia
    # Pontos de corte (média entre potências consecutivas) e regra do empate:
    # nos pontos médios exatos (60, 85, 125, 200, 325, 700) prevalece a maior
    # perda. Por isso usamos `<` (estrito) e o valor maior cai na branch seguinte.
    # 1000 W tem default de 110 W (seção 6.3) e é tratado no início.
    formula = (
        f'IF({f}="LED",0,'
        f'IF({p}=1000,110,'
        f'IF({p}<60,12,'        # <60 → 50W (12W); 60 empate → 14W
        f'IF({p}<85,14,'        # <85 → 70W (14W); 85 empate → 17W
        f'IF({p}<125,17,'       # <125 → 100W (17W); 125 empate → 22W
        f'IF({p}<200,22,'       # <200 → 150W (22W); 200 empate → 30W
        f'IF({p}<325,30,'       # <325 → 250W (30W); 325 empate → 38W
        f'IF({p}<700,38,'       # <700 → 400W (38W); 700+ → 1000W default
        f"110)))))))))"
    )
    return formula


__all__ = [
    "TABELA_REATOR",
    "PERDA_1000W_DEFAULT",
    "ALERTA_1000W",
    "perda_reator",
    "aplicar_reator",
    "teve_1000w",
    "formula_reator_excel",
]
