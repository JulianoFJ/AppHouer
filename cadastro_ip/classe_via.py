"""
Propagação de Classe Via por logradouro (seção 10).

A inspeção classifica cada ponto inspecionado com sua Classe Via (NBR 5101).
Como a inspeção é amostral, propaga-se a classe para todos os pontos do
cadastro na mesma rua, com restrições:

  - APENAS as classes C0, C1, C2, C3, M1, M2 são propagáveis.
  - Outras classes (C4, C5, etc.) ficam em branco — não propagam.
  - Classe Pedestre (P1–P6) NUNCA propaga — vai só onde a inspeção mediu.
  - Ruas sem nenhum ponto inspecionado → todos pontos em branco.
  - Divergência na mesma rua → maioria; empate → mais restritiva (menor número).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from .normalizacao import chave_logradouro, limpar_id_serie


CLASSES_PROPAGAVEIS = {"C0", "C1", "C2", "C3", "M1", "M2"}

# Severidade ordinal — número menor = mais restritiva (exigente)
RANK_RESTRITIVIDADE = {
    "C0": 0, "C1": 1, "C2": 2, "C3": 3,
    "M1": 0, "M2": 1,
}


@dataclass
class ResultadoPropagacao:
    df_cadastro_com_classe: pd.DataFrame   # cadastro + coluna 'classe_via' preenchida ou ""
    pontos_sem_classe: int
    pontos_com_classe: int
    logradouros_divergentes: list[dict]    # [{rua, classes_observadas, escolhida, contagens}]
    aviso_sem_bairro: bool
    quantidades_por_classe: dict[str, int] = field(default_factory=dict)


def _normalizar_classe(c) -> str | None:
    """Aceita 'C3', 'c3', 'C 3', 'CLASSE C3', etc."""
    if c is None or (isinstance(c, float) and pd.isna(c)):
        return None
    s = str(c).upper().strip()
    m = re.search(r"\b([CM])\s*([0-9])\b", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s or None


def _escolher_classe_por_rua(classes_inspecao: list[str]) -> str | None:
    """
    Dado o conjunto de classes observadas nos pontos inspecionados de uma rua,
    retorna a classe a propagar — ou None se nenhuma for propagável.
    """
    propagaveis = [c for c in classes_inspecao if c in CLASSES_PROPAGAVEIS]
    if not propagaveis:
        return None
    contagens = Counter(propagaveis)
    max_freq = max(contagens.values())
    candidatas = [c for c, n in contagens.items() if n == max_freq]
    if len(candidatas) == 1:
        return candidatas[0]
    # Empate → mais restritiva (menor rank)
    candidatas.sort(key=lambda c: RANK_RESTRITIVIDADE.get(c, 99))
    return candidatas[0]


def propagar(
    cadastro: pd.DataFrame,
    inspecao: pd.DataFrame,
    col_logradouro: str = "logradouro",
    col_bairro: str = "bairro",
    col_classe_insp: str = "classe_via",
    col_id: str = "id_ponto",
) -> ResultadoPropagacao:
    """
    Propaga a classe via da inspeção para todos os pontos do cadastro na mesma rua,
    conforme regras da seção 10.

    Se a inspeção não tiver coluna de logradouro, faz lookup no cadastro pelo
    id_ponto para descobrir em qual rua cada ponto inspecionado está.

    Returns:
        ResultadoPropagacao com o cadastro enriquecido e estatísticas para o relatório.
    """
    aviso_sem_bairro = col_bairro not in cadastro.columns

    # ── Sem coluna de classe na inspeção: nada a propagar ───────────────────
    if col_classe_insp not in inspecao.columns:
        cad = cadastro.copy()
        cad["classe_via"] = ""
        return ResultadoPropagacao(
            df_cadastro_com_classe=cad,
            pontos_sem_classe=len(cad),
            pontos_com_classe=0,
            logradouros_divergentes=[],
            aviso_sem_bairro=aviso_sem_bairro,
        )

    insp = inspecao.copy()

    # Enriquece a inspeção com logradouro e/ou bairro vindos do cadastro
    # (lookup por id_ponto) sempre que faltar. Sem isso, a chave de logradouro
    # gerada na inspeção fica diferente da do cadastro e a propagação não bate.
    if col_id in insp.columns and col_id in cadastro.columns:
        cols_lookup = [col_id]
        if col_logradouro not in insp.columns and col_logradouro in cadastro.columns:
            cols_lookup.append(col_logradouro)
        if col_bairro not in insp.columns and col_bairro in cadastro.columns:
            cols_lookup.append(col_bairro)
        if len(cols_lookup) > 1:
            lookup = cadastro[cols_lookup].copy()
            lookup[col_id] = limpar_id_serie(lookup[col_id])
            insp[col_id] = limpar_id_serie(insp[col_id])
            insp = insp.merge(lookup, on=col_id, how="left", suffixes=("", "_cad"))

    if col_logradouro not in insp.columns:
        cad = cadastro.copy()
        cad["classe_via"] = ""
        return ResultadoPropagacao(
            df_cadastro_com_classe=cad,
            pontos_sem_classe=len(cad),
            pontos_com_classe=0,
            logradouros_divergentes=[],
            aviso_sem_bairro=aviso_sem_bairro,
        )

    insp["_classe"] = insp[col_classe_insp].map(_normalizar_classe)
    if col_bairro in insp.columns:
        insp["_chave"] = [
            chave_logradouro(log, bai) for log, bai in zip(insp[col_logradouro], insp[col_bairro])
        ]
    else:
        insp["_chave"] = insp[col_logradouro].map(chave_logradouro)

    # Agrupa classes inspecionadas por chave de logradouro
    mapa_classe: dict[str, str] = {}
    divergencias: list[dict] = []
    for chave, grupo in insp.dropna(subset=["_classe"]).groupby("_chave"):
        classes = [c for c in grupo["_classe"].tolist() if c]
        if not classes:
            continue
        escolhida = _escolher_classe_por_rua(classes)
        if escolhida is not None:
            mapa_classe[chave] = escolhida
        # Registra divergência se houver mais de uma classe distinta
        classes_unicas = set(classes)
        if len(classes_unicas) > 1:
            divergencias.append(
                {
                    "rua": chave,
                    "classes_observadas": sorted(classes_unicas),
                    "escolhida": escolhida or "(em branco — nenhuma propagável)",
                    "contagens": dict(Counter(classes)),
                }
            )

    # ── Aplica ao cadastro ──────────────────────────────────────────────────
    cad = cadastro.copy()
    if col_bairro in cad.columns:
        cad["_chave"] = [
            chave_logradouro(log, bai) for log, bai in zip(cad[col_logradouro], cad[col_bairro])
        ] if col_logradouro in cad.columns else [""] * len(cad)
    else:
        cad["_chave"] = cad[col_logradouro].map(chave_logradouro) if col_logradouro in cad.columns else [""] * len(cad)

    cad["classe_via"] = cad["_chave"].map(lambda k: mapa_classe.get(k, ""))
    cad.drop(columns=["_chave"], inplace=True)

    qtd_classe = cad["classe_via"].value_counts().to_dict()
    qtd_classe.pop("", None)
    pontos_com_classe = int(sum(qtd_classe.values()))
    pontos_sem_classe = int(len(cad) - pontos_com_classe)

    return ResultadoPropagacao(
        df_cadastro_com_classe=cad,
        pontos_sem_classe=pontos_sem_classe,
        pontos_com_classe=pontos_com_classe,
        logradouros_divergentes=divergencias,
        aviso_sem_bairro=aviso_sem_bairro,
        quantidades_por_classe=qtd_classe,
    )


__all__ = [
    "CLASSES_PROPAGAVEIS",
    "ResultadoPropagacao",
    "propagar",
]
