"""
Identificação do município de um cadastro, contra a lista oficial do IBGE.

Por que existe: município e UF não são só rótulo de cabeçalho. A UF **entra no cálculo**
— é ela que estreita as zonas UTM candidatas quando o cadastro vem projetado (ver
`coordenadas.py`), e uma sigla digitada errada degrada essa desambiguação em silêncio.
E o nome do município é carimbado na planilha que vai para o cliente, onde erro de
digitação é constrangimento, não detalhe.

A saída disso é não deixar o operador digitar: a escolha sai de uma lista fechada de
5.570 municípios, a UF é derivada dela, e quando o próprio cadastro traz a coluna de
município a escolha já vem pronta.
"""

from __future__ import annotations

import pandas as pd

from .normalizacao import _slug, detectar_colunas

UFS: tuple[str, ...] = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)

COLUNAS_LISTA = ["ente", "uf", "rotulo", "norm"]


def listar_municipios() -> pd.DataFrame:
    """
    Os municípios do cadastro do SICONFI, já versionado no repositório.

    Usa `entes_em_cache`, que **nunca vai à rede**: abrir a ferramenta de amostragem não
    pode depender da API do Tesouro estar de pé, nem pagar um timeout por isso. Sem o
    parquet devolve vazio, e cabe à UI degradar para campo de texto livre.
    """
    try:
        from hub_municipios import siconfi
        entes = siconfi.entes_em_cache()
    except Exception:
        return pd.DataFrame(columns=COLUNAS_LISTA)
    if entes is None or entes.empty or not {"ente", "uf"}.issubset(entes.columns):
        return pd.DataFrame(columns=COLUNAS_LISTA)

    lista = entes[["ente", "uf"]].dropna().astype(str).copy()
    lista["rotulo"] = lista["ente"].str.strip() + "/" + lista["uf"].str.strip()
    lista["norm"] = lista["ente"].map(_slug)
    return (lista[COLUNAS_LISTA]
            .sort_values("rotulo", kind="stable")
            .reset_index(drop=True))


def sugerir(cadastro: pd.DataFrame | None, lista: pd.DataFrame) -> str | None:
    """
    Rótulo `Município/UF` deduzido do próprio cadastro, ou None.

    O valor considerado é o **mais frequente**, não o da primeira linha: base
    consolidada de consórcio mistura municípios, e a primeira linha não representa o
    lote. Homônimo em vários estados (há dezenas de "Bom Jesus") é desempatado pela
    coluna de UF do cadastro; sem ela, não se escolhe — devolve None e o operador decide.
    """
    if cadastro is None or cadastro.empty or lista.empty:
        return None

    achadas = detectar_colunas(cadastro, obrigatorios=[], recomendados=["municipio", "uf"])
    col_mun = achadas.mapeados.get("municipio")
    if not col_mun or col_mun not in cadastro.columns:
        return None

    valores = cadastro[col_mun].dropna().astype(str).str.strip()
    valores = valores[valores != ""]
    if valores.empty:
        return None

    candidatos = lista[lista["norm"] == _slug(valores.mode().iloc[0])]
    if candidatos.empty:
        return None
    if len(candidatos) == 1:
        return str(candidatos.iloc[0]["rotulo"])

    col_uf = achadas.mapeados.get("uf")
    if col_uf and col_uf in cadastro.columns:
        ufs = cadastro[col_uf].dropna().astype(str).str.strip().str.upper()
        ufs = ufs[ufs.isin(UFS)]
        if not ufs.empty:
            pela_uf = candidatos[candidatos["uf"].str.upper() == ufs.mode().iloc[0]]
            if len(pela_uf) == 1:
                return str(pela_uf.iloc[0]["rotulo"])
    return None   # homônimo sem desempate: melhor perguntar que chutar o estado errado


def separar_rotulo(texto: str) -> tuple[str, str]:
    """
    Quebra `Município/UF` digitado à mão em (nome, UF). UF inválida vira vazia.

    Existe para o município que não está na lista do IBGE (distrito, nome antigo, base
    de teste). Sem sufixo reconhecível a UF fica **vazia em vez de adivinhada**: uma UF
    errada escolheria a zona UTM errada, o que é pior que não ter UF nenhuma.
    """
    texto = (texto or "").strip()
    if not texto:
        return "", ""
    nome, separador, sufixo = texto.rpartition("/")
    sufixo = sufixo.strip().upper()
    if separador and nome.strip() and sufixo in UFS:
        return nome.strip(), sufixo
    return texto, ""


__all__ = ["UFS", "listar_municipios", "separar_rotulo", "sugerir"]
