"""
Leitura da planilha de cadastro enviada pela UI.

Mesma heurística de detecção de cabeçalho usada na página de Análise de Cadastro —
cadastro municipal costuma vir com uma ou duas linhas de título antes do cabeçalho
real, e ler com `header=0` nesse caso produz colunas chamadas "Unnamed: 3". Aqui a
função vive no pacote (e não dentro do script da página) para poder ser testada sem
subir o Streamlit.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


# Tokens que denunciam o cabeçalho real de um cadastro de IP.
_TOKENS_CABECALHO = {
    "etiqueta", "id_ponto", "id pip", "identificador", "idg", "cod_id", "cod id",
    "tecnologia", "tipo de lampada", "tipo lampada", "logradouro", "endereco",
    "latitude", "longitude", "bairro", "classe", "potencia",
}


def _slug(valor) -> str:
    texto = unicodedata.normalize("NFD", str(valor))
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", texto.lower()).strip()


def detectar_linha_cabecalho(df_bruto: pd.DataFrame, max_linhas: int = 10) -> int:
    """Índice (0-based) da linha que contém o cabeçalho real. Default: 0."""
    for indice in range(min(max_linhas, len(df_bruto))):
        valores = df_bruto.iloc[indice].dropna().astype(str).tolist()
        slugs = {_slug(v) for v in valores}
        acertos = sum(1 for token in _TOKENS_CABECALHO if any(token in s for s in slugs))
        if acertos >= 2:
            return indice
    return 0


def ler_planilha(arquivo) -> pd.DataFrame:
    """
    Lê .xlsx/.xls/.csv (caminho ou objeto de upload do Streamlit) em um DataFrame.

    Levanta a exceção original em caso de falha — a UI é quem decide como reportar.
    """
    nome = str(getattr(arquivo, "name", arquivo)).lower()
    if nome.endswith(".csv"):
        return pd.read_csv(arquivo)

    bruto = pd.read_excel(arquivo, sheet_name=0, header=None)
    linha = detectar_linha_cabecalho(bruto)
    if hasattr(arquivo, "seek"):
        arquivo.seek(0)
    return pd.read_excel(arquivo, sheet_name=0, header=linha)


__all__ = ["ler_planilha", "detectar_linha_cabecalho"]
