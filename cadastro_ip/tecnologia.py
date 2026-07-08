"""
Classificação de tecnologia (seção 5).

Duas famílias:
  - LED — qualquer variante contendo "LED"
  - Convencional — tudo que não é LED

E normalização dos códigos de tecnologias convencionais para o padrão da Houer:
VS, VM, VMT, MT, FL, IN, mais categorias especiais de luminária (Globo,
Ornamental, Rebatedor, Projetor).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


# ── Códigos padronizados (seção 5.1) ──────────────────────────────────────────
CODIGO_DESC = {
    "VS":  "Vapor de Sódio",
    "VM":  "Vapor de Mercúrio",
    "VMT": "Vapor Metálico (Iodetos Metálicos)",
    "MT":  "Mista",
    "FL":  "Fluorescente",
    "IN":  "Incandescente",
    "LED": "LED",
}

# Categorias especiais de luminária (seção 5.2) — convencionais com potência possivelmente nula
CATEGORIAS_LUMINARIA = {"GLOBO", "ORNAMENTAL", "REBATEDOR", "PROJETOR"}

# Mapeamento de variantes de entrada → código padronizado.
# Slug aplicado (lower, sem acento, sem pontuação) antes da busca.
VARIANTES_PARA_CODIGO: dict[str, str] = {
    # LED
    "led": "LED",
    "lampada led": "LED",
    "luminaria led": "LED",
    # Vapor de Sódio
    "vs": "VS",
    "sodio": "VS",
    "vapor de sodio": "VS",
    "vapor sodio": "VS",
    "vsap": "VS",
    "na": "VS",
    "hps": "VS",
    # Vapor de Mercúrio
    "vm": "VM",
    "mercurio": "VM",
    "vapor de mercurio": "VM",
    "vapor mercurio": "VM",
    "mv": "VM",
    "hg": "VM",
    "vma": "VM",
    "vme": "VM",        # Vapor Mercúrio Elétrica/Eletrônica
    "vmd": "VM",        # variante regional
    "lvm": "VM",        # Lâmpada Vapor de Mercúrio
    # Vapor Metálico
    "vmt": "VMT",
    "metalico": "VMT",
    "vapor metalico": "VMT",
    "metal halide": "VMT",
    "mh": "VMT",
    "iodetos metalicos": "VMT",
    "iodeto metalico": "VMT",
    "im": "VMT",
    "vmi": "VMT",       # Vapor Metálico Iodetos
    "lvm metalica": "VMT",
    # Mista
    "mt": "MT",
    "mista": "MT",
    "mix": "MT",
    "mis": "MT",
    # Fluorescente
    "fl": "FL",
    "fluorescente": "FL",
    "flu": "FL",
    # Incandescente
    "in": "IN",
    "incandescente": "IN",
    "inc": "IN",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _slug(s: str) -> str:
    s = _strip_accents(str(s)).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def codigo_padronizado(valor) -> str | None:
    """
    Recebe uma string bruta de tecnologia e retorna o código padronizado
    (LED, VS, VM, VMT, MT, FL, IN) ou o nome de uma categoria especial de
    luminária em caixa alta (GLOBO, ORNAMENTAL, REBATEDOR, PROJETOR).

    Retorna None quando o valor é vazio ou irreconhecível — o caller decide
    o que fazer (ex: registrar no relatório).
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    bruto = str(valor).strip()
    if not bruto:
        return None
    sl = _slug(bruto)

    # LED tem prioridade — qualquer variante contendo "led" é LED (seção 5)
    if "led" in sl.split() or sl.startswith("led ") or sl.endswith(" led") or sl == "led":
        return "LED"

    # Categorias especiais de luminária
    for cat in CATEGORIAS_LUMINARIA:
        if cat.lower() in sl:
            return cat

    # Match exato pela tabela de variantes
    if sl in VARIANTES_PARA_CODIGO:
        return VARIANTES_PARA_CODIGO[sl]

    # Match parcial (token a token) — útil quando a célula traz "Lâmpada Vapor de Sódio 250W"
    tokens = sl.split()
    # Tentar substrings progressivamente menores
    for tamanho in range(min(4, len(tokens)), 0, -1):
        for inicio in range(0, len(tokens) - tamanho + 1):
            sub = " ".join(tokens[inicio : inicio + tamanho])
            if sub in VARIANTES_PARA_CODIGO:
                return VARIANTES_PARA_CODIGO[sub]

    return None


def familia(codigo: str | None) -> str:
    """Retorna 'LED' ou 'Convencional' a partir do código padronizado. Outro: 'Desconhecido'."""
    if codigo is None:
        return "Desconhecido"
    if codigo == "LED":
        return "LED"
    if codigo in CODIGO_DESC or codigo in CATEGORIAS_LUMINARIA:
        return "Convencional"
    return "Desconhecido"


def aplicar_classificacao(df: pd.DataFrame, coluna_origem: str = "tecnologia") -> pd.DataFrame:
    """
    Adiciona ao df duas colunas:
      - 'codigo_tecnologia' — código padronizado (LED, VS, VM, etc.)
      - 'familia_tecnologia' — 'LED', 'Convencional' ou 'Desconhecido'

    Não altera a coluna original — preserva o valor bruto para auditoria.
    """
    if coluna_origem not in df.columns:
        raise KeyError(f"Coluna de tecnologia '{coluna_origem}' não encontrada no DataFrame.")
    df = df.copy()
    df["codigo_tecnologia"] = df[coluna_origem].map(codigo_padronizado)
    df["familia_tecnologia"] = df["codigo_tecnologia"].map(familia)
    return df


def codigos_desconhecidos(df: pd.DataFrame, coluna_origem: str = "tecnologia") -> dict[str, int]:
    """
    Retorna { codigo_bruto: quantidade } para os valores presentes na coluna
    de tecnologia que NÃO foram reconhecidos pelo classificador.

    Use no relatório de execução para listar ao usuário quais códigos novos
    apareceram e quantos pontos cada um afeta — em vez de mascarar em "SEM CLASS.".
    """
    if coluna_origem not in df.columns:
        return {}
    contagem: dict[str, int] = {}
    for valor in df[coluna_origem]:
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            continue
        bruto = str(valor).strip()
        if not bruto:
            continue
        if codigo_padronizado(bruto) is None:
            contagem[bruto] = contagem.get(bruto, 0) + 1
    return contagem


def normalizacoes_aplicadas(df: pd.DataFrame, coluna_origem: str = "tecnologia") -> dict[str, int]:
    """
    Retorna contagem de normalizações por par (entrada bruta → código).
    Útil para o relatório (seção 12.4) registrar quais codificações foram aplicadas.
    """
    if "codigo_tecnologia" not in df.columns:
        df = aplicar_classificacao(df, coluna_origem)
    contagem: dict[str, int] = {}
    for bruto, codigo in zip(df[coluna_origem], df["codigo_tecnologia"]):
        if codigo is None or (isinstance(codigo, float) and pd.isna(codigo)):
            continue
        bruto_s = str(bruto).strip() if bruto is not None else ""
        codigo_s = str(codigo).strip()
        if bruto_s.upper() != codigo_s.upper() and codigo_s and codigo_s.lower() != "none":
            key = f"{bruto_s} → {codigo_s}"
            contagem[key] = contagem.get(key, 0) + 1
    return contagem


__all__ = [
    "CODIGO_DESC",
    "CATEGORIAS_LUMINARIA",
    "codigo_padronizado",
    "familia",
    "aplicar_classificacao",
    "normalizacoes_aplicadas",
    "codigos_desconhecidos",
]
