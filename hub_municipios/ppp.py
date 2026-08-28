"""
Base de PPPs de iluminação pública já contratadas no Brasil.

181 contratos, 22 UFs, 4,87 milhões de pontos sob concessão — 25,2% do parque
nacional mapeado pela BDGD. Traz concessionária, acionistas, valor, vigência, data de
assinatura e cobertura de telegestão.

Serve a dois propósitos na triagem:
  · marcar quem já tem contrato, para não entrar como alvo de nova estruturação;
  · ancorar o custo de referência em contrato assinado, e não em premissa: a despesa
    implícita mediana é R$ 34,23/ponto.mês (quartis R$ 19,73–46,43).

Fonte: aba `Planilha1` de `CLP.xlsx`, no Drive compartilhado do time. Para atualizar,
rode `py -m hub_municipios._importar_ppp`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config

ARQUIVO = config.DATA_PACOTE / "ppp_existentes.parquet"

_CACHE: Optional[pd.DataFrame] = None


def carregar(forcar: bool = False) -> pd.DataFrame:
    """Contratos de PPP de IP. DataFrame vazio se a base ainda não foi importada."""
    global _CACHE
    if _CACHE is not None and not forcar:
        return _CACHE
    if not ARQUIVO.exists():
        _CACHE = pd.DataFrame()
        return _CACHE
    try:
        df = pd.read_parquet(ARQUIVO)
        df["codigo_municipio"] = df["codigo_municipio"].astype(str)
    except Exception:
        df = pd.DataFrame()
    _CACHE = df
    return df


def do_municipio(codigo_municipio: str) -> Optional[pd.Series]:
    df = carregar()
    if df.empty:
        return None
    linha = df[df["codigo_municipio"] == str(codigo_municipio)]
    return None if linha.empty else linha.iloc[0]


def referencia_de_custo() -> dict:
    """Despesa contratada por ponto/mês — o benchmark derivado de contrato real."""
    df = carregar()
    if df.empty or "despesa_ponto_mes" not in df.columns:
        return {}
    d = pd.to_numeric(df["despesa_ponto_mes"], errors="coerce").dropna()
    if d.empty:
        return {}
    return {"n": int(len(d)), "p25": float(d.quantile(.25)),
            "mediana": float(d.median()), "p75": float(d.quantile(.75))}
