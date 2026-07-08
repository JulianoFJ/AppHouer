"""
Lookup do tempo de operação diário por município (seção 3).

Fonte: Resolução Homologatória ANEEL nº 2.590/2019 (Anexo I).
O PDF tem ~5.500 municípios — extraído uma única vez para `data/aneel_2590.csv`
com colunas: codigo_ibge, municipio, uf, horas, minutos.

Para extrair o PDF na primeira execução, use `extrair_pdf()` neste módulo
(consome `tabula-py` ou `pdfplumber` — instalar sob demanda).

Fallback: se o lookup falhar (município ausente da base ou base ainda não
extraída), a UI deve perguntar o valor ao usuário no formato HHhMMmin.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


# Caminho da base extraída do PDF
_AQUI = Path(__file__).parent
DATA_DIR = _AQUI / "data"
CSV_ANEEL = DATA_DIR / "aneel_2590.csv"


@dataclass
class TempoOperacao:
    municipio: str
    uf: str
    horas: int
    minutos: int
    codigo_ibge: int | None = None

    @property
    def formato_hhmm(self) -> str:
        return f"{self.horas:02d}h{self.minutos:02d}min"

    @property
    def total_horas_decimal(self) -> float:
        return self.horas + self.minutos / 60.0


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _slug(s: str) -> str:
    s = _strip_accents(str(s)).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


@lru_cache(maxsize=1)
def _carregar_base() -> pd.DataFrame | None:
    """Carrega o CSV uma única vez (cache). Retorna None se não existir."""
    if not CSV_ANEEL.exists():
        return None
    df = pd.read_csv(CSV_ANEEL, dtype={"codigo_ibge": "Int64"})
    df["_mun_slug"] = df["municipio"].map(_slug)
    df["_uf"] = df["uf"].str.upper().str.strip()
    return df


def base_disponivel() -> bool:
    """Indica se a base ANEEL já foi extraída do PDF e está pronta para uso."""
    return _carregar_base() is not None


def buscar(municipio: str, uf: str, codigo_ibge: int | None = None) -> TempoOperacao | None:
    """
    Retorna o tempo de operação para um município+UF.

    Estratégia:
      1. Se houver codigo_ibge, prioriza match exato pelo IBGE (desempate de homônimos).
      2. Senão, match por slug(municipio) + UF.
      3. Se não encontrar, retorna None (caller deve perguntar ao usuário).
    """
    base = _carregar_base()
    if base is None:
        return None

    if codigo_ibge is not None:
        hit = base[base["codigo_ibge"] == int(codigo_ibge)]
        if not hit.empty:
            row = hit.iloc[0]
            return TempoOperacao(
                municipio=row["municipio"],
                uf=row["_uf"],
                horas=int(row["horas"]),
                minutos=int(row["minutos"]),
                codigo_ibge=int(row["codigo_ibge"]) if pd.notna(row["codigo_ibge"]) else None,
            )

    mun_slug = _slug(municipio)
    uf_norm = str(uf).upper().strip()
    hit = base[(base["_mun_slug"] == mun_slug) & (base["_uf"] == uf_norm)]
    if hit.empty:
        return None

    row = hit.iloc[0]
    return TempoOperacao(
        municipio=row["municipio"],
        uf=row["_uf"],
        horas=int(row["horas"]),
        minutos=int(row["minutos"]),
        codigo_ibge=int(row["codigo_ibge"]) if pd.notna(row["codigo_ibge"]) else None,
    )


def parse_hhmm(texto: str) -> tuple[int, int] | None:
    """
    Aceita formatos comuns que o usuário possa digitar manualmente:
      '11h26min', '11h26', '11:26', '11 26', '11.43' (decimal).
    Retorna (horas, minutos) ou None se irreconhecível.
    """
    if texto is None:
        return None
    s = str(texto).strip().lower()
    if not s:
        return None

    # Decimal: '11.43' → 11h25min (0.43 * 60 ≈ 26)
    m = re.fullmatch(r"(\d{1,2})[.,](\d{1,3})", s)
    if m and "h" not in s and ":" not in s:
        horas = int(m.group(1))
        frac = float("0." + m.group(2))
        return horas, int(round(frac * 60))

    # Formato com separadores: 11h26min, 11h26, 11:26, 11 26
    m = re.fullmatch(r"(\d{1,2})\s*[h:]\s*(\d{1,2})\s*(min)?", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.fullmatch(r"(\d{1,2})\s+(\d{1,2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Só horas, ex: '11'
    m = re.fullmatch(r"(\d{1,2})", s)
    if m:
        return int(m.group(1)), 0

    return None


# ── Extração do PDF (executar uma única vez) ──────────────────────────────────
# Cada página do Anexo I é uma tabela visual com 4 colunas: IBGE | UF | Município
# | HHhMMmin. O `pdfplumber.extract_tables()` retorna vazio para esse PDF (não
# há grades reconhecíveis), então extraímos via texto linear + regex.
_PADRAO_LINHA_PDF = re.compile(
    r"^(\d{7})\s+"          # código IBGE (7 dígitos)
    r"([A-Z]{2})\s+"        # UF (2 letras)
    r"(.+?)\s+"             # município (qualquer coisa, lazy)
    r"(\d{1,2})h(\d{2})min" # tempo HHhMMmin
    r"\s*$"
)


def extrair_pdf(pdf_path: str | Path, csv_destino: str | Path | None = None) -> Path:
    """
    Extrai os dados do Anexo I da Resolução 2.590/2019 e grava em CSV.

    Requer `pdfplumber` instalado (`pip install pdfplumber`). O parser opera
    sobre o texto linear extraído (não tenta detectar tabelas — o PDF não
    tem grade reconhecível) e casa cada linha contra o padrão
    `IBGE UF MUNICIPIO HHhMMmin`.

    Returns:
        Path do CSV gerado (com colunas: codigo_ibge, municipio, uf, horas, minutos).
    Raises:
        RuntimeError se nenhuma linha for reconhecida (layout pode ter mudado).
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "Para extrair o PDF da ANEEL é necessário `pdfplumber`. "
            "Instale com: pip install pdfplumber"
        ) from exc

    pdf_path = Path(pdf_path)
    csv_destino = Path(csv_destino) if csv_destino else CSV_ANEEL
    csv_destino.parent.mkdir(parents=True, exist_ok=True)

    registros: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text() or ""
            for linha in texto.split("\n"):
                m = _PADRAO_LINHA_PDF.match(linha.strip())
                if m:
                    registros.append({
                        "codigo_ibge": int(m.group(1)),
                        "uf": m.group(2),
                        "municipio": m.group(3).strip(),
                        "horas": int(m.group(4)),
                        "minutos": int(m.group(5)),
                    })

    if not registros:
        raise RuntimeError(
            f"Nenhuma linha de município reconhecida em {pdf_path}. "
            "O layout do PDF pode ter mudado — verifique se o padrão "
            "'IBGE UF MUNICIPIO HHhMMmin' ainda se aplica."
        )

    df = pd.DataFrame(registros).drop_duplicates(subset=["codigo_ibge"])
    # Ordena por UF + município para facilitar revisão manual do CSV
    df = df.sort_values(["uf", "municipio"]).reset_index(drop=True)
    df.to_csv(csv_destino, index=False)
    _carregar_base.cache_clear()
    return csv_destino


__all__ = [
    "TempoOperacao",
    "buscar",
    "base_disponivel",
    "parse_hhmm",
    "extrair_pdf",
    "CSV_ANEEL",
]
