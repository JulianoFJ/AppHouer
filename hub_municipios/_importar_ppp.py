"""
Importa a base de PPPs de iluminação pública já contratadas para o pacote.

Fonte: aba `Planilha1` de `CLP.xlsx`, mantida pelo time no Drive compartilhado
(PVSC - Assistente de Pré-Viabilidade). São ~181 contratos com concessionária,
população, pontos de luz, valor, vigência, data de assinatura e acionistas.

Script one-shot, reexecutável:

    py -m hub_municipios._importar_ppp
    py -m hub_municipios._importar_ppp --arquivo "C:/caminho/CLP.xlsx"

Saída: `hub_municipios/data/ppp_existentes.parquet`, versionado com o pacote.

O trabalho real aqui é **casar o nome do município com o código IBGE**. A planilha
identifica o ente por texto ("Açailândia (MA)"), e todo o resto do Hub trabalha por
código de 7 dígitos. O casamento usa o cadastro de entes do SICONFI, normalizado
(sem acento, sem caixa), restrito à UF que vem entre parênteses — sem a UF, nomes
repetidos entre estados (Bom Jesus, Santa Luzia, Bonito) casariam com o município
errado silenciosamente.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config, siconfi

CAMINHO_PADRAO = Path(
    r"G:\Drives compartilhados\Head de Energia\06. Projetos"
    r"\PVSC - Assistente de Pré-Viabilidade\CLP\CLP.xlsx"
)
ABA = "Planilha1"
DESTINO = config.DATA_PACOTE / "ppp_existentes.parquet"

# "Açailândia (MA)" -> ("Açailândia", "MA")
PADRAO_MUNICIPIO_UF = re.compile(r"^\s*(?P<nome>.+?)\s*\((?P<uf>[A-Za-z]{2})\)\s*$")

# Grafias da planilha que não existem no cadastro do IBGE. Conferido em 28/08/2026
# contra o cadastro de entes do SICONFI — são erros de digitação na fonte, não
# municípios diferentes. Mantido como tabela explícita para que a correção fique
# auditável e para não silenciar um município novo que deixe de casar no futuro.
CORRECOES_DE_GRAFIA = {
    ("tome acu", "PA"): "1508001",        # planilha: "Tomé Açu"; IBGE: "Tomé-Açu"
    ("angical", "PI"): "2200608",         # planilha: "Angical";  IBGE: "Angical do Piauí"
    ("ponta gossa", "PR"): "4119905",     # planilha: "Ponta Gossa"; IBGE: "Ponta Grossa"
}

RENOMEAR = {
    "MUNICÍPIO": "municipio_uf",
    "CONCESSIONÁRIA": "concessionaria",
    "POPULAÇÃO": "populacao_contrato",
    "PONTOS DE LUZ": "pontos_luz_contrato",
    "TELEGESTÃO": "telegestao_bruto",
    "VALOR DO CONTRATO (R$ MILHÕES)": "valor_contrato_milhoes",
    "VIGÊNCIA (ANOS) ": "vigencia_anos",
    "ASSINATURA": "assinatura",
    "ACIONISTAS": "acionistas",
}


def _telegestao(valor) -> Optional[float]:
    """
    A coluna mistura três convenções: "SIM"/"NÃO" e uma fração de cobertura (0,1 a 1).
    Normaliza tudo para fração de pontos telegeridos; devolve None quando não dá.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip().upper()
    if texto in ("SIM", "S", "TRUE"):
        return 1.0
    if texto in ("NÃO", "NAO", "N", "FALSE"):
        return 0.0
    try:
        fracao = float(str(valor).replace(",", "."))
    except ValueError:
        return None
    return fracao if 0.0 <= fracao <= 1.0 else None


def _data_assinatura(valor):
    """A coluna vem como serial do Excel (42922 = 2017-07-05) ou como data."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return pd.NaT
    if isinstance(valor, (int, float)) and 20000 < float(valor) < 60000:
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(valor))
    return pd.to_datetime(valor, errors="coerce", dayfirst=True)


def importar(arquivo: Path = CAMINHO_PADRAO) -> pd.DataFrame:
    bruto = pd.read_excel(arquivo, sheet_name=ABA)
    df = bruto.rename(columns=RENOMEAR)
    df = df[[c for c in RENOMEAR.values() if c in df.columns]].copy()
    df = df[df["municipio_uf"].notna()]

    extraido = df["municipio_uf"].astype(str).str.extract(PADRAO_MUNICIPIO_UF)
    df["municipio"] = extraido["nome"]
    df["uf"] = extraido["uf"].str.upper()

    df["telegestao"] = df["telegestao_bruto"].map(_telegestao)
    df["assinatura"] = df["assinatura"].map(_data_assinatura)
    df["ano_assinatura"] = df["assinatura"].dt.year
    for col in ("populacao_contrato", "pontos_luz_contrato",
                "valor_contrato_milhoes", "vigencia_anos"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Despesa implícita do contrato, em R$ por ponto por mês — é o benchmark que o
    # Hub usa como referência de custo de PPP, derivado de contratos reais e não de
    # premissa arbitrária.
    denominador = df["pontos_luz_contrato"] * df["vigencia_anos"] * 12
    df["despesa_ponto_mes"] = (df["valor_contrato_milhoes"] * 1e6 /
                               denominador.replace(0, pd.NA))

    # ── casamento com o código IBGE ──────────────────────────────────────────
    entes = siconfi.carregar_entes()
    if entes.empty:
        raise SystemExit("Cadastro de entes do SICONFI indisponível — sem ele não há "
                         "como atribuir o código IBGE aos contratos.")
    chave = (entes["_ente_norm"] + "|" + entes["uf"].astype(str).str.upper())
    mapa = dict(zip(chave, entes["cod_ibge"]))

    df["_norm"] = df["municipio"].map(siconfi.normalizar)
    df["_chave"] = df["_norm"] + "|" + df["uf"].fillna("")
    df["codigo_municipio"] = df["_chave"].map(mapa)

    # aplica as correções de grafia só onde o casamento por nome falhou
    faltantes = df["codigo_municipio"].isna()
    if faltantes.any():
        df.loc[faltantes, "codigo_municipio"] = [
            CORRECOES_DE_GRAFIA.get((n, u))
            for n, u in zip(df.loc[faltantes, "_norm"], df.loc[faltantes, "uf"])
        ]

    colunas = ["codigo_municipio", "municipio", "uf", "concessionaria", "acionistas",
               "pontos_luz_contrato", "populacao_contrato", "valor_contrato_milhoes",
               "vigencia_anos", "despesa_ponto_mes", "telegestao",
               "assinatura", "ano_assinatura"]
    return df[colunas].sort_values(["uf", "municipio"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arquivo", default=str(CAMINHO_PADRAO),
                    help="caminho da CLP.xlsx (padrão: Drive compartilhado do time)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    arquivo = Path(args.arquivo)
    if not arquivo.exists():
        print(f"Arquivo não encontrado: {arquivo}")
        return 1

    df = importar(arquivo)
    sem_codigo = df[df["codigo_municipio"].isna()]

    print(f"{len(df)} contratos de PPP importados de {arquivo.name}")
    print(f"  casados com código IBGE: {len(df) - len(sem_codigo)}")
    if not sem_codigo.empty:
        print(f"  SEM código IBGE ({len(sem_codigo)}) — conferir grafia na planilha:")
        for r in sem_codigo.itertuples():
            print(f"    · {r.municipio} / {r.uf}")

    d = df["despesa_ponto_mes"].dropna()
    if not d.empty:
        print(f"\n  despesa contratada (R$/ponto.mês): mediana {d.median():,.2f} · "
              f"quartis {d.quantile(.25):,.2f}–{d.quantile(.75):,.2f}")
    print(f"  pontos de luz sob contrato: {df['pontos_luz_contrato'].sum():,.0f}")
    print(f"  UFs cobertas: {df['uf'].nunique()}")

    config.garantir_pastas()
    df.to_parquet(DESTINO, index=False)
    print(f"\n  gravado: {DESTINO}  ({DESTINO.stat().st_size / 1e3:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
