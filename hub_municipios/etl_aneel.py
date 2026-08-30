"""
ETL offline das tarifas B4a da ANEEL → `hub_municipios/data/tarifas_b4a.parquet`.

Roda FORA do Streamlit, como o `etl_bdgd`, e pelo mesmo motivo: o portal da ANEEL cai
com frequência (handshake TLS recusado) e tarifa muda uma vez por ano. O app lê só o
parquet agregado, de poucos KB, versionado no repositório.

USO
---
    py -m hub_municipios.etl_aneel --listar
        Imprime o schema REAL de cada recurso, sem gravar nada. Rode isto primeiro:
        os nomes de campo em `config` vêm do dicionário de metadados publicado, não de
        uma resposta lida, e `--listar` é o que confirma.

    py -m hub_municipios.etl_aneel
        Baixa tarifas homologadas + SAMP do ano mais recente, cruza e grava o parquet.

    py -m hub_municipios.etl_aneel --ano 2024
        Fixa o ano do SAMP.

    py -m hub_municipios.etl_aneel --csv tarifas.csv --csv-samp samp.csv
        Caminho de contingência: quando o TLS da ANEEL recusar a conexão, baixe os CSV
        pelo navegador e aponte para eles. O tratamento é idêntico.

RETOMÁVEL
---------
Cada recurso baixado é gravado cru em `dados/aneel/cache/` antes do processamento, e
reaproveitado nas execuções seguintes (`--refazer` ignora o cache). O download é o
gargalo: o recurso de tarifas tem centenas de milhares de linhas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from . import aneel_tarifas, bdgd, config


def _cache(nome: str) -> Path:
    config.garantir_pastas()
    return config.ANEEL_CACHE / nome


def _ler_cache(nome: str, refazer: bool) -> Optional[pd.DataFrame]:
    caminho = _cache(nome)
    if refazer or not caminho.exists():
        return None
    try:
        df = pd.read_parquet(caminho)
        print(f"  cache: {nome} ({len(df):,} linhas)")
        return df
    except Exception:
        return None


def _gravar_cache(df: pd.DataFrame, nome: str) -> None:
    if not df.empty:
        df.to_parquet(_cache(nome), index=False)


def _obter(resource_id: str, nome: str, refazer: bool, csv: Optional[str],
           filtros: Optional[dict] = None) -> pd.DataFrame:
    if csv:
        print(f"  lendo CSV local: {csv}")
        return pd.read_csv(csv, sep=None, engine="python", dtype=str,
                           encoding="utf-8-sig", on_bad_lines="skip")
    cacheado = _ler_cache(nome, refazer)
    if cacheado is not None:
        return cacheado
    print(f"  baixando recurso {resource_id} ...")
    df = aneel_tarifas.baixar_recurso(resource_id, filtros=filtros)
    print(f"  {len(df):,} linhas")
    _gravar_cache(df, nome)
    return df


def _diag(d: aneel_tarifas.Diagnostico) -> None:
    print(f"  [{d.fonte}] {d.linhas_lidas:,} lidas -> {d.linhas_uteis:,} distribuidoras")
    if d.colunas_faltando:
        print(f"    colunas não encontradas: {', '.join(d.colunas_faltando)}")
    for aviso in d.avisos:
        print(f"    ! {aviso}")


def listar(args: argparse.Namespace) -> int:
    """Imprime o schema real de cada recurso. Não grava nada."""
    print("== Recurso de tarifas homologadas ==")
    try:
        amostra = aneel_tarifas.baixar_recurso(config.ANEEL_RECURSO_TARIFAS, limite=200)
        print(f"  colunas: {list(amostra.columns)}")
        for c in ("DscSubGrupo", "DscClasse", "DscBaseTarifaria"):
            achada = aneel_tarifas._coluna(amostra, c)
            if achada:
                print(f"  {achada}: {sorted(amostra[achada].astype(str).unique())[:12]}")
        print(amostra.head(3).to_string())
    except aneel_tarifas.ErroANEEL as exc:
        print(f"  FALHOU: {exc}")

    print("\n== Recursos do SAMP ==")
    try:
        recursos = aneel_tarifas.listar_recursos_samp()
        print(recursos.sort_values("ano", ascending=False).head(12).to_string(index=False))
        alvo = recursos.dropna(subset=["ano"]).sort_values("ano").iloc[-1]
        amostra = aneel_tarifas.baixar_recurso(alvo["resource_id"], limite=200)
        print(f"\n  colunas de {alvo['nome']}: {list(amostra.columns)}")
        classe = aneel_tarifas._coluna(amostra, "DscClasseConsumo", "DscClasse", "Classe")
        if classe:
            print(f"  {classe}: {sorted(amostra[classe].astype(str).unique())[:12]}")
        print(amostra.head(3).to_string())
    except aneel_tarifas.ErroANEEL as exc:
        print(f"  FALHOU: {exc}")
    return 0


def executar(args: argparse.Namespace) -> int:
    config.garantir_pastas()

    print("== Tarifas homologadas (sem tributos) ==")
    try:
        bruto = _obter(config.ANEEL_RECURSO_TARIFAS, "tarifas_bruto.parquet",
                       args.refazer, args.csv)
    except aneel_tarifas.ErroANEEL as exc:
        print(f"  FALHOU: {exc}", file=sys.stderr)
        return 1
    homologadas, d1 = aneel_tarifas.normalizar_tarifas(bruto)
    _diag(d1)

    print("\n== SAMP (tarifa faturada, com tributos) ==")
    samp = pd.DataFrame()
    try:
        if args.csv_samp:
            bruto_samp = _obter("", "samp_bruto.parquet", args.refazer, args.csv_samp)
            ano = args.ano
        else:
            recursos = aneel_tarifas.listar_recursos_samp()
            # O datastore só indexa os recursos CSV; os .parquet do mesmo ano não têm
            # tabela consultável por filtro.
            recursos = recursos[recursos["formato"].astype(str).str.upper() == "CSV"]
            recursos = recursos.dropna(subset=["ano"])
            if args.ano:
                recursos = recursos[recursos["ano"] == args.ano]
            if recursos.empty:
                raise aneel_tarifas.ErroANEEL("Nenhum recurso do SAMP para o ano pedido.")
            alvo = recursos.sort_values("ano").iloc[-1]
            ano = int(alvo["ano"])
            print(f"  ano {ano} ({alvo['nome']})")
            bruto_samp = _obter(
                alvo["resource_id"], f"samp_{ano}_bruto.parquet", args.refazer, None,
                filtros={"DscClasseConsumoMercado": config.ANEEL_CLASSE_IP})
        samp, d2 = aneel_tarifas.normalizar_samp(bruto_samp, ano)
        _diag(d2)
    except aneel_tarifas.ErroANEEL as exc:
        print(f"  FALHOU: {exc}", file=sys.stderr)
        print("  Seguindo só com a tarifa homologada — a página vai avisar que a "
              "tarifa faturada não está disponível.", file=sys.stderr)

    if homologadas.empty and samp.empty:
        print("\nNada a gravar.", file=sys.stderr)
        return 1

    if homologadas.empty:
        final = samp
    elif samp.empty:
        final = homologadas
    else:
        final = homologadas.merge(samp, on="sigla_distribuidora", how="outer")

    final = _anexar_codigos(final)

    for col in aneel_tarifas.COLUNAS_TARIFAS:
        if col not in final.columns:
            final[col] = pd.NA
    final = final[aneel_tarifas.COLUNAS_TARIFAS]
    final.to_parquet(config.TARIFAS_B4A, index=False)

    print(f"\n== Gravado: {config.TARIFAS_B4A} ==")
    print(f"  {len(final):,} distribuidoras")
    _resumo(final)
    return 0


def _anexar_codigos(final: pd.DataFrame) -> pd.DataFrame:
    """
    Casa a sigla da ANEEL com o `cod_distribuidora`/`distribuidora` que a BDGD usa.

    Sem isso o join município → tarifa depende de bater NOME de distribuidora, que
    diverge entre as duas bases ("CEMIG-D" x "Cemig-D", "NEOENERGIA COELBA" x
    "Neoenergia_Coelba"). O código ANEEL é o mesmo dos dois lados e é o que amarra.
    """
    try:
        parque = bdgd.carregar_municipios()
    except Exception:
        parque = None
    if parque is None or parque.empty or "cod_distribuidora" not in parque.columns:
        print("  ! BDGD indisponível: parquet sai sem cod_distribuidora e o casamento "
              "no portal vai depender do nome.")
        return final

    # O sentido do casamento e da BDGD PARA a ANEEL, nao o contrario: sao as 38
    # distribuidoras da BDGD que precisam de tarifa. As outras ~77 siglas do recurso sao
    # cooperativas e permissionarias sem municipio na base, e ficar tentando casa-las
    # so gera ruido no relatorio.
    nomes = sorted({n.strip() for d in parque["distribuidora"].dropna().unique()
                    for n in str(d).split("+") if n.strip()})
    siglas = final["sigla_distribuidora"].dropna().tolist()

    codigos = (parque.dropna(subset=["cod_distribuidora"])
                     .drop_duplicates("distribuidora")
                     .set_index("distribuidora")["cod_distribuidora"].to_dict())

    de_para, sem_par = [], []
    for nome in nomes:
        sigla = aneel_tarifas.casar_sigla(nome, siglas)
        if sigla is None:
            sem_par.append(nome)
            continue
        de_para.append({"sigla_distribuidora": sigla, "distribuidora": nome,
                        "cod_distribuidora": codigos.get(nome)})

    print(f"  casamento com a BDGD: {len(de_para)} de {len(nomes)} distribuidoras")
    if sem_par:
        print(f"    SEM TARIFA: {', '.join(sem_par)}")
        print("    Complete em ALIAS_BDGD_ANEEL (aneel_tarifas.py) se a sigla existir.")

    if not de_para:
        return final
    ponte = pd.DataFrame(de_para).drop_duplicates("sigla_distribuidora")
    return final.merge(ponte, on="sigla_distribuidora", how="left")


def _resumo(final: pd.DataFrame) -> None:
    com = pd.to_numeric(final.get("tarifa_com_tributos"), errors="coerce").dropna()
    sem = pd.to_numeric(final.get("tarifa_sem_tributos"), errors="coerce").dropna()
    if len(sem):
        print(f"  sem tributos : mediana R$ {sem.median():.4f}/kWh "
              f"(min {sem.min():.4f}, max {sem.max():.4f})")
    if len(com):
        print(f"  com tributos : mediana R$ {com.median():.4f}/kWh "
              f"(min {com.min():.4f}, max {com.max():.4f})")
    if len(com) and len(sem):
        par = final.dropna(subset=["tarifa_com_tributos", "tarifa_sem_tributos"])
        if len(par):
            razao = (par["tarifa_com_tributos"] / par["tarifa_sem_tributos"]).median()
            print(f"  gross-up implícito (com ÷ sem): {razao:.3f}x  "
                  "— esperado ~1,25 a 1,45 (PIS/COFINS + ICMS + bandeiras)")
            if not 1.05 <= razao <= 1.90:
                print("  ! Fora do esperado: confira se as duas colunas estão na mesma "
                      "unidade antes de publicar o parquet.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="etl_aneel",
        description="Baixa e agrega as tarifas B4a de iluminação pública da ANEEL.")
    p.add_argument("--listar", action="store_true",
                   help="imprime o schema real dos recursos e sai, sem gravar")
    p.add_argument("--ano", type=int, default=None,
                   help="ano do recurso do SAMP (padrão: o mais recente)")
    p.add_argument("--refazer", action="store_true",
                   help="ignora o cache de brutos e baixa de novo")
    p.add_argument("--csv", default=None,
                   help="CSV local de tarifas homologadas, em vez da API")
    p.add_argument("--csv-samp", default=None, dest="csv_samp",
                   help="CSV local do SAMP, em vez da API")
    args = p.parse_args(argv)
    return listar(args) if args.listar else executar(args)


if __name__ == "__main__":
    raise SystemExit(main())
