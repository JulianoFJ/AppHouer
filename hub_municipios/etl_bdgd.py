"""
ETL offline da BDGD — roda fora do Streamlit, uma vez por base recebida.

    py -m hub_municipios.etl_bdgd --listar           # o que há para processar
    py -m hub_municipios.etl_bdgd                    # processa o que falta
    py -m hub_municipios.etl_bdgd --base Cemig       # filtra por nome
    py -m hub_municipios.etl_bdgd --paralelo 4       # 4 extrações simultâneas
    py -m hub_municipios.etl_bdgd --consolidar       # só reconsolida o que já foi extraído
    py -m hub_municipios.etl_bdgd --reextrair        # ignora o parquet intermediário

Entrada : `dados/bdgd/brutos/*.gdb` + repositórios extras de `config.BDGD_PASTAS_EXTRA`
          (por padrão, o acervo nacional no Drive compartilhado do time).
Saída   : `hub_municipios/data/bdgd_municipios.parquet` (centenas de KB)
          `hub_municipios/data/bdgd_tecnologia.parquet`

**Retomável.** O agregado de cada base é gravado assim que fica pronto, em
`dados/bdgd/processados/agregado_<slug>.parquet`, e a consolidação lê esses arquivos.
Interromper no meio não custa o que já foi feito; rodar de novo continua de onde parou.

**Custo.** Base local: ~55 s para 15,9 GB. Base no Drive compartilhado: ~7 min para
5,65 GB — o gargalo é o download sob demanda do Drive File Stream, não a CPU. Por isso as
bases são processadas da menor para a maior (valor entregue mais cedo) e o paralelismo
ajuda: são conexões de rede simultâneas, não trabalho de processador.

Requer GDAL (`ogr2ogr` + `ogrinfo`) — OSGeo4W ou QGIS.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd

from . import bdgd, config


def _tabela(df: pd.DataFrame, colunas: List[str], n: int = 10) -> str:
    presentes = [c for c in colunas if c in df.columns]
    return df[presentes].head(n).to_string(index=False, float_format=lambda v: f"{v:,.1f}")


def _ja_processada(base: bdgd.BaseBDGD) -> bool:
    return (config.BDGD_PROCESSADOS / f"agregado_{base.slug}.parquet").exists()


def _processar_uma(base: bdgd.BaseBDGD, reextrair: bool) -> bdgd.ResultadoETL:
    caminho = bdgd.extrair_pip(base, sobrescrever=reextrair)
    res = bdgd.agregar(caminho, base)
    bdgd.salvar_agregado_base(res)
    return res


def _resumo_final() -> int:
    dados = bdgd.consolidar_de_disco()
    mun = dados["municipios"]
    if mun.empty:
        print("\nNenhum agregado em disco para consolidar.")
        return 3

    caminhos = bdgd.gravar_derivado(dados)
    print(f"\n{'=' * 78}\nCONSOLIDADO NACIONAL\n{'=' * 78}")
    print(f"  {len(mun):,} municípios · {int(mun['pontos_ip'].sum()):,} pontos de IP · "
          f"{mun['consumo_kwh_ano'].sum() / 1e6:,.1f} GWh/ano")
    print(f"  potência média por ponto: {mun['potencia_media_w'].median():,.0f} W (mediana municipal)")
    if "perc_led" in mun.columns and mun["perc_led"].notna().any():
        pond = ((mun["perc_led"] * mun["pontos_ip"]).sum()
                / mun.loc[mun["perc_led"].notna(), "pontos_ip"].sum())
        print(f"  parque em LED: {pond * 100:,.1f}% dos pontos (ponderado) · "
              f"{mun['perc_led'].median() * 100:,.1f}% (mediana municipal)")
    print(f"  horas equivalentes: {mun['horas_equivalentes_ano'].median():,.0f} h/ano (mediana)")

    print("\n  Arquivos gravados:")
    for nome, caminho in caminhos.items():
        print(f"    {nome:<12} {caminho.name}  ({caminho.stat().st_size / 1e3:,.0f} KB)")

    por_uf = (mun.assign(uf_cod=mun["codigo_municipio"].str[:2])
                 .groupby("uf_cod")
                 .agg(municipios=("codigo_municipio", "nunique"),
                      pontos=("pontos_ip", "sum"))
                 .sort_values("pontos", ascending=False))
    print(f"\n  Cobertura por UF (código IBGE): {len(por_uf)} estados")
    print("  " + por_uf.head(12).to_string().replace("\n", "\n  "))
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Processa BDGDs (.gdb) em um agregado municipal de iluminação pública.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--listar", action="store_true", help="só lista as bases encontradas")
    ap.add_argument("--base", help="filtra pelo nome do arquivo (substring)")
    ap.add_argument("--reextrair", action="store_true",
                    help="refaz a extração mesmo se o parquet intermediário existir")
    ap.add_argument("--refazer", action="store_true",
                    help="reprocessa bases que já têm agregado gravado")
    ap.add_argument("--consolidar", action="store_true",
                    help="apenas reconsolida os agregados já em disco")
    ap.add_argument("--pasta", action="append",
                    help="pasta de .gdb (repetível; padrão: local + repositórios extras)")
    ap.add_argument("--paralelo", type=int, default=1, metavar="N",
                    help="extrações simultâneas (útil em base de rede; padrão 1)")
    ap.add_argument("--limite", type=int, metavar="N",
                    help="processa no máximo N bases nesta execução")
    args = ap.parse_args(argv)

    # O console do Windows abre em cp1252 e quebra em acento/seta — sem isso o ETL morre
    # com UnicodeEncodeError no meio do processamento. `line_buffering` é o que permite
    # acompanhar o progresso quando a saída é redirecionada para arquivo: sem ele o Python
    # bufferiza em bloco e o log fica vazio durante horas de processamento.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

    config.garantir_pastas()

    if args.consolidar:
        return _resumo_final()

    pastas = [Path(p) for p in args.pasta] if args.pasta else config.pastas_bdgd()
    bases = bdgd.descobrir_bases(pastas)

    if args.base:
        alvo = args.base.lower()
        bases = [b for b in bases if alvo in b.caminho.name.lower()]

    if not bases:
        print("Nenhum .gdb encontrado em:")
        for p in pastas:
            print(f"  {p}  {'(inexistente)' if not p.exists() else ''}")
        print("\nColoque as bases da ANEEL numa dessas pastas, mantendo o nome original:")
        print("  Cemig-D_4950_2024-12-31_V11_20250929-1522.gdb")
        return 1

    # Menor primeiro: entrega municípios cedo e deixa as pesadas para o fim.
    tamanhos = {b.slug: bdgd.tamanho_base(b) for b in bases}
    bases.sort(key=lambda b: tamanhos[b.slug])

    pendentes = [b for b in bases if args.refazer or not _ja_processada(b)]
    prontas = len(bases) - len(pendentes)
    if args.limite:
        pendentes = pendentes[:args.limite]

    total_gb = sum(tamanhos[b.slug] for b in bases) / 1e9
    pend_gb = sum(tamanhos[b.slug] for b in pendentes) / 1e9
    print(f"{len(bases)} base(s) em {len(pastas)} pasta(s) · {total_gb:,.1f} GB no acervo")
    print(f"  {prontas} já processada(s) · {len(pendentes)} pendente(s) ({pend_gb:,.1f} GB a ler)\n")

    for b in bases:
        marca = "ok " if _ja_processada(b) and not args.refazer else "-> "
        print(f"  {marca}{tamanhos[b.slug] / 1e9:7.2f} GB  {b.rotulo}")

    if args.listar:
        return 0

    if not pendentes:
        print("\nNada pendente. Reconsolidando o que já existe.")
        return _resumo_final()

    try:
        print(f"\nGDAL: {bdgd.localizar_ogr2ogr()}")
    except bdgd.OgrIndisponivel as exc:
        print(f"\nERRO: {exc}")
        return 2

    # O gargalo é rede (Drive File Stream), então threads bastam: cada uma fica
    # bloqueada num subprocess ogr2ogr, sem disputar GIL.
    workers = max(1, args.paralelo)
    print(f"Processando {len(pendentes)} base(s) com {workers} extração(ões) simultânea(s)…\n")

    t0 = time.time()
    falhas: List[str] = []
    concluidas = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = {pool.submit(_processar_uma, b, args.reextrair): b for b in pendentes}
        for fut in as_completed(futuros):
            base = futuros[fut]
            concluidas += 1
            prefixo = f"[{concluidas}/{len(pendentes)}]"
            try:
                res = fut.result()
            except Exception as exc:
                falhas.append(f"{base.rotulo}: {type(exc).__name__}: {exc}")
                print(f"{prefixo} FALHA  {base.rotulo}\n         {exc}")
                continue

            aviso = f"  ⚠ {len(res.avisos)} aviso(s)" if res.avisos else ""
            print(f"{prefixo} {base.rotulo}: {res.pontos_lidos:,} pontos "
                  f"({res.pontos_desativados:,} desativados) → "
                  f"{len(res.municipios):,} municípios{aviso}")
            for a in res.avisos:
                print(f"         ⚠ {a}")

    print(f"\nExtração concluída em {(time.time() - t0) / 60:,.1f} min.")
    if falhas:
        print(f"\n{len(falhas)} base(s) com falha:")
        for f in falhas:
            print(f"  · {f}")
        print("Rode o comando de novo para tentar apenas as que faltam.")

    return _resumo_final()


if __name__ == "__main__":
    sys.exit(main())
