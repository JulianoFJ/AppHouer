"""
ETL offline: cadastro ponto a ponto de IP a partir da BDGD.

    py -m cadastro_bdgd.extracao 3162955             # um município pelo código IBGE
    py -m cadastro_bdgd.extracao --base Cemig        # a distribuidora INTEIRA, de uma vez
    py -m cadastro_bdgd.extracao --bases             # que .gdb estão visíveis
    py -m cadastro_bdgd.extracao --listar            # o que já foi extraído
    py -m cadastro_bdgd.extracao --publicar-existentes   # manda os locais para o repositório

Entrada : o `.gdb` da distribuidora (pasta local ou acervo do Drive — os mesmos
          caminhos que `hub_municipios.config.pastas_bdgd()` varre).
Saída   : um parquet por município. Grava em `dados/bdgd/cadastros/` (fora do
          repositório); com `--publicar`, em `cadastro_bdgd/data/cadastros/`, que é a
          pasta versionada e a única que existe no Streamlit Cloud.

**Roda fora do Streamlit**, como o ETL do Hub e pelo mesmo motivo: ler File Geodatabase
exige GDAL, e o `ogr2ogr` é chamado por subprocess porque o Smart App Control do
Windows bloqueia a DLL do `pyogrio`. Nenhuma dependência geoespacial entra no
`requirements.txt` do portal.

De onde vem a coordenada
------------------------
A PIP **não tem geometria** (`Geometry: None` no ogrinfo) e nenhum campo de endereço.
O que ela tem é `PN_CON`, o ponto notável de conexão, que é chave para a entidade
PONNOT — essa sim uma camada de pontos em SIRGAS 2000. O join `PIP.PN_CON` ->
`PONNOT.COD_ID` casou **100% dos 10.590 pontos** de Ponta Porã/MS em 04/09/2026, sem
duplicidade de COD_ID.

Cuidado com `PAC`: parece a chave natural e casou **0%** — é o ponto de atendimento da
UC, com codificação própria, não um ponto notável.

Uma passada por distribuidora, não uma por município
----------------------------------------------------
`--base` existe porque o modo município a município engana. O `WHERE MUN=` do ogr2ogr
varre a tabela inteira de qualquer forma, e quando a base está no Drive compartilhado
cada varredura rebaixa os 15,9 GB sob demanda: um único município da Cemig-D passou de
uma hora em 04/09/2026. Numa passada só, o download é pago uma vez e os 766 municípios
saem juntos. Base local (Energisa MS, 4,4 GB) leva ~45 s por município e o modo direto
serve bem.

Tamanho do artefato
-------------------
O parquet publicado é enxuto (`COLUNAS_PUBLICADAS`, dtypes reduzidos, zstd): ~13 bytes
por ponto, contra ~35 do cadastro completo. São 142 KB para Ponta Porã e ~47 KB no
município mediano — daí MG inteiro dar ~33 MB e caber no repositório.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from hub_municipios import bdgd, config

from . import caminhos

# Colunas da PIP que interessam ao cadastro de campo. `PN_CON` é a chave da coordenada
# e não existia na extração do Hub — é o que este ETL acrescenta.
COLUNAS_PIP = [
    "COD_ID", "PN_CON", "MUN", "CAR_INST", "TIPO_LAMP", "POT_LAMP",
    "PERDA_REAT", "PERDA_RELE", "SIT_ATIV", "ARE_LOC", "CONTROLE",
    "TIP_SIST", "DAT_CON", "GRU_TAR", "CLAS_SUB",
]
COLUNAS_PIP_OBRIGATORIAS = ["COD_ID", "PN_CON", "MUN"]

COLUNAS_PONNOT = ["COD_ID", "MUN", "TIP_PN", "TIP_INST", "ARE_LOC"]
COLUNAS_PONNOT_OBRIGATORIAS = ["COD_ID", "MUN"]


class MunicipioSemBase(RuntimeError):
    """Não há .gdb da distribuidora que atende o município."""


# ── Descoberta ───────────────────────────────────────────────────────────────

def distribuidora_do_municipio(codigo_ibge: str) -> str:
    """
    Qual distribuidora atende o município, segundo o agregado do Hub.

    O agregado já responde isso para 5.417 municípios e é lido do parquet versionado,
    então não custa rede nem varredura de .gdb.
    """
    municipios = pd.read_parquet(config.BDGD_MUNICIPIOS)
    linha = municipios[municipios["codigo_municipio"] == str(codigo_ibge)]
    if linha.empty:
        raise MunicipioSemBase(
            f"O município {codigo_ibge} não aparece no agregado da BDGD. "
            "Confira o código IBGE ou rode o ETL do Hub para a distribuidora dele."
        )
    return str(linha.iloc[0]["distribuidora"])


def base_do_municipio(codigo_ibge: str) -> bdgd.BaseBDGD:
    """Encontra o .gdb da distribuidora do município, entre as pastas conhecidas."""
    alvo = distribuidora_do_municipio(codigo_ibge)
    bases = bdgd.descobrir_bases(config.pastas_bdgd())
    if not bases:
        raise MunicipioSemBase(
            "Nenhum .gdb encontrado. Verifique dados/bdgd/brutos/ e o acervo em "
            "config.BDGD_PASTAS_EXTRA."
        )

    def normaliza(texto: str) -> str:
        return "".join(c for c in texto.lower() if c.isalnum())

    alvo_n = normaliza(alvo)
    candidatas = [b for b in bases if normaliza(b.distribuidora) == alvo_n]
    if not candidatas:
        candidatas = [b for b in bases
                      if alvo_n in normaliza(b.caminho.name)
                      or normaliza(b.distribuidora) in alvo_n]
    if not candidatas:
        disponiveis = ", ".join(sorted({b.distribuidora for b in bases}))
        raise MunicipioSemBase(
            f"O município {codigo_ibge} é atendido por '{alvo}', e não há .gdb dessa "
            f"distribuidora nas pastas varridas.\nDisponíveis: {disponiveis}"
        )
    # Mais de uma versão da mesma distribuidora: vence a data-base mais recente, a
    # mesma regra que o ETL do Hub aplica ao consolidar.
    return sorted(candidatas, key=lambda b: b.data_base)[-1]


# ── Extração ─────────────────────────────────────────────────────────────────

def _colunas_existentes(gdb: Path, camada: str, desejadas: Sequence[str],
                        obrigatorias: Sequence[str]) -> list[str]:
    """
    Intersecta as colunas desejadas com o schema real da camada.

    Pedir coluna inexistente faz o `ogr2ogr` abortar com "Unrecognized field name" e
    perder a base inteira. O schema da BDGD muda entre versões — a M10/2017 não tem os
    campos de lâmpada —, então o SELECT é montado depois de olhar o schema, que é a
    mesma precaução que `hub_municipios.bdgd` toma.
    """
    ogrinfo = str(Path(bdgd.localizar_ogr2ogr()).with_name("ogrinfo.exe"))
    saida = subprocess.run([ogrinfo, "-so", str(gdb), camada],
                           capture_output=True, text=True, errors="replace")
    schema = saida.stdout
    presentes = [c for c in desejadas if f"\n{c}:" in schema]
    faltando = [c for c in obrigatorias if c not in presentes]
    if faltando:
        raise RuntimeError(
            f"A camada {camada} de {gdb.name} não tem {', '.join(faltando)} — "
            "esta versão da BDGD não permite montar o cadastro georreferenciado."
        )
    return presentes


def _extrair_csv(gdb: Path, sql: str, destino: Path, com_xy: bool) -> pd.DataFrame:
    ogr2ogr = bdgd.localizar_ogr2ogr()
    comando = [ogr2ogr, "-f", "CSV", str(destino), str(gdb), "-sql", sql]
    if com_xy:
        comando += ["-lco", "GEOMETRY=AS_XY"]
    processo = subprocess.run(comando, capture_output=True, text=True, errors="replace")
    if processo.returncode != 0 or not destino.exists():
        raise RuntimeError(f"ogr2ogr falhou: {processo.stderr.strip()[:500]}")
    return pd.read_csv(destino, dtype=str, low_memory=False)


def extrair(codigo_ibge: str, base: Optional[bdgd.BaseBDGD] = None,
            verboso: bool = True) -> pd.DataFrame:
    """Extrai PIP + PONNOT do município e devolve o cadastro cru, já com coordenada."""
    codigo_ibge = str(codigo_ibge).strip()
    base = base or base_do_municipio(codigo_ibge)
    gdb = base.caminho

    def log(msg: str) -> None:
        if verboso:
            print(f"  {msg}", flush=True)

    log(f"base: {base.rotulo}")
    cols_pip = _colunas_existentes(gdb, "PIP", COLUNAS_PIP, COLUNAS_PIP_OBRIGATORIAS)
    cols_pn = _colunas_existentes(gdb, "PONNOT", COLUNAS_PONNOT,
                                  COLUNAS_PONNOT_OBRIGATORIAS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.time()
        log("extraindo PIP…")
        pip = _extrair_csv(
            gdb, f"SELECT {', '.join(cols_pip)} FROM PIP WHERE MUN='{codigo_ibge}'",
            tmp / "pip.csv", com_xy=False)
        log(f"PIP: {len(pip):,} pontos ({time.time() - t0:,.0f} s)".replace(",", "."))

        if pip.empty:
            raise MunicipioSemBase(
                f"A PIP de {base.distribuidora} não tem nenhum ponto com MUN='{codigo_ibge}'."
            )

        t1 = time.time()
        log("extraindo PONNOT…")
        ponnot = _extrair_csv(
            gdb, f"SELECT {', '.join(cols_pn)} FROM PONNOT WHERE MUN='{codigo_ibge}'",
            tmp / "ponnot.csv", com_xy=True)
        log(f"PONNOT: {len(ponnot):,} pontos ({time.time() - t1:,.0f} s)".replace(",", "."))

    return _montar(pip, ponnot, base, codigo_ibge)


def _montar(pip: pd.DataFrame, ponnot: pd.DataFrame, base: bdgd.BaseBDGD,
            codigo_ibge: str) -> pd.DataFrame:
    """Faz o join da coordenada, infere a tecnologia e normaliza os nomes de coluna."""
    coordenadas = (ponnot[["COD_ID", "X", "Y"]]
                   .rename(columns={"COD_ID": "PN_CON", "X": "longitude", "Y": "latitude"}))
    # O PONNOT pode trazer o mesmo COD_ID em versões inconsistentes da base; ficar com
    # a primeira ocorrência evita multiplicar linhas do cadastro num join 1:N silencioso.
    coordenadas = coordenadas.drop_duplicates(subset="PN_CON")

    cadastro = pip.merge(coordenadas, on="PN_CON", how="left")
    for eixo in ("latitude", "longitude"):
        cadastro[eixo] = pd.to_numeric(cadastro[eixo], errors="coerce")

    for numerica in ("CAR_INST", "POT_LAMP", "PERDA_REAT", "PERDA_RELE"):
        if numerica in cadastro.columns:
            cadastro[numerica] = pd.to_numeric(cadastro[numerica], errors="coerce")

    # Tecnologia pela assinatura física do código, reusando a inferência já validada do
    # Hub — o domínio de TIPO_LAMP varia por distribuidora e não vem na BDGD.
    if {"TIPO_LAMP", "PERDA_REAT", "POT_LAMP"} <= set(cadastro.columns):
        mapa = bdgd.inferir_tecnologias(cadastro)
        cadastro["TIPO_LAMP"] = cadastro["TIPO_LAMP"].astype(str)
        cadastro = cadastro.merge(mapa[["tipo_lamp", "tecnologia"]],
                                  left_on="TIPO_LAMP", right_on="tipo_lamp", how="left")
        cadastro["tecnologia"] = cadastro["tecnologia"].fillna("Não informado")
        cadastro = cadastro.drop(columns=["tipo_lamp"])
    else:
        cadastro["tecnologia"] = "Não informado"

    renomear = {
        "COD_ID": "id_ponto", "CAR_INST": "carga_instalada_kw",
        "POT_LAMP": "potencia_lampada_w", "PERDA_REAT": "perda_reator_w",
        "PERDA_RELE": "perda_rele_w", "TIP_SIST": "tipo_sistema",
        "DAT_CON": "data_conexao", "SIT_ATIV": "situacao",
    }
    cadastro = cadastro.rename(columns={k: v for k, v in renomear.items()
                                        if k in cadastro.columns})

    # ARE_LOC: UB = urbano, NU = não urbano. Vira booleano porque é o que a Tabela 1 da
    # NBR 5101 consome como luminância ambiente.
    if "ARE_LOC" in cadastro.columns:
        cadastro["area_urbana"] = cadastro["ARE_LOC"].astype(str).str.upper().eq("UB")
    else:
        cadastro["area_urbana"] = pd.NA

    if "CONTROLE" in cadastro.columns:
        cadastro["telegestao"] = pd.to_numeric(cadastro["CONTROLE"],
                                               errors="coerce").fillna(0).gt(0)

    cadastro["municipio_ibge"] = codigo_ibge
    cadastro["distribuidora"] = base.distribuidora
    cadastro["data_base_bdgd"] = base.data_base
    cadastro["versao_bdgd"] = base.versao
    return cadastro


# Colunas que vão para o parquet publicado. As demais (PN_CON, TIPO_LAMP, GRU_TAR,
# CLAS_SUB, perdas) servem à montagem e à inferência de tecnologia, não à inspeção de
# campo, e sair com elas custaria o triplo do espaço num artefato versionado.
COLUNAS_PUBLICADAS = [
    "id_ponto", "latitude", "longitude", "tecnologia", "potencia_lampada_w",
    "carga_instalada_kw", "area_urbana", "telegestao", "tipo_sistema",
    "situacao", "data_conexao", "municipio_ibge", "distribuidora",
    "data_base_bdgd", "versao_bdgd",
]

# float32 dá ~1 cm de resolução em grau decimal — muito além da incerteza da própria
# BDGD. `category` porque tecnologia e tipo de sistema têm meia dúzia de valores
# repetidos milhares de vezes.
DTYPES_ENXUTOS = {
    "latitude": "float32", "longitude": "float32",
    "potencia_lampada_w": "float32", "carga_instalada_kw": "float32",
    "tecnologia": "category", "tipo_sistema": "category",
    "situacao": "category", "distribuidora": "category",
}


def enxugar(cadastro: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz o cadastro ao que vai para disco: 24 colunas viram 15, e ~35 bytes por ponto
    viram ~13. É o que faz MG inteiro caber no repositório em vez de estourá-lo.
    """
    presentes = [c for c in COLUNAS_PUBLICADAS if c in cadastro.columns]
    enxuto = cadastro[presentes].copy()
    for coluna, tipo in DTYPES_ENXUTOS.items():
        if coluna in enxuto.columns:
            try:
                enxuto[coluna] = enxuto[coluna].astype(tipo)
            except (TypeError, ValueError):
                pass                      # coluna com lixo: fica como está, não trava
    return enxuto


def salvar(cadastro: pd.DataFrame, codigo_ibge: str, publicar: bool = False) -> Path:
    caminhos.garantir_pastas()
    destino = caminhos.caminho_para_gravar(str(codigo_ibge), publicar=publicar)
    enxugar(cadastro).to_parquet(destino, index=False, compression="zstd")
    return destino


def carregar(codigo_ibge: str) -> Optional[pd.DataFrame]:
    """Lê o cadastro já extraído, ou None se não houver. É o que o portal usa."""
    destino = caminhos.caminho_cadastro(str(codigo_ibge))
    return pd.read_parquet(destino) if destino else None


def extrair_base_inteira(base: bdgd.BaseBDGD, publicar: bool = False,
                         verboso: bool = True) -> dict[str, int]:
    """
    Extrai a distribuidora inteira numa passada e grava um parquet por município.

    **É por isso que este modo existe.** Extrair município a município parece mais
    econômico e é o oposto: o `WHERE MUN=` do ogr2ogr varre a tabela toda de qualquer
    jeito, e quando a base está no Drive compartilhado cada varredura rebaixa os 15,9 GB
    sob demanda. Medido em 04/09/2026: um único município da Cemig-D passou de uma hora.
    Numa passada só, o download é pago uma vez e os 766 municípios saem juntos.

    Devolve {código IBGE: pontos gravados}.
    """
    def log(msg: str) -> None:
        if verboso:
            print(f"  {msg}", flush=True)

    gdb = base.caminho
    cols_pip = _colunas_existentes(gdb, "PIP", COLUNAS_PIP, COLUNAS_PIP_OBRIGATORIAS)
    cols_pn = _colunas_existentes(gdb, "PONNOT", COLUNAS_PONNOT,
                                  COLUNAS_PONNOT_OBRIGATORIAS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.time()
        log("extraindo a PIP inteira…")
        pip = _extrair_csv(gdb, f"SELECT {', '.join(cols_pip)} FROM PIP",
                           tmp / "pip.csv", com_xy=False)
        log(f"PIP: {len(pip):,} pontos ({time.time() - t0:,.0f} s)".replace(",", "."))

        t1 = time.time()
        log("extraindo o PONNOT inteiro…")
        ponnot = _extrair_csv(gdb, f"SELECT {', '.join(cols_pn)} FROM PONNOT",
                              tmp / "ponnot.csv", com_xy=True)
        log(f"PONNOT: {len(ponnot):,} pontos ({time.time() - t1:,.0f} s)".replace(",", "."))

    gravados: dict[str, int] = {}
    for codigo, fatia in pip.groupby("MUN"):
        codigo = str(codigo).strip()
        if not codigo or codigo.lower() in ("nan", "none"):
            continue
        # O PONNOT é filtrado pelo município junto com a PIP: manter a tabela inteira
        # no merge de cada município multiplicaria o custo por 766.
        pn_municipio = ponnot[ponnot["MUN"].astype(str).str.strip() == codigo]
        cadastro = _montar(fatia, pn_municipio, base, codigo)
        salvar(cadastro, codigo, publicar=publicar)
        gravados[codigo] = len(cadastro)
    log(f"{len(gravados):,} municípios gravados".replace(",", "."))
    return gravados


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extrai o cadastro ponto a ponto de IP a partir da BDGD.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("municipios", nargs="*", help="códigos IBGE (7 dígitos)")
    ap.add_argument("--base", metavar="NOME",
                    help="extrai a distribuidora INTEIRA numa passada e grava um "
                         "parquet por município (substring do nome do .gdb)")
    ap.add_argument("--listar", action="store_true", help="lista o que já foi extraído")
    ap.add_argument("--bases", action="store_true", help="lista as bases .gdb visíveis")
    ap.add_argument("--publicar", action="store_true",
                    help="grava direto na pasta versionada, para o município ir ao ar")
    ap.add_argument("--publicar-existentes", nargs="*", metavar="IBGE",
                    help="move cadastros já extraídos para a pasta versionada "
                         "(sem argumento, publica todos os locais)")
    ap.add_argument("--forcar", action="store_true", help="refaz mesmo se já existir")
    args = ap.parse_args(argv)

    # O console do Windows abre em cp1252 e morre com UnicodeEncodeError em acento —
    # a mesma precaução do ETL do Hub, pelo mesmo motivo.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

    if args.bases:
        bases = bdgd.descobrir_bases(config.pastas_bdgd())
        print(f"{len(bases)} base(s) visível(is):")
        for b in sorted(bases, key=lambda x: x.distribuidora):
            print(f"  {b.distribuidora:<28} {b.data_base}  {b.versao}")
        return 0

    if args.listar:
        existentes = caminhos.municipios_disponiveis()
        if not existentes:
            print("Nenhum cadastro extraído.")
            print(f"  publicados: {caminhos.CADASTROS_PUBLICADOS}")
            print(f"  locais    : {caminhos.CADASTROS_LOCAIS}")
            return 0
        publicados = sum(1 for c in existentes if caminhos.esta_publicado(c))
        print(f"{len(existentes)} cadastro(s) · {publicados} publicado(s) (vão ao ar) · "
              f"{len(existentes) - publicados} só local(is)")
        total_kb = 0
        for codigo in existentes:
            arquivo = caminhos.caminho_cadastro(codigo)
            linhas = len(pd.read_parquet(arquivo, columns=["id_ponto"]))
            kb = arquivo.stat().st_size / 1e3
            total_kb += kb
            marca = "pub " if caminhos.esta_publicado(codigo) else "    "
            print(f"  {marca}{codigo}  {linhas:>8,} pontos  {kb:>7,.0f} KB"
                  .replace(",", "."))
        print(f"  {'':<4}{'total':<7}{'':>19}{total_kb:>9,.0f} KB".replace(",", "."))
        return 0

    if args.publicar_existentes is not None:
        alvos = args.publicar_existentes or [
            c for c in caminhos.municipios_disponiveis()
            if not caminhos.esta_publicado(c)
        ]
        if not alvos:
            print("Nada a publicar.")
            return 0
        for codigo in alvos:
            try:
                destino = caminhos.publicar(str(codigo).strip())
                print(f"  publicado {codigo} -> {destino.name}")
            except FileNotFoundError as exc:
                print(f"  FALHA {codigo}: {exc}")
        return 0

    if args.base:
        alvo = args.base.lower()
        bases = [b for b in bdgd.descobrir_bases(config.pastas_bdgd())
                 if alvo in b.caminho.name.lower() or alvo in b.distribuidora.lower()]
        if not bases:
            print(f"Nenhuma base casa com '{args.base}'. Use --bases para ver as opções.")
            return 1
        if len(bases) > 1:
            print(f"'{args.base}' casa com {len(bases)} bases; seja mais específico:")
            for b in bases:
                print(f"  {b.caminho.name}")
            return 1

        base = bases[0]
        print(f"\n{base.rotulo}")
        t0 = time.time()
        try:
            gravados = extrair_base_inteira(base, publicar=args.publicar)
        except Exception as exc:                        # noqa: BLE001
            print(f"  FALHA: {type(exc).__name__}: {exc}")
            return 1
        pontos = sum(gravados.values())
        print(f"  OK {len(gravados):,} municípios · {pontos:,} pontos · "
              f"{(time.time() - t0) / 60:,.1f} min".replace(",", "."))
        return 0

    if not args.municipios:
        ap.error("informe códigos IBGE, ou --base NOME, ou --listar")

    falhas = 0
    for codigo in args.municipios:
        codigo = str(codigo).strip()
        print(f"\n{codigo}")
        existente = caminhos.caminho_cadastro(codigo)
        if existente and not args.forcar:
            print(f"  já existe ({existente.name}); use --forcar para refazer")
            continue
        try:
            t0 = time.time()
            cadastro = extrair(codigo)
            caminho = salvar(cadastro, codigo, publicar=args.publicar)
            com_coord = int(cadastro["latitude"].notna().sum())
            print(f"  OK {len(cadastro):,} pontos · {com_coord:,} com coordenada "
                  f"({com_coord / len(cadastro):.0%}) · {time.time() - t0:,.0f} s"
                  .replace(",", "."))
            print(f"  -> {caminho}")
        except Exception as exc:                        # noqa: BLE001
            falhas += 1
            print(f"  FALHA: {type(exc).__name__}: {exc}")

    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
