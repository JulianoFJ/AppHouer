"""
Ingestão da BDGD (Base de Dados Geográfica da Distribuidora — ANEEL) para o Hub.

Só interessa a entidade **PIP** (Ponto de Iluminação Pública): uma linha por unidade
consumidora de IP, com potência, tipo de lâmpada, perdas e energia faturada mês a mês.

Por que ogr2ogr e não geopandas
-------------------------------
A leitura de File Geodatabase exige GDAL. Nesta máquina o Smart App Control do Windows
bloqueia a DLL do `pyogrio` ("Uma política de Controle de Aplicativo bloqueou este
arquivo"), então usamos o **executável** `ogr2ogr` do OSGeo4W/QGIS por subprocess. Isso
tem dois efeitos bons: nenhuma dependência binária entra no `requirements.txt` do portal
(os outros módulos ficam intocados) e o ETL roda offline, fora do Streamlit.

Fluxo
-----
    .gdb (dezenas de GB)  --ogr2ogr-->  parquet bruto da PIP (dezenas de MB)
                          --pandas -->  agregado por município (algumas centenas de KB)

O agregado é o único artefato que o portal lê.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import config

# ── colunas da PIP ───────────────────────────────────────────────────────────
# O schema da PIP MUDA entre versões da BDGD. Conferido nas duas bases em mãos:
#   Cemig-D V11 (2024) : tem TIPO_LAMP, POT_LAMP, PERDA_REAT/RELE/OUTR, ARE_LOC, CONTROLE
#   Energisa_MT M10 (2017): NÃO tem nenhum desses — só carga instalada e energia mensal
# Por isso a extração consulta o schema real antes de montar o SELECT: pedir uma coluna
# inexistente faz o ogr2ogr abortar a base inteira ("Unrecognized field name").
MESES = [f"ENE_{i:02d}" for i in range(1, 13)]

COLUNAS_OBRIGATORIAS = ["MUN", "CAR_INST", *MESES]
COLUNAS_OPCIONAIS = [
    "DIST", "CLAS_SUB", "SIT_ATIV", "ARE_LOC", "GRU_TAR",
    "TIPO_LAMP", "POT_LAMP", "PERDA_REAT", "PERDA_RELE", "PERDA_OUTR",
    "CONTROLE", "TIP_SIST", "DAT_CON",
]
COLUNAS_PIP = COLUNAS_OBRIGATORIAS + COLUNAS_OPCIONAIS

# Nome do arquivo BDGD: Distribuidora_CodANEEL_DataBase_Versao_Timestamp.gdb
PADRAO_NOME = re.compile(
    r"^(?P<distribuidora>.+?)_(?P<cod_dist>\d+)_(?P<data_base>\d{4}-\d{2}-\d{2})_(?P<versao>[A-Za-z]\d+)"
)

CANDIDATOS_OGR = [
    r"C:\OSGeo4W\bin\ogr2ogr.exe",
    r"C:\OSGeo4W64\bin\ogr2ogr.exe",
]

# Prefixo do código IBGE → UF. A BDGD identifica o município só pelo código; num acervo
# nacional a UF é o filtro primário da interface, e derivá-la do próprio código evita
# depender do cadastro do SICONFI para exibir o parque.
UF_POR_PREFIXO_IBGE = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}


class OgrIndisponivel(RuntimeError):
    """GDAL/ogr2ogr não encontrado — o ETL não tem como abrir o .gdb."""


def localizar_ogr2ogr() -> str:
    """
    Acha o executável do GDAL. Ordem: variável OGR2OGR, PATH, instalações conhecidas
    do OSGeo4W e do QGIS.
    """
    env = os.environ.get("OGR2OGR")
    if env and Path(env).exists():
        return env

    no_path = shutil.which("ogr2ogr")
    if no_path:
        return no_path

    for cand in CANDIDATOS_OGR:
        if Path(cand).exists():
            return cand

    for base in (Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)")):
        if base.exists():
            for qgis in sorted(base.glob("QGIS*")):
                exe = qgis / "bin" / "ogr2ogr.exe"
                if exe.exists():
                    return str(exe)

    raise OgrIndisponivel(
        "ogr2ogr não encontrado. Instale o OSGeo4W (https://qgis.org/download) ou o QGIS, "
        "ou aponte a variável de ambiente OGR2OGR para o executável."
    )


# ── descoberta das bases ─────────────────────────────────────────────────────

@dataclass
class BaseBDGD:
    caminho: Path
    distribuidora: str
    cod_dist: str
    data_base: str
    versao: str

    @property
    def ano_base(self) -> int:
        return int(self.data_base[:4])

    @property
    def rotulo(self) -> str:
        return f"{self.distribuidora} · {self.data_base} · {self.versao}"

    @property
    def slug(self) -> str:
        limpo = re.sub(r"[^A-Za-z0-9]+", "_", self.distribuidora).strip("_").lower()
        return f"{limpo}_{self.cod_dist}_{self.ano_base}"


def tamanho_base(base: "BaseBDGD") -> int:
    """Bytes da geodatabase. Num Drive virtual isso lê metadados, não baixa conteúdo."""
    try:
        return sum(f.stat().st_size for f in base.caminho.iterdir() if f.is_file())
    except OSError:
        return 0


def descobrir_bases(pastas: Optional[Sequence[Path]] = None) -> List[BaseBDGD]:
    """
    Lista os .gdb das pastas informadas (padrão: local + repositórios extras),
    extraindo distribuidora/data-base/versão do nome.

    Quando a mesma distribuidora aparece em mais de uma pasta, mantém a de data-base
    mais recente — o acervo nacional traz a V11/2024 de bases que localmente ainda
    estão numa versão antiga.
    """
    if pastas is None:
        pastas = config.pastas_bdgd()
    elif isinstance(pastas, (str, Path)):
        pastas = [Path(pastas)]

    bases: List[BaseBDGD] = []
    for pasta in pastas:
        pasta = Path(pasta)
        if not pasta.exists():
            continue
        for caminho in sorted(pasta.glob("*.gdb")):
            m = PADRAO_NOME.match(caminho.stem)
            if m:
                bases.append(BaseBDGD(caminho=caminho, **m.groupdict()))
            else:
                # nome fora do padrão da ANEEL: ainda assim utilizável
                bases.append(BaseBDGD(caminho=caminho, distribuidora=caminho.stem,
                                      cod_dist="0", data_base="0000-00-00", versao="?"))

    melhores: Dict[str, BaseBDGD] = {}
    for b in bases:
        chave = b.cod_dist if b.cod_dist != "0" else b.distribuidora.lower()
        atual = melhores.get(chave)
        if atual is None or b.data_base > atual.data_base:
            melhores[chave] = b
    return sorted(melhores.values(), key=lambda b: b.distribuidora.lower())


# ── extração ─────────────────────────────────────────────────────────────────

def colunas_disponiveis(base: BaseBDGD) -> List[str]:
    """Lê o schema real da camada PIP via `ogrinfo -so`."""
    exe = localizar_ogr2ogr()
    ogrinfo = str(Path(exe).with_name("ogrinfo.exe" if exe.lower().endswith(".exe") else "ogrinfo"))
    if not Path(ogrinfo).exists():
        return COLUNAS_PIP      # sem ogrinfo, tenta o conjunto completo

    proc = subprocess.run([ogrinfo, "-so", str(base.caminho), "PIP"],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Não foi possível ler o schema de {base.caminho.name}:\n"
                           f"{proc.stderr.strip() or proc.stdout.strip()}")

    campos = []
    for linha in proc.stdout.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s+\w+", linha.strip())
        if m:
            campos.append(m.group(1))
    return campos


def extrair_pip(base: BaseBDGD, destino: Optional[Path] = None,
                sobrescrever: bool = False) -> Path:
    """
    Exporta a camada PIP do .gdb para Parquet, apenas com as colunas que a base tem.
    Referência de custo: Cemig-D V11 (2,37 M pontos) leva ~55 s e gera ~70 MB.
    """
    destino = Path(destino or config.BDGD_PROCESSADOS / f"pip_{base.slug}.parquet")
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not sobrescrever:
        return destino

    existentes = set(colunas_disponiveis(base))
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in existentes]
    if faltando:
        raise RuntimeError(
            f"A camada PIP de {base.caminho.name} não tem os campos mínimos "
            f"{faltando}. Sem eles não há como montar o parque municipal."
        )
    selecionadas = [c for c in COLUNAS_PIP if c in existentes]

    exe = localizar_ogr2ogr()
    sql = f"SELECT {', '.join(selecionadas)} FROM PIP"
    cmd = [exe, "-f", "Parquet", str(destino), str(base.caminho), "-sql", sql]

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not destino.exists():
        raise RuntimeError(
            f"Falha ao extrair PIP de {base.caminho.name}:\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return destino


# ── classificação tecnológica ────────────────────────────────────────────────
# A BDGD não traz o domínio de TIPO_LAMP embutido, e o código varia entre
# distribuidoras. Em vez de fixar uma tabela, o rótulo é INFERIDO da assinatura física
# de cada código, o que funciona em qualquer base e é auditável:
#
#   perda de reator ≈ 0            -> LED (driver integrado, sem reator declarado)
#   potências 80/125/250/400 W     -> vapor de mercúrio (série normalizada da tecnologia)
#   potências 70/100/150/250/400 W -> vapor de sódio
#   perda baixa, série mista       -> vapor metálico
#
# A tabela inferida é gravada junto com as evidências (n, potência modal, perda média)
# para conferência contra o dicionário da distribuidora.

SERIE_MERCURIO = {80.0, 125.0}       # série normalizada exclusiva do vapor de mercúrio
SERIE_SODIO = {70.0, 100.0, 150.0}   # compartilhada com o vapor metálico
LIMITE_PERDA_LED = 1.0               # W — abaixo disso não há reator eletromagnético
LIMITE_PERDA_METALICO = 10.0         # W — reator de multivapor perde bem menos que o de VS


def _rotular_codigo(perda_reat: float, potencias_dominantes: set) -> str:
    # A ordem importa: 70/100/150 W servem tanto a sódio quanto a metálico, e o que
    # separa os dois é a perda do reator (~17 W no sódio, ~5 W no multivapor).
    if perda_reat < LIMITE_PERDA_LED:
        return "LED"
    if potencias_dominantes & SERIE_MERCURIO:
        return "Vapor de mercúrio"
    if perda_reat < LIMITE_PERDA_METALICO:
        return "Vapor metálico"
    if potencias_dominantes & SERIE_SODIO:
        return "Vapor de sódio"
    return "Descarga (não identificada)"


def inferir_tecnologias(df: pd.DataFrame) -> pd.DataFrame:
    """Mapa TIPO_LAMP -> tecnologia, com as evidências que sustentam cada rótulo."""
    if not {"TIPO_LAMP", "PERDA_REAT", "POT_LAMP"} <= set(df.columns):
        return pd.DataFrame(columns=["tipo_lamp", "tecnologia", "pontos",
                                     "pot_modal_w", "pot_mediana_w",
                                     "perda_reator_media_w"])

    linhas = []
    for codigo, g in df.groupby("TIPO_LAMP", dropna=False):
        codigo_txt = str(codigo).strip()
        perda = float(g["PERDA_REAT"].mean() or 0.0)
        modais = set(g["POT_LAMP"].value_counts().head(2).index.astype(float))
        modas = g["POT_LAMP"].mode()
        linhas.append({
            "tipo_lamp": codigo_txt,
            # código em branco não é evidência de nada: perda zero aqui significa
            # campo não preenchido, não driver eletrônico.
            "tecnologia": "Não informado" if not codigo_txt
                          else _rotular_codigo(perda, modais),
            "pontos": int(len(g)),
            "pot_modal_w": float(modas.iloc[0]) if not modas.empty else None,
            "pot_mediana_w": float(g["POT_LAMP"].median()),
            "perda_reator_media_w": round(perda, 2),
        })
    return (pd.DataFrame(linhas)
            .sort_values("pontos", ascending=False)
            .reset_index(drop=True))


def detectar_fator_carga(df: pd.DataFrame) -> tuple[float, str]:
    """
    `CAR_INST` deveria vir em **kW**, mas algumas distribuidoras declaram em **W**.

    A detecção usa `POT_LAMP`, que é sempre em W e é o campo confiável: numa base
    correta, POT_LAMP/CAR_INST ≈ 1000 (a carga é a potência da lâmpada mais perdas,
    convertida para kW). Quando a razão dá ≈ 1, a carga está em W.

    Caso real (Equatorial_GO V11/2024): CAR_INST mediana 80,0 e POT_LAMP 80,0 — razão 1.
    Sem a correção, o parque aparece com 154 kW por luminária e 4 horas de operação por
    ano; com ela, 154 W e 4.000 h/ano, que é exatamente o esperado. O erro é simétrico
    (carga 1000× maior, horas 1000× menores), o que também serve de conferência.
    """
    if "POT_LAMP" not in df.columns:
        return 1.0, ""
    car = pd.to_numeric(df["CAR_INST"], errors="coerce").median()
    pot = pd.to_numeric(df["POT_LAMP"], errors="coerce").median()
    if not car or not pot or car <= 0 or pot <= 0:
        return 1.0, ""

    razao = pot / car
    if 0.5 <= razao <= 2.0:
        return 0.001, (
            f"CAR_INST declarada em W, não em kW (POT_LAMP/CAR_INST = {razao:.2f}, "
            "esperado ~1.000). Carga dividida por 1.000 para normalizar."
        )
    if 500 <= razao <= 2000:
        return 1.0, ""
    return 1.0, (
        f"Relação POT_LAMP/CAR_INST = {razao:,.1f} fora do esperado (~1 ou ~1.000): a "
        "unidade da carga instalada não pôde ser confirmada. Carga mantida como declarada."
    )


def _aplicar_override(mapa: pd.DataFrame) -> pd.DataFrame:
    """
    Permite corrigir a inferência sem tocar no código: basta criar
    `hub_municipios/data/tipo_lamp_override.csv` com colunas tipo_lamp,tecnologia.
    """
    caminho = config.DATA_PACOTE / "tipo_lamp_override.csv"
    if not caminho.exists():
        return mapa
    try:
        over = pd.read_csv(caminho, dtype={"tipo_lamp": str})
    except Exception:
        return mapa
    if not {"tipo_lamp", "tecnologia"} <= set(over.columns):
        return mapa
    mapa = mapa.merge(over[["tipo_lamp", "tecnologia"]], on="tipo_lamp",
                      how="left", suffixes=("", "_override"))
    mapa["tecnologia"] = mapa["tecnologia_override"].fillna(mapa["tecnologia"])
    return mapa.drop(columns=["tecnologia_override"])


# ── agregação por município ──────────────────────────────────────────────────

@dataclass
class ResultadoETL:
    base: BaseBDGD
    municipios: pd.DataFrame
    tecnologia: pd.DataFrame
    mapa_tecnologias: pd.DataFrame
    pontos_lidos: int = 0
    pontos_desativados: int = 0
    avisos: List[str] = field(default_factory=list)


def agregar(caminho_pip: Path, base: BaseBDGD) -> ResultadoETL:
    """
    Lê o parquet da PIP e devolve dois agregados por município:
      - `municipios`: pontos, carga, consumo, potência média, % urbano
      - `tecnologia`: pontos e carga por município × tecnologia (formato longo)
    """
    df = pd.read_parquet(caminho_pip)
    total_lido = len(df)
    avisos: List[str] = []

    # Só pontos ativos: 'DS' são desativados e distorceriam o parque.
    if "SIT_ATIV" in df.columns:
        desativados = int((df["SIT_ATIV"] != "AT").sum())
        df = df[df["SIT_ATIV"] == "AT"]
    else:
        desativados = 0
        avisos.append("Campo SIT_ATIV ausente: não foi possível excluir pontos desativados.")

    df["MUN"] = df["MUN"].astype(str).str.extract(r"(\d{7})", expand=False)
    invalidos = int(df["MUN"].isna().sum())
    if invalidos:
        avisos.append(f"{invalidos:,} pontos com código IBGE inválido foram descartados.")
    df = df.dropna(subset=["MUN"])

    for col in ("CAR_INST", "POT_LAMP", "PERDA_REAT", *MESES):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # CAR_INST deveria vir em kW; algumas distribuidoras declaram em W (ver
    # detectar_fator_carga). Normaliza ANTES de qualquer cálculo derivado.
    fator, aviso_carga = detectar_fator_carga(df)
    if fator != 1.0:
        df["CAR_INST"] = df["CAR_INST"] * fator
    if aviso_carga:
        avisos.append(aviso_carga)

    df["kwh_ano"] = df[MESES].sum(axis=1)
    df["carga_w"] = df["CAR_INST"] * 1000.0     # agora garantidamente em kW

    # Bases anteriores à V-alguma-coisa não trazem TIPO_LAMP/POT_LAMP/PERDA_REAT
    # (conferido: Energisa_MT M10/2017). Sem esses campos não há mix tecnológico.
    tem_tecnologia = {"TIPO_LAMP", "PERDA_REAT", "POT_LAMP"} <= set(df.columns)
    if tem_tecnologia:
        mapa = _aplicar_override(inferir_tecnologias(df))
        df["TIPO_LAMP"] = df["TIPO_LAMP"].astype(str)
        df = df.merge(mapa[["tipo_lamp", "tecnologia"]],
                      left_on="TIPO_LAMP", right_on="tipo_lamp", how="left")
        df["tecnologia"] = df["tecnologia"].fillna("Não informado")
    else:
        mapa = inferir_tecnologias(df)      # vazio
        df["tecnologia"] = "Não informado"
        avisos.append(
            f"A BDGD {base.versao} ({base.data_base}) não traz TIPO_LAMP/POT_LAMP/"
            "PERDA_REAT: o mix tecnológico e a potência de lâmpada não puderam ser "
            "apurados para esta distribuidora."
        )

    agregacoes = {
        "pontos_ip": ("MUN", "size"),
        "carga_instalada_kw": ("CAR_INST", "sum"),
        "consumo_kwh_ano": ("kwh_ano", "sum"),
        "potencia_media_w": ("carga_w", "mean"),
    }
    if "POT_LAMP" in df.columns:
        agregacoes["potencia_lampada_media_w"] = ("POT_LAMP", "mean")
    ag = (df.groupby("MUN").agg(**agregacoes)
            .reset_index().rename(columns={"MUN": "codigo_municipio"}))

    if "ARE_LOC" in df.columns:
        urb = (df.assign(_u=(df["ARE_LOC"] == "UB").astype(int))
                 .groupby("MUN")["_u"].mean().rename("perc_urbano"))
        ag = ag.merge(urb, left_on="codigo_municipio", right_index=True, how="left")
    else:
        ag["perc_urbano"] = pd.NA

    if tem_tecnologia:
        led = (df.assign(_l=(df["tecnologia"] == "LED").astype(int))
                 .groupby("MUN")["_l"].mean().rename("perc_led"))
        ag = ag.merge(led, left_on="codigo_municipio", right_index=True, how="left")
    else:
        ag["perc_led"] = pd.NA

    ag["consumo_kwh_ponto_ano"] = ag["consumo_kwh_ano"] / ag["pontos_ip"]
    ag["horas_equivalentes_ano"] = (ag["consumo_kwh_ano"] /
                                    ag["carga_instalada_kw"].replace(0, pd.NA))
    ag["distribuidora"] = base.distribuidora
    ag["cod_distribuidora"] = base.cod_dist
    ag["data_base_bdgd"] = base.data_base
    ag["versao_bdgd"] = base.versao
    ag["ano_base_bdgd"] = base.ano_base

    tec = (df.groupby(["MUN", "tecnologia"])
             .agg(pontos=("MUN", "size"),
                  carga_kw=("CAR_INST", "sum"),
                  consumo_kwh_ano=("kwh_ano", "sum"))
             .reset_index()
             .rename(columns={"MUN": "codigo_municipio"}))
    tec["distribuidora"] = base.distribuidora
    tec["ano_base_bdgd"] = base.ano_base

    # Sanidade física: IP opera ~11-12 h/dia (≈ 4.000-4.400 h/ano). Fora disso, o
    # consumo declarado não fecha com a carga instalada e o dado merece desconfiança.
    mediana_horas = float(ag["horas_equivalentes_ano"].median(skipna=True) or 0)
    if not (3000 <= mediana_horas <= 5000):
        avisos.append(
            f"Horas equivalentes medianas = {mediana_horas:,.0f} h/ano, fora da faixa "
            "esperada de 3.000–5.000 h. Consumo e carga instalada podem estar inconsistentes."
        )

    return ResultadoETL(base=base, municipios=ag, tecnologia=tec,
                        mapa_tecnologias=mapa, pontos_lidos=total_lido,
                        pontos_desativados=desativados, avisos=avisos)


# ── consolidação e leitura pelo portal ───────────────────────────────────────

def salvar_agregado_base(res: ResultadoETL) -> None:
    """
    Persiste o agregado de UMA base. É o que torna o ETL retomável: processar o acervo
    nacional leva horas (o gargalo é o download do Drive), e uma queda na 30ª base não
    pode custar as 29 anteriores.
    """
    config.BDGD_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    res.municipios.to_parquet(
        config.BDGD_PROCESSADOS / f"agregado_{res.base.slug}.parquet", index=False)
    if not res.tecnologia.empty:
        res.tecnologia.to_parquet(
            config.BDGD_PROCESSADOS / f"tecnologia_{res.base.slug}.parquet", index=False)


def _apenas_mais_recentes(arquivos: List[Path]) -> List[Path]:
    """
    Um agregado por distribuidora, o de data-base mais nova.

    O slug termina em `<coddist>_<ano>`, então processar duas versões da mesma
    distribuidora (a M10/2017 e a V11/2024 da Energisa MT, por exemplo) deixaria os
    dois agregados em disco — e a consolidação somaria os parques como se fossem
    concessionárias distintas, dobrando os pontos daquele estado.
    """
    melhor: Dict[str, tuple] = {}
    for arq in arquivos:
        partes = arq.stem.split("_")
        if len(partes) >= 2 and partes[-1].isdigit() and partes[-2].isdigit():
            chave, ano = partes[-2], int(partes[-1])
        else:
            chave, ano = arq.stem, 0
        if chave not in melhor or ano > melhor[chave][0]:
            melhor[chave] = (ano, arq)
    return [arq for _, arq in sorted(melhor.values(), key=lambda t: t[1].name)]


def consolidar_de_disco() -> Dict[str, pd.DataFrame]:
    """Junta todos os agregados por base já gravados, sem reprocessar geodatabase."""
    mun_arqs = _apenas_mais_recentes(sorted(config.BDGD_PROCESSADOS.glob("agregado_*.parquet")))
    tec_arqs = _apenas_mais_recentes(sorted(config.BDGD_PROCESSADOS.glob("tecnologia_*.parquet")))
    if not mun_arqs:
        return {"municipios": pd.DataFrame(), "tecnologia": pd.DataFrame()}

    mun = pd.concat([pd.read_parquet(a) for a in mun_arqs], ignore_index=True)
    tec = (pd.concat([pd.read_parquet(a) for a in tec_arqs], ignore_index=True)
           if tec_arqs else pd.DataFrame())
    return _consolidar_frames(mun, tec)


def consolidar(resultados: List[ResultadoETL]) -> Dict[str, pd.DataFrame]:
    """
    Junta várias distribuidoras. Um município pode aparecer em mais de uma base
    (fronteira de área de concessão) — nesse caso os parques são somados, e a
    distribuidora vira uma lista.
    """
    if not resultados:
        return {"municipios": pd.DataFrame(), "tecnologia": pd.DataFrame()}

    mun = pd.concat([r.municipios for r in resultados], ignore_index=True)
    tec = pd.concat([r.tecnologia for r in resultados], ignore_index=True)
    return _consolidar_frames(mun, tec)


def _consolidar_frames(mun: pd.DataFrame, tec: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Um município pode aparecer em mais de uma base (fronteira de área de concessão) —
    nesse caso os parques são somados e a distribuidora vira uma lista. Sem isso, o
    município ficaria com o parque de apenas uma das concessionárias.
    """
    if mun.empty:
        return {"municipios": mun, "tecnologia": tec}

    mun = mun.copy()
    mun["codigo_municipio"] = mun["codigo_municipio"].astype(str)

    duplicados = mun["codigo_municipio"].duplicated(keep=False)
    if duplicados.any():
        somas = {"pontos_ip": "sum", "carga_instalada_kw": "sum", "consumo_kwh_ano": "sum"}
        agrupado = mun.groupby("codigo_municipio").agg({
            **somas,
            "distribuidora": lambda s: " + ".join(sorted(set(s))),
            "cod_distribuidora": lambda s: " + ".join(sorted(set(s.astype(str)))),
            "data_base_bdgd": "max",
            "versao_bdgd": "last",
            "ano_base_bdgd": "max",
        }).reset_index()
        # médias ponderadas pelo nº de pontos
        for col in ("potencia_media_w", "potencia_lampada_media_w", "perc_urbano", "perc_led"):
            if col in mun.columns:
                pond = (mun.assign(_p=mun[col] * mun["pontos_ip"])
                          .groupby("codigo_municipio")
                          .agg(_p=("_p", "sum"), _n=("pontos_ip", "sum")))
                agrupado[col] = agrupado["codigo_municipio"].map(pond["_p"] / pond["_n"])
        agrupado["consumo_kwh_ponto_ano"] = (agrupado["consumo_kwh_ano"] /
                                             agrupado["pontos_ip"])
        agrupado["horas_equivalentes_ano"] = (agrupado["consumo_kwh_ano"] /
                                              agrupado["carga_instalada_kw"].replace(0, pd.NA))
        mun = agrupado

    mun["uf"] = mun["codigo_municipio"].str[:2].map(UF_POR_PREFIXO_IBGE)
    return {"municipios": mun.reset_index(drop=True), "tecnologia": tec}


def gravar_derivado(dados: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
    config.garantir_pastas()
    caminhos = {}
    if not dados["municipios"].empty:
        dados["municipios"].to_parquet(config.BDGD_MUNICIPIOS, index=False)
        caminhos["municipios"] = config.BDGD_MUNICIPIOS
    if not dados["tecnologia"].empty:
        dados["tecnologia"].to_parquet(config.BDGD_TECNOLOGIA, index=False)
        caminhos["tecnologia"] = config.BDGD_TECNOLOGIA
    return caminhos


def carregar_municipios() -> pd.DataFrame:
    """Agregado por município — é o que o portal lê. Vazio se o ETL nunca rodou."""
    if not config.BDGD_MUNICIPIOS.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(config.BDGD_MUNICIPIOS)
    except Exception:
        return pd.DataFrame()


def carregar_tecnologia() -> pd.DataFrame:
    if not config.BDGD_TECNOLOGIA.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(config.BDGD_TECNOLOGIA)
    except Exception:
        return pd.DataFrame()
