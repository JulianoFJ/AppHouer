"""
Tarifas de iluminação pública (subgrupo B4a) do Portal de Dados Abertos da ANEEL.

SÃO DUAS TARIFAS, E ELAS NÃO SÃO INTERCAMBIÁVEIS
------------------------------------------------
    tarifa_sem_tributos  TUSD + TE da resolução homologatória (REH). É o número que
                         aparece no site da distribuidora e em "tarifa B4a". Cemig:
                         ~R$ 0,39/kWh.
    tarifa_com_tributos  R$/kWh EFETIVAMENTE FATURADO — receita faturada com PIS,
                         COFINS e ICMS dividida pelo mercado da classe Iluminação
                         Pública. Cemig: ~R$ 0,51/kWh.

Usar a primeira onde cabe a segunda subestima o custo de energia do Poder Concedente
em ~30%. Em 28/08/2026 isso produziu uma divergência de fator 2 entre a triagem de
São José da Lapa e o modelo econômico-financeiro de Matozinhos — a outra metade veio
de comparar valor anual com acumulado nominal de ciclo (ver `indicadores`).

A ANEEL documenta que a receita faturada do SAMP **inclui bandeiras tarifárias** e
**exclui COSIP/CIP**, que é exatamente a definição correta de custo de energia de IP
para um EVTE: a CIP é receita do município, não custo do serviço.

FONTES
------
    1. Tarifas de aplicação das distribuidoras de energia elétrica (CKAN, um recurso)
       https://dadosabertos.aneel.gov.br/dataset/tarifas-distribuidoras-energia-eletrica
       Campos: SigAgente, DscSubGrupo, DscClasse, VlrTUSD, VlrTE, DatInicioVigencia,
               DatFimVigencia, DscREH, DscBaseTarifaria, DscModalidadeTarifaria.
    2. SAMP — Sistema de Acompanhamento de Informações de Mercado (um recurso por ano)
       https://dadosabertos.aneel.gov.br/dataset/samp
       Mercado (MWh) e receita faturada com tributos, por distribuidora e classe.

O painel "Luz na Tarifa" (portalrelatorios.aneel.gov.br/luznatarifa) mostra o mesmo
dado, mas é Power BI embarcado, sem API estável — não serve de fonte programática.

POR QUE ESTE MÓDULO NÃO É CHAMADO PELO STREAMLIT
------------------------------------------------
O servidor da ANEEL derruba o handshake TLS com frequência
(`SSL: UNEXPECTED_EOF_WHILE_READING`, testado em 28/08/2026 por curl e por fetch, das
duas vezes sem resposta). Tarifa muda uma vez por ano, no reajuste. Portanto: ETL
offline (`etl_aneel.py`) → `data/tarifas_b4a.parquet` versionado → o portal só lê o
parquet. Mesmo desenho da BDGD, pelo mesmo motivo.

TOLERÂNCIA A SCHEMA
-------------------
Os nomes de campo abaixo vêm do dicionário de metadados publicado, não de uma resposta
lida. `_coluna()` casa por normalização (sem acento, sem caixa, sem separador), de modo
que uma renomeação cosmética no CKAN não quebra o ETL — e o que não casar aparece no
`--listar`, que imprime o schema real recebido.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

from . import config

TIMEOUT = 120
PAGINA = 32000

COLUNAS_TARIFAS = [
    "cod_distribuidora", "sigla_distribuidora", "distribuidora",
    "vigencia_inicio", "vigencia_fim", "reh",
    "tarifa_sem_tributos", "tusd", "te",
    "tarifa_com_tributos", "tarifa_samp_sem_tributos",
    "ano_samp", "mercado_mwh", "receita_energia", "receita_bandeiras", "tributos",
    "receita_com_tributos",
]


# ── normalização de nomes de coluna ─────────────────────────────────────────

def _chave(texto: Any) -> str:
    """Nome de coluna reduzido a letras e dígitos minúsculos, sem acento."""
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


# De-para entre o nome que a BDGD usa e a sigla de agente da ANEEL. Não é derivável:
# a ANEEL identifica as Energisa por acrônimo regulatório (EAC, EMS, EMT, EMR, ENF,
# EPB, ERO, ESE, ESS, ETO), a Enel SP ainda pela razão social antiga (ELETROPAULO), e
# as duas distribuidoras do Norte pelo controlador atual (ÂMBAR). Sem esta tabela o
# casamento por nome acerta 16 das 38 distribuidoras da BDGD.
# Conferido contra a lista completa de 115 siglas do recurso em 28/08/2026.
ALIAS_BDGD_ANEEL = {
    "Amazonas_Energia": "ÂMBAR AMAZONAS",
    "CEA_Equatorial": "CEA",
    "CEEE_Equatorial": "CEEE-D",
    "Celesc-Dis": "CELESC",
    "CPFL_Piratininga": "CPFL-PIRATINING",
    "Enel_SP": "ELETROPAULO",
    "Energisa_AC": "EAC",
    "Energisa_Minas_Rio": "EMR",
    "Energisa_MS": "EMS",
    "Energisa_MT": "EMT",
    "Energisa_Nova_Friburgo": "ENF",
    "Energisa_PB": "EPB",
    "Energisa_RO": "ERO",
    "Energisa_SE": "ESE",
    "Energisa_Sul-Sudeste": "ESS",
    "Energisa_TO": "ETO",
    "Light": "LIGHT SESA",
    "Neoenergia_Coelba": "COELBA",
    "Neoenergia_Cosern": "COSERN",
    "Neoenergia_Elektro": "ELEKTRO",
    "Neoenergia_Pernambuco": "NEOENERGIA PE",
    "Roraima_Energia": "ÂMBAR ENERGIA RR",
}


def casar_sigla(nome_bdgd: str, siglas_aneel: Iterable[str]) -> Optional[str]:
    """
    Sigla da ANEEL correspondente a um nome de distribuidora da BDGD.

    Três tentativas, nesta ordem: a tabela explícita, igualdade por `_chave`
    (`CPFL_Paulista` == `CPFL-PAULISTA`), e contenção de uma chave na outra
    (`Cemig-D` ⊂ `CEMIG-D`), ficando com o candidato mais curto para não deixar
    `RGE` casar com `RGE SUL` quando `RGE` existe.
    """
    disponiveis = {_chave(s): s for s in siglas_aneel if s}
    if not nome_bdgd:
        return None

    alias = ALIAS_BDGD_ANEEL.get(str(nome_bdgd).strip())
    if alias and _chave(alias) in disponiveis:
        return disponiveis[_chave(alias)]

    chave = _chave(nome_bdgd)
    if chave in disponiveis:
        return disponiveis[chave]

    contidos = [orig for k, orig in disponiveis.items()
                if k and (k in chave or chave in k)]
    return min(contidos, key=len) if contidos else None


def _coluna(df: pd.DataFrame, *candidatos: str) -> Optional[str]:
    """Primeira coluna do DataFrame que casa com algum candidato, por `_chave`."""
    mapa = {_chave(c): c for c in df.columns}
    for cand in candidatos:
        achado = mapa.get(_chave(cand))
        if achado is not None:
            return achado
    return None


def _numero(serie: pd.Series) -> pd.Series:
    """
    Converte para float aceitando o decimal por VÍRGULA do CKAN da ANEEL.

    O portal serve `0,39` em campo de texto em vários recursos. `pd.to_numeric` sozinho
    devolve NaN para tudo e o ETL grava um parquet inteiro de nulos sem reclamar — é a
    mesma armadilha do cadastro que guarda carga como texto.
    """
    if serie.dtype.kind in "if":
        return serie.astype(float)
    texto = serie.astype(str).str.strip()
    # só troca vírgula por ponto quando não há ponto de milhar junto
    tem_ambos = texto.str.contains(r"\.", regex=True) & texto.str.contains(",", regex=True)
    limpo = texto.where(~tem_ambos, texto.str.replace(".", "", regex=False))
    limpo = limpo.str.replace(",", ".", regex=False)
    return pd.to_numeric(limpo, errors="coerce")


def _data(serie: pd.Series) -> pd.Series:
    """
    Datas do CKAN, que vêm em ISO pela API e em dd/mm/aaaa nos CSV baixados à mão.

    ISO primeiro; o que sobrar em NaT tenta dd/mm/aaaa. Passar `dayfirst=True` em cima
    de ISO é o caminho para NaT silencioso — e uma vigência nula derruba o filtro de
    "REH mais recente", trazendo de volta tarifas de 2010 para a mediana.
    """
    iso = pd.to_datetime(serie, errors="coerce", format="ISO8601")
    if iso.isna().any():
        br = pd.to_datetime(serie, errors="coerce", dayfirst=True, format="mixed")
        iso = iso.fillna(br)
    return iso


# ── cliente CKAN ────────────────────────────────────────────────────────────

def _sessao() -> requests.Session:
    s = requests.Session()
    politica = Retry(total=5, backoff_factor=1.5,
                     status_forcelist=[429, 500, 502, 503, 504],
                     allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=politica))
    s.headers.update({"User-Agent": "Plataforma-IP/1.0"})
    return s


class ErroANEEL(RuntimeError):
    """Falha ao falar com o portal da ANEEL, com a causa preservada na mensagem."""


def _get(sessao: requests.Session, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = sessao.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.SSLError as exc:
        raise ErroANEEL(
            "Handshake TLS recusado pelo servidor da ANEEL. É falha conhecida e "
            "intermitente do portal, não da rede local — tente de novo mais tarde ou "
            "baixe o CSV pelo navegador e aponte o ETL com --csv."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ErroANEEL(f"Falha ao consultar {url}: {exc}") from exc
    corpo = r.json()
    if not corpo.get("success"):
        raise ErroANEEL(f"CKAN respondeu sem sucesso para {url}: {corpo.get('error')}")
    return corpo["result"]


def baixar_recurso(resource_id: str, sessao: Optional[requests.Session] = None,
                   limite: Optional[int] = None,
                   filtros: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Baixa um recurso do datastore do CKAN, paginado, e devolve o DataFrame cru.

    Sem tratamento de schema: o que vier, vem. Quem interpreta são as funções
    `_normalizar_*`, e `--listar` imprime exatamente isto.
    """
    sessao = sessao or _sessao()
    url = f"{config.ANEEL_CKAN}/datastore_search"
    registros: List[Dict[str, Any]] = []
    offset = 0
    while True:
        tamanho = PAGINA if limite is None else min(PAGINA, limite - len(registros))
        if tamanho <= 0:
            break
        params: Dict[str, Any] = {"resource_id": resource_id,
                                  "limit": tamanho, "offset": offset}
        if filtros:
            # O filtro roda no servidor: sem ele o SAMP são 1,37 milhão de linhas por
            # ano, contra 21 mil da classe de iluminação pública.
            params["filters"] = json.dumps(filtros, ensure_ascii=False)
        res = _get(sessao, url, params)
        lote = res.get("records", [])
        registros.extend(lote)
        if len(lote) < tamanho:
            break
        offset += len(lote)
        time.sleep(0.2)
    df = pd.DataFrame(registros)
    return df.drop(columns=[c for c in ("_id",) if c in df.columns])


def listar_recursos_samp(sessao: Optional[requests.Session] = None) -> pd.DataFrame:
    """Recursos do dataset SAMP — um por ano — com id, nome e ano inferido."""
    sessao = sessao or _sessao()
    res = _get(sessao, f"{config.ANEEL_CKAN}/package_show",
               {"id": config.ANEEL_DATASET_SAMP})
    linhas = []
    for r in res.get("resources", []):
        nome = str(r.get("name", ""))
        anos = [int(t) for t in "".join(c if c.isdigit() else " " for c in nome).split()
                if len(t) == 4 and 2000 <= int(t) <= 2100]
        linhas.append({"resource_id": r.get("id"), "nome": nome,
                       "formato": r.get("format"), "ano": anos[0] if anos else None})
    return pd.DataFrame(linhas)


# ── normalização das duas fontes ────────────────────────────────────────────

@dataclass
class Diagnostico:
    """O que o ETL conseguiu casar e o que não conseguiu — vai para o relatório."""
    fonte: str
    linhas_lidas: int = 0
    linhas_uteis: int = 0
    colunas_faltando: List[str] = None
    avisos: List[str] = None

    def __post_init__(self) -> None:
        self.colunas_faltando = self.colunas_faltando or []
        self.avisos = self.avisos or []


def normalizar_tarifas(bruto: pd.DataFrame) -> tuple[pd.DataFrame, Diagnostico]:
    """
    Recurso de tarifas homologadas → uma linha por distribuidora com a B4a vigente.

    Filtra subgrupo B4a e classe Iluminação Pública, soma TUSD + TE e fica com a
    vigência mais recente de cada distribuidora. A base tarifária "Tarifa de Aplicação"
    é a que vale; "Base Econômica" é insumo de cálculo regulatório e entraria duplicando.
    """
    diag = Diagnostico("tarifas homologadas", linhas_lidas=len(bruto))
    if bruto.empty:
        diag.avisos.append("Recurso de tarifas veio vazio.")
        return pd.DataFrame(columns=COLUNAS_TARIFAS), diag

    col = {
        "sigla": _coluna(bruto, "SigAgente", "SigAgenteAcessante"),
        "subgrupo": _coluna(bruto, "DscSubGrupo", "SubGrupo"),
        "classe": _coluna(bruto, "DscClasse", "Classe"),
        "tusd": _coluna(bruto, "VlrTUSD", "TUSD"),
        "te": _coluna(bruto, "VlrTE", "TE"),
        "inicio": _coluna(bruto, "DatInicioVigencia", "InicioVigencia"),
        "fim": _coluna(bruto, "DatFimVigencia", "FimVigencia"),
        "reh": _coluna(bruto, "DscREH", "REH"),
        "base": _coluna(bruto, "DscBaseTarifaria", "BaseTarifaria"),
        "posto": _coluna(bruto, "NomPostoTarifario", "PostoTarifario"),
        "subclasse": _coluna(bruto, "DscSubClasse", "SubClasse"),
        "unidade": _coluna(bruto, "DscUnidadeTerciaria", "UnidadeTerciaria"),
    }
    diag.colunas_faltando = [k for k, v in col.items() if v is None]
    for obrigatoria in ("sigla", "subgrupo", "tusd", "te"):
        if col[obrigatoria] is None:
            diag.avisos.append(
                f"Coluna obrigatória '{obrigatoria}' não encontrada. Schema recebido: "
                f"{list(bruto.columns)}")
            return pd.DataFrame(columns=COLUNAS_TARIFAS), diag

    df = bruto.copy()

    # O B4a NÃO está em DscSubGrupo — o subgrupo é "B4", e o "a" vive na SUBCLASSE
    # ("Iluminação pública – B4a"). Filtrar por subgrupo == "B4a" devolve zero linhas
    # sem erro nenhum, que é a falha silenciosa clássica deste tipo de ETL. Quando a
    # subclasse existe, ela manda; senão, cai para o subgrupo.
    # B4b é bulbo de lâmpada — tarifa mais cara e escopo diferente. Fica de fora.
    chave_sub = df[col["subclasse"]].map(_chave) if col["subclasse"] is not None else None
    # A subclasse só manda quando ela de fato carrega a distinção B4a/B4b. Se a coluna
    # existe mas nunca menciona "b4", ela não é a discriminante e o subgrupo assume.
    # Sem essa condição, um recurso só de B4b cairia no fallback e entraria como se
    # fosse B4a — que é o inverso exato do que a distinção serve para evitar.
    if chave_sub is not None and chave_sub.str.contains("b4", na=False).any():
        alvo = chave_sub.str.endswith(_chave(config.ANEEL_SUBCLASSE_IP), na=False)
    else:
        alvo = df[col["subgrupo"]].map(_chave) == _chave(config.ANEEL_SUBGRUPO_IP)

    if col["classe"]:
        classe_ip = df[col["classe"]].map(_chave) == _chave(config.ANEEL_CLASSE_IP)
        if classe_ip.any():
            alvo &= classe_ip
    if col["base"]:
        # "Tarifa de Aplicação" é a que vale; "Base Econômica" é insumo regulatório e
        # entraria duplicando cada linha com um valor parecido, porém errado.
        aplicacao = df[col["base"]].map(_chave).str.contains("aplicacao", na=False)
        if aplicacao.any():
            alvo &= aplicacao
    df = df[alvo]
    if df.empty:
        diag.avisos.append(
            f"Nenhuma linha de {config.ANEEL_SUBCLASSE_IP}. Subgrupos presentes: "
            f"{sorted(bruto[col['subgrupo']].astype(str).unique())[:20]}"
            + (f" | subclasses: {sorted(bruto[col['subclasse']].astype(str).unique())[:10]}"
               if col["subclasse"] else ""))
        return pd.DataFrame(columns=COLUNAS_TARIFAS), diag

    df["tusd"] = _numero(df[col["tusd"]])
    df["te"] = _numero(df[col["te"]])

    # Conversão de unidade pela coluna que a declara, não por adivinhação. O recurso
    # traz IP em R$/MWh; a heurística de magnitude fica só como rede de segurança para
    # quando DscUnidadeTerciaria sumir do schema.
    if col["unidade"] is not None:
        em_mwh = df[col["unidade"]].map(_chave) == _chave("MWh")
        df.loc[em_mwh, ["tusd", "te"]] /= 1000.0
        if em_mwh.any():
            diag.avisos.append(
                f"{int(em_mwh.sum())} linhas em R$/MWh convertidas para R$/kWh "
                "(unidade declarada em DscUnidadeTerciaria).")
        nao_energia = ~em_mwh & df[col["unidade"]].map(_chave).isin({_chave("kW")})
        if nao_energia.any():
            # Tarifa de demanda em R$/kW não é tarifa de energia e não pode somar.
            df = df[~nao_energia]
    elif (df["tusd"].fillna(0) + df["te"].fillna(0)).median() > 10:
        df[["tusd", "te"]] /= 1000.0
        diag.avisos.append("Sem coluna de unidade; magnitude indica R$/MWh, convertido.")

    df["tarifa_sem_tributos"] = df["tusd"].fillna(0) + df["te"].fillna(0)

    df["vigencia_inicio"] = _data(df[col["inicio"]]) if col["inicio"] else pd.NaT
    df["vigencia_fim"] = _data(df[col["fim"]]) if col["fim"] else pd.NaT
    df["sigla_distribuidora"] = df[col["sigla"]].astype(str).str.strip().str.upper()
    df["reh"] = df[col["reh"]].astype(str) if col["reh"] else None

    # Uma linha por distribuidora: a vigência mais recente. Postos tarifários e
    # modalidades diferentes do B4a convencional colapsam pela mediana, que é robusta
    # a uma linha esdrúxula solta no recurso.
    df = df[df["tarifa_sem_tributos"] > 0]
    df = df.sort_values("vigencia_inicio")
    ultima = df.groupby("sigla_distribuidora")["vigencia_inicio"].transform("max")
    df = df[(df["vigencia_inicio"] == ultima) | df["vigencia_inicio"].isna()]

    saida = (df.groupby("sigla_distribuidora")
               .agg(tarifa_sem_tributos=("tarifa_sem_tributos", "median"),
                    tusd=("tusd", "median"), te=("te", "median"),
                    vigencia_inicio=("vigencia_inicio", "max"),
                    vigencia_fim=("vigencia_fim", "max"),
                    reh=("reh", "first"))
               .reset_index())
    diag.linhas_uteis = len(saida)
    return saida, diag


# Métricas do SAMP que compõem a fatura de iluminação pública. O recurso é FORMATO
# LONGO: uma linha por (distribuidora, subclasse, competência, métrica), com o número
# em `VlrMercado` e o nome da métrica em `DscDetalheMercado`.
#
# `Receita Energia (R$)` JÁ INCLUI os tributos — ICMS, PIS/PASEP e COFINS são o
# detalhamento do que está DENTRO dela, cobrados "por dentro" como manda a legislação
# brasileira, e não parcelas a somar. Conferido na Cemig-D em 28/08/2026: receita ÷
# energia dá R$ 0,4729/kWh e, subtraindo os tributos, R$ 0,3424/kWh — contra
# R$ 0,3399/kWh da resolução homologatória, 0,7% de diferença. Somar os tributos por
# cima daria R$ 0,60/kWh e inflaria o custo de energia em 27%.
SAMP_ENERGIA = ("Energia TE (kWh)", "Energia Consumida (kWh)", "Energia TUSD (kWh)")
SAMP_RECEITA = "Receita Energia (R$)"
SAMP_BANDEIRAS = "Receita Bandeiras (R$)"
SAMP_TRIBUTOS = ("ICMS (R$)", "PIS/PASEP (R$)", "COFINS (R$)", "PIS/COFINS (R$)")

# Faixa física do R$/kWh faturado de iluminação pública no Brasil.
SAMP_TARIFA_MIN, SAMP_TARIFA_MAX = 0.20, 2.00


def normalizar_samp(bruto: pd.DataFrame, ano: Optional[int] = None
                    ) -> tuple[pd.DataFrame, Diagnostico]:
    """
    Recurso SAMP de um ano → tarifa faturada de iluminação pública, com e sem tributos.

        tarifa_com_tributos      = (Receita Energia + Bandeiras) ÷ energia faturada
        tarifa_samp_sem_tributos = idem, menos ICMS + PIS/PASEP + COFINS

    A segunda é CONFERÊNCIA, não produto: ela tem de reproduzir a tarifa da resolução
    homologatória, que vem de outra fonte. Batendo as duas, a primeira está certa. É o
    mesmo teste de preço unitário implícito que pegou o erro da tabela de energia do
    Recife: normalizar os dois lados pela mesma unidade antes de aceitar o par.

    AGREGAÇÃO POR MEDIANA MENSAL, NÃO POR SOMA. O SAMP tem meses com erro de ordem de
    grandeza na declaração: a Cemig-D declarou R$ 388 milhões de receita de IP em
    junho/2026 contra ~R$ 36 milhões em todos os outros meses, e a soma anual sozinha
    devolvia R$ 1,25/kWh. A mediana das competências é imune a isso, e o mês descartado
    continua aparecendo no diagnóstico.

    Filtra subclasse B4a (B4b é bulbo de lâmpada), `NomTipoMercado == "Regular"` e
    `DscOpcaoEnergia == "CATIVO"`: refaturamento entra negativo e os sistemas de
    compensação de GD contabilizam energia injetada, que não é consumo faturado de IP.

    É média da área de concessão inteira, não do município — serve de default de
    triagem; a fatura do município sempre ganha dela. E é ela que absorve, sem
    modelagem, a variação de ICMS entre estados e as isenções de IP.
    """
    diag = Diagnostico(f"SAMP {ano or ''}".strip(), linhas_lidas=len(bruto))
    vazio = pd.DataFrame(columns=["sigla_distribuidora", "tarifa_com_tributos"])
    if bruto.empty:
        diag.avisos.append("Recurso SAMP veio vazio.")
        return vazio, diag

    col = {
        "sigla": _coluna(bruto, "SigAgenteDistribuidora", "SigAgente", "Distribuidora"),
        "classe": _coluna(bruto, "DscClasseConsumoMercado", "DscClasseConsumo", "DscClasse"),
        "subclasse": _coluna(bruto, "DscSubClasseConsumidor", "DscSubClasse"),
        "detalhe": _coluna(bruto, "DscDetalheMercado", "DetalheMercado"),
        "valor": _coluna(bruto, "VlrMercado", "Valor"),
        "tipo": _coluna(bruto, "NomTipoMercado", "TipoMercado"),
        "opcao": _coluna(bruto, "DscOpcaoEnergia", "OpcaoEnergia"),
        "competencia": _coluna(bruto, "DatCompetencia", "Competencia"),
    }
    diag.colunas_faltando = [k for k, v in col.items() if v is None]
    for obrigatoria in ("sigla", "detalhe", "valor"):
        if col[obrigatoria] is None:
            diag.avisos.append(
                f"SAMP sem a coluna '{obrigatoria}'. Schema recebido: {list(bruto.columns)}")
            return vazio, diag

    df = bruto.copy()

    def _restringe(chave: str, teste, descricao: str) -> None:
        """Aplica um filtro só quando ele casa com alguma linha; senão avisa e segue."""
        nonlocal df
        if col[chave] is None:
            return
        mascara = teste(df[col[chave]])
        if mascara.any():
            df = df[mascara]
        else:
            diag.avisos.append(f"Filtro de {descricao} não casou; seguiu sem ele.")

    _restringe("classe",
               lambda s: s.map(_chave).str.contains(_chave("iluminacao"), na=False),
               "classe de consumo")
    if df.empty:
        diag.avisos.append("Nenhuma linha de iluminação pública.")
        return vazio, diag
    _restringe("subclasse",
               lambda s: s.map(_chave).str.endswith(_chave(config.ANEEL_SUBCLASSE_IP),
                                                    na=False),
               "subclasse B4a")
    _restringe("tipo", lambda s: s.map(_chave) == _chave("Regular"), "tipo de mercado")
    _restringe("opcao", lambda s: s.map(_chave) == _chave("CATIVO"), "opção de energia")

    df = df.copy()
    df["_valor"] = _numero(df[col["valor"]])
    df["sigla_distribuidora"] = df[col["sigla"]].astype(str).str.strip().str.upper()
    df["_competencia"] = (df[col["competencia"]].astype(str) if col["competencia"]
                          else "unica")
    df["_metrica"] = df[col["detalhe"]].map(_chave)

    painel = (df.pivot_table(index=["sigla_distribuidora", "_competencia"],
                             columns="_metrica", values="_valor", aggfunc="sum")
                .reset_index())

    def _metrica(*nomes: str) -> pd.Series:
        """Soma das métricas pedidas que existirem no painel; NA quando nenhuma existe."""
        presentes = [_chave(n) for n in nomes if _chave(n) in painel.columns]
        if not presentes:
            return pd.Series(pd.NA, index=painel.index, dtype="Float64")
        return painel[presentes].sum(axis=1, min_count=1).astype("Float64")

    energia = pd.Series(pd.NA, index=painel.index, dtype="Float64")
    for nome in SAMP_ENERGIA:
        energia = energia.combine_first(_metrica(nome))

    painel = painel.assign(
        energia_kwh=energia,
        receita_energia=_metrica(SAMP_RECEITA),
        receita_bandeiras=_metrica(SAMP_BANDEIRAS).fillna(0.0),
        tributos=_metrica(*SAMP_TRIBUTOS).fillna(0.0),
    )
    painel = painel[painel["energia_kwh"].notna() & (painel["energia_kwh"] > 0)
                    & painel["receita_energia"].notna() & (painel["receita_energia"] > 0)]
    if painel.empty:
        diag.avisos.append(
            "Nenhuma competência com energia e receita positivas. Métricas presentes: "
            f"{sorted(df[col['detalhe']].astype(str).unique())[:25]}")
        return vazio, diag

    painel["receita_com_tributos"] = painel["receita_energia"] + painel["receita_bandeiras"]
    painel["_com"] = painel["receita_com_tributos"] / painel["energia_kwh"]
    painel["_sem"] = ((painel["receita_com_tributos"] - painel["tributos"])
                      / painel["energia_kwh"])

    # A competência fora da faixa física é descartada ANTES de agregar: é ela que carrega
    # o erro de declaração, e mantê-la contamina até a mediana quando há poucos meses.
    plausivel = painel["_com"].between(SAMP_TARIFA_MIN, SAMP_TARIFA_MAX)
    if (~plausivel).any():
        diag.avisos.append(
            f"{int((~plausivel).sum())} competências descartadas por tarifa fora de "
            f"R$ {SAMP_TARIFA_MIN:.2f}–{SAMP_TARIFA_MAX:.2f}/kWh (erro de declaração).")
    painel = painel[plausivel]
    if painel.empty:
        diag.avisos.append("Nenhuma competência plausível sobrou.")
        return vazio, diag

    ag = (painel.groupby("sigla_distribuidora")
                .agg(tarifa_com_tributos=("_com", "median"),
                     tarifa_samp_sem_tributos=("_sem", "median"),
                     energia_kwh=("energia_kwh", "sum"),
                     receita_energia=("receita_energia", "sum"),
                     receita_bandeiras=("receita_bandeiras", "sum"),
                     tributos=("tributos", "sum"),
                     receita_com_tributos=("receita_com_tributos", "sum"),
                     competencias=("_com", "size"))
                .reset_index())
    ag["mercado_mwh"] = ag["energia_kwh"] / 1000.0
    ag["ano_samp"] = ano

    diag.linhas_uteis = len(ag)
    return ag, diag


# ── leitura pelo portal ─────────────────────────────────────────────────────

def carregar() -> pd.DataFrame:
    """
    Tarifas B4a por distribuidora, do parquet versionado. Vazio se o ETL não rodou.

    Nunca vai à rede: o Streamlit lê só o derivado. Ver `etl_aneel.py`.
    """
    caminho = config.TARIFAS_B4A
    if not caminho.exists():
        return pd.DataFrame(columns=COLUNAS_TARIFAS)
    try:
        return pd.read_parquet(caminho)
    except Exception:
        return pd.DataFrame(columns=COLUNAS_TARIFAS)


def tarifa_do_municipio(cod_ibge: str, parque: Optional[pd.DataFrame] = None,
                        tarifas: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Tarifa B4a do município, pela distribuidora que atende o parque dele.

    O join já existe: a BDGD traz `cod_distribuidora` (código ANEEL) e `distribuidora`
    por município. Municípios atendidos por mais de uma distribuidora aparecem com o
    nome composto ("Cemig-D + Light"); nesse caso a tarifa sai da MÉDIA das que casarem,
    e o resultado vem marcado com `composta=True` para a página avisar.

    Devolve None quando não há tarifa apurada — e aí a página fica com o default.
    """
    from . import bdgd

    tarifas = carregar() if tarifas is None else tarifas
    if tarifas.empty:
        return None
    if parque is None:
        parque = bdgd.carregar_municipios()
    if parque is None or parque.empty:
        return None

    linha = parque[parque["codigo_municipio"].astype(str) == str(cod_ibge)]
    if linha.empty:
        return None
    linha = linha.iloc[0]

    achados = pd.DataFrame()
    cod = linha.get("cod_distribuidora")
    if pd.notna(cod) and "cod_distribuidora" in tarifas.columns:
        achados = tarifas[tarifas["cod_distribuidora"].astype(str) == str(cod)]
    if achados.empty:
        nomes = [n.strip() for n in str(linha.get("distribuidora") or "").split("+")]
        chaves = {_chave(n) for n in nomes if n}
        if chaves and "distribuidora" in tarifas.columns:
            achados = tarifas[tarifas["distribuidora"].map(_chave).isin(chaves)]
        if achados.empty and chaves and "sigla_distribuidora" in tarifas.columns:
            achados = tarifas[tarifas["sigla_distribuidora"].map(_chave).isin(chaves)]
    if achados.empty:
        return None

    com = pd.to_numeric(achados.get("tarifa_com_tributos"), errors="coerce").dropna()
    sem = pd.to_numeric(achados.get("tarifa_sem_tributos"), errors="coerce").dropna()
    return {
        "tarifa_com_tributos": float(com.mean()) if len(com) else None,
        "tarifa_sem_tributos": float(sem.mean()) if len(sem) else None,
        "distribuidora": achados.iloc[0].get("distribuidora")
                         or achados.iloc[0].get("sigla_distribuidora"),
        "reh": achados.iloc[0].get("reh"),
        "vigencia_inicio": achados.iloc[0].get("vigencia_inicio"),
        "ano_samp": achados.iloc[0].get("ano_samp"),
        "composta": len(achados) > 1,
    }
