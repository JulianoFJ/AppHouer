"""
Cliente do SICONFI (API Data Lake do Tesouro Nacional) para a receita de COSIP.

Extrai do DCA — Anexo I-C (Receitas Orçamentárias) a Contribuição para o Custeio do
Serviço de Iluminação Pública, por município e exercício.

ATENÇÃO — QUEBRA DE SÉRIE NO PLANO DE CONTAS (verificado na API em 28/08/2026):
    até 2017      : "1.2.3.0.00.00.00 - Contribuição para Custeio do Serviço de I. P."
    2018 em diante: "1.2.4.0.00.0.0  - Contribuição para o Custeio do Serviço de I. P."
Muda o código E o texto (o artigo "o" só existe na redação nova). Filtrar pelo texto
literal da conta devolve VAZIO para 2017 e anteriores, e esse vazio costuma ser lido
como "o município não arrecada COSIP" — apagando exercícios inteiros da série. Por isso
o casamento aqui é por padrão de nome + classe de conta orçamentária.

O resultado sempre carrega um `status`, para distinguir "arrecadou zero" de "não
declarou" e de "falha de API". Sem essa distinção, média e série histórica ficam
contaminadas por ausência de dado.
"""

from __future__ import annotations

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

from . import config

BASE_URL = "http://apidatalake.tesouro.gov.br/ords/siconfi/tt"
ANEXO = "DCA-Anexo I-C"
NR_PERIODO = 6

PADRAO_CONTA_COSIP = re.compile(r"contribuic\w*\s+(?:para\s+)?(?:o\s+)?(?:custeio.*)?ilumina")
PREFIXO_RECEITA_ORCAMENTARIA = "ro"   # exclui a intraorçamentária "RI", que duplicaria

COL_BRUTA = "receitas brutas realizadas"
COL_DEDUCAO = "outras deducoes da receita"
COL_LIQUIDA = "receitas realizadas liquidas"

# A dedução é declarada como valor positivo no layout do Anexo I-C. Usar o módulo evita
# que um ente que declarou com sinal negativo infle artificialmente a receita líquida.
DEDUCAO_EM_MODULO = True

HEADERS = {
    "User-Agent": "Plataforma Houer IP - Hub de Municipios (uso responsavel)",
    "Accept": "application/json",
}
TIMEOUT = 60
LIMIT_PAGINA = 500
SLEEP_PAGINA = 0.35
WORKERS_PADRAO = 4

COLUNAS = [
    "ano_exercicio", "codigo_municipio", "municipio", "uf",
    "cod_conta", "conta", "receita_bruta", "deducoes", "cosip_liquida",
    "cosip_liquida_api", "populacao", "status", "observacao",
]


# ── utilitários ──────────────────────────────────────────────────────────────

def normalizar(texto: Any) -> str:
    if texto is None:
        return ""
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def so_digitos(valor: Any) -> str:
    """Trata códigos IBGE vindos como float do pandas (3550308.0), texto ou com máscara."""
    if valor is None:
        return ""
    s = str(valor).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"\D", "", s)


def _sessao() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


SESSAO = _sessao()


def _paginar(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Percorre a paginação ORDS (offset/limit)."""
    itens: List[Dict[str, Any]] = []
    offset = 0
    url = f"{BASE_URL}{endpoint}"
    while True:
        resp = SESSAO.get(url, params=dict(params, offset=offset, limit=LIMIT_PAGINA),
                          timeout=TIMEOUT)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        lote = resp.json().get("items", [])
        if not lote:
            break
        itens.extend(lote)
        if len(lote) < LIMIT_PAGINA:
            break
        offset += LIMIT_PAGINA
        time.sleep(SLEEP_PAGINA)
    return itens


# ── cadastro de entes ────────────────────────────────────────────────────────

_ENTES: Optional[pd.DataFrame] = None


def carregar_entes(forcar: bool = False) -> pd.DataFrame:
    """
    Cadastro de municípios do SICONFI (cod_ibge, ente, uf, população). Persistido em
    parquet no pacote — são ~5.600 linhas, e evita uma chamada de rede por sessão.
    """
    global _ENTES
    if _ENTES is not None and not forcar:
        return _ENTES

    cache = config.ENTES_CACHE
    if cache.exists() and not forcar:
        try:
            _ENTES = pd.read_parquet(cache)
            return _ENTES
        except Exception:
            pass

    try:
        itens = _paginar("/entes", {})
    except requests.exceptions.RequestException:
        itens = []

    df = pd.DataFrame(itens)
    if not df.empty:
        df["cod_ibge"] = df["cod_ibge"].apply(so_digitos)
        if "esfera" in df.columns:
            df = df[df["esfera"].astype(str).str.upper().str.startswith("M")]
        df = df[df["cod_ibge"].str.len() == 7].copy()
        df["_ente_norm"] = df["ente"].apply(normalizar)
        try:
            config.garantir_pastas()
            df.to_parquet(cache, index=False)
        except Exception:
            pass
    _ENTES = df
    return df


def buscar_municipio(termo: str, uf: Optional[str] = None) -> pd.DataFrame:
    """Resolve código IBGE (7 dígitos) ou nome parcial em municípios."""
    df = carregar_entes()
    if df.empty:
        return pd.DataFrame(columns=["cod_ibge", "ente", "uf"])

    codigo = so_digitos(termo)
    if len(codigo) == 7:
        return df[df["cod_ibge"] == codigo][["cod_ibge", "ente", "uf"]]

    alvo = normalizar(termo)
    if not alvo:
        return pd.DataFrame(columns=["cod_ibge", "ente", "uf"])
    sel = df[df["_ente_norm"].str.contains(re.escape(alvo), na=False)]
    if uf:
        sel = sel[sel["uf"].astype(str).str.upper() == uf.strip().upper()]
    exatos = sel[sel["_ente_norm"] == alvo]          # nome exato tem prioridade
    sel = pd.concat([exatos, sel[~sel.index.isin(exatos.index)]])
    return sel[["cod_ibge", "ente", "uf"]]


def identificar(cod_ibge: str) -> Dict[str, str]:
    df = carregar_entes()
    if df.empty:
        return {"ente": "", "uf": ""}
    linha = df[df["cod_ibge"] == cod_ibge]
    if linha.empty:
        return {"ente": "", "uf": ""}
    return {"ente": str(linha.iloc[0].get("ente", "")), "uf": str(linha.iloc[0].get("uf", ""))}


# ── extração da COSIP ────────────────────────────────────────────────────────

def _e_linha_cosip(linha: Dict[str, Any]) -> bool:
    if not PADRAO_CONTA_COSIP.search(normalizar(linha.get("conta"))):
        return False
    cod = normalizar(linha.get("cod_conta"))
    return cod.startswith(PREFIXO_RECEITA_ORCAMENTARIA) or not cod


def _raiz_conta(cod_conta: Any) -> str:
    """'RO1.2.4.0.00.0.0' -> '124'. Identifica a conta sintética da árvore."""
    digitos = re.sub(r"\D", "", str(cod_conta or ""))
    return digitos.rstrip("0") or digitos


def _somente_raizes(linhas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Descarta desdobramentos (principal, dívida ativa) quando a conta sintética também
    foi declarada — somar ambos contaria o valor duas vezes.
    """
    chaves = {_raiz_conta(l.get("cod_conta")) for l in linhas}
    raizes = {c for c in chaves if not any(c != o and c.startswith(o) for o in chaves)}
    return [l for l in linhas if _raiz_conta(l.get("cod_conta")) in raizes]


def consultar(cod_ibge: str, ano: int) -> Dict[str, Any]:
    """
    Um par (município, ano). Devolve SEMPRE um registro, com `status`:
      OK · SEM_DADO_NO_ANEXO · ENTE_NAO_DECLAROU · ERRO_API
    """
    cod_ibge = so_digitos(cod_ibge)
    ident = identificar(cod_ibge)
    reg: Dict[str, Any] = dict.fromkeys(COLUNAS)
    reg.update({
        "ano_exercicio": ano, "codigo_municipio": cod_ibge,
        "municipio": ident["ente"], "uf": ident["uf"],
        "cod_conta": "", "conta": "", "status": "", "observacao": "",
    })

    params = {"an_exercicio": ano, "id_ente": cod_ibge,
              "no_anexo": ANEXO, "nr_periodo": NR_PERIODO}
    try:
        dados = _paginar("/dca", params)
    except requests.exceptions.RequestException as exc:
        reg["status"] = "ERRO_API"
        reg["observacao"] = f"{type(exc).__name__}: {exc}"
        return reg

    if not dados:
        reg["status"] = "ENTE_NAO_DECLAROU"
        reg["observacao"] = "API não retornou o anexo para este ente/exercício."
        return reg

    amostra = dados[0]
    reg["municipio"] = reg["municipio"] or amostra.get("instituicao") or ""
    reg["uf"] = reg["uf"] or amostra.get("uf") or ""
    try:
        reg["populacao"] = int(amostra.get("populacao") or 0) or None
    except (TypeError, ValueError):
        reg["populacao"] = None

    linhas = [l for l in dados if _e_linha_cosip(l)]
    if not linhas:
        reg["status"] = "SEM_DADO_NO_ANEXO"
        reg["observacao"] = "Anexo disponível, porém sem a rubrica de COSIP."
        return reg

    linhas = _somente_raizes(linhas)
    bruta = deducao = liquida_api = None
    for linha in linhas:
        col = normalizar(linha.get("coluna"))
        try:
            valor = float(linha.get("valor") or 0.0)
        except (TypeError, ValueError):
            continue
        if COL_BRUTA in col:
            bruta = (bruta or 0.0) + valor
            reg["cod_conta"] = linha.get("cod_conta", "")
            reg["conta"] = linha.get("conta", "")
        elif COL_DEDUCAO in col:
            deducao = (deducao or 0.0) + valor
        elif COL_LIQUIDA in col:
            liquida_api = (liquida_api or 0.0) + valor

    if not reg["cod_conta"]:
        reg["cod_conta"] = linhas[0].get("cod_conta", "")
        reg["conta"] = linhas[0].get("conta", "")

    bruta_v = bruta or 0.0
    ded_v = abs(deducao or 0.0) if DEDUCAO_EM_MODULO else (deducao or 0.0)
    reg["receita_bruta"] = bruta_v
    reg["deducoes"] = ded_v
    reg["cosip_liquida"] = bruta_v - ded_v
    reg["cosip_liquida_api"] = liquida_api
    reg["status"] = "OK"

    avisos = []
    if len({_raiz_conta(l.get("cod_conta")) for l in linhas}) > 1:
        avisos.append("Mais de uma conta raiz de COSIP declarada; valores somados.")
    if liquida_api is not None and abs(liquida_api - reg["cosip_liquida"]) > 0.01:
        fmt = lambda v: f"{v:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
        avisos.append(f"Divergência vs. coluna líquida da API: R$ {fmt(liquida_api)} "
                      f"declarado x R$ {fmt(reg['cosip_liquida'])} calculado.")
    reg["observacao"] = " ".join(avisos)
    return reg


def consultar_muitos(
    codigos: Sequence[str],
    anos: Sequence[int],
    workers: int = WORKERS_PADRAO,
    progresso=None,
) -> pd.DataFrame:
    """Todos os pares (município, ano). `progresso` é callable(feitos, total, texto)."""
    codigos = [c for c in (so_digitos(x) for x in codigos) if len(c) == 7]
    pares = [(c, a) for a in anos for c in codigos]
    if not pares:
        return pd.DataFrame(columns=COLUNAS)

    carregar_entes()   # aquece o cache antes do paralelismo
    resultados: List[Dict[str, Any]] = []
    feitos = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futuros = {pool.submit(consultar, c, a): (c, a) for c, a in pares}
        for fut in as_completed(futuros):
            cod, ano = futuros[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = dict.fromkeys(COLUNAS)
                res.update({"ano_exercicio": ano, "codigo_municipio": cod,
                            "status": "ERRO_API",
                            "observacao": f"{type(exc).__name__}: {exc}"})
            resultados.append(res)
            feitos += 1
            if progresso:
                progresso(feitos, len(pares),
                          f"{res.get('municipio') or cod}/{ano} → {res['status']}")

    df = pd.DataFrame(resultados)
    for col in COLUNAS:
        if col not in df.columns:
            df[col] = None
    return (df[COLUNAS]
            .sort_values(["uf", "municipio", "codigo_municipio", "ano_exercicio"],
                         kind="stable")
            .reset_index(drop=True))


# ── cache local de consultas ─────────────────────────────────────────────────

def _caminho_cache() -> "pd.io.parquet.os.PathLike":
    return config.SICONFI_CACHE / "cosip.parquet"


def carregar_cache() -> pd.DataFrame:
    caminho = _caminho_cache()
    if caminho.exists():
        try:
            return pd.read_parquet(caminho)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUNAS)


def gravar_cache(df: pd.DataFrame) -> None:
    """Mescla no cache, mantendo o registro mais recente de cada par município/ano."""
    if df.empty:
        return
    config.garantir_pastas()
    atual = carregar_cache()
    combinado = pd.concat([atual, df], ignore_index=True)
    combinado = combinado.drop_duplicates(
        subset=["codigo_municipio", "ano_exercicio"], keep="last"
    )
    try:
        combinado.to_parquet(_caminho_cache(), index=False)
    except Exception:
        pass


def consultar_com_cache(
    codigos: Sequence[str],
    anos: Sequence[int],
    workers: int = WORKERS_PADRAO,
    progresso=None,
    revalidar: bool = False,
) -> pd.DataFrame:
    """
    Igual a `consultar_muitos`, mas reaproveita o cache local. Consultas que falharam
    (ERRO_API) nunca são cacheadas como resposta válida — são sempre refeitas.
    """
    codigos = [c for c in (so_digitos(x) for x in codigos) if len(c) == 7]
    anos = list(anos)
    cache = carregar_cache()

    if revalidar or cache.empty:
        faltantes = [(c, a) for a in anos for c in codigos]
        aproveitado = pd.DataFrame(columns=COLUNAS)
    else:
        cache = cache[cache["status"] != "ERRO_API"]
        chaves_cache = set(zip(cache["codigo_municipio"], cache["ano_exercicio"]))
        faltantes = [(c, a) for a in anos for c in codigos if (c, a) not in chaves_cache]
        aproveitado = cache[
            cache["codigo_municipio"].isin(codigos) & cache["ano_exercicio"].isin(anos)
        ]

    if faltantes:
        novos = consultar_muitos(
            sorted({c for c, _ in faltantes}),
            sorted({a for _, a in faltantes}),
            workers=workers, progresso=progresso,
        )
        gravar_cache(novos)
        resultado = pd.concat([aproveitado, novos], ignore_index=True)
    else:
        resultado = aproveitado.copy()

    resultado = resultado.drop_duplicates(
        subset=["codigo_municipio", "ano_exercicio"], keep="last"
    )
    return (resultado
            .sort_values(["uf", "municipio", "codigo_municipio", "ano_exercicio"],
                         kind="stable")
            .reset_index(drop=True))
