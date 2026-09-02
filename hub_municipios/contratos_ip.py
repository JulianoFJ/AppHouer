"""
Contratos municipais de iluminação pública — PNCP (Portal Nacional de Contratações).

Complementa `despesas.py`. A despesa do SICONFI é classificada por **função de governo**,
que não isola iluminação pública: o melhor que se consegue lá é um envelope (25.752
Energia Elétrica, ou 15.452 Serviços Urbanos). O PNCP resolve pelo outro lado — ele
publica o **contrato**, com objeto descrito em texto livre, valor global e vigência.
É onde aparece, nominalmente, "manutenção do parque de iluminação pública".

## Por que o resultado é evidência, e não número de cálculo

Três limitações medidas na API em 01/09/2026, todas tratadas aqui mas nenhuma
eliminável:

  1. **A busca é fuzzy.** `q="manutenção iluminação pública"` casa por relevância, não
     por termo obrigatório: entre os contratos de IP de MG veio um de cessão de "Salão
     de Festas" em Belo Horizonte. Por isso todo resultado passa por `_e_relevante`,
     que exige o radical de iluminação no objeto — e mesmo assim o que sai é uma lista
     para conferência humana, não um total somado às cegas.
  2. **`municipio_id` do PNCP não é o código IBGE.** Belo Horizonte é 2310 no PNCP e
     3106200 no IBGE. Não há, na API de busca, filtro por código IBGE; o casamento é
     por nome normalizado + UF, com todo o risco de homônimo que isso carrega
     (há Bom Jesus em oito estados).
  3. **Rate limiting agressivo.** Chamadas seguidas levam `ConnectionReset` sem
     resposta HTTP — não é 429, a conexão simplesmente cai. Daí a sessão com retry e
     backoff exponencial, e o intervalo entre chamadas.

O contrato encontrado também não é necessariamente o vigente nem o único: município
grande costuma ter vários (manutenção, eficientização, telegestão), e a vigência
precisa ser lida. Some-se a isso que a adesão ao PNCP só passou a ser obrigatória com
a Lei 14.133/2021, então contrato antigo pode simplesmente não estar lá.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


URL_BUSCA = "https://pncp.gov.br/api/search/"
TIMEOUT = 60
PAUSA_ENTRE_CHAMADAS = 1.5
TENTATIVAS_CONEXAO = 4

# Termos que caracterizam contrato de iluminação pública. O radical "ilumina" é o que
# manda; os demais entram para pegar contrato descrito por objeto (parque, luminária).
TERMOS_IP = ["ilumina", "luminaria", "luminária", "parque luminotecnico",
             "parque luminotécnico", "iluminacao publica", "iluminação pública"]
# Termos que denunciam contrato de OUTRA natureza que casou por relevância.
TERMOS_EXCLUSAO = ["salao de festas", "salão de festas", "show", "iluminacao cenica",
                   "iluminação cênica", "natal"]

CONSULTA_PADRAO = "manutenção iluminação pública"

COLUNAS = [
    "municipio", "uf", "orgao", "orgao_cnpj", "objeto",
    "valor_global", "data_inicio_vigencia", "data_fim_vigencia",
    "modalidade", "ano", "numero_controle_pncp", "link",
]


def _sessao() -> requests.Session:
    """
    Sessão com retry e backoff — obrigatória, não conveniência.

    O PNCP derruba conexões seguidas sem devolver status HTTP (`ConnectionReset`), o
    que o `Retry` do urllib3 sozinho não cobre: ele reage a status de erro, e aqui não
    há status. Por isso o retry do urllib3 é combinado com o laço em `_buscar_pagina`.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Plataforma IP - Hub de Municipios (uso responsavel)",
        "Accept": "application/json",
    })
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"])))
    return s


SESSAO = _sessao()


def _slug(texto: Any) -> str:
    t = unicodedata.normalize("NFD", str(texto or ""))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()


def _e_relevante(objeto: str, orgao: str = "") -> bool:
    """Filtra o ruído da busca fuzzy: exige o radical de iluminação no objeto."""
    alvo = _slug(objeto) + " " + _slug(orgao)
    if any(_slug(t) in alvo for t in TERMOS_EXCLUSAO):
        return False
    return any(_slug(t) in alvo for t in TERMOS_IP)


def _formulacoes(consulta: str, municipio: Optional[str]) -> List[str]:
    """
    Variações da consulta a tentar, em ordem de precisão.

    O ranqueamento do PNCP é sensível à formulação: procurando o contrato de Itabira,
    "manutenção iluminação pública" filtrado pelo município devolveu zero, enquanto
    "MUNICIPIO DE ITABIRA iluminação" devolveu 21 registros do mesmo município. Incluir
    o nome do órgão na consulta é o que faz o portal ranquear os contratos daquela
    prefeitura acima dos milhares de contratos homônimos do país inteiro.
    """
    if not municipio:
        return [consulta]
    return [
        f"MUNICIPIO DE {municipio} iluminação pública",
        f"{municipio} {consulta}",
        consulta,
    ]


def _buscar_pagina(consulta: str, uf: Optional[str], pagina: int,
                   tam_pagina: int) -> List[Dict[str, Any]]:
    params = {"q": consulta, "tipos_documento": "contrato", "pagina": pagina,
              "tam_pagina": tam_pagina, "ordenacao": "-data"}
    if uf:
        params["ufs"] = uf
    for tentativa in range(TENTATIVAS_CONEXAO):
        try:
            resp = SESSAO.get(URL_BUSCA, params=params, timeout=TIMEOUT)
            if resp.status_code != 200:
                return []
            return resp.json().get("items", []) or []
        except requests.exceptions.RequestException:
            if tentativa == TENTATIVAS_CONEXAO - 1:
                raise
            time.sleep(2.0 * (tentativa + 1))
    return []


def _montar(item: Dict[str, Any]) -> Dict[str, Any]:
    controle = item.get("numero_controle_pncp") or ""
    return {
        "municipio": item.get("municipio_nome"),
        "uf": item.get("uf"),
        "orgao": item.get("orgao_nome"),
        "orgao_cnpj": item.get("orgao_cnpj"),
        "objeto": (item.get("description") or "").strip(),
        "valor_global": item.get("valor_global"),
        "data_inicio_vigencia": item.get("data_inicio_vigencia"),
        "data_fim_vigencia": item.get("data_fim_vigencia"),
        "modalidade": item.get("modalidade_licitacao_nome"),
        "ano": item.get("ano"),
        "numero_controle_pncp": controle,
        "link": f"https://pncp.gov.br/app/contratos/{controle}" if controle else None,
    }


def buscar(
    municipio: Optional[str] = None,
    uf: Optional[str] = None,
    consulta: str = CONSULTA_PADRAO,
    paginas: int = 2,
    tam_pagina: int = 20,
) -> pd.DataFrame:
    """
    Busca contratos de iluminação pública no PNCP.

    Args:
        municipio: nome do município para filtrar o resultado (casamento por nome
            normalizado — a API de busca não aceita código IBGE).
        uf: sigla da UF, usada como filtro na própria API (reduz muito o ruído).
        consulta: termo de busca textual.
        paginas: quantas páginas percorrer. Cada página é uma chamada, e o PNCP
            limita a taxa — não aumentar sem necessidade.

    Returns:
        DataFrame com as colunas de `COLUNAS`, já filtrado por relevância e ordenado
        do contrato mais recente para o mais antigo. Vazio se nada for encontrado ou
        se a API não responder.
    """
    alvo_municipio = _slug(municipio) if municipio else None
    registros: List[Dict[str, Any]] = []

    for consulta_atual in _formulacoes(consulta, municipio):
        for pagina in range(1, max(paginas, 1) + 1):
            try:
                itens = _buscar_pagina(consulta_atual, uf, pagina, tam_pagina)
            except requests.exceptions.RequestException:
                break
            if not itens:
                break
            for item in itens:
                if not _e_relevante(item.get("description") or "",
                                    item.get("orgao_nome") or ""):
                    continue
                if alvo_municipio and _slug(item.get("municipio_nome")) != alvo_municipio:
                    continue
                registros.append(_montar(item))
            if len(itens) < tam_pagina:
                break
            time.sleep(PAUSA_ENTRE_CHAMADAS)
        if alvo_municipio and registros:
            break               # já achou o município: não gasta mais chamada
        time.sleep(PAUSA_ENTRE_CHAMADAS)

    df = pd.DataFrame(registros, columns=COLUNAS)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["numero_controle_pncp"], keep="first")
    return df.sort_values("data_inicio_vigencia", ascending=False).reset_index(drop=True)


def resumir(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Resumo dos contratos encontrados, para o cabeçalho da UI e para o slide.

    `valor_total` soma os contratos listados e por isso NÃO é o gasto anual do
    município com IP: contratos têm vigências diferentes, podem se sobrepor, podem
    ser aditivos de um mesmo objeto e o valor global cobre todo o prazo, não um ano.
    Serve como ordem de grandeza da contratação, sempre com essa ressalva à vista.
    """
    if df is None or df.empty:
        return {"quantidade": 0, "valor_total": None, "maior_contrato": None,
                "vigencia_mais_recente": None}
    valores = pd.to_numeric(df["valor_global"], errors="coerce")
    return {
        "quantidade": int(len(df)),
        "valor_total": float(valores.sum()) if valores.notna().any() else None,
        "maior_contrato": float(valores.max()) if valores.notna().any() else None,
        "vigencia_mais_recente": df["data_inicio_vigencia"].dropna().max()
        if df["data_inicio_vigencia"].notna().any() else None,
    }


def _data(valor) -> Optional[pd.Timestamp]:
    try:
        convertida = pd.to_datetime(valor, errors="coerce")
        return None if pd.isna(convertida) else convertida
    except Exception:
        return None


def vigentes(df: pd.DataFrame, referencia: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Filtra os contratos vigentes na data de referência (hoje, por padrão).

    Contrato encerrado não serve de comparação com a contraprestação proposta — o
    município pode já ter licitado outro, ou estar sem contrato. Sem data de fim, o
    contrato é mantido: é mais comum a data faltar do que o contrato ser eterno, e
    descartar por ausência de campo esconderia contratação real.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(columns=COLUNAS)
    hoje = referencia or pd.Timestamp.today().normalize()
    fim = pd.to_datetime(df["data_fim_vigencia"], errors="coerce")
    return df[fim.isna() | (fim >= hoje)].reset_index(drop=True)


def valor_mensal_equivalente(contrato) -> Optional[float]:
    """
    Converte o valor global do contrato em equivalente mensal pelo prazo contratual.

    O PNCP publica o **valor global**, que cobre toda a vigência — comparar esse número
    com uma contraprestação mensal seria erro de ordem de grandeza (um contrato de 12
    meses pareceria doze vezes mais caro do que é). A divisão pelo prazo é a única forma
    de pôr os dois na mesma base, e o resultado é sempre rotulado como *equivalente*,
    porque desembolso real raramente é linear.

    Devolve None quando falta valor ou quando o prazo não é apurável.
    """
    try:
        valor = float(contrato.get("valor_global") or 0)
    except (TypeError, ValueError):
        return None
    if valor <= 0:
        return None
    inicio, fim = _data(contrato.get("data_inicio_vigencia")), _data(contrato.get("data_fim_vigencia"))
    if inicio is None or fim is None or fim <= inicio:
        return None
    meses = (fim - inicio).days / 30.44
    return valor / meses if meses >= 1 else None


def principal(df: pd.DataFrame) -> Optional[dict]:
    """
    Escolhe o contrato mais representativo para citar num slide.

    Critério: entre os vigentes, o de maior valor global — que é o contrato de
    manutenção do parque, e não a compra pontual de luminárias que também casa na
    busca. Devolve o registro acrescido de `valor_mensal_equivalente`, ou None.
    """
    candidatos = vigentes(df)
    if candidatos is None or candidatos.empty:
        return None
    valores = pd.to_numeric(candidatos["valor_global"], errors="coerce")
    if not valores.notna().any():
        return None
    escolhido = candidatos.loc[valores.idxmax()].to_dict()
    escolhido["valor_mensal_equivalente"] = valor_mensal_equivalente(escolhido)
    return escolhido


# ── Cache em disco ───────────────────────────────────────────────────────────
def _caminho_cache():
    return config.SICONFI_CACHE / "contratos_ip_pncp.parquet"


def carregar_cache() -> pd.DataFrame:
    caminho = _caminho_cache()
    if caminho.exists():
        try:
            return pd.read_parquet(caminho)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUNAS + ["_chave"])


def buscar_com_cache(municipio: str, uf: str, consulta: str = CONSULTA_PADRAO,
                     revalidar: bool = False, **kwargs) -> pd.DataFrame:
    """
    Igual a `buscar`, reaproveitando o cache local.

    O cache é por (município, UF, consulta) e existe sobretudo para poupar a API —
    que, além de limitar a taxa, é a parte mais lenta da montagem da apresentação.
    """
    chave = f"{_slug(municipio)}|{_slug(uf)}|{_slug(consulta)}"
    cache = carregar_cache()
    if not revalidar and not cache.empty and "_chave" in cache.columns:
        guardado = cache[cache["_chave"] == chave]
        if not guardado.empty:
            return guardado.drop(columns=["_chave"]).reset_index(drop=True)

    df = buscar(municipio=municipio, uf=uf, consulta=consulta, **kwargs)
    if not df.empty:
        config.garantir_pastas()
        novo = df.copy()
        novo["_chave"] = chave
        combinado = pd.concat([cache[cache.get("_chave") != chave] if not cache.empty
                               else cache, novo], ignore_index=True)
        try:
            combinado.to_parquet(_caminho_cache(), index=False)
        except Exception:
            pass
    return df


__all__ = [
    "COLUNAS", "CONSULTA_PADRAO", "TERMOS_IP",
    "buscar", "buscar_com_cache", "resumir", "carregar_cache",
    "vigentes", "valor_mensal_equivalente", "principal",
]
