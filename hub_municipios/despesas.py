"""
Despesa municipal com energia elétrica e serviços urbanos — SICONFI, DCA Anexo I-E.

Complementa `siconfi.py`, que traz a **receita** de COSIP (Anexo I-C). Sem a despesa, a
"sobra" calculada pelo Hub é uma conta de um lado só: sabe-se quanto o município
arrecada de CIP, mas não quanto ele efetivamente gasta com a luz. É essa despesa que
torna palpável a pergunta que a prefeitura sempre faz — "a CIP paga a conta?".

Mesma API do datalake do Tesouro já usada pelo módulo de receita, só mudando o anexo.

## O que o Anexo I-E entrega, e o que ele NÃO entrega

A classificação é **funcional** (função e subfunção do orçamento), não por objeto de
contrato. Duas linhas interessam:

  - **25.752 — Energia Elétrica**: toda a energia elétrica paga pelo município. Inclui
    a iluminação pública, mas também escolas, postos de saúde, prédios administrativos,
    poços e bombas. É um **teto** para o gasto de IP, nunca o valor exato.
  - **15.452 — Serviços Urbanos**: onde costuma ser empenhado o contrato de manutenção
    da IP — junto com limpeza urbana, capina, praças, cemitério e feiras. Também é um
    envelope, e mais poluído que o de energia.

Nenhuma das duas isola a iluminação pública. Isso é limitação da própria classificação
funcional, não do módulo: o plano de contas nacional não tem subfunção de IP. Por isso
todo indicador derivado daqui sai rotulado como envelope, e a UI diz de onde veio.
Quem quiser o contrato de manutenção nominal precisa do PNCP — ver `contratos_ip.py`.

## Qual coluna de execução usar

A API devolve o mesmo valor em quatro estágios: Empenhadas, Liquidadas, Pagas e
Inscrição de Restos a Pagar. O padrão aqui é **Liquidadas**, porque liquidação é o
reconhecimento do fato gerador (o serviço foi prestado) e é o estágio comparável com a
receita arrecadada do Anexo I-C. "Empenhadas" superestima (empenho não executado é
anulado) e "Pagas" subestima em município que empurra pagamento para restos a pagar —
diferença que em Mateus Leme/2023 foi de 24% entre empenhado e pago só na energia.
A coluna é parâmetro, para quem preferir outro critério.
"""

from __future__ import annotations

import concurrent.futures as futuros
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

from . import config, siconfi


ANEXO = "DCA-Anexo I-E"
NR_PERIODO = 6

# Colunas de execução orçamentária devolvidas pela API, do mais amplo ao mais restrito.
COLUNA_EMPENHADA = "Despesas Empenhadas"
COLUNA_LIQUIDADA = "Despesas Liquidadas"
COLUNA_PAGA = "Despesas Pagas"
COLUNAS_EXECUCAO = [COLUNA_EMPENHADA, COLUNA_LIQUIDADA, COLUNA_PAGA]
COLUNA_PADRAO = COLUNA_LIQUIDADA

# Códigos da classificação funcional que interessam ao IP.
COD_FUNCAO_ENERGIA = "25"
COD_SUBFUNCAO_ENERGIA = "25.752"          # Energia Elétrica
# Alguns municípios lançam o gasto de energia como programa de conservação, na 751 em
# vez da 752. Itabira/2025 é assim: R$ 17,6 mi inteiros em "25.751 - Conservação de
# Energia" e nada na 752. Ignorar a 751 fazia o módulo concluir que o município não
# declarava energia e cair para o envelope de Serviços Urbanos, com o rótulo errado.
COD_SUBFUNCAO_CONSERVACAO = "25.751"      # Conservação de Energia
COD_FUNCAO_URBANISMO = "15"
COD_SUBFUNCAO_SERVICOS_URBANOS = "15.452"

COLUNAS = [
    "codigo_municipio", "municipio", "uf", "ano_exercicio",
    "coluna_execucao",
    "despesa_energia_eletrica",      # 25.752 ou 25.751 — envelope: IP + prédios + poços
    "despesa_funcao_energia",        # 25     — função inteira
    "despesa_servicos_urbanos",      # 15.452 — envelope: IP + limpeza + praças
    "despesa_funcao_urbanismo",      # 15     — função inteira
    "despesa_total",                 # Despesas exceto intraorçamentárias
    "envelope_ip_reais",             # melhor teto disponível para o gasto anual com IP
    "origem_despesa_energia",        # de qual linha funcional saiu esse envelope
    "status", "observacao",
]

WORKERS_PADRAO = 6


def _valor(linha: Dict[str, Any]) -> Optional[float]:
    try:
        return float(linha.get("valor") or 0.0)
    except (TypeError, ValueError):
        return None


def _codigo_da_conta(linha: Dict[str, Any]) -> str:
    """
    Extrai o código funcional do texto da conta.

    A API traz `conta` no formato "25.752 - Energia Elétrica" e `cod_conta` sempre
    igual a "TotalDespesas" neste anexo — ou seja, o código útil está no texto, não
    no campo de código. Daí a extração ser pelo prefixo antes do hífen.
    """
    texto = str(linha.get("conta") or "").strip()
    return texto.split("-", 1)[0].strip() if "-" in texto else texto


def consultar(cod_ibge: str, ano: int, coluna: str = COLUNA_PADRAO) -> Dict[str, Any]:
    """
    Despesa por função de um par (município, ano).

    Devolve SEMPRE um registro, com `status`:
      OK · SEM_DADO_NO_ANEXO · ENTE_NAO_DECLAROU · ERRO_API
    """
    cod_ibge = siconfi.so_digitos(cod_ibge)
    ident = siconfi.identificar(cod_ibge)
    reg: Dict[str, Any] = dict.fromkeys(COLUNAS)
    reg.update({
        "codigo_municipio": cod_ibge, "ano_exercicio": ano,
        "municipio": ident["ente"], "uf": ident["uf"],
        "coluna_execucao": coluna, "status": "", "observacao": "",
    })

    params = {"an_exercicio": ano, "id_ente": cod_ibge,
              "no_anexo": ANEXO, "nr_periodo": NR_PERIODO}
    try:
        dados = siconfi._paginar("/dca", params)
    except requests.exceptions.RequestException as exc:
        reg["status"] = "ERRO_API"
        reg["observacao"] = f"{type(exc).__name__}: {exc}"
        return reg

    if not dados:
        reg["status"] = "ENTE_NAO_DECLAROU"
        reg["observacao"] = "API não retornou o Anexo I-E para este ente/exercício."
        return reg

    amostra = dados[0]
    reg["municipio"] = reg["municipio"] or amostra.get("instituicao") or ""
    reg["uf"] = reg["uf"] or amostra.get("uf") or ""

    alvo = siconfi.normalizar(coluna)
    mapa = {
        COD_SUBFUNCAO_ENERGIA: "despesa_energia_eletrica",
        COD_SUBFUNCAO_CONSERVACAO: "despesa_energia_eletrica",
        COD_FUNCAO_ENERGIA: "despesa_funcao_energia",
        COD_SUBFUNCAO_SERVICOS_URBANOS: "despesa_servicos_urbanos",
        COD_FUNCAO_URBANISMO: "despesa_funcao_urbanismo",
    }
    encontrou = False
    total = 0.0
    for linha in dados:
        if siconfi.normalizar(linha.get("coluna")) != alvo:
            continue
        valor = _valor(linha)
        if valor is None:
            continue
        codigo = _codigo_da_conta(linha)
        if codigo in mapa:
            reg[mapa[codigo]] = valor
            if codigo in (COD_SUBFUNCAO_ENERGIA, COD_SUBFUNCAO_CONSERVACAO):
                reg["_subfuncao_energia"] = str(linha.get("conta") or codigo).strip()
            encontrou = True
        # O anexo não tem linha "Total Geral": o total é a soma das duas linhas de
        # topo, "Despesas Exceto Intraorçamentárias" e "Despesas Intraorçamentárias".
        elif siconfi.normalizar(linha.get("conta")).startswith("despesas "):
            total += valor
    if total:
        reg["despesa_total"] = total

    if not encontrou:
        reg["status"] = "SEM_DADO_NO_ANEXO"
        reg["observacao"] = (
            f"Anexo disponível, mas sem as funções 15 nem 25 na coluna “{coluna}”."
        )
        return reg

    reg["status"] = "OK"
    _resolver_envelope_energia(reg)
    return reg


def _resolver_envelope_energia(reg: Dict[str, Any]) -> None:
    """
    Decide de qual linha funcional sai o envelope de gasto com energia, e registra a
    origem — nunca deixa o número solto sem dizer de onde veio.

    Município pequeno costuma abrir a função 25 (Energia) com a subfunção 25.752. Já
    município grande frequentemente NÃO declara a função 25 e empenha a energia da
    iluminação dentro de 15.452 (Serviços Urbanos): Belo Horizonte/2023 é assim — zero
    na função 25 e R$ 177 mi em Serviços Urbanos. Tratar a ausência da 25 como "não
    gastou com energia" seria erro grosseiro; o correto é cair para o envelope
    disponível e dizer que ele é mais poluído.
    """
    if reg.get("despesa_energia_eletrica") is not None:
        reg["envelope_ip_reais"] = reg["despesa_energia_eletrica"]
        reg["origem_despesa_energia"] = reg.pop("_subfuncao_energia", None) or \
            "25.752 - Energia Elétrica"
        return
    if reg.get("despesa_funcao_energia") is not None:
        reg["despesa_energia_eletrica"] = reg["despesa_funcao_energia"]
        reg["envelope_ip_reais"] = reg["despesa_funcao_energia"]
        reg["origem_despesa_energia"] = "25 - Energia (função inteira)"
        reg["observacao"] = "Sem a subfunção 25.752; usou-se a função 25 inteira."
        return
    if reg.get("despesa_servicos_urbanos") is not None:
        # `despesa_energia_eletrica` fica vazia de propósito: Serviços Urbanos NÃO é
        # gasto com energia, e preencher esse campo com ele seria mentir no rótulo.
        # O valor vai para o envelope, que é assumidamente um teto poluído.
        reg["envelope_ip_reais"] = reg["despesa_servicos_urbanos"]
        reg["origem_despesa_energia"] = "15.452 - Serviços Urbanos (sem função 25)"
        reg["observacao"] = (
            "O município não declarou a função 25 (Energia). O envelope disponível é "
            "15.452 (Serviços Urbanos), que soma iluminação pública, limpeza urbana, "
            "praças e cemitério — serve de teto, não de gasto com energia."
        )
        return
    reg["origem_despesa_energia"] = None


def consultar_muitos(
    codigos: Sequence[str],
    anos: Sequence[int],
    coluna: str = COLUNA_PADRAO,
    workers: int = WORKERS_PADRAO,
    progresso=None,
) -> pd.DataFrame:
    """Consulta em paralelo. `progresso(feitos, total)` é chamado a cada conclusão."""
    pares = [(siconfi.so_digitos(c), a) for a in anos for c in codigos
             if len(siconfi.so_digitos(c)) == 7]
    if not pares:
        return pd.DataFrame(columns=COLUNAS)

    registros: List[Dict[str, Any]] = []
    with futuros.ThreadPoolExecutor(max_workers=workers) as executor:
        tarefas = {executor.submit(consultar, c, a, coluna): (c, a) for c, a in pares}
        for feitos, tarefa in enumerate(futuros.as_completed(tarefas), start=1):
            try:
                registros.append(tarefa.result())
            except Exception as exc:            # nunca derruba a triagem inteira
                cod, ano = tarefas[tarefa]
                registros.append({
                    "codigo_municipio": cod, "ano_exercicio": ano,
                    "coluna_execucao": coluna, "status": "ERRO_API",
                    "observacao": f"{type(exc).__name__}: {exc}",
                })
            if progresso:
                progresso(feitos, len(pares))
    return pd.DataFrame(registros, columns=COLUNAS)


# ── Cache em disco, no mesmo padrão da receita ───────────────────────────────
def _caminho_cache():
    return config.SICONFI_CACHE / "despesas_funcao.parquet"


def carregar_cache() -> pd.DataFrame:
    caminho = _caminho_cache()
    if caminho.exists():
        try:
            return pd.read_parquet(caminho)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUNAS)


def gravar_cache(df: pd.DataFrame) -> None:
    """Mescla no cache mantendo o registro mais recente de cada município/ano/coluna."""
    if df.empty:
        return
    config.garantir_pastas()
    combinado = pd.concat([carregar_cache(), df], ignore_index=True)
    combinado = combinado.drop_duplicates(
        subset=["codigo_municipio", "ano_exercicio", "coluna_execucao"], keep="last"
    )
    try:
        combinado.to_parquet(_caminho_cache(), index=False)
    except Exception:
        pass


def pendencias_no_cache(codigos: Sequence[str], anos: Sequence[int],
                        coluna: str = COLUNA_PADRAO) -> int:
    """Quantos pares (município, ano) ainda exigiriam ida à API. Ver o par em `siconfi`."""
    codigos = [c for c in (siconfi.so_digitos(x) for x in codigos) if len(c) == 7]
    if not codigos:
        return 0
    cache = carregar_cache()
    if cache.empty:
        return len(codigos) * len(list(anos))
    validos = cache[(cache["coluna_execucao"] == coluna) & (cache["status"] != "ERRO_API")]
    ja_tem = set(zip(validos["codigo_municipio"], validos["ano_exercicio"]))
    return sum(1 for a in anos for c in codigos if (c, a) not in ja_tem)


def consultar_com_cache(
    codigos: Sequence[str],
    anos: Sequence[int],
    coluna: str = COLUNA_PADRAO,
    workers: int = WORKERS_PADRAO,
    progresso=None,
    revalidar: bool = False,
) -> pd.DataFrame:
    """
    Igual a `consultar_muitos`, reaproveitando o cache local.

    Consulta que falhou (ERRO_API) nunca é cacheada como resposta válida — é sempre
    refeita, senão uma queda momentânea da API viraria "município sem dado" para sempre.
    """
    codigos = [c for c in (siconfi.so_digitos(x) for x in codigos) if len(c) == 7]
    anos = list(anos)
    cache = carregar_cache()

    if revalidar or cache.empty:
        aproveitado = pd.DataFrame(columns=COLUNAS)
        faltantes = [(c, a) for a in anos for c in codigos]
    else:
        valido = cache[
            (cache["coluna_execucao"] == coluna)
            & (cache["status"] != "ERRO_API")
            & (cache["codigo_municipio"].isin(codigos))
            & (cache["ano_exercicio"].isin(anos))
        ]
        aproveitado = valido.copy()
        ja_tem = set(zip(valido["codigo_municipio"], valido["ano_exercicio"]))
        faltantes = [(c, a) for a in anos for c in codigos if (c, a) not in ja_tem]

    if not faltantes:
        return aproveitado.reset_index(drop=True)

    novos = consultar_muitos(
        sorted({c for c, _ in faltantes}), sorted({a for _, a in faltantes}),
        coluna=coluna, workers=workers, progresso=progresso,
    )
    gravar_cache(novos[novos["status"] != "ERRO_API"])
    return pd.concat([aproveitado, novos], ignore_index=True).reset_index(drop=True)


__all__ = [
    "ANEXO", "COLUNAS", "COLUNAS_EXECUCAO", "COLUNA_PADRAO",
    "COLUNA_EMPENHADA", "COLUNA_LIQUIDADA", "COLUNA_PAGA",
    "COD_SUBFUNCAO_ENERGIA", "COD_SUBFUNCAO_CONSERVACAO",
    "COD_SUBFUNCAO_SERVICOS_URBANOS",
    "consultar", "consultar_muitos", "consultar_com_cache",
    "carregar_cache", "gravar_cache",
]
