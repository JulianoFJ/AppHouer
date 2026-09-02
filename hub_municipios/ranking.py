"""
Ordenação e filtragem da carteira de municípios.

A triagem do Hub produz um painel com dezenas de indicadores por município. Para
priorizar esforço comercial e montar apresentação, é preciso ordenar esse painel por
critérios diferentes conforme a pergunta — e as perguntas são de naturezas distintas:

  - **"onde tem contrato grande?"** ordena por parque e contraprestação potencial;
  - **"onde a PPP fecha com folga?"** ordena por sobra da COSIP;
  - **"onde a arrecadação está defasada?"** ordena por R$/ponto ASCENDENTE — o
    município mal-arrecadado é o que mais precisa de revisão da lei de CIP, e é
    oportunidade de assessoria, não de descarte.

Por isso cada critério carrega o sentido natural da sua ordenação, em vez de assumir
que "maior é melhor" para tudo.

A lógica vive aqui, e não na página, para poder ser testada sem subir o Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class Criterio:
    """Um eixo de ordenação da carteira."""

    rotulo: str
    coluna: str
    formato: str          # "reais" | "reais_compacto" | "numero" | "percentual" | "decimal"
    maior_primeiro: bool  # sentido natural da ordenação
    ajuda: str


CRITERIOS: dict[str, Criterio] = {
    c.rotulo: c for c in [
        Criterio("Contraprestação potencial (R$/mês)", "contraprestacao_mes",
                 "reais_compacto", True,
                 "Tamanho do contrato: pontos × custo por ponto. É o eixo comercial."),
        Criterio("Pontos de IP", "pontos_ip", "numero", True,
                 "Tamanho do parque. Proxy de porte do projeto e do CAPEX."),
        Criterio("População", "populacao", "numero", True,
                 "Porte do município. Cuidado: população alta com parque pequeno "
                 "costuma indicar cadastro da distribuidora desatualizado."),
        Criterio("Arrecadação de COSIP (R$/ano)", "cosip_liquida", "reais_compacto", True,
                 "Receita líquida de CIP declarada ao SICONFI no exercício."),
        Criterio("Arrecadação por ponto (R$/ponto.mês)", "cosip_ponto_mes", "reais", True,
                 "O indicador que decide bancabilidade: quanto a CIP arrecada por "
                 "ponto por mês, a comparar com o custo da PPP por ponto."),
        Criterio("Arrecadação por ponto (R$/ponto.ano)", "cosip_por_ponto_ano",
                 "reais", True, "Mesma medida em base anual."),
        Criterio("Arrecadação por habitante (R$/hab.ano)", "cosip_por_habitante",
                 "reais", True,
                 "Carga da CIP sobre o cidadão. Valor alto com serviço ruim é "
                 "argumento político forte para a concessão."),
        Criterio("Sobra da COSIP (%)", "sobra_percentual", "percentual", True,
                 "Quanto da CIP sobra depois de paga a contraprestação. Sobra "
                 "negativa significa que a CIP não banca a PPP sozinha."),
        Criterio("Sobra da COSIP (R$/ano)", "sobra_reais_ano", "reais_compacto", True,
                 "A mesma sobra em reais — o que o município mantém em caixa."),
        Criterio("Economia de energia (R$/ano)", "economia_reais_ano", "reais_compacto",
                 True, "Economia no ano 1 da modernização, em reais constantes."),
        Criterio("Economia no ciclo (R$)", "economia_ciclo_reais", "reais_compacto", True,
                 "Acumulado nominal do prazo, pela soma da série geométrica. É o "
                 "número que conversa com o EVTE."),
        Criterio("CIP cobre da conta de luz (%)", "cosip_cobre_energia", "percentual",
                 False,
                 "COSIP arrecadada ÷ despesa de energia declarada ao SICONFI. "
                 "Ordenado do menor para o maior: quem cobre menos tem o problema "
                 "orçamentário maior, e a conversa mais urgente."),
        Criterio("Potência média (W/ponto)", "potencia_media_w", "decimal", True,
                 "Potência média do parque. Alta indica parque legado a eficientizar."),
        Criterio("Parque em LED (%)", "perc_led", "percentual", False,
                 "Ordenado do menor: parque com pouco LED é onde ainda há ganho de "
                 "eficientização a capturar."),
        Criterio("Pontos por mil habitantes", "pontos_por_mil_hab", "decimal", True,
                 "Densidade de iluminação. Fora da faixa de 20 a 40 costuma denunciar "
                 "problema de cadastro, não característica da cidade."),
    ]
}

CRITERIO_PADRAO = "Contraprestação potencial (R$/mês)"


def criterio(rotulo: str) -> Criterio:
    """Busca um critério pelo rótulo, caindo no padrão se não existir."""
    return CRITERIOS.get(rotulo, CRITERIOS[CRITERIO_PADRAO])


def filtrar(
    painel: pd.DataFrame,
    ufs: Optional[Sequence[str]] = None,
    viabilidades: Optional[Sequence[str]] = None,
    populacao_min: Optional[float] = None,
    populacao_max: Optional[float] = None,
    pontos_min: Optional[float] = None,
    pontos_max: Optional[float] = None,
    cosip_min: Optional[float] = None,
    cosip_ponto_mes_min: Optional[float] = None,
    incluir_com_ppp: bool = True,
    somente_dado_plausivel: bool = False,
) -> pd.DataFrame:
    """
    Aplica os filtros da carteira.

    `somente_dado_plausivel` descarta município com declaração de CIP implausível ou
    potência média fora da faixa física — os que o Hub já marca como suspeitos. É o
    filtro a ligar antes de gerar apresentação para cliente: número errado num slide
    custa mais caro que município a menos na lista.
    """
    if painel is None or painel.empty:
        return painel if painel is not None else pd.DataFrame()

    df = painel.copy()

    def _num(coluna: str) -> pd.Series:
        return pd.to_numeric(df[coluna], errors="coerce") if coluna in df.columns \
            else pd.Series([pd.NA] * len(df), index=df.index)

    if ufs:
        df = df[df["uf"].isin(list(ufs))]
    if viabilidades and "viabilidade" in df.columns:
        df = df[df["viabilidade"].isin(list(viabilidades))]
    if not incluir_com_ppp and "tem_ppp" in df.columns:
        df = df[~df["tem_ppp"].fillna(False).astype(bool)]

    if somente_dado_plausivel:
        for coluna in ("declaracao_implausivel", "potencia_implausivel"):
            if coluna in df.columns:
                df = df[~df[coluna].fillna(False).astype(bool)]

    # Faixas numéricas: linha sem o dado é mantida, porque filtro por faixa não deve
    # funcionar como filtro de completude — quem quiser isso liga o de plausibilidade.
    def _faixa(coluna: str, minimo, maximo=None) -> None:
        nonlocal df
        valores = pd.to_numeric(df[coluna], errors="coerce") if coluna in df.columns else None
        if valores is None:
            return
        if minimo is not None:
            df = df[valores.isna() | (valores >= minimo)]
            valores = pd.to_numeric(df[coluna], errors="coerce")
        if maximo is not None:
            df = df[valores.isna() | (valores <= maximo)]

    _faixa("populacao", populacao_min, populacao_max)
    _faixa("pontos_ip", pontos_min, pontos_max)
    _faixa("cosip_liquida", cosip_min)
    _faixa("cosip_ponto_mes", cosip_ponto_mes_min)
    return df.reset_index(drop=True)


def rankear(
    painel: pd.DataFrame,
    rotulo_criterio: str = CRITERIO_PADRAO,
    maior_primeiro: Optional[bool] = None,
    top: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ordena a carteira por um critério e devolve a coluna `posicao` (1 = primeiro).

    Município sem o dado do critério vai sempre para o fim, independentemente do
    sentido da ordenação: ausência de informação não é desempenho ruim nem bom, e
    deixá-lo no topo de uma ordenação ascendente seria enganoso.
    """
    if painel is None or painel.empty:
        return painel if painel is not None else pd.DataFrame()

    crit = criterio(rotulo_criterio)
    if crit.coluna not in painel.columns:
        return painel.copy()

    ascendente = (not crit.maior_primeiro) if maior_primeiro is None else (not maior_primeiro)
    df = painel.copy()
    df["_ordem"] = pd.to_numeric(df[crit.coluna], errors="coerce")
    df = df.sort_values("_ordem", ascending=ascendente, na_position="last", kind="stable")
    df = df.drop(columns=["_ordem"])
    df.insert(0, "posicao", range(1, len(df) + 1))
    return (df.head(top) if top else df).reset_index(drop=True)


def formatar_valor(valor, formato: str) -> str:
    """Formata um valor conforme o tipo do critério, no padrão brasileiro."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)) or pd.isna(valor):
        return "—"
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return str(valor)

    if formato == "percentual":
        return f"{numero:.1%}".replace(".", ",")
    if formato == "numero":
        return f"{numero:,.0f}".replace(",", ".")
    if formato == "decimal":
        return f"{numero:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if formato == "reais":
        return "R$ " + f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if formato == "reais_compacto":
        for limite, sufixo in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
            if abs(numero) >= limite:
                return ("R$ " + f"{numero / limite:,.1f}".replace(".", ",") + sufixo)
        return "R$ " + f"{numero:,.0f}".replace(",", ".")
    return str(valor)


# ── Score comercial de leads ─────────────────────────────────────────────────
# Pesos dos três eixos escolhidos em 01/09/2026. Somam 1,0.
PESO_SOBRA = 0.40
PESO_TAMANHO = 0.35
PESO_LEGADO = 0.25
# Multiplicador aplicado a quem já tem PPP assinada.
FATOR_PPP_CONTRATADA = 0.4

CLASSE_QUENTE = "Quente"
CLASSE_MORNO = "Morno"
CLASSE_FRIO = "Frio"
ORDEM_TEMPERATURA = [CLASSE_QUENTE, CLASSE_MORNO, CLASSE_FRIO]


def _percentil(serie: pd.Series) -> pd.Series:
    """
    Posição relativa de cada valor na carteira, de 0 a 1.

    Percentil, e não normalização min-max, porque um único município gigante achataria
    todos os outros perto de zero — a carteira brasileira tem São Paulo e tem município
    de 1.200 pontos na mesma lista. O que interessa ao time comercial é a ordem.
    """
    valores = pd.to_numeric(serie, errors="coerce")
    if valores.notna().sum() <= 1:
        return pd.Series([0.5 if pd.notna(v) else 0.0 for v in valores],
                         index=serie.index, dtype=float)
    return valores.rank(pct=True).fillna(0.0)


def score_comercial(painel: pd.DataFrame) -> pd.DataFrame:
    """
    Pontua cada município como lead e classifica em Quente / Morno / Frio.

    Três eixos, definidos pelo usuário:

      1. **Sobra da CIP** (peso 0,40) — projeto que fecha com folga e ainda comporta
         soluções digitais é o que menos depende de negociação difícil.
      2. **Tamanho do contrato potencial** (peso 0,35) — receita.
      3. **Parque legado sem PPP** (peso 0,25) — potência média alta e pouco LED com
         concessão ainda não contratada: tese de eficientização fácil de sustentar.
         Município que **já tem PPP** zera este eixo, porque o ativo está tomado.

    O score é relativo à carteira analisada: o mesmo município pode ser Quente numa
    lista de vizinhos e Morno numa lista nacional. É intencional — a pergunta que o
    time faz é "por onde começo desta lista", não "qual a nota absoluta".

    Returns:
        Cópia do painel com `score_comercial` (0 a 100), `temperatura` e as três
        parcelas, para que a priorização possa ser auditada e contestada.
    """
    if painel is None or painel.empty:
        return painel if painel is not None else pd.DataFrame()

    df = painel.copy()
    sobra = _percentil(df.get("sobra_percentual", pd.Series(dtype=float)))
    tamanho = _percentil(df.get("contraprestacao_mes", pd.Series(dtype=float)))

    potencia = _percentil(df.get("potencia_media_w", pd.Series(dtype=float)))
    led = pd.to_numeric(df.get("perc_led", pd.Series(dtype=float)), errors="coerce")
    # Pouco LED = mais oportunidade, daí o complemento.
    falta_led = _percentil(1 - led.fillna(led.median() if led.notna().any() else 0.5))
    legado = (potencia + falta_led) / 2
    if "tem_ppp" in df.columns:
        legado = legado.where(~df["tem_ppp"].fillna(False).astype(bool), 0.0)

    df["score_sobra"] = (sobra * PESO_SOBRA * 100).round(1)
    df["score_tamanho"] = (tamanho * PESO_TAMANHO * 100).round(1)
    df["score_legado"] = (legado * PESO_LEGADO * 100).round(1)
    df["score_comercial"] = (df["score_sobra"] + df["score_tamanho"]
                             + df["score_legado"]).round(1)

    # Município com dado suspeito não pode liderar lista de prospecção: a abordagem
    # seria feita em cima de número que não se sustenta na primeira reunião.
    for coluna in ("declaracao_implausivel", "potencia_implausivel"):
        if coluna in df.columns:
            df.loc[df[coluna].fillna(False).astype(bool), "score_comercial"] *= 0.5

    # PPP vigente rebaixa o lead de forma decisiva, não marginal. Zerar apenas o eixo
    # de legado deixava um município grande com concessão assinada em 1º lugar como
    # "quente" — e o slide dizendo "ativo tomado" logo abaixo. O ativo está contratado
    # por 20 e poucos anos: cabe abordagem de escopo complementar, não prospecção.
    com_ppp = (df["tem_ppp"].fillna(False).astype(bool) if "tem_ppp" in df.columns
               else pd.Series(False, index=df.index))
    df.loc[com_ppp, "score_comercial"] *= FATOR_PPP_CONTRATADA
    df["score_comercial"] = df["score_comercial"].round(1)

    limite_quente = df["score_comercial"].quantile(0.75)
    limite_morno = df["score_comercial"].quantile(0.40)
    df["temperatura"] = pd.cut(
        df["score_comercial"],
        bins=[-0.01, limite_morno, limite_quente, 1000],
        labels=[CLASSE_FRIO, CLASSE_MORNO, CLASSE_QUENTE],
    ).astype(str)
    # Trava dura: com PPP contratada, nunca "Quente", por melhor que sejam os números.
    df.loc[com_ppp & (df["temperatura"] == CLASSE_QUENTE), "temperatura"] = CLASSE_MORNO
    return df.sort_values("score_comercial", ascending=False).reset_index(drop=True)


__all__ = ["Criterio", "CRITERIOS", "CRITERIO_PADRAO", "criterio",
           "filtrar", "rankear", "formatar_valor",
           "score_comercial", "ORDEM_TEMPERATURA",
           "CLASSE_QUENTE", "CLASSE_MORNO", "CLASSE_FRIO"]
