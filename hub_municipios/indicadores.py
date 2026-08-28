"""
Cruzamento COSIP (SICONFI) × parque de IP (BDGD) e os indicadores derivados.

A pergunta que o módulo responde é a de triagem de PPP de iluminação pública:
**a arrecadação da COSIP sustenta o serviço?** Para isso não basta o valor absoluto —
ele só significa alguma coisa dividido pelo parque que precisa custear.

Indicadores
-----------
- `cosip_por_ponto_ano`   R$/ponto/ano — a métrica que dialoga direto com a
                          contraprestação de uma PPP, cotada na mesma unidade.
- `cosip_por_habitante`   R$/hab/ano — compara entes de portes diferentes.
- `custo_energia_estimado` consumo BDGD × tarifa B4a.
- `cobertura_energia`     quantas vezes a COSIP cobre a conta de energia. Abaixo de
                          1,0 a contribuição não paga nem a energia — não há espaço
                          para O&M nem para investimento sem outra fonte.
- `saldo_apos_energia`    o que sobra por ano para O&M, modernização e contraprestação.
- `economia_potencial_*`  ganho de um retrofit integral em LED, à potência de referência.

Todos os cruzamentos carregam `defasagem_anos` — a distância entre o exercício da COSIP
e a data-base da BDGD. Comparar arrecadação de 2024 com parque de 2017 é possível, mas
o leitor precisa saber que está fazendo isso.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config

COLUNAS_INDICADORES = [
    "codigo_municipio", "municipio", "uf", "ano_exercicio",
    "cosip_liquida", "receita_bruta", "deducoes", "populacao", "status",
    "pontos_ip", "carga_instalada_kw", "consumo_kwh_ano", "potencia_media_w",
    "perc_led", "perc_urbano", "consumo_kwh_ponto_ano", "horas_equivalentes_ano",
    "distribuidora", "ano_base_bdgd", "defasagem_anos",
    "cosip_por_ponto_ano", "cosip_por_habitante", "pontos_por_mil_hab",
    "custo_energia_estimado", "cobertura_energia", "saldo_apos_energia",
    "saldo_por_ponto_ano", "consumo_led_kwh_ano",
    "economia_potencial_kwh_ano", "economia_potencial_reais_ano",
    "custo_energia_pos_retrofit", "espaco_pos_retrofit",
    "espaco_ponto_mes_atual", "espaco_ponto_mes_pos_retrofit",
    "declaracao_implausivel", "consumo_bdgd_suspeito",
]

# Faixa física de operação da IP: relé fotoelétrico liga 11–12 h/dia, ou seja
# 4.000–4.400 h/ano. Fora de 3.000–5.000 h o consumo ou a carga declarados à ANEEL estão
# inconsistentes, e todo indicador que depende do consumo (custo de energia, cobertura,
# economia potencial) deixa de valer para aquele município. Casos reais encontrados em
# 28/08/2026: Demei/RS 6.259 h, Cocel/PR 5.340 h, Roraima Energia 2.336 h.
HORAS_MIN_PLAUSIVEL, HORAS_MAX_PLAUSIVEL = 3000.0, 5000.0

# Piso de plausibilidade da declaração: R$/ponto/ano abaixo do qual o valor não pode ser
# arrecadação real — é erro de preenchimento do DCA. R$ 12/ponto/ano equivale a R$ 1 por
# ponto por MÊS; nenhuma lei de COSIP em vigor produz isso. Caso observado em 28/08/2026:
# Passos/MG declarou R$ 134,83 (2020) e R$ 16,35 (2025) para um parque de 15.573 pontos.
# Sem esse filtro, o município entra em qualquer ranking como "o pior do estado" e
# distorce mediana, gráfico e conclusão.
PISO_PLAUSIBILIDADE_POR_PONTO = 12.0


def cruzar(
    cosip: pd.DataFrame,
    parque: Optional[pd.DataFrame] = None,
    tarifa_kwh: float = config.TARIFA_B4A_PADRAO,
    potencia_led_w: float = config.POTENCIA_LED_REFERENCIA_W,
) -> pd.DataFrame:
    """
    Junta a COSIP (uma linha por município/ano) ao parque de IP (uma linha por
    município) e calcula os indicadores. Municípios sem BDGD são preservados, com as
    colunas de parque vazias — ausência de base da distribuidora não pode sumir com o
    município da análise.
    """
    from . import bdgd  # import tardio: evita exigir a BDGD para usar só a COSIP

    if cosip.empty:
        return pd.DataFrame(columns=COLUNAS_INDICADORES)

    if parque is None:
        parque = bdgd.carregar_municipios()

    df = cosip.copy()
    df["codigo_municipio"] = df["codigo_municipio"].astype(str)

    if parque is not None and not parque.empty:
        parque = parque.copy()
        parque["codigo_municipio"] = parque["codigo_municipio"].astype(str)
        colunas_parque = [c for c in parque.columns if c != "codigo_municipio"]
        df = df.merge(parque, on="codigo_municipio", how="left", suffixes=("", "_bdgd"))
    else:
        for col in ("pontos_ip", "carga_instalada_kw", "consumo_kwh_ano",
                    "potencia_media_w", "perc_led", "perc_urbano",
                    "consumo_kwh_ponto_ano", "horas_equivalentes_ano",
                    "distribuidora", "ano_base_bdgd"):
            df[col] = pd.NA

    # ── indicadores ──────────────────────────────────────────────────────────
    pontos = pd.to_numeric(df.get("pontos_ip"), errors="coerce")
    pop = pd.to_numeric(df.get("populacao"), errors="coerce")
    cosip_liq = pd.to_numeric(df.get("cosip_liquida"), errors="coerce")
    consumo = pd.to_numeric(df.get("consumo_kwh_ano"), errors="coerce")

    df["cosip_por_ponto_ano"] = cosip_liq / pontos.replace(0, pd.NA)
    df["cosip_por_habitante"] = cosip_liq / pop.replace(0, pd.NA)
    df["pontos_por_mil_hab"] = pontos / (pop.replace(0, pd.NA) / 1000.0)

    df["custo_energia_estimado"] = consumo * float(tarifa_kwh)
    df["cobertura_energia"] = cosip_liq / df["custo_energia_estimado"].replace(0, pd.NA)
    df["saldo_apos_energia"] = cosip_liq - df["custo_energia_estimado"]
    df["saldo_por_ponto_ano"] = df["saldo_apos_energia"] / pontos.replace(0, pd.NA)

    # Retrofit integral: mantém as horas de operação observadas no próprio município
    # (o regime de acionamento não muda com a troca da luminária).
    horas_declaradas = pd.to_numeric(df.get("horas_equivalentes_ano"), errors="coerce")
    horas = horas_declaradas.fillna(config.HORAS_OPERACAO_ANO)
    df["consumo_led_kwh_ano"] = pontos * float(potencia_led_w) * horas / 1000.0
    df["economia_potencial_kwh_ano"] = (consumo - df["consumo_led_kwh_ano"]).clip(lower=0)
    df["economia_potencial_reais_ano"] = df["economia_potencial_kwh_ano"] * float(tarifa_kwh)

    # ── Espaço para contraprestação ──────────────────────────────────────────
    # A eficientização é o que financia a PPP de IP: o retrofit derruba a conta de
    # energia, e a diferença é o que passa a caber na contraprestação. Por isso o espaço
    # RELEVANTE não é o saldo de hoje, e sim o saldo depois do retrofit.
    #
    # Premissa embutida: modelo em que a ENERGIA CONTINUA COM O MUNICÍPIO e a
    # contraprestação remunera CAPEX + O&M — o arranjo mais comum no Brasil. Se a
    # concessionária assumir a energia, o espaço passa a ser a COSIP inteira e estes
    # números viram piso, não teto.
    df["custo_energia_pos_retrofit"] = df["consumo_led_kwh_ano"] * float(tarifa_kwh)
    df["espaco_pos_retrofit"] = cosip_liq - df["custo_energia_pos_retrofit"]

    # R$/ponto/mês é a unidade em que a contraprestação de PPP de IP é cotada no
    # mercado — é o número que dialoga direto com uma proposta.
    pontos_mes = (pontos.replace(0, pd.NA) * 12.0)
    df["espaco_ponto_mes_atual"] = df["saldo_apos_energia"] / pontos_mes
    df["espaco_ponto_mes_pos_retrofit"] = df["espaco_pos_retrofit"] / pontos_mes

    if "ano_base_bdgd" in df.columns:
        df["defasagem_anos"] = (pd.to_numeric(df["ano_exercicio"], errors="coerce") -
                                pd.to_numeric(df["ano_base_bdgd"], errors="coerce"))
    else:
        df["defasagem_anos"] = pd.NA

    df["declaracao_implausivel"] = (
        df["cosip_por_ponto_ano"].notna()
        & (df["cosip_por_ponto_ano"] < PISO_PLAUSIBILIDADE_POR_PONTO)
    )
    # Usa as horas DECLARADAS, não as preenchidas com o default: preencher a lacuna com
    # o valor de referência não pode fabricar um "dado consistente".
    df["consumo_bdgd_suspeito"] = (
        horas_declaradas.notna()
        & ~horas_declaradas.between(HORAS_MIN_PLAUSIVEL, HORAS_MAX_PLAUSIVEL)
    )

    for col in COLUNAS_INDICADORES:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUNAS_INDICADORES].reset_index(drop=True)


def ressalvas(linha: pd.Series) -> list[str]:
    """
    Ressalvas que devem acompanhar o número quando ele sai da tela. São as mesmas que
    um parecer de due diligence exigiria — não são decorativas.
    """
    avisos: list[str] = []

    status = linha.get("status")
    if status == "SEM_DADO_NO_ANEXO":
        avisos.append(
            "O município não declarou a rubrica de COSIP neste exercício. Isso pode ser "
            "ausência de lei instituidora ou falha de declaração — confirme na legislação "
            "municipal antes de concluir que não há arrecadação."
        )
    elif status == "ENTE_NAO_DECLAROU":
        avisos.append("A API do SICONFI não retornou o Anexo I-C para este exercício.")

    defasagem = linha.get("defasagem_anos")
    if pd.notna(defasagem) and abs(float(defasagem)) >= 2:
        avisos.append(
            f"A COSIP é do exercício {int(linha['ano_exercicio'])} e o parque é da BDGD de "
            f"{int(linha['ano_base_bdgd'])} — {abs(int(defasagem))} anos de defasagem. "
            "Os indicadores por ponto misturam períodos."
        )

    if pd.isna(linha.get("pontos_ip")):
        avisos.append(
            "Sem BDGD para este município: os indicadores por ponto não foram calculados. "
            "Processe a base da distribuidora que atende o ente."
        )

    if bool(linha.get("consumo_bdgd_suspeito")):
        horas = float(linha.get("horas_equivalentes_ano"))
        comparacao = "acima" if horas > HORAS_MAX_PLAUSIVEL else "abaixo"
        avisos.append(
            f"**Consumo da BDGD inconsistente.** As horas equivalentes de operação dão "
            f"{formatar_numero(horas)} h/ano ({formatar_numero(horas / 365, 1)} h/dia), "
            f"{comparacao} da faixa física da iluminação pública "
            f"({formatar_numero(HORAS_MIN_PLAUSIVEL)}–"
            f"{formatar_numero(HORAS_MAX_PLAUSIVEL)} h). Consumo ou carga instalada "
            "declarados à ANEEL estão errados: **o custo de energia, a cobertura da COSIP "
            "e a economia potencial não valem para este município.** Número de pontos e "
            "carga instalada seguem utilizáveis."
        )

    # `is True` não serve aqui: o pandas devolve numpy.bool_, que não é o singleton True.
    if bool(linha.get("declaracao_implausivel")):
        avisos.append(
            f"**Declaração implausível.** O valor informado ao SICONFI equivale a "
            f"{formatar_moeda(linha.get('cosip_por_ponto_ano'))} por ponto por ANO — abaixo do "
            f"piso técnico de {formatar_moeda(PISO_PLAUSIBILIDADE_POR_PONTO)}/ponto/ano. "
            "Nenhuma lei de COSIP produz isso: trata-se de erro de preenchimento do DCA pelo "
            "ente (valor em unidade errada ou lançamento equivocado), não de baixa "
            "arrecadação. Não use este número — busque o balancete municipal."
        )
    elif not bool(linha.get("consumo_bdgd_suspeito")):
        cobertura = linha.get("cobertura_energia")
        if pd.notna(cobertura) and float(cobertura) < 1.0:
            avisos.append(
                "A COSIP arrecadada não cobre sequer o custo estimado de energia. Uma PPP nesse "
                "ente depende de aporte de recursos ordinários ou de revisão da lei da COSIP."
            )

    return avisos


def _br(texto: str) -> str:
    """Troca o separador americano pelo brasileiro num número já formatado."""
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def formatar_moeda(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return "R$ " + _br(f"{float(v):,.2f}")


def formatar_numero(v, casas: int = 0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return _br(f"{float(v):,.{casas}f}")
