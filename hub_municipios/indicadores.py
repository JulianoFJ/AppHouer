"""
Triagem de pré-viabilidade de PPP de iluminação pública.

Reproduz a metodologia que o time aplica à mão na planilha `CLP.xlsx` e no painel
"SMART RMBH", com as duas bases já integradas: a arrecadação de COSIP declarada ao
SICONFI e o parque de IP da BDGD/ANEEL.

A conta
-------
    arrecadação por ponto (R$/ponto.mês) = COSIP líquida ÷ (pontos × 12)
    sobra (R$/ponto.mês)                 = arrecadação por ponto − custo da PPP
    sobra (%)                            = sobra ÷ arrecadação por ponto

O **custo da PPP em R$ por ponto por mês** é o parâmetro que governa a triagem, e é
informado pelo usuário. Ele substituiu a combinação anterior "tarifa R$/kWh × consumo
da BDGD" nessa conta, por dois motivos:

1. é o número que o time já usa (R$ 38/ponto.mês na CLP, R$ 32 no estudo da RMBH) e
   que o mercado cota — a mediana dos 181 contratos de PPP assinados é R$ 34,23;
2. o consumo em kWh da BDGD é **inutilizável em 1.820 dos 5.417 municípios** (33,6%,
   6,7 milhões de pontos): Neoenergia Elektro declara 0 h/ano de operação, Copel 13 h,
   Enel CE 13.415 h. Amarrar o indicador financeiro ao consumo condenava um terço do
   país a número errado. Contagem de pontos e carga instalada seguem confiáveis.

Classificação de viabilidade
----------------------------
    Já possui PPP   — contrato de PPP de IP vigente (base de 181 contratos)
    Viável          — sobra positiva E arrecadação acima do corte
    Viabilizável    — sobra positiva, mas arrecadação abaixo do corte: precisa de
                      consórcio, agregação regional ou revisão da lei da COSIP
    Não viável      — a COSIP não cobre a contraprestação de referência
    Não possui CIP  — o município não declarou arrecadação de COSIP

O corte de arrecadação (R$ 4,5 milhões/ano por padrão) é o mesmo da CLP, e foi
conferido contra o painel da RMBH: Brumadinho (R$ 4,57 mi) é "Viável"; Caeté
(R$ 1,88 mi), com sobra por ponto igualmente positiva, é "Viabilizável".

Energia e emissões
------------------
A tarifa (R$/kWh) e o fator de emissão do SIN (tCO2/MWh) alimentam um bloco separado:
custo da conta de luz, economia do retrofit e emissões evitadas. Eles NÃO entram na
conta da sobra — essa continua em R$/ponto.mês, imune ao consumo declarado.

A base de consumo é o **declarado à ANEEL** sempre que ele for plausível (3.000–5.000 h
equivalentes), caindo para carga × horas onde não for — o campo de energia da BDGD é
inutilizável em um terço dos municípios. `origem_consumo` diz qual dos dois valeu.

A tarifa é o **R$/kWh faturado, com tributos**, não a da resolução homologatória. Ver
`config.TARIFA_ENERGIA_PADRAO`: confundir as duas subestima o custo em ~30%.

Anual × ciclo
-------------
`economia_reais_ano` é o ano 1 em reais constantes. `economia_ciclo_reais` é o
acumulado NOMINAL do prazo, pela soma da série geométrica (`fator_acumulado`) — que é
o número comparável a um EVTE. Multiplicar o anual pelo prazo subestima em 36% num
ciclo de 22 anos a 4% a.a.

Quando a potência média por ponto está fora da faixa física (`potencia_implausivel`),
TODO o bloco de energia sai como nulo. São 1.295 municípios, quase um quarto da base.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config, estimativa

# ── parâmetros de classificação ─────────────────────────────────────────────
# Corte de arrecadação anual abaixo do qual o projeto não se sustenta sozinho —
# escala pequena demais para o custo de transação de uma PPP. Da metodologia da CLP.
CORTE_ARRECADACAO_PADRAO = 4_500_000.0

# Horas anuais de operação para converter carga em energia. 4.160 h/ano (11,4 h/dia) é
# o valor medido na BDGD da Cemig-D e coerente com acionamento por relé fotoelétrico.
HORAS_OPERACAO = config.HORAS_OPERACAO_ANO

# Piso de plausibilidade da declaração de COSIP, em R$ por ponto por ano. Abaixo disso
# o valor não é arrecadação, é erro de preenchimento do DCA. Caso real: Passos/MG
# declarou R$ 16,35 no exercício de 2025 para um parque de 15.573 pontos.
PISO_PLAUSIBILIDADE_POR_PONTO = 12.0

# Faixa física de operação da IP. Fora dela, o consumo declarado pela distribuidora é
# inconsistente — hoje isso não afeta a triagem (que não usa consumo), mas continua
# sinalizado porque invalida a leitura de eficiência do parque.
HORAS_MIN_PLAUSIVEL, HORAS_MAX_PLAUSIVEL = 3000.0, 5000.0

# Faixa física da potência média por ponto. Fora dela, o bloco de energia é suprimido.
POTENCIA_MIN_PLAUSIVEL_W = config.POTENCIA_MIN_PLAUSIVEL_W
POTENCIA_MAX_PLAUSIVEL_W = config.POTENCIA_MAX_PLAUSIVEL_W

ORIGEM_CONSUMO_DECLARADO = "Declarado à ANEEL"
ORIGEM_CONSUMO_DERIVADO = "Derivado da carga"

VIABILIDADE_JA_TEM_PPP = "Já possui PPP"
VIABILIDADE_VIAVEL = "Viável"
VIABILIDADE_VIABILIZAVEL = "Viabilizável"
VIABILIDADE_NAO_VIAVEL = "Não viável"
VIABILIDADE_SEM_CIP = "Não possui CIP"

COLUNAS_INDICADORES = [
    # identificação
    "codigo_municipio", "municipio", "uf", "ano_exercicio", "status",
    # arrecadação
    "cosip_liquida", "receita_bruta", "deducoes", "populacao",
    # parque
    "pontos_ip", "origem_pontos", "carga_instalada_kw", "potencia_media_w",
    "perc_led", "distribuidora", "ano_base_bdgd", "defasagem_anos",
    "consumo_kwh_ano", "horas_equivalentes_ano",
    # triagem
    "cosip_ponto_mes", "cosip_por_ponto_ano", "cosip_por_habitante",
    "pontos_por_mil_hab", "custo_ppp_ponto_mes",
    "contraprestacao_mes", "contraprestacao_ano",
    "sobra_ponto_mes", "sobra_percentual", "sobra_reais_ano",
    # modernização
    "tarifa_energia", "consumo_estimado_kwh_ano", "origem_consumo", "custo_energia_ano",
    "custo_energia_ponto_mes",
    "potencia_futura_w", "economia_percentual", "economia_kwh_ano",
    "economia_reais_ano", "consumo_pos_retrofit_kwh_ano",
    "prazo_concessao_anos", "reajuste_anual", "economia_ciclo_reais",
    "custo_energia_ciclo_reais",
    "fator_emissao", "co2_evitado_t_ano",
    # despesa declarada ao SICONFI (Anexo I-E) — envelope, não gasto de IP puro
    "despesa_energia_declarada", "envelope_ip_declarado", "origem_despesa_energia",
    "despesa_energia_ponto_mes", "cosip_cobre_energia", "ip_sobre_energia_declarada",
    # classificação e qualidade do dado
    "viabilidade", "tem_ppp", "concessionaria_ppp", "ano_ppp",
    "declaracao_implausivel", "consumo_bdgd_suspeito", "potencia_implausivel",
]


def fator_acumulado(prazo_anos: int, reajuste: float) -> float:
    """
    Fator que converte um valor ANUAL do ano 1 em acumulado NOMINAL do ciclo.

    É a soma da série geométrica Σ (1+r)^(k-1), k=1..n — não `n`. A diferença não é
    detalhe: 22 anos a 4% a.a. dão 34,25, ou seja, multiplicar pelo prazo subestima o
    acumulado em 36%. Foi o que separou os R$ 8,73 mi da plataforma dos R$ 17 mi do
    modelo econômico-financeiro de Matozinhos, junto com a tarifa faturada.

    Com reajuste zero degenera para o prazo, que é o comportamento esperado.
    """
    n = int(prazo_anos)
    r = float(reajuste)
    if n <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return float(n)
    return ((1.0 + r) ** n - 1.0) / r


def _consumo_base(df: pd.DataFrame, carga_kw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Base de consumo anual, em kWh, e a origem dela.

    Prioridade ao **consumo declarado à ANEEL** pela distribuidora, que é medição e não
    modelo — mas só quando ele é fisicamente plausível, aferido pelas horas equivalentes
    (consumo ÷ carga) na faixa de 3.000–5.000 h/ano. Quando não é, cai para
    carga × HORAS_OPERACAO.

    O fallback existe porque o campo de energia da BDGD é inutilizável em um terço dos
    municípios: Neoenergia Elektro declara 0 h/ano equivalentes, Copel 13 h, Enel CE
    13.415 h. Onde o declarado presta, ele ganha — em São José da Lapa são 1.274 MWh
    medidos contra 1.127 MWh derivados, 11,5% de diferença que ia toda para a economia.
    """
    declarado = _numerica(df, "consumo_kwh_ano")
    derivado = carga_kw * HORAS_OPERACAO

    # As horas são RECALCULADAS de consumo ÷ carga, não lidas da coluna. A coluna vem do
    # ETL e pode estar dessincronizada do consumo desta linha; a razão calculada aqui é
    # sempre coerente com os dois números que de fato entram na conta.
    horas = declarado / carga_kw.replace(0, pd.NA)
    horas = horas.fillna(_numerica(df, "horas_equivalentes_ano"))

    presta = (declarado.notna() & (declarado > 0)
              & horas.notna() & horas.between(HORAS_MIN_PLAUSIVEL, HORAS_MAX_PLAUSIVEL))

    consumo = derivado.where(~presta, declarado)
    origem = pd.Series(ORIGEM_CONSUMO_DERIVADO, index=df.index, dtype="object")
    origem[presta] = ORIGEM_CONSUMO_DECLARADO
    origem[consumo.isna()] = None
    return consumo, origem


def _numerica(df: pd.DataFrame, coluna: str) -> pd.Series:
    """
    Coluna como Series numérica, mesmo quando ela não existe no DataFrame.

    `pd.to_numeric(df.get("x"))` devolve um escalar quando a coluna falta, e o escalar
    não tem `.notna()` nem alinha em operação vetorizada — o erro só aparece com um
    parque parcial (BDGD antiga sem campos de lâmpada, ou município só com COSIP).
    """
    if coluna not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[coluna], errors="coerce")


def _aplicar_despesa_declarada(
    df: pd.DataFrame,
    despesas_declaradas: Optional[pd.DataFrame],
    pontos_validos: pd.Series,
    cosip_liq: pd.Series,
) -> None:
    """
    Cruza a despesa declarada ao SICONFI (Anexo I-E) com a receita e a estimativa.

    Dois indicadores saem daqui, e são de naturezas diferentes:

      - `cosip_cobre_energia` = COSIP arrecadada ÷ despesa de energia declarada. É a
        resposta à pergunta que toda prefeitura faz — "a CIP paga a conta de luz?".
        Em Mateus Leme/2023 deu 71,5%: não paga.
      - `ip_sobre_energia_declarada` = custo de energia de IP estimado pela BDGD ÷
        despesa de energia declarada. É teste de consistência: a IP costuma responder
        por parte relevante da conta de energia de um município, mas **nunca por mais
        que 100%** — passar disso significa que a estimativa da BDGD, a tarifa ou a
        declaração ao Tesouro estão erradas, e o número não deve ser usado sem apurar.

    Ambos só são calculados quando o município declarou a função 25 (Energia). Onde
    o envelope veio de 15.452 (Serviços Urbanos), a razão não teria significado — o
    denominador incluiria limpeza urbana, capina e cemitério. Ver `despesas.py`, que
    mede isso em 45% dos municípios da amostra de MG.
    """
    for col in ("despesa_energia_declarada", "envelope_ip_declarado",
                "origem_despesa_energia", "despesa_energia_ponto_mes",
                "cosip_cobre_energia", "ip_sobre_energia_declarada"):
        df[col] = pd.NA

    if despesas_declaradas is None or despesas_declaradas.empty:
        return

    dd = despesas_declaradas.copy()
    dd["codigo_municipio"] = dd["codigo_municipio"].astype(str)
    dd = dd[dd["status"] == "OK"]
    if dd.empty:
        return
    dd = dd.drop_duplicates(subset=["codigo_municipio", "ano_exercicio"], keep="last")

    chave = ["codigo_municipio", "ano_exercicio"]
    colunas = chave + ["despesa_energia_eletrica", "envelope_ip_reais",
                       "origem_despesa_energia"]
    df_chave = df[chave].copy()
    df_chave["ano_exercicio"] = pd.to_numeric(df_chave["ano_exercicio"], errors="coerce")
    dd["ano_exercicio"] = pd.to_numeric(dd["ano_exercicio"], errors="coerce")
    juntado = df_chave.merge(dd[colunas], on=chave, how="left")

    energia = pd.to_numeric(juntado["despesa_energia_eletrica"], errors="coerce")
    df["despesa_energia_declarada"] = energia.values
    df["envelope_ip_declarado"] = pd.to_numeric(
        juntado["envelope_ip_reais"], errors="coerce").values
    df["origem_despesa_energia"] = juntado["origem_despesa_energia"].values

    energia_valida = energia.replace(0, pd.NA)
    df["despesa_energia_ponto_mes"] = (energia / (pontos_validos.values * 12.0)).values
    df["cosip_cobre_energia"] = (cosip_liq.values / energia_valida).values
    custo_ip = _numerica(df, "custo_energia_ano")
    df["ip_sobre_energia_declarada"] = (custo_ip.values / energia_valida).values


def cruzar(
    cosip: pd.DataFrame,
    parque: Optional[pd.DataFrame] = None,
    custo_ppp_ponto_mes: float = config.CUSTO_PPP_PONTO_MES_PADRAO,
    potencia_futura_w: float = config.POTENCIA_FUTURA_PADRAO_W,
    corte_arrecadacao: float = CORTE_ARRECADACAO_PADRAO,
    tarifa_energia: float = config.TARIFA_ENERGIA_PADRAO,
    fator_emissao: float = config.FATOR_EMISSAO_SIN_PADRAO,
    prazo_concessao_anos: int = config.PRAZO_CONCESSAO_ANOS_PADRAO,
    reajuste_anual: float = config.REAJUSTE_ANUAL_PADRAO,
    estimar_sem_bdgd: bool = True,
    despesas_declaradas: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Junta COSIP (uma linha por município/ano) ao parque de IP (uma linha por município)
    e devolve a triagem completa.

    Municípios sem BDGD ganham parque estimado pela população, sempre marcados em
    `origem_pontos`. Nenhum município é descartado por falta de base da distribuidora.
    """
    from . import bdgd, ppp as ppp_mod   # import tardio: evita ciclo e carga desnecessária

    if cosip is None or cosip.empty:
        return pd.DataFrame(columns=COLUNAS_INDICADORES)

    if parque is None:
        parque = bdgd.carregar_municipios()

    df = cosip.copy()
    df["codigo_municipio"] = df["codigo_municipio"].astype(str)

    if parque is not None and not parque.empty:
        parque = parque.copy()
        parque["codigo_municipio"] = parque["codigo_municipio"].astype(str)
        colunas = [c for c in ("codigo_municipio", "pontos_ip", "carga_instalada_kw",
                               "consumo_kwh_ano", "potencia_media_w", "perc_led",
                               "horas_equivalentes_ano", "distribuidora", "ano_base_bdgd")
                   if c in parque.columns]
        df = df.merge(parque[colunas], on="codigo_municipio", how="left")
    else:
        for col in ("pontos_ip", "carga_instalada_kw", "consumo_kwh_ano",
                    "potencia_media_w", "perc_led", "horas_equivalentes_ano",
                    "distribuidora", "ano_base_bdgd"):
            df[col] = pd.NA

    # ── parque medido ou estimado, sempre etiquetado ────────────────────────
    if estimar_sem_bdgd:
        df = estimativa.completar_parque(df)
    else:
        df["origem_pontos"] = pd.to_numeric(df["pontos_ip"], errors="coerce").notna().map(
            {True: estimativa.ORIGEM_MEDIDA, False: None})

    pontos = _numerica(df, "pontos_ip")
    pop = _numerica(df, "populacao")
    cosip_liq = _numerica(df, "cosip_liquida")
    pontos_validos = pontos.replace(0, pd.NA)

    # ── a conta da triagem ──────────────────────────────────────────────────
    custo = float(custo_ppp_ponto_mes)
    df["custo_ppp_ponto_mes"] = custo
    df["cosip_ponto_mes"] = cosip_liq / (pontos_validos * 12.0)
    df["cosip_por_ponto_ano"] = cosip_liq / pontos_validos
    df["cosip_por_habitante"] = cosip_liq / pop.replace(0, pd.NA)
    df["pontos_por_mil_hab"] = pontos / (pop.replace(0, pd.NA) / 1000.0)

    df["contraprestacao_mes"] = pontos * custo
    df["contraprestacao_ano"] = df["contraprestacao_mes"] * 12.0
    df["sobra_ponto_mes"] = df["cosip_ponto_mes"] - custo
    df["sobra_percentual"] = df["sobra_ponto_mes"] / df["cosip_ponto_mes"].replace(0, pd.NA)
    df["sobra_reais_ano"] = cosip_liq - df["contraprestacao_ano"]

    # ── energia, modernização e emissões ────────────────────────────────────
    pot_atual = _numerica(df, "potencia_media_w")
    carga_kw = _numerica(df, "carga_instalada_kw")
    # sem carga medida (parque estimado), deriva da potência de referência por ponto
    carga_kw = carga_kw.fillna(pontos * pot_atual / 1000.0)

    # A potência média por ponto só é utilizável dentro da faixa física. Fora dela o
    # bloco de energia inteiro é suprimido — ver POTENCIA_*_PLAUSIVEL_W no config.
    df["potencia_implausivel"] = pot_atual.notna() & ~pot_atual.between(
        POTENCIA_MIN_PLAUSIVEL_W, POTENCIA_MAX_PLAUSIVEL_W)
    utilizavel = ~df["potencia_implausivel"]

    df["consumo_estimado_kwh_ano"], df["origem_consumo"] = _consumo_base(df, carga_kw)

    df["tarifa_energia"] = float(tarifa_energia)
    df["custo_energia_ano"] = df["consumo_estimado_kwh_ano"] * float(tarifa_energia)
    df["custo_energia_ponto_mes"] = df["custo_energia_ano"] / (pontos_validos * 12.0)

    # A economia é a redução PROPORCIONAL da potência média do município, aplicada à
    # base de consumo. Vale para o parque inteiro, não para um recorte de tecnologia:
    # a decisão é do usuário e está registrada — recortar por "só o que não é LED"
    # seria premissa nova, e premissa nova não entra sem pedido.
    df["potencia_futura_w"] = float(potencia_futura_w)
    df["economia_percentual"] = ((pot_atual - float(potencia_futura_w)) /
                                 pot_atual.replace(0, pd.NA)).clip(lower=0)
    df["economia_kwh_ano"] = df["consumo_estimado_kwh_ano"] * df["economia_percentual"]
    df["economia_reais_ano"] = df["economia_kwh_ano"] * float(tarifa_energia)
    df["consumo_pos_retrofit_kwh_ano"] = (df["consumo_estimado_kwh_ano"]
                                          - df["economia_kwh_ano"]).clip(lower=0)

    # Acumulado do ciclo, NOMINAL e reajustado — o número que conversa com o EVTE.
    # Soma da série geométrica, não anual × prazo (ver config.PRAZO_CONCESSAO_*).
    df["prazo_concessao_anos"] = int(prazo_concessao_anos)
    df["reajuste_anual"] = float(reajuste_anual)
    df["economia_ciclo_reais"] = df["economia_reais_ano"] * fator_acumulado(
        prazo_concessao_anos, reajuste_anual)
    df["custo_energia_ciclo_reais"] = df["custo_energia_ano"] * fator_acumulado(
        prazo_concessao_anos, reajuste_anual)

    # Emissões evitadas. A matriz elétrica brasileira é predominantemente renovável,
    # então o ganho ambiental de economizar kWh aqui é modesto em tCO2 — bem menor do
    # que a mesma economia renderia num país de matriz fóssil. Reportar sem inflar.
    df["fator_emissao"] = float(fator_emissao)
    df["co2_evitado_t_ano"] = df["economia_kwh_ano"] / 1000.0 * float(fator_emissao)

    # Potência implausível contamina TODO o bloco derivado dela. Zerar só a economia
    # deixaria a conta de luz errada na tela, que é pior do que não mostrar nada.
    for col in ("consumo_estimado_kwh_ano", "custo_energia_ano", "custo_energia_ponto_mes",
                "economia_percentual", "economia_kwh_ano", "economia_reais_ano",
                "consumo_pos_retrofit_kwh_ano", "economia_ciclo_reais",
                "custo_energia_ciclo_reais", "co2_evitado_t_ano"):
        df[col] = df[col].where(utilizavel)
    df.loc[~utilizavel, "origem_consumo"] = None

    # ── qualidade do dado ───────────────────────────────────────────────────
    df["declaracao_implausivel"] = (
        df["cosip_por_ponto_ano"].notna()
        & (df["cosip_por_ponto_ano"] < PISO_PLAUSIBILIDADE_POR_PONTO)
    )
    horas = _numerica(df, "horas_equivalentes_ano")
    df["consumo_bdgd_suspeito"] = (
        horas.notna() & ~horas.between(HORAS_MIN_PLAUSIVEL, HORAS_MAX_PLAUSIVEL))

    if "ano_base_bdgd" in df.columns:
        df["defasagem_anos"] = (pd.to_numeric(df["ano_exercicio"], errors="coerce") -
                                pd.to_numeric(df["ano_base_bdgd"], errors="coerce"))
    else:
        df["defasagem_anos"] = pd.NA

    # ── PPP existente ───────────────────────────────────────────────────────
    contratos = ppp_mod.carregar()
    if not contratos.empty:
        c = contratos.drop_duplicates("codigo_municipio").set_index("codigo_municipio")
        df["tem_ppp"] = df["codigo_municipio"].isin(c.index)
        df["concessionaria_ppp"] = df["codigo_municipio"].map(c["concessionaria"])
        df["ano_ppp"] = df["codigo_municipio"].map(c["ano_assinatura"])
    else:
        df["tem_ppp"] = False
        df["concessionaria_ppp"] = pd.NA
        df["ano_ppp"] = pd.NA

    _aplicar_despesa_declarada(df, despesas_declaradas, pontos_validos, cosip_liq)

    df["viabilidade"] = _classificar(df, corte_arrecadacao)

    for col in COLUNAS_INDICADORES:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUNAS_INDICADORES].reset_index(drop=True)


def _classificar(df: pd.DataFrame, corte: float) -> pd.Series:
    """
    Ordem das regras importa: contrato vigente vence qualquer cálculo, e ausência de
    COSIP vence a conta de sobra (não dá para dividir uma arrecadação que não existe).
    """
    sobra = pd.to_numeric(df["sobra_ponto_mes"], errors="coerce")
    arrecadacao = pd.to_numeric(df["cosip_liquida"], errors="coerce")
    tem_cosip = arrecadacao.notna() & (arrecadacao > 0) & ~df["declaracao_implausivel"]

    classe = pd.Series(VIABILIDADE_SEM_CIP, index=df.index, dtype=object)
    tem_escala = arrecadacao >= corte

    # Abaixo do corte, o que falta é ESCALA — o município pode ser viabilizado por
    # consórcio regional ou revisão da lei da COSIP, tenha a sobra sinal positivo ou
    # negativo. Só é "Não viável" quem já tem escala e ainda assim não cobre a
    # contraprestação: aí o problema é a arrecadação por ponto, não o tamanho.
    # Conferido contra o painel SMART RMBH: Caeté (R$ 30,00/ponto contra custo de
    # R$ 32,00, portanto sobra negativa, mas só R$ 1,88 mi/ano) é "Viabilizável" lá.
    classe[tem_cosip & ~tem_escala] = VIABILIDADE_VIABILIZAVEL
    classe[tem_cosip & tem_escala & (sobra > 0)] = VIABILIDADE_VIAVEL
    classe[tem_cosip & tem_escala & (sobra <= 0)] = VIABILIDADE_NAO_VIAVEL
    classe[df["tem_ppp"].fillna(False).astype(bool)] = VIABILIDADE_JA_TEM_PPP
    return classe


def ressalvas(linha: pd.Series) -> list[str]:
    """Ressalvas que precisam acompanhar o número quando ele sai da tela."""
    avisos: list[str] = []

    status = linha.get("status")
    if status == "SEM_DADO_NO_ANEXO":
        avisos.append(
            "O município não declarou COSIP neste exercício. Pode ser ausência de lei "
            "instituidora ou falha de declaração — confirme na legislação municipal "
            "antes de concluir que não há arrecadação."
        )
    elif status == "ENTE_NAO_DECLAROU":
        avisos.append("A API do SICONFI não retornou o Anexo I-C para este exercício.")

    if bool(linha.get("declaracao_implausivel")):
        avisos.append(
            f"**Declaração implausível.** O valor informado equivale a "
            f"{formatar_moeda(linha.get('cosip_por_ponto_ano'))} por ponto por ANO, abaixo "
            f"do piso técnico de {formatar_moeda(PISO_PLAUSIBILIDADE_POR_PONTO)}. É erro de "
            "preenchimento do DCA, não baixa arrecadação. Busque o balancete municipal."
        )

    if linha.get("origem_pontos") == estimativa.ORIGEM_ESTIMADA:
        avisos.append(
            "**Parque estimado, não medido.** Este município não tem BDGD processada; os "
            "pontos de IP foram estimados pela população, com a densidade mediana da faixa "
            "de porte dele. Todos os indicadores por ponto herdam essa incerteza."
        )

    if bool(linha.get("tem_ppp")):
        conc = linha.get("concessionaria_ppp")
        ano = linha.get("ano_ppp")
        detalhe = f" ({conc}" + (f", {int(ano)}" if pd.notna(ano) else "") + ")" if pd.notna(conc) else ""
        avisos.append(f"**O município já tem PPP de iluminação pública{detalhe}.** "
                      "A triagem de arrecadação segue válida como referência, mas o ente "
                      "não é alvo de nova estruturação.")

    defasagem = linha.get("defasagem_anos")
    if pd.notna(defasagem) and abs(float(defasagem)) >= 2:
        avisos.append(
            f"A COSIP é do exercício {int(linha['ano_exercicio'])} e o parque é da BDGD de "
            f"{int(linha['ano_base_bdgd'])} — {abs(int(defasagem))} anos de defasagem."
        )

    if bool(linha.get("potencia_implausivel")):
        avisos.append(
            f"**Parque com potência fisicamente impossível.** A distribuidora "
            f"({linha.get('distribuidora') or 'não identificada'}) declara "
            f"{formatar_numero(linha.get('potencia_media_w'))} W por ponto, fora da faixa "
            f"de {formatar_numero(POTENCIA_MIN_PLAUSIVEL_W)}–"
            f"{formatar_numero(POTENCIA_MAX_PLAUSIVEL_W)} W. A carga instalada da base "
            "está inflada por um fator de escala, então o bloco de energia inteiro foi "
            "suprimido. Contagem de pontos e arrecadação seguem válidas."
        )

    elif bool(linha.get("consumo_bdgd_suspeito")):
        horas = float(linha.get("horas_equivalentes_ano"))
        avisos.append(
            f"O consumo declarado pela distribuidora é inconsistente "
            f"({formatar_numero(horas)} h/ano de operação equivalente, fora da faixa "
            f"física de {formatar_numero(HORAS_MIN_PLAUSIVEL)}–"
            f"{formatar_numero(HORAS_MAX_PLAUSIVEL)} h). Não afeta a triagem, que não usa "
            "consumo; o bloco de energia caiu para a carga instalada × "
            f"{formatar_numero(HORAS_OPERACAO)} h/ano de referência."
        )

    return avisos


# ── formatação pt-BR ────────────────────────────────────────────────────────

def _br(texto: str) -> str:
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def formatar_moeda(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return "R$ " + _br(f"{float(v):,.2f}")


def formatar_numero(v, casas: int = 0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return _br(f"{float(v):,.{casas}f}")


def formatar_percentual(v, casas: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return _br(f"{float(v) * 100:,.{casas}f}") + "%"
