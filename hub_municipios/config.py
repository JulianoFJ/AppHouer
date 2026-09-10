"""
Caminhos e parâmetros do Hub de Municípios.

Os caminhos de dados BRUTOS/intermediários (`DADOS`, `BDGD_BRUTOS`, `SICONFI_CACHE`,
etc.) vieram de `caminhos_dados.py`, na raiz do projeto — são genéricos, usados também
por `cadastro_bdgd`/`cadastro_ip`, e por isso não moram só aqui (até 10/09/2026,
moravam: `cadastro_bdgd`/`cadastro_ip` importavam direto deste módulo, um pacote de
domínio que não é o deles — via de mão dupla que este arquivo resolve reexportando em
vez de duplicar, para não quebrar os ~10 pontos dentro do próprio Hub que já chamam
`config.garantir_pastas()`/`config.pastas_bdgd()`).

O que fica só aqui: o derivado LEVE e publicável do próprio Hub (`DATA_PACOTE` e o que
vive dentro dele) e os parâmetros de negócio do Hub — nenhum dos dois é genérico o
bastante para o módulo compartilhado.

Layout de dados:

    Plataforma_IP/
      dados/                          <- fora do repositório git (o repo é AppHouer/)
        bdgd/brutos/                  <- os .gdb entram aqui (dezenas de GB)
        bdgd/processados/             <- parquet intermediário por distribuidora
        siconfi/cache/                <- cadastro de entes + consultas COSIP
      AppHouer/
        hub_municipios/data/          <- derivado LEVE, versionável e publicável
          bdgd_municipios.parquet     <- agregado por município (poucos MB)

A separação é proposital: o bruto é grande demais para versionar ou subir em deploy;
o derivado agregado é pequeno e acompanha o app.
"""

from __future__ import annotations

from pathlib import Path

import caminhos_dados

PACOTE = Path(__file__).resolve().parent

# ── Dados brutos e intermediários (fora do repositório) — reexportados ───────
DADOS = caminhos_dados.DADOS
BDGD_BRUTOS = caminhos_dados.BDGD_BRUTOS
BDGD_PROCESSADOS = caminhos_dados.BDGD_PROCESSADOS
SICONFI_CACHE = caminhos_dados.SICONFI_CACHE
ANEEL_CACHE = caminhos_dados.ANEEL_CACHE
BDGD_PASTAS_EXTRA = caminhos_dados.BDGD_PASTAS_EXTRA
pastas_bdgd = caminhos_dados.pastas_bdgd

# ── Derivado leve, que acompanha o app ──────────────────────────────────────
DATA_PACOTE = PACOTE / "data"
BDGD_MUNICIPIOS = DATA_PACOTE / "bdgd_municipios.parquet"
BDGD_TECNOLOGIA = DATA_PACOTE / "bdgd_tecnologia.parquet"
ENTES_CACHE = DATA_PACOTE / "entes_siconfi.parquet"
TARIFAS_B4A = DATA_PACOTE / "tarifas_b4a.parquet"


def garantir_pastas() -> None:
    """Cria as pastas de dados (genéricas + derivado do Hub) se não existirem. Idempotente."""
    caminhos_dados.garantir_pastas()
    DATA_PACOTE.mkdir(parents=True, exist_ok=True)


# ── Parâmetros de negócio ───────────────────────────────────────────────────
# Custo de referência da PPP de iluminação pública, em R$ por ponto por MÊS. É o
# único parâmetro financeiro da triagem, e substituiu a combinação anterior
# "tarifa R$/kWh x consumo da BDGD". A troca resolve dois problemas:
#   1. é o número que o time já usa e que o mercado cota — R$ 38 na planilha CLP,
#      R$ 32 no estudo da RMBH, mediana de R$ 34,23 nos 181 contratos assinados;
#   2. o consumo em kWh da BDGD é inutilizável em 1.820 dos 5.417 municípios
#      (33,6%), e amarrar o indicador a ele condenava um terço do país a número
#      errado. Contagem de pontos e carga instalada seguem confiáveis.
CUSTO_PPP_PONTO_MES_PADRAO = 38.00

POTENCIA_FUTURA_PADRAO_W = 60.0

# Tarifa de energia da iluminação pública (subgrupo B4a), em R$/kWh COM TRIBUTOS —
# isto é, o R$/kWh EFETIVAMENTE FATURADO: total pago no mês ÷ kWh faturado no mês.
# Usada SÓ para o bloco de energia: custo da conta de luz, economia do retrofit e
# emissões evitadas. NÃO entra na conta da sobra da CIP, que trabalha em R$/ponto.mês.
#
# ATENÇÃO — não é a tarifa da resolução homologatória. A REH publica TUSD+TE SEM
# tributos; a Cemig B4a, por exemplo, sai a ~R$ 0,39/kWh na REH e a ~R$ 0,51/kWh
# faturado depois de PIS/COFINS e ICMS. Confundir as duas subestima o custo de energia
# em ~30% — foi exatamente o que ocorreu ao comparar São José da Lapa com o modelo de
# Matozinhos em 28/08/2026. `aneel_tarifas.py` busca as DUAS na ANEEL e a página
# preenche este campo com a faturada. A faixa nacional vai de ~R$ 0,45 a ~R$ 0,95.
TARIFA_ENERGIA_PADRAO = 0.72

# Fator MÉDIO de emissão de CO2 da geração elétrica do SIN, em tCO2 por MWh.
#
# É o fator de "inventários corporativos" publicado pelo MCTI no SIRENE — o mesmo que
# o Programa Brasileiro GHG Protocol adota para inventário de escopo 2. Valor de 2024:
# 0,0545 tCO2/MWh.
#
# NÃO confundir com o fator da MARGEM DE OPERAÇÃO, também publicado pelo MCTI e bem
# menor (0,0215 a 0,0289 tCO2/MWh nos primeiros meses de 2025). Aquele serve a projetos
# de MDL/crédito de carbono; este, a inventário corporativo — que é o caso de uso aqui.
#
# O valor oscila com o despacho térmico: sobe em ano hidrológico seco e cai em ano
# chuvoso. Para dimensionar a ordem de grandeza: 0,0545 no Brasil contra 0,3674 nos
# Estados Unidos e 0,321 na Alemanha. A matriz brasileira é ~88% renovável, então a
# mesma economia de kWh evita aqui cerca de um sexto das emissões que evitaria lá.
FATOR_EMISSAO_SIN_PADRAO = 0.0545

# Horas de operação anuais observadas na BDGD da Cemig-D (2024): 4.160 h/ano,
# ou 11,4 h/dia — coerente com acionamento por relé fotoelétrico. É o FALLBACK: o
# consumo entra pelo campo declarado à ANEEL sempre que ele for plausível, e só cai
# para carga × estas horas quando não for (ver indicadores.consumo_base).
HORAS_OPERACAO_ANO = 4160.0

# ── Acumulado do ciclo da concessão ─────────────────────────────────────────
# O valor ANUAL de economia não conversa com um EVTE, que fala em acumulado de ciclo
# e em fluxo NOMINAL reajustado. Multiplicar o anual pelo prazo subestima em ~56% num
# ciclo de 22 anos a 4% a.a.: o fator correto é a soma da série geométrica, 34,25.
# Caso que expôs isso: Matozinhos, R$ 396,6 mil/ano a R$ 0,39 × 22 = R$ 8,73 mi contra
# os R$ 17 mi do modelo econômico — a diferença é tarifa faturada (×1,31) e reajuste
# (×1,56), não erro de cálculo físico.
PRAZO_CONCESSAO_ANOS_PADRAO = 22
REAJUSTE_ANUAL_PADRAO = 0.04

# ── Plausibilidade física do parque ─────────────────────────────────────────
# Faixa da potência média por ponto. Fora dela o dado da distribuidora é lixo e o bloco
# de energia inteiro precisa ser suprimido, não exibido: medido em 28/08/2026, 1.295 dos
# 5.417 municípios (5,04 milhões de pontos) estão fora — Neoenergia Elektro declara
# 208.625 W/ponto, Copel 37.736 W, CPFL Paulista e RGE ~1.250 W. A causa é CAR_INST e
# POT_LAMP inflados pelo MESMO fator, o que faz a razão entre eles passar no teste de
# `bdgd.detectar_fator_carga` e o erro entrar limpo. O contra-teste é
# `horas_equivalentes_ano` (514 h na CPFL, 13 h na Copel).
POTENCIA_MIN_PLAUSIVEL_W = 50.0
POTENCIA_MAX_PLAUSIVEL_W = 400.0

# ── ANEEL: Portal de Dados Abertos (CKAN) ───────────────────────────────────
# São DUAS tarifas, e a plataforma precisa das duas (ver TARIFA_ENERGIA_PADRAO).
#
#  1. Homologada, SEM tributos — dataset "Tarifas de aplicação das distribuidoras de
#     energia elétrica". Schema CONFERIDO na API em 28/08/2026, e ele NÃO é o que a
#     documentação sugere: o subgrupo é "B4" (não "B4a"), e o B4a aparece só em
#     DscSubClasse, como "Iluminação pública – B4a". Filtrar por DscSubGrupo == "B4a"
#     devolve zero linhas em silêncio. Os valores vêm em R$/MWh, com a unidade
#     declarada em DscUnidadeTerciaria. Campos: SigAgente, DscSubGrupo, DscSubClasse,
#     DscClasse, DscBaseTarifaria, DscUnidadeTerciaria, VlrTUSD, VlrTE,
#     DatInicioVigencia, DatFimVigencia, DscREH. Histórico desde 2010, 10.788 linhas
#     de iluminação pública.
#  2. Faturada, COM tributos — dataset SAMP (um recurso por ano), que traz mercado em
#     MWh e receita faturada com PIS/ICMS/COFINS por distribuidora e classe de consumo.
#     A razão receita ÷ mercado da classe "Iluminação Pública" é o R$/kWh pago. A ANEEL
#     documenta que essa receita INCLUI bandeiras tarifárias e EXCLUI COSIP/CIP — que é
#     exatamente a definição correta de custo de energia de IP para um EVTE.
#
# O servidor da ANEEL derruba o handshake TLS com alguma frequência
# (SSL: UNEXPECTED_EOF_WHILE_READING), e por isso o ETL é offline e retomável, com o
# CSV bruto guardado em ANEEL_CACHE — nunca chamada ao vivo dentro do Streamlit.
ANEEL_CKAN = "https://dadosabertos.aneel.gov.br/api/3/action"
ANEEL_RECURSO_TARIFAS = "fcf2906c-7c32-4b9b-a637-054e7a5234f4"
ANEEL_DATASET_SAMP = "samp"
ANEEL_SUBGRUPO_IP = "B4"
ANEEL_SUBCLASSE_IP = "B4a"     # via pública; B4b é bulbo de lâmpada, tarifa diferente
ANEEL_CLASSE_IP = "Iluminação pública"
