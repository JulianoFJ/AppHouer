"""
Hub de Municípios — arrecadação de COSIP × parque de iluminação pública.

Cruza duas bases públicas para responder à pergunta de triagem de uma PPP de IP:
a contribuição arrecadada sustenta o serviço?

  · SICONFI / DCA Anexo I-C (Tesouro Nacional) — receita de COSIP declarada.
  · BDGD / entidade PIP (ANEEL) — pontos, carga, consumo e tecnologia do parque.

Esta página é apenas a UI; a lógica vive em `hub_municipios/`.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hub_municipios import bdgd, config, indicadores, malhas, siconfi

# ── Paleta ───────────────────────────────────────────────────────────────────
# Categórica validada para fundo escuro (surface #12192b): banda de lightness
# OKLCH 0.48–0.67, croma ≥ 0.10, ΔE de CVD ≥ 8 em todos os pares adjacentes.
# A cor segue a TECNOLOGIA, nunca a posição no ranking — filtrar a lista não
# repinta as séries sobreviventes.
COR_TECNOLOGIA = {
    "LED": "#1F9ED1",                       # teal da marca — a tecnologia-alvo
    "Vapor de sódio": "#C08420",            # âmbar, a luz amarelada do sódio
    "Vapor de mercúrio": "#8B5CF6",         # violeta, a luz azulada do mercúrio
    "Vapor metálico": "#17A672",
}
COR_AUSENTE = "#64748B"                     # ausência de dado não é identidade

COR_COSIP = "#1F9ED1"
COR_ENERGIA = "#C08420"
ESCALA_LED = ["#12384A", "#186F96", "#1F9ED1", "#5BC0E8"]   # sequencial, 1 hue

# Status — reservados, nunca reutilizados como "série 4"
STATUS_BOM, STATUS_ATENCAO, STATUS_CRITICO = "#17A672", "#C08420", "#DC5B52"

TINTA_PRIMARIA, TINTA_SECUNDARIA, TINTA_FRACA = "#f8fafc", "#cbd5e1", "#94a3b8"
GRADE, SUPERFICIE = "#1f2937", "rgba(0,0,0,0)"

ANO_ATUAL = datetime.now().year


def _br(texto: str) -> str:
    """Troca o separador americano pelo brasileiro em um número já formatado."""
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _compacto(v) -> str:
    """
    R$ 236,4 mi — o `st.metric` trunca strings longas com reticências, e um valor de
    nove dígitos não cabe. Abrevia sem perder a ordem de grandeza.
    """
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    for limite, sufixo in ((1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if abs(v) >= limite:
            return "R$ " + _br(f"{v/limite:,.1f}") + f" {sufixo}"
    return indicadores.formatar_moeda(v)


def _pct(v, casas: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return _br(f"{float(v)*100:,.{casas}f}") + "%"


def _layout_base(fig: go.Figure, altura: int = 320) -> go.Figure:
    """Grade e eixos recessivos; texto sempre em tinta neutra, nunca na cor da série."""
    fig.update_layout(
        height=altura,
        # decimal com vírgula, milhar com ponto — o default do plotly é o americano
        separators=",.",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family="Inter, sans-serif", color=TINTA_SECUNDARIA, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA_SECUNDARIA)),
        hoverlabel=dict(bgcolor="#12192b", bordercolor=GRADE,
                        font=dict(color=TINTA_PRIMARIA, family="Inter, sans-serif")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRADE,
                     tickfont=dict(color=TINTA_FRACA))
    fig.update_yaxes(showgrid=True, gridcolor=GRADE, zeroline=False, linecolor=GRADE,
                     tickfont=dict(color=TINTA_FRACA))
    return fig


# ── Dados (com cache) ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _entes() -> pd.DataFrame:
    return siconfi.carregar_entes()


@st.cache_data(show_spinner=False)
def _parque() -> pd.DataFrame:
    return bdgd.carregar_municipios()


@st.cache_data(show_spinner=False)
def _tecnologia() -> pd.DataFrame:
    return bdgd.carregar_tecnologia()


@st.cache_data(show_spinner="Consultando o SICONFI…")
def _cosip(codigos: tuple, anos: tuple) -> pd.DataFrame:
    return siconfi.consultar_com_cache(list(codigos), list(anos))


@st.cache_data(show_spinner="Carregando a malha do IBGE…")
def _malha(uf: str):
    return malhas.carregar(uf)


# Indicadores oferecidos no mapa. Todos são magnitude, então usam o MESMO sequencial de
# uma hue (o teal da marca, claro → escuro). Em "Parque em LED" a escala é invertida:
# o que interessa achar no mapa é o parque LEGADO, e ele precisa ser o extremo saturado.
ESCALA_MAGNITUDE = ["#0E2A3A", "#186F96", "#1F9ED1", "#7FCDEA"]
MAPA_INDICADORES = {
    "Carga média por ponto (W)": {
        "coluna": "potencia_media_w", "sufixo": " W", "casas": 0, "inverter": False,
    },
    "COSIP por ponto (R$/ano)": {
        "coluna": "cosip_por_ponto_ano", "prefixo": "R$ ", "casas": 0, "inverter": False,
    },
    "Parque em LED (%)": {
        "coluna": "perc_led_pct", "sufixo": "%", "casas": 0, "inverter": True,
    },
    "Consumo por ponto (kWh/ano)": {
        "coluna": "consumo_kwh_ponto_ano", "sufixo": " kWh", "casas": 0, "inverter": False,
    },
    "Pontos de IP": {
        "coluna": "pontos_ip", "casas": 0, "inverter": False,
    },
}


def _excel(painel: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        painel.to_excel(writer, sheet_name="Indicadores", index=False)
    return buffer.getvalue()


# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 2.2rem; font-weight: 800;
                    background: linear-gradient(90deg, #ffffff, #00A9E0);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: -0.5px;">Hub de Municípios</div>
        <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 0.3rem;">
            Arrecadação de COSIP (SICONFI/DCA) cruzada com o parque de iluminação
            pública (BDGD/ANEEL) — indicadores por ponto, por habitante e cobertura do custeio
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

parque = _parque()
if parque.empty:
    st.warning(
        "**Nenhuma BDGD processada ainda.** A consulta de COSIP funciona normalmente, mas "
        "os indicadores por ponto de IP dependem da base da distribuidora.\n\n"
        f"1. Coloque os arquivos `.gdb` em `{config.BDGD_BRUTOS}`\n"
        "2. Rode `py -m hub_municipios.etl_bdgd` a partir de `app/`",
        icon="📂",
    )

# ── Filtros: uma linha acima de tudo ─────────────────────────────────────────
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    anos = c1.multiselect(
        "Exercícios",
        options=list(range(ANO_ATUAL, 2012, -1)),
        default=[ANO_ATUAL - 2, ANO_ATUAL - 1],
        help="A COSIP vem do DCA anual. O exercício mais recente costuma levar meses "
             "para ser publicado pelo Tesouro.",
    )
    tarifa = c2.number_input(
        "Tarifa B4a (R$/kWh)", min_value=0.10, max_value=3.00,
        value=config.TARIFA_B4A_PADRAO, step=0.01, format="%.2f",
        help="Default de triagem. A tarifa real vem da resolução homologatória da "
             "distribuidora e do regime tributário do município.",
    )
    pot_led = c3.number_input(
        "LED de referência (W)", min_value=20.0, max_value=250.0,
        value=config.POTENCIA_LED_REFERENCIA_W, step=5.0,
        help="Potência média por ponto após retrofit integral, usada no potencial de "
             "eficientização. Faixa de mercado: 45–75 W em via urbana.",
    )

if not anos:
    st.info("Selecione ao menos um exercício.")
    st.stop()

aba_municipio, aba_mapa, aba_comparar, aba_base = st.tabs(
    ["🔎  Município", "🗺️  Mapa", "📊  Comparar", "🗄️  Base de dados"]
)


# ═════════════════════════════════════════════════════════════════════════════
# ABA 1 — ficha do município
# ═════════════════════════════════════════════════════════════════════════════
with aba_municipio:
    # Município escolhido no mapa. Precisa ser consumido AQUI, antes de o text_input
    # existir: escrever na chave de um widget já instanciado levanta
    # StreamlitAPIException e derruba a página inteira. O mapa apenas grava a chave e
    # dispara rerun; a leitura acontece no início do ciclo seguinte.
    vindo_do_mapa = st.session_state.pop("hm_do_mapa", None)
    if vindo_do_mapa:
        st.session_state["hm_busca"] = str(vindo_do_mapa)
        st.session_state["hm_uf"] = "(todas)"     # o código IBGE já identifica a UF

    b1, b2 = st.columns([3, 1])
    termo = b1.text_input("Município", placeholder="Nome ou código IBGE (7 dígitos)",
                          key="hm_busca")
    uf_filtro = b2.selectbox(
        "UF", ["(todas)"] + sorted(_entes()["uf"].dropna().unique().tolist())
        if not _entes().empty else ["(todas)"], key="hm_uf")

    escolhido = None
    if termo.strip():
        achados = siconfi.buscar_municipio(termo, None if uf_filtro == "(todas)" else uf_filtro)
        if achados.empty:
            st.warning("Nenhum município encontrado. Verifique a grafia ou use o código IBGE.")
        else:
            opcoes = {f"{r.ente} / {r.uf}  ·  {r.cod_ibge}": r.cod_ibge
                      for r in achados.head(50).itertuples()}
            rotulo = st.selectbox(f"{len(achados)} resultado(s)", list(opcoes),
                                  key="hm_escolha")
            escolhido = opcoes[rotulo]

    if escolhido:
        cosip = _cosip((escolhido,), tuple(sorted(anos)))
        painel = indicadores.cruzar(cosip, parque, tarifa, pot_led)

        if painel.empty:
            st.error("Nenhum dado retornado para este município.")
        else:
            recente = painel.sort_values("ano_exercicio").iloc[-1]

            st.markdown(f"### {recente['municipio']} / {recente['uf']}")
            cap = [f"IBGE {recente['codigo_municipio']}"]
            if pd.notna(recente.get("populacao")):
                cap.append(f"{indicadores.formatar_numero(recente['populacao'])} habitantes")
            if pd.notna(recente.get("distribuidora")):
                cap.append(f"{recente['distribuidora']} · BDGD {recente['ano_base_bdgd']}")
            st.caption("  ·  ".join(cap))

            # ── KPIs ────────────────────────────────────────────────────────
            k = st.columns(4)
            k[0].metric(f"COSIP líquida {int(recente['ano_exercicio'])}",
                        _compacto(recente["cosip_liquida"]),
                        help=indicadores.formatar_moeda(recente["cosip_liquida"]))
            k[1].metric("Pontos de IP",
                        indicadores.formatar_numero(recente.get("pontos_ip")))
            k[2].metric("R$ / ponto / ano",
                        indicadores.formatar_moeda(recente.get("cosip_por_ponto_ano")))
            k[3].metric("R$ / habitante / ano",
                        indicadores.formatar_moeda(recente.get("cosip_por_habitante")))

            k2 = st.columns(4)
            cobertura = recente.get("cobertura_energia")
            k2[0].metric("Cobertura da energia",
                         _br(f"{float(cobertura):.2f}") + "×" if pd.notna(cobertura) else "—",
                         help="COSIP líquida ÷ custo estimado de energia. Abaixo de 1,00× a "
                              "contribuição não paga nem a conta de luz do parque.")
            k2[1].metric("Saldo após energia",
                         _compacto(recente.get("saldo_apos_energia")),
                         help="O que sobra por ano para O&M, modernização e contraprestação. "
                              + indicadores.formatar_moeda(recente.get("saldo_apos_energia")))
            k2[2].metric("Consumo por ponto",
                         f"{indicadores.formatar_numero(recente.get('consumo_kwh_ponto_ano'))} kWh/ano")
            perc_led = recente.get("perc_led")
            k2[3].metric("Parque em LED",
                         _pct(perc_led, 0 if pd.notna(perc_led) and float(perc_led) >= 0.9995
                              else 1))

            k3 = st.columns(4)
            k3[0].metric(
                "Carga média por ponto",
                f"{indicadores.formatar_numero(recente.get('potencia_media_w'), 1)} W",
                help="Carga instalada declarada à ANEEL ÷ número de pontos. Inclui as perdas "
                     "de reator e relé, por isso fica acima da potência nominal da lâmpada. "
                     "É o indicador direto do estado tecnológico do parque: acima de ~130 W "
                     "predomina descarga; abaixo de ~80 W, LED.",
            )
            k3[1].metric(
                "Potência nominal da lâmpada",
                f"{indicadores.formatar_numero(recente.get('potencia_lampada_media_w'), 1)} W"
                if pd.notna(recente.get("potencia_lampada_media_w")) else "—",
                help="Média de POT_LAMP na BDGD, sem perdas. Ausente em versões antigas da base.",
            )
            k3[2].metric(
                "Carga instalada total",
                f"{indicadores.formatar_numero(recente.get('carga_instalada_kw'), 0)} kW"
                if pd.notna(recente.get("carga_instalada_kw")) else "—",
            )
            k3[3].metric(
                "Pontos por mil habitantes",
                indicadores.formatar_numero(recente.get("pontos_por_mil_hab"), 1),
                help="Densidade do parque. A faixa usual em município urbanizado é 60–90 "
                     "pontos por mil habitantes; muito abaixo disso sugere cadastro incompleto "
                     "na BDGD ou demanda reprimida.",
            )

            # ── Ressalvas ───────────────────────────────────────────────────
            avisos = indicadores.ressalvas(recente)
            for aviso in avisos:
                st.warning(aviso, icon="⚠️")

            st.divider()
            g1, g2 = st.columns([3, 2])

            # ── Série histórica: COSIP × custo de energia, MESMA escala R$ ──
            with g1:
                st.markdown("**COSIP arrecadada × custo estimado de energia**")
                st.caption("valores em R$ milhões correntes")
                serie = painel[painel["status"] == "OK"].sort_values("ano_exercicio")
                if serie.empty:
                    st.info("Sem exercícios com COSIP declarada entre os anos selecionados.")
                else:
                    # Eixo em R$ milhões: em valores de nove dígitos os rótulos do eixo
                    # colidem, e a ordem de grandeza é o que importa aqui.
                    fig = go.Figure()
                    fig.add_bar(
                        x=serie["ano_exercicio"].astype(str),
                        y=serie["cosip_liquida"] / 1e6,
                        customdata=serie["cosip_liquida"],
                        name="COSIP líquida", marker_color=COR_COSIP,
                        marker_cornerradius=4, width=0.55,
                        hovertemplate="COSIP: R$ %{customdata:,.0f}<extra></extra>",
                    )
                    if serie["custo_energia_estimado"].notna().any():
                        fig.add_scatter(
                            x=serie["ano_exercicio"].astype(str),
                            y=serie["custo_energia_estimado"] / 1e6,
                            customdata=serie["custo_energia_estimado"],
                            name="Custo de energia (estimado)", mode="lines+markers",
                            line=dict(color=COR_ENERGIA, width=2),
                            marker=dict(size=9, color=COR_ENERGIA,
                                        line=dict(width=2, color="#12192b")),
                            hovertemplate="Energia: R$ %{customdata:,.0f}<extra></extra>",
                        )
                    fig.update_layout(hovermode="x unified", bargap=0.35)
                    _layout_base(fig, 330)
                    # sem type="category" o plotly lê "2024" como número e cria ticks
                    # intermediários sem sentido ("2.023,5")
                    fig.update_xaxes(type="category")
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.caption(
                        "Ambas as séries em R$ correntes, na mesma escala. O custo de energia "
                        "usa o parque de um único ano-base da BDGD, projetado sobre todos os "
                        "exercícios — não é série histórica de consumo."
                    )

            # ── Mix tecnológico ─────────────────────────────────────────────
            with g2:
                st.markdown("**Composição do parque**")
                tec = _tecnologia()
                tec_mun = (tec[tec["codigo_municipio"] == escolhido]
                           if not tec.empty else pd.DataFrame())
                tec_mun = tec_mun[tec_mun["tecnologia"] != "Não informado"] \
                    if not tec_mun.empty else tec_mun
                if tec_mun.empty:
                    st.info(
                        "Sem mix tecnológico para este município. BDGDs anteriores à versão "
                        "V não trazem os campos de lâmpada (TIPO_LAMP/POT_LAMP)."
                    )
                else:
                    tec_mun = tec_mun.sort_values("pontos")
                    total = tec_mun["pontos"].sum()
                    fig = go.Figure(go.Bar(
                        x=tec_mun["pontos"], y=tec_mun["tecnologia"], orientation="h",
                        marker_color=[COR_TECNOLOGIA.get(t, COR_AUSENTE)
                                      for t in tec_mun["tecnologia"]],
                        marker_cornerradius=4,
                        text=[_pct(p / total) for p in tec_mun["pontos"]],
                        textposition="outside",
                        textfont=dict(color=TINTA_SECUNDARIA),
                        hovertemplate="%{y}: %{x:,.0f} pontos<extra></extra>",
                    ))
                    _layout_base(fig, 330)
                    fig.update_xaxes(showgrid=True, gridcolor=GRADE,
                                     title_text="pontos", range=[0, total * 1.18])
                    fig.update_yaxes(showgrid=False)
                    fig.update_layout(showlegend=False, bargap=0.35)
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.caption(
                        "Tecnologia inferida da assinatura física de cada código TIPO_LAMP "
                        "(perda de reator + série normalizada de potência), não de tabela fixa."
                    )

            # ── Espaço para contraprestação ─────────────────────────────────
            economia = recente.get("economia_potencial_reais_ano")
            suspeito = bool(recente.get("consumo_bdgd_suspeito"))
            if pd.notna(economia) and float(economia) > 0 and not suspeito:
                st.divider()
                st.markdown("**Espaço para contraprestação**")
                st.caption(
                    "Quanto da COSIP sobra depois de pagar a energia — hoje e depois de um "
                    "retrofit integral. É o teto de partida para dimensionar uma PPP."
                )
                e = st.columns(4)
                e[0].metric("Sobra hoje (após energia)",
                            _compacto(recente.get("saldo_apos_energia")),
                            help=indicadores.formatar_moeda(recente.get("saldo_apos_energia"))
                                 + " por ano")
                e[1].metric("Sobra pós-retrofit",
                            _compacto(recente.get("espaco_pos_retrofit")),
                            delta=_compacto(economia) + "/ano de energia economizada",
                            help="A eficientização é o que financia a PPP: o retrofit derruba "
                                 "a conta de energia e a diferença passa a caber na "
                                 "contraprestação.")
                e[2].metric("R$/ponto/mês hoje",
                            indicadores.formatar_moeda(recente.get("espaco_ponto_mes_atual")),
                            help="Unidade em que a contraprestação de PPP de IP é cotada no "
                                 "mercado — dialoga direto com uma proposta.")
                e[3].metric("R$/ponto/mês pós-retrofit",
                            indicadores.formatar_moeda(
                                recente.get("espaco_ponto_mes_pos_retrofit")))

                st.warning(
                    "**Este número é teto de arrecadação, não espaço fiscal disponível.** "
                    "Antes de virar contraprestação viável falta descontar: (i) a **O&M atual** "
                    "do parque, que a prefeitura já paga — manutenção corretiva, equipe, "
                    "veículos; (ii) eventual **passivo com a distribuidora**; (iii) a "
                    "**inadimplência**, que aqui já está embutida porque o dado é arrecadado e "
                    "não faturado. A premissa embutida é a de que a **energia continua com o "
                    "município** e a contraprestação remunera CAPEX + O&M — o arranjo mais "
                    "comum no Brasil. Se a concessionária assumir a energia, estes valores "
                    "viram piso, não teto. E o retrofit assume "
                    f"{pot_led:,.0f} W/ponto sem tratar demanda reprimida nem projeto "
                    "luminotécnico: é triagem, não dimensionamento.",
                    icon="⚠️",
                )
                st.caption(
                    f"Retrofit a {pot_led:,.0f} W/ponto economizaria "
                    f"{indicadores.formatar_numero(recente['economia_potencial_kwh_ano']/1000)} "
                    f"MWh/ano — {_pct(float(economia)/float(recente['custo_energia_estimado']), 0)} "
                    "da conta de energia atual."
                )

            with st.expander("Tabela completa"):
                st.dataframe(painel, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️  Baixar indicadores (.xlsx)", _excel(painel),
                file_name=f"hub_{recente['codigo_municipio']}_{'_'.join(map(str, sorted(anos)))}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ═════════════════════════════════════════════════════════════════════════════
# ABA 2 — mapa por município
# ═════════════════════════════════════════════════════════════════════════════
with aba_mapa:
    if parque.empty or "uf" not in parque.columns:
        st.info("O mapa depende do parque da BDGD. Processe ao menos uma base "
                "(`py -m hub_municipios.etl_bdgd`).")
    else:
        ufs_com_parque = sorted(parque["uf"].dropna().unique().tolist())
        m1, m2, m3 = st.columns([1, 2, 2])
        uf_mapa = m1.selectbox("UF", ufs_com_parque,
                               index=ufs_com_parque.index("MG") if "MG" in ufs_com_parque else 0,
                               key="hm_mapa_uf")
        nome_ind = m2.selectbox("Colorir por", list(MAPA_INDICADORES), key="hm_mapa_ind")
        cfg_ind = MAPA_INDICADORES[nome_ind]
        precisa_cosip = cfg_ind["coluna"] == "cosip_por_ponto_ano"
        ano_mapa = m3.selectbox("Exercício da COSIP", sorted(anos, reverse=True),
                                key="hm_mapa_ano",
                                help="Usado só quando o indicador escolhido é a COSIP.",
                                disabled=not precisa_cosip)

        malha = _malha(uf_mapa)
        if malha is None:
            st.warning(
                f"Não foi possível obter a malha municipal de {uf_mapa} na API do IBGE. "
                "Verifique a conexão — a malha é baixada uma vez por UF e fica em cache.",
                icon="🌐",
            )
        else:
            dados_uf = parque[parque["uf"] == uf_mapa].copy()
            dados_uf["perc_led_pct"] = dados_uf["perc_led"] * 100

            # A COSIP não está no agregado da BDGD: só é consultada quando o indicador
            # escolhido exige, para não disparar centenas de chamadas ao SICONFI à toa.
            if precisa_cosip:
                cosip_uf = _cosip(tuple(dados_uf["codigo_municipio"]), (ano_mapa,))
                painel_uf = indicadores.cruzar(cosip_uf, dados_uf, tarifa, pot_led)
                painel_uf = painel_uf[painel_uf["declaracao_implausivel"] != True]  # noqa: E712
                dados_uf = dados_uf.merge(
                    painel_uf[["codigo_municipio", "cosip_por_ponto_ano", "municipio"]],
                    on="codigo_municipio", how="left")
            else:
                nomes = _entes().set_index("cod_ibge")["ente"]
                dados_uf["municipio"] = dados_uf["codigo_municipio"].map(nomes)

            coluna = cfg_ind["coluna"]
            dados_uf = dados_uf[dados_uf[coluna].notna()]
            if dados_uf.empty:
                st.info(f"Sem dados de “{nome_ind}” para {uf_mapa}.")
            else:
                pref, suf = cfg_ind.get("prefixo", ""), cfg_ind.get("sufixo", "")
                casas = cfg_ind.get("casas", 0)
                rotulos = [
                    f"{r.municipio or r.codigo_municipio}<br>{pref}"
                    f"{indicadores.formatar_numero(getattr(r, coluna), casas)}{suf}<br>"
                    f"{indicadores.formatar_numero(r.pontos_ip)} pontos de IP"
                    for r in dados_uf.itertuples()
                ]
                # go.Choroplethmap (não o Choropleth de projeção) casa os polígonos por
                # `featureidkey`, dá pan/zoom de mapa real e não exige token de mapbox —
                # é o que permite clicar num município pequeno para selecioná-lo.
                centro = malhas.centro_aproximado(malha)
                # A escala é cortada nos percentis 5–95: um único município com dado
                # extremo (visto em MG: 1.370 W/ponto contra mediana de 136 W) achata o
                # gradiente e o mapa inteiro fica de uma cor só. Os valores reais seguem
                # no hover e nos KPIs — o corte é só da rampa de cor.
                serie = dados_uf[coluna].astype(float)
                z_min, z_max = serie.quantile(0.05), serie.quantile(0.95)
                if z_min == z_max:
                    z_min, z_max = serie.min(), serie.max()
                cortou = bool(serie.min() < z_min or serie.max() > z_max)

                fig = go.Figure()

                # Camada de contexto: TODOS os municípios da UF em cinza neutro. Sem ela
                # só aparecem os que têm BDGD processada, o mapa fica com meia dúzia de
                # manchas soltas e some a referência geográfica do estado. Ausência de
                # dado precisa ser visível — e distinguível de valor baixo.
                nomes_uf = _entes().set_index("cod_ibge")["ente"]
                com_dado = set(dados_uf["codigo_municipio"])
                sem_dado = sorted(malhas.codigos_da_malha(malha) - com_dado)
                if sem_dado:
                    fig.add_trace(go.Choroplethmap(
                        geojson=malha, featureidkey="properties.codarea",
                        locations=sem_dado, z=[0] * len(sem_dado),
                        colorscale=[[0, "#1c2536"], [1, "#1c2536"]],
                        showscale=False, hoverinfo="text",
                        marker=dict(line=dict(color="#0b111e", width=0.3), opacity=0.55),
                        # customdata também aqui: clicar num município sem BDGD ainda é
                        # útil — a consulta de COSIP no SICONFI não depende do parque.
                        customdata=[[c] for c in sem_dado],
                        text=[f"{nomes_uf.get(c, c)}<br>sem BDGD processada" for c in sem_dado],
                        hovertemplate="%{text}<extra></extra>",
                    ))

                fig.add_trace(go.Choroplethmap(
                    geojson=malha,
                    featureidkey="properties.codarea",
                    locations=dados_uf["codigo_municipio"],
                    z=serie,
                    zmin=z_min, zmax=z_max,
                    colorscale=ESCALA_MAGNITUDE,
                    reversescale=cfg_ind.get("inverter", False),
                    marker=dict(line=dict(color="#0b111e", width=0.4), opacity=0.9),
                    colorbar=dict(title=dict(text=nome_ind.split(" (")[0],
                                             font=dict(color=TINTA_FRACA, size=11)),
                                  tickfont=dict(color=TINTA_FRACA), thickness=12, len=0.75),
                    customdata=dados_uf[["codigo_municipio"]],
                    text=rotulos, hovertemplate="%{text}<extra></extra>",
                ))
                fig.update_layout(
                    map=dict(style="carto-darkmatter", center=centro,
                             zoom=malhas.zoom_aproximado(malha)),
                    height=560, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor=SUPERFICIE, separators=",.",
                    font=dict(family="Inter, sans-serif", color=TINTA_SECUNDARIA),
                    hoverlabel=dict(bgcolor="#12192b", bordercolor=GRADE,
                                    font=dict(color=TINTA_PRIMARIA)),
                )

                evento = st.plotly_chart(fig, use_container_width=True,
                                         config={"displayModeBar": False},
                                         on_select="rerun", key="hm_mapa_chart")

                sel = (evento.get("selection", {}) or {}).get("points", []) if evento else []
                if sel:
                    # O clique pode cair na camada de contexto (trace 0) ou na de dados;
                    # `customdata` existe nas duas justamente para o código sair igual.
                    ponto = sel[0]
                    cod = None
                    if ponto.get("customdata"):
                        cod = ponto["customdata"][0]
                    else:
                        idx, curva = ponto.get("point_index"), ponto.get("curve_number", 0)
                        origem = sem_dado if (sem_dado and curva == 0) else \
                            dados_uf["codigo_municipio"].tolist()
                        if idx is not None and idx < len(origem):
                            cod = origem[idx]

                    # NÃO escrever em `hm_busca` aqui: essa é a chave do text_input da aba
                    # Município, que o Streamlit já instanciou neste mesmo ciclo (st.tabs
                    # executa o conteúdo de todas as abas), e escrever na chave de um
                    # widget já criado levanta StreamlitAPIException — derrubando a página
                    # inteira. O código vai para uma chave PRÓPRIA e a aba Município o
                    # consome no início do ciclo seguinte, antes de criar o widget.
                    # A guarda usa uma chave PERSISTENTE (`hm_ultimo_clique`), não a que a
                    # aba Município consome: o Streamlit preserva a seleção do gráfico
                    # entre reruns, então comparar com a chave consumida faria o clique
                    # ser reprocessado a cada ciclo — loop infinito de st.rerun().
                    if cod and st.session_state.get("hm_ultimo_clique") != str(cod):
                        st.session_state["hm_ultimo_clique"] = str(cod)
                        st.session_state["hm_do_mapa"] = str(cod)
                        st.rerun()
                    elif cod:
                        linha = dados_uf[dados_uf["codigo_municipio"] == cod]
                        nome = (linha.iloc[0].get("municipio") or cod) if not linha.empty else cod
                        st.success(
                            f"**{nome}** (IBGE {cod}) enviado para a aba **🔎 Município**.",
                            icon="📍",
                        )

                st.caption(
                    f"**{len(dados_uf)} de {len(dados_uf) + len(sem_dado)} municípios "
                    f"de {uf_mapa}** têm dado de “{nome_ind}”; os "
                    f"{len(sem_dado)} em cinza ainda não têm BDGD processada. "
                    "**Clique em qualquer município** para levá-lo à aba Município — a "
                    "consulta de COSIP funciona mesmo sem o parque. "
                    "Malha: IBGE, subdivisão municipal, qualidade mínima."
                    + (f"  A rampa de cor está cortada nos percentis 5–95 "
                       f"({pref}{indicadores.formatar_numero(z_min, casas)}{suf} a "
                       f"{pref}{indicadores.formatar_numero(z_max, casas)}{suf}) para que "
                       "outliers não achatem o gradiente; os valores reais estão no hover."
                       if cortou else "")
                )

                r = st.columns(4)
                r[0].metric("Municípios no mapa", indicadores.formatar_numero(len(dados_uf)))
                r[1].metric("Mediana",
                            f"{pref}{indicadores.formatar_numero(dados_uf[coluna].median(), casas)}{suf}")
                maior = dados_uf.nlargest(1, coluna).iloc[0]
                menor = dados_uf.nsmallest(1, coluna).iloc[0]
                r[2].metric("Maior", f"{pref}{indicadores.formatar_numero(maior[coluna], casas)}{suf}",
                            help=str(maior.get("municipio") or maior["codigo_municipio"]))
                r[3].metric("Menor", f"{pref}{indicadores.formatar_numero(menor[coluna], casas)}{suf}",
                            help=str(menor.get("municipio") or menor["codigo_municipio"]))


# ═════════════════════════════════════════════════════════════════════════════
# ABA 3 — comparação entre municípios
# ═════════════════════════════════════════════════════════════════════════════
with aba_comparar:
    st.markdown("**Selecione os municípios** — por UF, por lista ou via planilha de códigos IBGE.")

    modo = st.radio("Origem", ["Por UF", "Planilha de códigos"],
                    horizontal=True, label_visibility="collapsed", key="hm_modo")

    codigos: list[str] = []
    if modo == "Por UF":
        c1, c2 = st.columns([1, 3])
        entes = _entes()
        ufs = sorted(entes["uf"].dropna().unique().tolist()) if not entes.empty else []
        uf = c1.selectbox("UF", ufs, index=ufs.index("MG") if "MG" in ufs else 0,
                          key="hm_cmp_uf")
        candidatos = entes[entes["uf"] == uf].copy()
        if not parque.empty:
            com_parque = set(parque["codigo_municipio"])
            candidatos["tem_bdgd"] = candidatos["cod_ibge"].isin(com_parque)
            só_bdgd = c2.checkbox(
                "Somente municípios com BDGD processada", value=True, key="hm_so_bdgd",
                help="Sem BDGD não há indicador por ponto — só COSIP absoluta e per capita.",
            )
            if só_bdgd:
                candidatos = candidatos[candidatos["tem_bdgd"]]
            # Ordena pelos MAIORES parques. Cortar a lista na ordem do cadastro daria
            # uma amostra arbitrária de municípios minúsculos, sem valor de triagem.
            tamanho = parque.set_index("codigo_municipio")["pontos_ip"]
            candidatos["_pontos"] = candidatos["cod_ibge"].map(tamanho).fillna(-1)
            candidatos = candidatos.sort_values("_pontos", ascending=False)
        limite = c2.slider("Máximo de municípios", 5, 200, 40, step=5, key="hm_limite",
                           help="Cada município/ano é uma consulta ao SICONFI. "
                                "O cache local evita repetir consultas já feitas.")
        codigos = candidatos["cod_ibge"].head(limite).tolist()
        c2.caption(f"{len(codigos)} município(s) de {len(candidatos)} disponíveis — "
                   "os de maior parque de IP primeiro.")
    else:
        arquivo = st.file_uploader("Planilha com uma coluna de códigos IBGE",
                                   type=["xlsx", "csv"], key="hm_upload")
        if arquivo is not None:
            try:
                df_in = (pd.read_csv(arquivo, dtype=str) if arquivo.name.endswith(".csv")
                         else pd.read_excel(arquivo, dtype=str))
                melhor, escore = None, 0
                for col in df_in.columns:
                    n = int(df_in[col].apply(
                        lambda v: len(siconfi.so_digitos(v)) == 7).sum())
                    if n > escore:
                        melhor, escore = col, n
                if melhor is None:
                    st.error("Nenhuma coluna com códigos IBGE de 7 dígitos foi encontrada.")
                else:
                    codigos = list(dict.fromkeys(
                        c for c in df_in[melhor].map(siconfi.so_digitos) if len(c) == 7))
                    st.success(f"{len(codigos)} código(s) lido(s) da coluna “{melhor}”.")
            except Exception as exc:
                st.error(f"Não foi possível ler a planilha: {exc}")

    if codigos and st.button("Consultar", type="primary", key="hm_consultar"):
        st.session_state["hm_codigos"] = codigos

    if st.session_state.get("hm_codigos"):
        codigos = st.session_state["hm_codigos"]
        cosip = _cosip(tuple(codigos), tuple(sorted(anos)))
        painel = indicadores.cruzar(cosip, parque, tarifa, pot_led)
        ok = painel[(painel["status"] == "OK") & painel["pontos_ip"].notna()]

        # Declarações implausíveis ficam FORA de mediana, scatter e ranking: um ente que
        # informou R$ 16 de COSIP no ano viraria "o pior do estado" e puxaria a mediana.
        suspeitos = ok[ok["declaracao_implausivel"] == True]        # noqa: E712
        ok = ok[ok["declaracao_implausivel"] != True]               # noqa: E712

        r1 = st.columns(4)
        r1[0].metric("Municípios", indicadores.formatar_numero(painel["codigo_municipio"].nunique()))
        r1[1].metric("Com COSIP declarada",
                     indicadores.formatar_numero(painel[painel.status == "OK"]["codigo_municipio"].nunique()))
        r1[2].metric("Pontos de IP somados",
                     indicadores.formatar_numero(ok.drop_duplicates("codigo_municipio")["pontos_ip"].sum()))
        r1[3].metric("R$/ponto/ano (mediana)",
                     indicadores.formatar_moeda(ok["cosip_por_ponto_ano"].median()))

        if not suspeitos.empty:
            nomes = ", ".join(sorted(suspeitos["municipio"].dropna().unique()))
            st.error(
                f"**{len(suspeitos)} declaração(ões) implausível(is) excluída(s) dos gráficos e "
                f"da mediana:** {nomes}. O valor informado ao SICONFI fica abaixo de "
                f"{indicadores.formatar_moeda(indicadores.PISO_PLAUSIBILIDADE_POR_PONTO)} por "
                "ponto por ano — é erro de preenchimento do DCA pelo ente, não baixa "
                "arrecadação. Os registros seguem visíveis na tabela, marcados na coluna "
                "`declaracao_implausivel`.",
                icon="🚩",
            )

        if ok.empty:
            st.info("Nenhum município com COSIP declarada **e** BDGD processada no recorte atual. "
                    "A tabela abaixo mostra o que foi retornado.")
        else:
            ano_foco = st.selectbox("Exercício em foco", sorted(ok["ano_exercicio"].unique(),
                                                               reverse=True), key="hm_ano_foco")
            foco = ok[ok["ano_exercicio"] == ano_foco]

            st.markdown("**Eficiência do parque × capacidade de custeio**")
            fig = go.Figure(go.Scatter(
                x=foco["consumo_kwh_ponto_ano"], y=foco["cosip_por_ponto_ano"],
                mode="markers", customdata=foco[["municipio", "pontos_ip", "perc_led"]],
                marker=dict(
                    size=foco["pontos_ip"], sizemode="area",
                    sizeref=2.0 * max(foco["pontos_ip"].max(), 1) / (42.0 ** 2), sizemin=8,
                    color=foco["perc_led"] * 100, colorscale=ESCALA_LED,
                    cmin=0, cmax=100, line=dict(width=2, color="#12192b"),
                    colorbar=dict(title=dict(text="% LED", font=dict(color=TINTA_FRACA)),
                                  tickfont=dict(color=TINTA_FRACA), thickness=12, len=0.8),
                ),
                hovertemplate=("<b>%{customdata[0]}</b><br>"
                               "Consumo: %{x:,.0f} kWh/ponto/ano<br>"
                               "COSIP: R$ %{y:,.0f}/ponto/ano<br>"
                               "Parque: %{customdata[1]:,.0f} pontos<extra></extra>"),
            ))
            mediana_y = foco["cosip_por_ponto_ano"].median()
            fig.add_hline(y=mediana_y, line_dash="dot", line_color=GRADE,
                          annotation_text="mediana R$/ponto",
                          annotation_font=dict(color=TINTA_FRACA, size=11))
            _layout_base(fig, 430)
            fig.update_xaxes(title_text="consumo por ponto (kWh/ano)", showgrid=True,
                             gridcolor=GRADE, tickformat=",.0f",
                             title_font=dict(color=TINTA_FRACA))
            fig.update_yaxes(title_text="COSIP por ponto (R$/ano)", tickprefix="R$ ",
                             tickformat=",.0f", title_font=dict(color=TINTA_FRACA))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "Cada bolha é um município; o tamanho é o número de pontos de IP. À direita "
                "está o parque que gasta mais energia por ponto; embaixo, o que arrecada menos "
                "por ponto. O quadrante inferior direito reúne os casos em que a COSIP tem de "
                "custear um parque caro — onde uma PPP costuma não fechar sem revisão da "
                "contribuição."
            )

            st.markdown(f"**Cobertura do custo de energia — {ano_foco}**")
            # Municípios com consumo inconsistente na BDGD apareceriam aqui como falsos
            # críticos: a cobertura baixa viria do consumo errado, não da arrecadação.
            confiavel = foco[foco["consumo_bdgd_suspeito"] != True]     # noqa: E712
            fora_da_faixa = len(foco) - len(confiavel)
            # Em barra horizontal o plotly empilha o primeiro item embaixo; ordenar
            # decrescente é o que põe o município MAIS crítico no topo da lista.
            rank = (confiavel.nsmallest(20, "cobertura_energia")
                             .sort_values("cobertura_energia", ascending=False))
            cores = [STATUS_CRITICO if c < 1 else STATUS_ATENCAO if c < 1.5 else STATUS_BOM
                     for c in rank["cobertura_energia"]]
            fig2 = go.Figure(go.Bar(
                x=rank["cobertura_energia"], y=rank["municipio"], orientation="h",
                marker_color=cores, marker_cornerradius=4,
                text=[_br(f"{c:.2f}") + "×" for c in rank["cobertura_energia"]],
                textposition="outside", textfont=dict(color=TINTA_SECUNDARIA),
                hovertemplate="%{y}: %{x:.2f}× a conta de energia<extra></extra>",
            ))
            fig2.add_vline(x=1.0, line_dash="dot", line_color=TINTA_FRACA)
            _layout_base(fig2, max(300, 26 * len(rank)))
            fig2.update_xaxes(title_text="COSIP ÷ custo de energia (×)", showgrid=True,
                              gridcolor=GRADE, title_font=dict(color=TINTA_FRACA))
            fig2.update_yaxes(showgrid=False)
            fig2.update_layout(showlegend=False, bargap=0.3)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "🔴 abaixo de 1,00× a COSIP não cobre a energia  ·  🟠 entre 1,00× e 1,50× "
                "cobre a energia mas sobra pouco para O&M  ·  🟢 acima de 1,50×. "
                "Os 20 municípios de menor cobertura."
                + (f"  {fora_da_faixa} município(s) ficaram fora do gráfico por consumo "
                   "inconsistente na BDGD — ver a aba Base de dados."
                   if fora_da_faixa else "")
            )

        st.markdown("**Tabela**")
        st.dataframe(painel, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️  Baixar comparativo (.xlsx)", _excel(painel),
            file_name=f"hub_comparativo_{'_'.join(map(str, sorted(anos)))}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="hm_dl_cmp",
        )


# ═════════════════════════════════════════════════════════════════════════════
# ABA 3 — estado das bases
# ═════════════════════════════════════════════════════════════════════════════
with aba_base:
    st.markdown("### Bases de dados do módulo")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SICONFI — DCA Anexo I-C**")
        entes = _entes()
        st.write(f"Cadastro de entes: **{len(entes):,}** municípios".replace(",", "."))
        cache = siconfi.carregar_cache()
        if cache.empty:
            st.caption("Nenhuma consulta em cache ainda.")
        else:
            st.write(f"Consultas em cache: **{len(cache):,}** pares município/ano".replace(",", "."))
            st.dataframe(cache["status"].value_counts().rename("pares").reset_index(),
                         hide_index=True, use_container_width=True)

    with c2:
        st.markdown("**BDGD — entidade PIP**")
        if parque.empty:
            st.caption("Nenhuma base processada.")
        else:
            # valores de 7+ dígitos são truncados pelo st.metric — abrevia
            pontos_tot = float(parque["pontos_ip"].sum())
            m = st.columns(3)
            m[0].metric("Municípios", indicadores.formatar_numero(len(parque)))
            m[1].metric("Pontos de IP", _br(f"{pontos_tot / 1e6:,.2f}") + " mi",
                        help=indicadores.formatar_numero(pontos_tot) + " pontos")
            m[2].metric("Consumo",
                        _br(f"{parque['consumo_kwh_ano'].sum() / 1e6:,.0f}") + " GWh/ano")
            if "perc_led" in parque.columns and parque["perc_led"].notna().any():
                validos = parque[parque["perc_led"].notna()]
                pond = (validos["perc_led"] * validos["pontos_ip"]).sum() / validos["pontos_ip"].sum()
                st.caption(f"Parque em LED: **{_pct(pond)}** dos pontos (ponderado) · "
                           f"{_pct(parque['perc_led'].median())} (mediana municipal)")

    if not parque.empty:
        st.markdown("**Cobertura por UF**")
        if "uf" in parque.columns:
            por_uf = (parque.groupby("uf")
                      .agg(municipios=("codigo_municipio", "nunique"),
                           pontos_ip=("pontos_ip", "sum"),
                           GWh_ano=("consumo_kwh_ano", lambda s: s.sum() / 1e6),
                           W_medio=("potencia_media_w", "median"),
                           perc_led=("perc_led", "median"))
                      .sort_values("pontos_ip", ascending=False).reset_index())
            por_uf["perc_led"] = por_uf["perc_led"] * 100      # a coluna vem em 0–1
            st.dataframe(por_uf, hide_index=True, use_container_width=True,
                         column_config={
                             "perc_led": st.column_config.NumberColumn("LED (mediana)",
                                                                       format="%.0f%%"),
                             "GWh_ano": st.column_config.NumberColumn("GWh/ano", format="%.1f"),
                             "W_medio": st.column_config.NumberColumn("W/ponto", format="%.0f"),
                         })

        # Consistência física por distribuidora: horas equivalentes = consumo ÷ carga.
        # IP com relé fotoelétrico opera 11-12 h/dia. Fora de 3.000-5.000 h/ano, o consumo
        # ou a carga declarados à ANEEL estão errados — e aí o custo de energia estimado
        # (logo, a cobertura da COSIP) não vale para os municípios daquela concessionária.
        por_dist = (parque.groupby(["distribuidora", "data_base_bdgd", "versao_bdgd"])
                    .agg(municipios=("codigo_municipio", "nunique"),
                         pontos_ip=("pontos_ip", "sum"),
                         GWh_ano=("consumo_kwh_ano", lambda s: s.sum() / 1e6),
                         horas_ano=("horas_equivalentes_ano", "median"),
                         W_ponto=("potencia_media_w", "median"))
                    .sort_values("pontos_ip", ascending=False).reset_index())
        por_dist["consistencia"] = por_dist["horas_ano"].apply(
            lambda h: "ok" if pd.notna(h) and 3000 <= h <= 5000 else "verificar")

        suspeitas = por_dist[por_dist["consistencia"] == "verificar"]
        if not suspeitas.empty:
            st.warning(
                f"**{len(suspeitas)} distribuidora(s) com consumo declarado fora da faixa "
                "física da IP** (3.000–5.000 h/ano de operação equivalente): "
                + ", ".join(f"{r.distribuidora} ({r.horas_ano:,.0f} h)".replace(",", ".")
                            for r in suspeitas.itertuples())
                + ". Nos municípios dessas concessionárias o **custo de energia estimado — e "
                "portanto a cobertura da COSIP — não é confiável**. O número de pontos e a "
                "carga instalada seguem utilizáveis.",
                icon="⚠️",
            )

        with st.expander(f"Por distribuidora ({parque['distribuidora'].nunique()})"):
            st.dataframe(
                por_dist, hide_index=True, use_container_width=True,
                column_config={
                    "GWh_ano": st.column_config.NumberColumn("GWh/ano", format="%.1f"),
                    "horas_ano": st.column_config.NumberColumn("h/ano equiv.", format="%.0f"),
                    "W_ponto": st.column_config.NumberColumn("W/ponto", format="%.0f"),
                })
            st.caption(
                "`h/ano equiv.` = consumo ÷ carga instalada, mediana municipal. É o teste de "
                "sanidade do dado: IP acionada por relé fotoelétrico opera 11–12 h/dia."
            )

    st.divider()
    st.markdown(
        f"""
**Como atualizar a BDGD**

1. Coloque os arquivos `.gdb` da ANEEL em `{config.BDGD_BRUTOS}`, mantendo o nome original
   (`Distribuidora_Código_DataBase_Versão_Timestamp.gdb`) — é dele que saem distribuidora,
   data-base e versão.
2. A partir de `app/`, rode `py -m hub_municipios.etl_bdgd`.

O ETL roda **fora** do portal e exige GDAL (OSGeo4W ou QGIS). O Streamlit lê apenas o
agregado municipal já processado, de algumas centenas de KB — nenhuma dependência
geoespacial entra no `requirements.txt` da plataforma.
"""
    )

    with st.expander("O que os indicadores significam — e o que eles não provam"):
        st.markdown(
            """
| Indicador | Cálculo | Leitura |
|---|---|---|
| **R$/ponto/ano** | COSIP líquida ÷ pontos de IP | Cotado na mesma unidade da contraprestação de uma PPP. |
| **R$/habitante/ano** | COSIP líquida ÷ população | Compara entes de portes diferentes. |
| **Cobertura da energia** | COSIP ÷ (consumo × tarifa) | Abaixo de 1,00× a contribuição não paga nem a conta de luz. |
| **Saldo após energia** | COSIP − custo de energia | O que resta para O&M, modernização e contraprestação. |
| **Economia potencial** | consumo atual − consumo a LED de referência | Triagem, não projeto luminotécnico. |

**Ressalvas que acompanham qualquer número daqui:**

1. **Valores nominais.** Série plurianual exige deflacionamento (IPCA/IGP-M).
2. **O DCA é declaratório e não auditado.** Reclassificação de rubrica pelo ente aparece
   como oscilação de receita — cruze com o balancete municipal antes de projetar.
3. **Receita arrecadada ≠ faturada.** A inadimplência da COSIP está embutida no dado; a
   base faturada só vem da concessionária de energia.
4. **A BDGD é declaratória da distribuidora** e tem data-base fixa. Comparar COSIP de um
   ano com parque de outro embute a defasagem — ela é sinalizada em cada ficha.
5. **COSIP arrecadada não é garantia de bancabilidade.** Ela limita o teto de
   contraprestação, mas o que sustenta o financiamento é o mecanismo de vinculação da
   arrecadação (conta vinculada, fundo garantidor), não o valor bruto.
6. **Ausência de COSIP no anexo não prova ausência de lei instituidora** — pode ser falha
   de declaração. Confirme na legislação municipal.
"""
        )
