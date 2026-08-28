"""
Hub de Municípios — triagem de pré-viabilidade de PPP de iluminação pública.

Automatiza a planilha `CLP.xlsx` e o painel "SMART RMBH": cruza a arrecadação de COSIP
declarada ao SICONFI com o parque de IP da BDGD/ANEEL e classifica a viabilidade.

Esta página é apenas a UI; a lógica vive em `hub_municipios/`.
"""

from __future__ import annotations

import io
import traceback
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hub_municipios import bdgd, config, estimativa, indicadores, malhas, ppp, siconfi

# ── Paleta ───────────────────────────────────────────────────────────────────
# Viabilidade é uma escala de decisão, não uma identidade arbitrária: cada classe
# recebe a cor do que ela significa para quem prospecta.
COR_VIABILIDADE = {
    indicadores.VIABILIDADE_VIAVEL: "#17A672",          # verde — alvo
    indicadores.VIABILIDADE_VIABILIZAVEL: "#1F9ED1",    # teal — alvo com ressalva
    indicadores.VIABILIDADE_NAO_VIAVEL: "#DC5B52",      # vermelho — arrecadação insuficiente
    indicadores.VIABILIDADE_JA_TEM_PPP: "#8B5CF6",      # violeta — fora do escopo
    indicadores.VIABILIDADE_SEM_CIP: "#64748B",         # cinza — sem dado de arrecadação
}
ORDEM_VIABILIDADE = [
    indicadores.VIABILIDADE_VIAVEL, indicadores.VIABILIDADE_VIABILIZAVEL,
    indicadores.VIABILIDADE_NAO_VIAVEL, indicadores.VIABILIDADE_JA_TEM_PPP,
    indicadores.VIABILIDADE_SEM_CIP,
]

# Categórica validada para fundo escuro (surface #12192b): banda de lightness OKLCH
# 0,48–0,67, croma >= 0,10, ΔE de CVD >= 8 em todos os pares adjacentes.
COR_TECNOLOGIA = {
    "LED": "#1F9ED1", "Vapor de sódio": "#C08420",
    "Vapor de mercúrio": "#8B5CF6", "Vapor metálico": "#17A672",
}
COR_AUSENTE = "#64748B"
ESCALA_MAGNITUDE = ["#0E2A3A", "#186F96", "#1F9ED1", "#7FCDEA"]

TINTA_PRIMARIA, TINTA_SECUNDARIA, TINTA_FRACA = "#f8fafc", "#cbd5e1", "#94a3b8"
GRADE, SUPERFICIE = "#1f2937", "rgba(0,0,0,0)"
ANO_ATUAL = datetime.now().year

fmt_moeda = indicadores.formatar_moeda
fmt_num = indicadores.formatar_numero
fmt_pct = indicadores.formatar_percentual


def _compacto(v) -> str:
    """R$ 236,4 mi — st.metric trunca string longa, e valor de 9 dígitos não cabe."""
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    for limite, sufixo in ((1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if abs(v) >= limite:
            return "R$ " + indicadores._br(f"{v / limite:,.1f}") + f" {sufixo}"
    return fmt_moeda(v)


# ── Estilo: rótulos de filtro legíveis ───────────────────────────────────────
# O tema escuro do portal deixa os labels de widget em cinza de baixo contraste, e a
# barra de filtros somia no fundo. Isto sobe o contraste só dos rótulos e dá peso ao
# painel lateral, sem tocar no CSS global do app.
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] label p,
      section[data-testid="stSidebar"] .stMarkdown p { color: #e2e8f0 !important; }
      section[data-testid="stSidebar"] label p { font-weight: 600 !important; font-size: .88rem !important; }
      div[data-testid="stMetric"] { background: rgba(18,25,43,.55); border-radius: 12px; }
      div[data-testid="stMetricLabel"] p { color: #94a3b8 !important; font-size: .78rem !important;
                                           text-transform: uppercase; letter-spacing: .04em; }
      button[data-baseweb="tab"] p { font-size: 1rem !important; font-weight: 600 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_data(show_spinner=False)
def _ppps() -> pd.DataFrame:
    return ppp.carregar()


@st.cache_data(show_spinner="Consultando o SICONFI…")
def _cosip(codigos: tuple, anos: tuple) -> pd.DataFrame:
    return siconfi.consultar_com_cache(list(codigos), list(anos))


@st.cache_data(show_spinner="Carregando a malha do IBGE…")
def _malha(uf: str):
    return malhas.carregar(uf)


def _excel(painel: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        painel.to_excel(writer, sheet_name="Triagem", index=False)
    return buffer.getvalue()


def _layout(fig: go.Figure, altura: int = 320) -> go.Figure:
    fig.update_layout(
        height=altura, separators=",.", margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=SUPERFICIE, plot_bgcolor=SUPERFICIE,
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


def _selo_viabilidade(classe: str) -> str:
    cor = COR_VIABILIDADE.get(classe, COR_AUSENTE)
    return (f'<span style="background:{cor}22;color:{cor};border:1px solid {cor}66;'
            f'padding:.25rem .7rem;border-radius:999px;font-size:.82rem;'
            f'font-weight:700;">{classe}</span>')


@contextmanager
def _aba_isolada(nome: str):
    """
    Contém uma falha dentro da aba onde ela aconteceu.

    O `st.tabs` executa o conteúdo de TODAS as abas no mesmo ciclo, então qualquer
    exceção sobe até o entry point e derruba a página inteira — o usuário perde até
    as abas que estavam funcionando, e a mensagem no Streamlit Cloud vem redigida.
    Isso já aconteceu duas vezes aqui (escrita em chave de widget já instanciado e
    dtype recusando ausência), então a proteção é estrutural, não pontual.

    As exceções de controle de fluxo do próprio Streamlit (`st.rerun`, `st.stop`)
    precisam passar intactas, senão a navegação para de funcionar.
    """
    try:
        yield
    except Exception as exc:
        if type(exc).__name__ in ("RerunException", "StopException", "RerunData"):
            raise
        st.error(f"**Esta aba falhou.** As demais seguem utilizáveis.\n\n"
                 f"`{type(exc).__name__}: {exc}`", icon="🚨")
        with st.expander("Detalhes técnicos"):
            st.code(traceback.format_exc(), language="text")


# ═════════════════════════════════════════════════════════════════════════════
# CABEÇALHO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:2.2rem;font-weight:800;
                  background:linear-gradient(90deg,#ffffff,#00A9E0);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:-.5px;">Hub de Municípios</div>
      <div style="font-size:.95rem;color:#94a3b8;margin-top:.3rem;">
        Triagem de pré-viabilidade de PPP de iluminação pública — arrecadação de COSIP
        (SICONFI) cruzada com o parque de IP (BDGD/ANEEL)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

parque = _parque()
contratos = _ppps()

# ═════════════════════════════════════════════════════════════════════════════
# PARÂMETROS — na barra lateral, sempre visíveis
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros da triagem")

    anos = st.multiselect(
        "Exercícios da COSIP",
        options=list(range(ANO_ATUAL, 2012, -1)),
        default=[ANO_ATUAL - 2, ANO_ATUAL - 1],
        help="A COSIP vem do DCA anual. O exercício mais recente leva meses para ser "
             "publicado pelo Tesouro.",
    )

    ref = ppp.referencia_de_custo()
    ajuda_custo = ("Quanto a PPP custa por ponto de iluminação, por mês. É o único "
                   "parâmetro financeiro da triagem: subtraído da arrecadação por "
                   "ponto, dá quanto sobra.")
    if ref:
        ajuda_custo += (f" Referência dos {ref['n']} contratos de PPP já assinados: "
                        f"mediana R$ {ref['mediana']:.2f}, "
                        f"quartis R$ {ref['p25']:.2f}–{ref['p75']:.2f}. "
                        "A planilha CLP usa R$ 38,00; o estudo da RMBH, R$ 32,00.")
    custo_ppp = st.number_input(
        "Custo da PPP (R$/ponto.mês)", min_value=1.0, max_value=300.0,
        value=config.CUSTO_PPP_PONTO_MES_PADRAO, step=1.0, format="%.2f",
        help=ajuda_custo,
    )

    corte = st.number_input(
        "Arrecadação mínima (R$ milhões/ano)", min_value=0.0, max_value=100.0,
        value=indicadores.CORTE_ARRECADACAO_PADRAO / 1e6, step=0.5, format="%.1f",
        help="Abaixo deste valor o município entra como “Viabilizável”: a escala não "
             "sustenta uma PPP sozinha, mas consórcio regional ou revisão da lei da "
             "COSIP podem viabilizar. A metodologia da CLP usa R$ 4,5 milhões.",
    ) * 1e6

    with st.expander("Energia e modernização"):
        pot_futura = st.number_input(
            "Potência média futura por ponto (W)", min_value=15.0, max_value=250.0,
            value=config.POTENCIA_FUTURA_PADRAO_W, step=5.0,
            help="Potência média por ponto depois da troca das luminárias. Serve para "
                 "estimar quanto de energia o município deixaria de gastar. Faixa de "
                 "mercado após LED: 45 a 75 W em via urbana.",
        )
        tarifa = st.number_input(
            "Tarifa de energia (R$/kWh)", min_value=0.10, max_value=3.00,
            value=config.TARIFA_ENERGIA_PADRAO, step=0.01, format="%.2f",
            help="Tarifa B4a com tributos. Usada só no bloco de energia — não entra na "
                 "conta da sobra da CIP. A tarifa real vem da resolução homologatória da "
                 "distribuidora e do tratamento de ICMS que o estado dá à iluminação "
                 "pública; a faixa nacional vai de ~R$ 0,45 a ~R$ 0,95.",
        )
        fator_co2 = st.number_input(
            "Fator de emissão do SIN (tCO₂/MWh)", min_value=0.0, max_value=1.0,
            value=config.FATOR_EMISSAO_SIN_PADRAO, step=0.0010, format="%.4f",
            help="Fator MÉDIO de emissão da geração elétrica do SIN, publicado pelo MCTI "
                 "no SIRENE e adotado pelo Programa Brasileiro GHG Protocol para "
                 "inventário de escopo 2. Valor de 2024: 0,0545 tCO₂/MWh. Não confundir "
                 "com o fator da margem de operação (0,02–0,03), que serve a projetos de "
                 "MDL, não a inventário. Oscila com o despacho térmico: sobe em ano seco. "
                 "Referência internacional: EUA 0,3674 e Alemanha 0,321 — a matriz "
                 "brasileira é ~88% renovável.",
        )

    st.markdown("---")
    st.caption(
        f"**{len(parque):,}** municípios com parque medido na BDGD  ·  "
        f"**{len(contratos)}** PPPs de IP contratadas".replace(",", ".")
    )

if not anos:
    st.info("Selecione ao menos um exercício na barra lateral.")
    st.stop()

aba_municipio, aba_mapa, aba_carteira, aba_ppp, aba_base = st.tabs(
    ["🔎 Município", "🗺️ Mapa", "📊 Carteira", "🤝 PPPs contratadas", "🗄️ Base de dados"]
)


# ═════════════════════════════════════════════════════════════════════════════
# ABA 1 — ficha do município
# ═════════════════════════════════════════════════════════════════════════════
with aba_municipio, _aba_isolada("Município"):
    # Município escolhido no mapa: consumido AQUI, antes de o text_input existir —
    # escrever na chave de um widget já instanciado levanta StreamlitAPIException.
    vindo_do_mapa = st.session_state.pop("hm_do_mapa", None)
    if vindo_do_mapa:
        st.session_state["hm_busca"] = str(vindo_do_mapa)
        st.session_state["hm_uf"] = "(todas)"

    b1, b2 = st.columns([3, 1])
    termo = b1.text_input("Buscar município", placeholder="Nome ou código IBGE (7 dígitos)",
                          key="hm_busca")
    ufs = sorted(_entes()["uf"].dropna().unique().tolist()) if not _entes().empty else []
    uf_filtro = b2.selectbox("UF", ["(todas)"] + ufs, key="hm_uf")

    escolhido = None
    if termo.strip():
        achados = siconfi.buscar_municipio(termo, None if uf_filtro == "(todas)" else uf_filtro)
        if achados.empty:
            st.warning("Nenhum município encontrado. Verifique a grafia ou use o código IBGE.")
        else:
            opcoes = {f"{r.ente} / {r.uf}  ·  {r.cod_ibge}": r.cod_ibge
                      for r in achados.head(50).itertuples()}
            rotulo = st.selectbox(f"{len(achados)} resultado(s)", list(opcoes), key="hm_escolha")
            escolhido = opcoes[rotulo]

    if escolhido:
        painel = indicadores.cruzar(_cosip((escolhido,), tuple(sorted(anos))),
                                    parque, custo_ppp, pot_futura, corte,
                                    tarifa, fator_co2)
        if painel.empty:
            st.error("Nenhum dado retornado para este município.")
        else:
            r = painel.sort_values("ano_exercicio").iloc[-1]

            cab1, cab2 = st.columns([3, 1])
            cab1.markdown(f"### {r['municipio']} / {r['uf']}")
            cab2.markdown(f"<div style='text-align:right;padding-top:1.1rem'>"
                          f"{_selo_viabilidade(r['viabilidade'])}</div>",
                          unsafe_allow_html=True)

            origem = r.get("origem_pontos")
            legenda = [f"IBGE {r['codigo_municipio']}"]
            if pd.notna(r.get("populacao")):
                legenda.append(f"{fmt_num(r['populacao'])} habitantes")
            if origem == estimativa.ORIGEM_MEDIDA and pd.notna(r.get("distribuidora")):
                legenda.append(f"{r['distribuidora']} · BDGD {int(r['ano_base_bdgd'])}")
            elif origem == estimativa.ORIGEM_ESTIMADA:
                legenda.append("parque **estimado** pela população")
            st.caption("  ·  ".join(legenda))

            # ── A conta, em quatro números ──────────────────────────────────
            st.markdown("#### A conta")
            k = st.columns(4, border=True)
            k[0].metric(f"COSIP {int(r['ano_exercicio'])}", _compacto(r["cosip_liquida"]),
                        help=fmt_moeda(r["cosip_liquida"]) + " no exercício")
            # a unidade vai no rótulo: o st.metric trunca o valor com reticências
            k[1].metric("Arrecadação por ponto.mês", fmt_moeda(r["cosip_ponto_mes"]),
                        help="COSIP líquida ÷ (pontos × 12)")
            k[2].metric("Custo da PPP por ponto.mês", fmt_moeda(r["custo_ppp_ponto_mes"]),
                        help="Parâmetro informado na barra lateral")
            sobra_pct = r.get("sobra_percentual")
            k[3].metric("SOBRA DA CIP", fmt_pct(sobra_pct),
                        delta=f"{fmt_moeda(r['sobra_ponto_mes'])}/ponto.mês"
                              if pd.notna(r.get("sobra_ponto_mes")) else None,
                        help="Quanto da arrecadação sobra depois de pagar a "
                             "contraprestação da PPP.")

            k2 = st.columns(4, border=True)
            k2[0].metric("Pontos de IP", fmt_num(r.get("pontos_ip")),
                         help=("Medido na BDGD" if origem == estimativa.ORIGEM_MEDIDA
                               else "Estimado pela população — não medido"))
            k2[1].metric("Contraprestação estimada", _compacto(r.get("contraprestacao_mes")),
                         help="pontos × custo da PPP, por mês")
            k2[2].metric("Sobra anual", _compacto(r.get("sobra_reais_ano")),
                         help="O que resta por ano para outras soluções — cidade "
                              "inteligente, expansão, contrapartidas.")
            k2[3].metric("Carga média por ponto",
                         f"{fmt_num(r.get('potencia_media_w'), 1)} W" if pd.notna(
                             r.get("potencia_media_w")) else "—",
                         help="Carga instalada ÷ pontos, incluindo perdas de reator e "
                              "relé. Acima de ~130 W predomina descarga; abaixo de "
                              "~80 W, LED.")

            # ── Modernização ────────────────────────────────────────────────
            # ── Energia hoje ────────────────────────────────────────────────
            if pd.notna(r.get("consumo_estimado_kwh_ano")):
                st.markdown("#### Energia")
                e = st.columns(4, border=True)
                e[0].metric("Consumo estimado",
                            f"{fmt_num(r['consumo_estimado_kwh_ano'] / 1000)} MWh/ano",
                            help="Carga instalada × 4.160 h/ano de operação. Derivado da "
                                 "carga, não do campo de energia da BDGD — que é "
                                 "inconsistente em um terço dos municípios.")
                e[1].metric("Custo da energia", _compacto(r.get("custo_energia_ano")) + "/ano",
                            help=f"À tarifa de {fmt_moeda(tarifa)}/kWh informada.")
                e[2].metric("Custo por ponto.mês",
                            fmt_moeda(r.get("custo_energia_ponto_mes")),
                            help="Para comparar com a arrecadação por ponto: é quanto da "
                                 "CIP a conta de luz consome.")
                declarado = r.get("consumo_kwh_ano")
                if pd.notna(declarado) and float(declarado) > 0:
                    desvio = (r["consumo_estimado_kwh_ano"] - declarado) / declarado
                    e[3].metric("Consumo declarado à ANEEL",
                                f"{fmt_num(declarado / 1000)} MWh/ano",
                                delta=f"{fmt_pct(-desvio)} vs. estimado",
                                delta_color="off")
                else:
                    e[3].metric("Consumo declarado à ANEEL", "—",
                                help="A distribuidora não declarou consumo utilizável.")

            # ── Modernização e emissões ─────────────────────────────────────
            econ_pct = r.get("economia_percentual")
            if pd.notna(econ_pct) and float(econ_pct) > 0:
                st.markdown("#### Se o parque fosse modernizado")
                m = st.columns(4, border=True)
                m[0].metric(f"Potência média hoje → {fmt_num(pot_futura)} W",
                            f"{fmt_num(r['potencia_media_w'], 0)} W")
                m[1].metric("Economia de energia", fmt_pct(econ_pct),
                            help="Redução proporcional da carga instalada.")
                m[2].metric("Economia anual",
                            _compacto(r.get("economia_reais_ano")),
                            delta=f"{fmt_num(r.get('economia_kwh_ano', 0) / 1000)} MWh/ano",
                            delta_color="off")
                m[3].metric("CO₂ evitado",
                            f"{fmt_num(r.get('co2_evitado_t_ano'), 1)} t/ano",
                            help=f"Economia de energia × fator de emissão do SIN "
                                 f"({fmt_num(fator_co2, 4)} tCO₂/MWh).")
                st.caption(
                    f"Trocar as luminárias levaria a potência média de "
                    f"{fmt_num(r['potencia_media_w'], 0)} W para {fmt_num(pot_futura)} W por "
                    f"ponto — **{fmt_pct(econ_pct)} menos energia**. "
                    "Estimativa de triagem: não substitui projeto luminotécnico nem "
                    "considera demanda reprimida. **A redução de CO₂ é modesta por mérito "
                    "da matriz brasileira**, predominantemente renovável: a mesma economia "
                    "de kWh num país de matriz fóssil evitaria várias vezes mais emissões. "
                    "O fator do SIN oscila com o despacho térmico e é publicado pelo MCTI."
                )

            for aviso in indicadores.ressalvas(r):
                st.warning(aviso, icon="⚠️")

            st.divider()
            g1, g2 = st.columns([3, 2])

            with g1:
                st.markdown("**Arrecadação por exercício**")
                st.caption("valores em R$ milhões correntes")
                serie = painel[painel["status"] == "OK"].sort_values("ano_exercicio")
                if serie.empty:
                    st.info("Nenhum exercício com COSIP declarada no período escolhido.")
                else:
                    fig = go.Figure()
                    fig.add_bar(x=serie["ano_exercicio"].astype(str),
                                y=serie["cosip_liquida"] / 1e6,
                                customdata=serie["cosip_liquida"],
                                name="COSIP arrecadada", marker_color="#1F9ED1",
                                marker_cornerradius=4, width=.55,
                                hovertemplate="COSIP: R$ %{customdata:,.0f}<extra></extra>")
                    fig.add_scatter(x=serie["ano_exercicio"].astype(str),
                                    y=serie["contraprestacao_ano"] / 1e6,
                                    customdata=serie["contraprestacao_ano"],
                                    name="Contraprestação da PPP", mode="lines+markers",
                                    line=dict(color="#C08420", width=2),
                                    marker=dict(size=9, color="#C08420",
                                                line=dict(width=2, color="#12192b")),
                                    hovertemplate="Contraprestação: R$ %{customdata:,.0f}"
                                                  "<extra></extra>")
                    fig.update_layout(hovermode="x unified", bargap=.35)
                    _layout(fig, 330)
                    fig.update_xaxes(type="category")
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.caption("A distância entre a barra e a linha é o que sobra da COSIP.")

            with g2:
                st.markdown("**Composição do parque**")
                tec = _tecnologia()
                tm = tec[tec["codigo_municipio"] == escolhido] if not tec.empty else pd.DataFrame()
                tm = tm[tm["tecnologia"] != "Não informado"] if not tm.empty else tm
                if tm.empty:
                    st.info("Sem mix tecnológico para este município — a BDGD da "
                            "distribuidora não traz os campos de lâmpada, ou o parque "
                            "foi estimado.")
                else:
                    tm = tm.sort_values("pontos")
                    total = tm["pontos"].sum()
                    fig = go.Figure(go.Bar(
                        x=tm["pontos"], y=tm["tecnologia"], orientation="h",
                        marker_color=[COR_TECNOLOGIA.get(t, COR_AUSENTE) for t in tm["tecnologia"]],
                        marker_cornerradius=4,
                        text=[fmt_pct(p / total) for p in tm["pontos"]],
                        textposition="outside", textfont=dict(color=TINTA_SECUNDARIA),
                        hovertemplate="%{y}: %{x:,.0f} pontos<extra></extra>"))
                    _layout(fig, 330)
                    fig.update_xaxes(showgrid=True, gridcolor=GRADE, title_text="pontos",
                                     range=[0, total * 1.2])
                    fig.update_yaxes(showgrid=False)
                    fig.update_layout(showlegend=False, bargap=.35)
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

            with st.expander("Todos os indicadores"):
                st.dataframe(painel, use_container_width=True, hide_index=True)
            st.download_button("⬇️  Baixar triagem (.xlsx)", _excel(painel),
                               file_name=f"triagem_{r['codigo_municipio']}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet")


# ═════════════════════════════════════════════════════════════════════════════
# ABA 2 — mapa
# ═════════════════════════════════════════════════════════════════════════════
with aba_mapa, _aba_isolada("Mapa"):
    if parque.empty or "uf" not in parque.columns:
        st.info("O mapa depende do parque da BDGD. Rode `py -m hub_municipios.etl_bdgd`.")
    else:
        ufs_disp = sorted(parque["uf"].dropna().unique().tolist())
        m1, m2 = st.columns([1, 3])
        uf_mapa = m1.selectbox("UF", ufs_disp,
                               index=ufs_disp.index("MG") if "MG" in ufs_disp else 0,
                               key="hm_mapa_uf")
        ano_mapa = m2.selectbox("Exercício", sorted(anos, reverse=True), key="hm_mapa_ano")

        malha = _malha(uf_mapa)
        if malha is None:
            st.warning("Não foi possível obter a malha do IBGE. Verifique a conexão.",
                       icon="🌐")
        else:
            codigos_uf = [c for c in _entes()[_entes().uf == uf_mapa]["cod_ibge"]]

            # Uma UF grande são centenas de consultas ao SICONFI. Abrir a aba não pode
            # disparar isso sozinho: o mapa monta com o que já está em cache e a busca
            # do que falta é um clique explícito.
            cache = siconfi.carregar_cache()
            if not cache.empty:
                ja_tem = set(cache[(cache["ano_exercicio"] == ano_mapa) &
                                   (cache["status"] != "ERRO_API")]["codigo_municipio"])
            else:
                ja_tem = set()
            faltam = [c for c in codigos_uf if c not in ja_tem]
            chave_completo = f"hm_mapa_completo_{uf_mapa}_{ano_mapa}"

            if faltam and not st.session_state.get(chave_completo):
                a1, a2 = st.columns([3, 1])
                a1.info(
                    f"**{len(codigos_uf) - len(faltam)} de {len(codigos_uf)} municípios de "
                    f"{uf_mapa}** já consultados. Faltam {len(faltam)} — são consultas ao "
                    "SICONFI, levam alguns minutos na primeira vez e ficam em cache depois.",
                    icon="ℹ️")
                if a2.button(f"Consultar os {len(faltam)}", key=f"hm_go_{uf_mapa}_{ano_mapa}",
                             type="primary"):
                    st.session_state[chave_completo] = True
                    st.rerun()
                consultar = [c for c in codigos_uf if c in ja_tem]
            else:
                consultar = codigos_uf

            # st.stop() aqui mataria a renderização das OUTRAS abas — o conteúdo de
            # todas roda no mesmo ciclo. O caminho vazio precisa ser um else.
            if not consultar:
                painel_uf = pd.DataFrame()
                st.info(f"Nenhum município de {uf_mapa} consultado ainda para {ano_mapa}. "
                        "Use o botão acima para carregar.")
            else:
                with st.spinner(f"Preparando {len(consultar)} municípios de {uf_mapa}…"):
                    painel_uf = indicadores.cruzar(_cosip(tuple(consultar), (ano_mapa,)),
                                                   parque, custo_ppp, pot_futura, corte,
                                    tarifa, fator_co2)

            if painel_uf.empty:
                st.info(f"Sem dados para {uf_mapa}.")
            else:
                resumo = painel_uf["viabilidade"].value_counts()
                cols = st.columns(len(ORDEM_VIABILIDADE), border=True)
                for col, classe in zip(cols, ORDEM_VIABILIDADE):
                    col.metric(classe, fmt_num(resumo.get(classe, 0)))

                fig = go.Figure()
                for classe in ORDEM_VIABILIDADE:
                    sub = painel_uf[painel_uf["viabilidade"] == classe]
                    if sub.empty:
                        continue
                    cor = COR_VIABILIDADE[classe]
                    fig.add_trace(go.Choroplethmap(
                        geojson=malha, featureidkey="properties.codarea",
                        locations=sub["codigo_municipio"], z=[1] * len(sub),
                        colorscale=[[0, cor], [1, cor]], showscale=False,
                        name=classe, legendgroup=classe, showlegend=False,
                        marker=dict(line=dict(color="#0b111e", width=.4), opacity=.85),
                        customdata=sub[["codigo_municipio"]],
                        text=[f"<b>{m}</b><br>{classe}<br>"
                              f"{fmt_moeda(a)}/ponto.mês · sobra {fmt_pct(s)}<br>"
                              f"{fmt_num(p)} pontos ({o or '—'})"
                              for m, a, s, p, o in zip(sub["municipio"], sub["cosip_ponto_mes"],
                                                       sub["sobra_percentual"], sub["pontos_ip"],
                                                       sub["origem_pontos"])],
                        hovertemplate="%{text}<extra></extra>"))

                fig.update_layout(
                    map=dict(style="carto-darkmatter",
                             center=malhas.centro_aproximado(malha),
                             zoom=malhas.zoom_aproximado(malha)),
                    height=560, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor=SUPERFICIE, separators=",.",
                    font=dict(family="Inter, sans-serif", color=TINTA_SECUNDARIA),
                    hoverlabel=dict(bgcolor="#12192b", bordercolor=GRADE,
                                    font=dict(color=TINTA_PRIMARIA)))

                evento = st.plotly_chart(fig, use_container_width=True,
                                         config={"displayModeBar": False},
                                         on_select="rerun", key="hm_mapa_chart")

                legenda = "  ".join(
                    f'<span style="color:{COR_VIABILIDADE[c]};font-size:1.1rem">■</span> '
                    f'<span style="color:#cbd5e1;font-size:.86rem">{c}</span>'
                    for c in ORDEM_VIABILIDADE)
                st.markdown(legenda, unsafe_allow_html=True)
                st.caption(f"{len(painel_uf)} municípios de {uf_mapa}. "
                           "**Clique em um município** para abrir a ficha dele.")

                sel = (evento.get("selection", {}) or {}).get("points", []) if evento else []
                if sel and sel[0].get("customdata"):
                    cod = sel[0]["customdata"][0]
                    # guarda persistente: o Streamlit preserva a seleção entre reruns, e
                    # comparar com a chave consumida geraria loop infinito de st.rerun()
                    if st.session_state.get("hm_ultimo_clique") != str(cod):
                        st.session_state["hm_ultimo_clique"] = str(cod)
                        st.session_state["hm_do_mapa"] = str(cod)
                        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ABA 3 — carteira
# ═════════════════════════════════════════════════════════════════════════════
with aba_carteira, _aba_isolada("Carteira"):
    st.markdown("**Triagem em lote** — por UF ou por planilha de códigos IBGE.")
    modo = st.radio("Origem", ["Por UF", "Planilha de códigos"], horizontal=True,
                    label_visibility="collapsed", key="hm_modo")

    codigos: list[str] = []
    if modo == "Por UF":
        entes = _entes()
        ufs = sorted(entes["uf"].dropna().unique().tolist()) if not entes.empty else []
        c1, c2 = st.columns([1, 3])
        uf = c1.selectbox("UF", ufs, index=ufs.index("MG") if "MG" in ufs else 0,
                          key="hm_cmp_uf")
        codigos = entes[entes["uf"] == uf]["cod_ibge"].tolist()
        c2.caption(f"{len(codigos)} municípios de {uf}. A consulta usa cache local — "
                   "a primeira rodada de uma UF grande leva alguns minutos.")
    else:
        arquivo = st.file_uploader("Planilha com uma coluna de códigos IBGE",
                                   type=["xlsx", "csv"], key="hm_upload")
        if arquivo is not None:
            try:
                df_in = (pd.read_csv(arquivo, dtype=str) if arquivo.name.endswith(".csv")
                         else pd.read_excel(arquivo, dtype=str))
                melhor, escore = None, 0
                for col in df_in.columns:
                    n = int(df_in[col].apply(lambda v: len(siconfi.so_digitos(v)) == 7).sum())
                    if n > escore:
                        melhor, escore = col, n
                if melhor is None:
                    st.error("Nenhuma coluna com códigos IBGE de 7 dígitos.")
                else:
                    codigos = list(dict.fromkeys(
                        c for c in df_in[melhor].map(siconfi.so_digitos) if len(c) == 7))
                    st.success(f"{len(codigos)} códigos lidos da coluna “{melhor}”.")
            except Exception as exc:
                st.error(f"Não foi possível ler a planilha: {exc}")

    if codigos and st.button("Executar triagem", type="primary", key="hm_go"):
        st.session_state["hm_codigos"] = codigos

    if st.session_state.get("hm_codigos"):
        ano_foco = st.selectbox("Exercício", sorted(anos, reverse=True), key="hm_ano_foco")
        painel = indicadores.cruzar(
            _cosip(tuple(st.session_state["hm_codigos"]), (ano_foco,)),
            parque, custo_ppp, pot_futura, corte,
                                    tarifa, fator_co2)

        resumo = painel["viabilidade"].value_counts()
        cols = st.columns(len(ORDEM_VIABILIDADE), border=True)
        for col, classe in zip(cols, ORDEM_VIABILIDADE):
            col.metric(classe, fmt_num(resumo.get(classe, 0)))

        alvos = painel[painel["viabilidade"].isin(
            [indicadores.VIABILIDADE_VIAVEL, indicadores.VIABILIDADE_VIABILIZAVEL])]
        if not alvos.empty:
            r2 = st.columns(3, border=True)
            r2[0].metric("Pontos nos alvos", fmt_num(alvos["pontos_ip"].sum()))
            r2[1].metric("Contraprestação potencial/mês",
                         _compacto(alvos["contraprestacao_mes"].sum()))
            r2[2].metric("Sobra mediana", fmt_pct(alvos["sobra_percentual"].median()))

            st.markdown(f"**Maiores oportunidades — {ano_foco}**")
            top = alvos.nlargest(20, "contraprestacao_mes").sort_values("contraprestacao_mes")
            fig = go.Figure(go.Bar(
                x=top["contraprestacao_mes"], y=top["municipio"], orientation="h",
                marker_color=[COR_VIABILIDADE[v] for v in top["viabilidade"]],
                marker_cornerradius=4,
                text=[fmt_pct(s) + " sobra" for s in top["sobra_percentual"]],
                textposition="outside", textfont=dict(color=TINTA_SECUNDARIA),
                customdata=top[["pontos_ip", "cosip_ponto_mes"]],
                hovertemplate="<b>%{y}</b><br>contraprestação R$ %{x:,.0f}/mês<br>"
                              "%{customdata[0]:,.0f} pontos · "
                              "R$ %{customdata[1]:,.2f}/ponto.mês<extra></extra>"))
            _layout(fig, max(320, 27 * len(top)))
            fig.update_xaxes(title_text="contraprestação estimada (R$/mês)", showgrid=True,
                             gridcolor=GRADE, tickformat=",.0f",
                             title_font=dict(color=TINTA_FRACA),
                             range=[0, top["contraprestacao_mes"].max() * 1.25])
            fig.update_yaxes(showgrid=False)
            fig.update_layout(showlegend=False, bargap=.3)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Ordenado pelo tamanho do contrato potencial. A cor é a classe de "
                       "viabilidade; o rótulo, quanto da COSIP sobra depois da contraprestação.")

        st.dataframe(painel, use_container_width=True, hide_index=True)
        st.download_button("⬇️  Baixar carteira (.xlsx)", _excel(painel),
                           file_name=f"triagem_carteira_{ano_foco}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet", key="hm_dl_cart")


# ═════════════════════════════════════════════════════════════════════════════
# ABA 4 — PPPs contratadas
# ═════════════════════════════════════════════════════════════════════════════
with aba_ppp, _aba_isolada("PPPs contratadas"):
    if contratos.empty:
        st.info("Base de PPPs não importada. Rode `py -m hub_municipios._importar_ppp`.")
    else:
        st.markdown("**Mercado de PPPs de iluminação pública já contratadas**")
        k = st.columns(4, border=True)
        k[0].metric("Contratos", fmt_num(len(contratos)))
        k[1].metric("Pontos sob contrato", fmt_num(contratos["pontos_luz_contrato"].sum()))
        k[2].metric("Despesa mediana",
                    f"{fmt_moeda(contratos['despesa_ponto_mes'].median())}/ponto.mês",
                    help="Valor do contrato ÷ (pontos × vigência × 12). É o benchmark "
                         "derivado de contrato assinado, não de premissa.")
        k[3].metric("Vigência mediana",
                    f"{fmt_num(contratos['vigencia_anos'].median())} anos")

        f1, f2 = st.columns([1, 3])
        ufs_ppp = ["(todas)"] + sorted(contratos["uf"].dropna().unique().tolist())
        uf_ppp = f1.selectbox("UF", ufs_ppp, key="hm_ppp_uf")
        vis = contratos if uf_ppp == "(todas)" else contratos[contratos["uf"] == uf_ppp]

        st.markdown("**Contratos assinados por ano**")
        por_ano = vis.dropna(subset=["ano_assinatura"]).groupby("ano_assinatura").size()
        if not por_ano.empty:
            fig = go.Figure(go.Bar(x=por_ano.index.astype(int).astype(str), y=por_ano.values,
                                   marker_color="#8B5CF6", marker_cornerradius=4,
                                   hovertemplate="%{x}: %{y} contratos<extra></extra>"))
            _layout(fig, 260)
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.dataframe(
            vis[["municipio", "uf", "concessionaria", "pontos_luz_contrato",
                 "valor_contrato_milhoes", "vigencia_anos", "despesa_ponto_mes",
                 "telegestao", "ano_assinatura", "acionistas"]]
            .sort_values("pontos_luz_contrato", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "valor_contrato_milhoes": st.column_config.NumberColumn("valor (R$ mi)",
                                                                        format="%.1f"),
                "despesa_ponto_mes": st.column_config.NumberColumn("R$/ponto.mês",
                                                                   format="%.2f"),
                "telegestao": st.column_config.ProgressColumn("telegestão", min_value=0,
                                                              max_value=1, format="%.0f%%"),
            })
        st.caption("Fonte: `CLP.xlsx`, aba Planilha1, mantida pelo time. Atualizar com "
                   "`py -m hub_municipios._importar_ppp`.")


# ═════════════════════════════════════════════════════════════════════════════
# ABA 5 — base de dados
# ═════════════════════════════════════════════════════════════════════════════
with aba_base, _aba_isolada("Base de dados"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SICONFI — DCA Anexo I-C**")
        st.write(f"Cadastro: **{len(_entes()):,}** municípios".replace(",", "."))
        cache = siconfi.carregar_cache()
        if cache.empty:
            st.caption("Nenhuma consulta em cache ainda.")
        else:
            st.write(f"Cache: **{len(cache):,}** pares município/ano".replace(",", "."))
            st.dataframe(cache["status"].value_counts().rename("pares").reset_index(),
                         hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**BDGD — entidade PIP**")
        if parque.empty:
            st.caption("Nenhuma base processada.")
        else:
            m = st.columns(3, border=True)
            m[0].metric("Municípios", fmt_num(len(parque)))
            m[1].metric("Pontos de IP",
                        indicadores._br(f"{parque['pontos_ip'].sum() / 1e6:,.2f}") + " mi")
            m[2].metric("Distribuidoras", fmt_num(parque["distribuidora"].nunique()))

    if not parque.empty and "uf" in parque.columns:
        st.markdown("**Cobertura por UF**")
        por_uf = (parque.groupby("uf")
                  .agg(municipios=("codigo_municipio", "nunique"),
                       pontos_ip=("pontos_ip", "sum"),
                       W_ponto=("potencia_media_w", "median"),
                       perc_led=("perc_led", "median"))
                  .sort_values("pontos_ip", ascending=False).reset_index())
        por_uf["perc_led"] = por_uf["perc_led"] * 100
        st.dataframe(por_uf, hide_index=True, use_container_width=True,
                     column_config={
                         "perc_led": st.column_config.NumberColumn("LED (mediana)",
                                                                   format="%.0f%%"),
                         "W_ponto": st.column_config.NumberColumn("W/ponto", format="%.0f")})

        por_dist = (parque.groupby("distribuidora")
                    .agg(municipios=("codigo_municipio", "nunique"),
                         pontos_ip=("pontos_ip", "sum"),
                         horas_ano=("horas_equivalentes_ano", "median"))
                    .sort_values("pontos_ip", ascending=False).reset_index())
        suspeitas = por_dist[~por_dist["horas_ano"].between(3000, 5000)]
        if not suspeitas.empty:
            st.warning(
                f"**{len(suspeitas)} de {len(por_dist)} distribuidoras declaram consumo "
                "fora da faixa física da IP** (3.000–5.000 h/ano de operação equivalente). "
                "Isso **não afeta a triagem**, que usa pontos e arrecadação — mas invalida "
                "a leitura de eficiência energética nesses municípios.", icon="⚠️")
        with st.expander(f"Por distribuidora ({len(por_dist)})"):
            st.dataframe(por_dist, hide_index=True, use_container_width=True,
                         column_config={"horas_ano": st.column_config.NumberColumn(
                             "h/ano equiv.", format="%.0f")})

    with st.expander("Metodologia e ressalvas"):
        ref = ppp.referencia_de_custo()
        st.markdown(f"""
**A conta da triagem**

```
arrecadação por ponto (R$/ponto.mês) = COSIP líquida ÷ (pontos × 12)
sobra (R$/ponto.mês)                 = arrecadação por ponto − custo da PPP
sobra (%)                            = sobra ÷ arrecadação por ponto
```

**Classificação**

| Classe | Critério |
|---|---|
| Já possui PPP | contrato vigente na base de {len(contratos)} PPPs |
| Viável | sobra positiva e arrecadação ≥ o corte |
| Viabilizável | arrecadação abaixo do corte — precisa de consórcio ou revisão da COSIP |
| Não viável | tem escala, mas a COSIP não cobre a contraprestação |
| Não possui CIP | sem arrecadação declarada |

**Origem dos pontos de IP** — a coluna `origem_pontos` diz sempre qual é:

- **BDGD**: medido na base da distribuidora ({len(parque):,} municípios).
- **Estimado**: calculado pela população, com a densidade mediana da faixa de porte do
  município. Calibrado sobre os {len(parque):,} municípios medidos — a densidade real vai
  de 148 pontos/mil habitantes nos municípios até 5 mil habitantes a 71 nos acima de
  500 mil. A regra manual de população ÷ 10 equivale a 100 pontos/mil e **subestima o
  parque em ~21% na mediana**, o que **infla** a arrecadação por ponto e faz o município
  parecer mais viável do que é.

**Ressalvas**

1. **Valores nominais.** Série plurianual exige deflacionamento.
2. **O DCA é declaratório e não auditado.** Cruze com o balancete municipal antes de projetar.
3. **Arrecadado ≠ faturado.** A inadimplência da COSIP já está embutida no dado.
4. **A sobra é teto de arrecadação, não espaço fiscal.** Falta descontar a O&M que a
   prefeitura já paga e eventual passivo com a distribuidora.
5. **O custo da PPP é premissa.** Referência dos contratos assinados:
   mediana R$ {ref.get('mediana', 0):.2f}, quartis R$ {ref.get('p25', 0):.2f}–{ref.get('p75', 0):.2f}.
6. **COSIP arrecadada não é garantia de bancabilidade** — o que sustenta o financiamento
   é o mecanismo de vinculação (conta vinculada, fundo garantidor), não o valor bruto.
""".replace(",", "."))
