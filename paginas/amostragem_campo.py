"""
Amostragem para Inspeção de Campo — recebe o cadastro do município e sorteia as duas
amostras (medição estrutural e medição de qualidade) dimensionadas pela NBR 5426.

Esta página é apenas a UI; a lógica vive em `amostragem_ip/`.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from acesso import registrar_acao
from amostragem_ip import amostrador, nbr5426, relatorio
from amostragem_ip.amostrador import GRUPO_ESTRUTURAL, GRUPO_QUALIDADE
from amostragem_ip.leitura import ler_planilha
from amostragem_ip.saidas import planilha_amostra
from amostragem_ip.vias import identificar_vias_principais, vias_para_dataframe
from cadastro_ip.normalizacao import detectar_colunas


# ── Estado da sessão ──────────────────────────────────────────────────────────
SS = {
    "cadastro": "am_cadastro",
    "nome_arquivo": "am_nome_arquivo",
    "resultado": "am_resultado",
    "xlsx_estrutural": "am_xlsx_estrutural",
    "xlsx_qualidade": "am_xlsx_qualidade",
    "relatorio": "am_relatorio",
}
for chave in SS.values():
    st.session_state.setdefault(chave, None)


# ── Estilo local ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .am-hero-title {
            font-size: 2.6rem; font-weight: 800;
            background: linear-gradient(90deg, #ffffff, #00A9E0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -1px; margin-bottom: 0.2rem;
        }
        .am-hero-sub { font-size: 1rem; color: #94a3b8; margin-bottom: 1.5rem; }
        .am-step {
            background: rgba(18, 25, 43, 0.6);
            border: 1px solid #1f2937;
            border-left: 4px solid #00A9E0;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .am-step-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.4rem; }
        .am-step-desc  { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="am-hero-title">🎯 Amostragem para Inspeção</div>
    <div class="am-hero-sub">
        Dimensiona a amostra de campo pela ABNT NBR 5426 e sorteia dois lotes disjuntos —
        60% medição estrutural, 40% medição de qualidade — garantindo pelo menos um ponto
        em cada classe de iluminação, em cada avenida ou rodovia estruturante e em toda a
        extensão do município.
    </div>
    """,
    unsafe_allow_html=True,
)


def _passo(titulo: str, descricao: str) -> None:
    st.markdown(
        f'<div class="am-step"><div class="am-step-title">{titulo}</div>'
        f'<div class="am-step-desc">{descricao}</div></div>',
        unsafe_allow_html=True,
    )


def _para_exibicao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara um DataFrame para `st.dataframe` sem disparar erro de serialização Arrow.

    Cadastro de distribuidora costuma ter tipos misturados na mesma coluna (a
    `Barramento`, por exemplo, traz `FS_180931006` ao lado de códigos numéricos). O
    PyArrow tenta inferir int64, falha, e o Streamlit se recupera reconvertendo a
    tabela inteira a cada rerun — funciona, mas enche o log de traceback e custa
    tempo. Converter as colunas mistas para texto na exibição resolve na origem.
    Não afeta o sorteio nem as planilhas: vale só para a prévia visual.
    """
    exibicao = df.copy()
    for coluna in exibicao.columns[exibicao.dtypes == "object"]:
        exibicao[coluna] = exibicao[coluna].astype(str)
    return exibicao


# ── Passo 1: cadastro do município ────────────────────────────────────────────
_passo(
    "Passo 1 — Cadastro do município",
    "Envie a base de cadastro de iluminação pública já com a coluna de classificação "
    "viária tratada. Aceita .xlsx, .xls e .csv.",
)

col_upload, col_ident = st.columns([2, 1])
with col_upload:
    arquivo = st.file_uploader(
        "📋 Base de cadastro de IP", type=["xlsx", "xls", "csv"], key="am_upload"
    )
with col_ident:
    municipio = st.text_input("Município", key="am_municipio", placeholder="Ex.: Matozinhos")
    uf = st.text_input("UF", key="am_uf", max_chars=2, placeholder="MG")

if arquivo is not None and st.session_state[SS["nome_arquivo"]] != arquivo.name:
    try:
        st.session_state[SS["cadastro"]] = ler_planilha(arquivo)
        st.session_state[SS["nome_arquivo"]] = arquivo.name
        st.session_state[SS["resultado"]] = None
    except Exception as exc:
        st.error(f"Erro ao ler **{arquivo.name}**: {exc}")

cadastro = st.session_state[SS["cadastro"]]
if cadastro is None:
    st.info("Envie o cadastro para continuar.")
    st.stop()

st.success(
    f"● {len(cadastro):,} pontos lidos de **{st.session_state[SS['nome_arquivo']]}** "
    f"({len(cadastro.columns)} colunas)".replace(",", ".")
)
with st.expander("Prévia do cadastro"):
    st.dataframe(_para_exibicao(cadastro.head(20)), use_container_width=True)


# ── Passo 2: mapeamento de colunas ────────────────────────────────────────────
_passo(
    "Passo 2 — Colunas do cadastro",
    "As colunas foram detectadas automaticamente pelo nome. Confira e ajuste — a "
    "classificação viária e o logradouro são o que garantem a abrangência da amostra.",
)

CONCEITOS = [
    ("id_ponto", "Identificador do ponto", True),
    ("classe_via", "Classificação viária", True),
    ("logradouro", "Logradouro", True),
    ("bairro", "Bairro", False),
    ("latitude", "Latitude", False),
    ("longitude", "Longitude", False),
]

deteccao = detectar_colunas(
    cadastro,
    obrigatorios=["id_ponto"],
    recomendados=["classe_via", "logradouro", "bairro", "latitude", "longitude"],
)
opcoes = ["— não usar —"] + list(cadastro.columns)
colunas: dict[str, str] = {}

grade = st.columns(3)
for indice, (conceito, rotulo, destaque) in enumerate(CONCEITOS):
    detectada = deteccao.mapeados.get(conceito)
    padrao = opcoes.index(detectada) if detectada in opcoes else 0
    with grade[indice % 3]:
        escolha = st.selectbox(
            f"{'**' if destaque else ''}{rotulo}{'**' if destaque else ''}",
            opcoes,
            index=padrao,
            key=f"am_col_{conceito}",
            help="Detectada automaticamente." if detectada else "Não detectada — selecione.",
        )
    if escolha != "— não usar —":
        colunas[conceito] = escolha

if "classe_via" not in colunas:
    st.warning(
        "Sem coluna de classificação viária a amostra perde a garantia de cobrir todas as "
        "classes — fica aleatória com dispersão geográfica apenas."
    )
if "logradouro" not in colunas:
    st.warning(
        "Sem logradouro não há como garantir ponto em cada avenida ou rodovia principal."
    )

base, ressalvas_preparacao = amostrador.preparar_base(cadastro, colunas)
tem_coordenadas = bool(base["_tem_coord"].any())


# ── Passo 3: dimensionamento pela NBR 5426 ────────────────────────────────────
_passo(
    "Passo 3 — Dimensionamento (ABNT NBR 5426)",
    "O tamanho vem da tabela da norma pelo tamanho do parque, nível de inspeção e NQA. "
    "O campo final vem pré-preenchido com esse número e pode ser aumentado — é prática "
    "levar folga sobre a norma para absorver perdas de campo.",
)

col_n1, col_n2, col_n3 = st.columns(3)
with col_n1:
    nivel = st.selectbox(
        "Nível de inspeção", nbr5426.NIVEIS,
        index=nbr5426.NIVEIS.index(nbr5426.NIVEL_PADRAO), key="am_nivel",
        help="Níveis gerais I, II (padrão da norma) e III; S-1 a S-4 são os especiais, "
             "de amostra reduzida, usados quando o ensaio por unidade é caro.",
    )
with col_n2:
    nqa = st.selectbox(
        "NQA (%)", nbr5426.NQAS, index=nbr5426.NQAS.index(nbr5426.NQA_PADRAO),
        key="am_nqa",
        help="Nível de Qualidade Aceitável: percentual de não conformidade que ainda "
             "faz o lote ser aceito. Quanto menor, mais rigoroso.",
    )
with col_n3:
    regime = st.selectbox(
        "Regime", nbr5426.REGIMES, index=0, key="am_regime",
        help="Severo endurece o critério de aceitação; atenuado reduz a amostra — só se "
             "aplica a fornecedor com histórico consistente de conformidade.",
    )

plano = nbr5426.plano(len(base), nivel=nivel, nqa=nqa, regime=regime)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Parque (lote)", f"{len(base):,}".replace(",", "."))
col_m2.metric("Letra-código", plano.letra_codigo)
col_m3.metric("Amostra pela norma", f"{plano.tamanho_amostra:,}".replace(",", "."))
col_m4.metric("Aceitação / rejeição", f"Ac {plano.numero_aceitacao} / Re {plano.numero_rejeicao}")
for observacao in plano.observacoes:
    st.caption(f"ℹ️ {observacao}")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    tamanho = st.number_input(
        "Tamanho final da amostra", min_value=2, max_value=int(len(base)),
        value=int(plano.tamanho_amostra), step=1, key="am_tamanho",
        help="Pré-preenchido com o número da norma. Aumente para levar folga de campo.",
    )
with col_t2:
    percentual_estrutural = st.slider(
        "% medição estrutural", min_value=10, max_value=90, value=60, step=5,
        key="am_prop", help="O restante vai para a medição de qualidade.",
    )
with col_t3:
    semente = st.number_input(
        "Semente do sorteio", min_value=0, max_value=10**9, value=2026, step=1,
        key="am_semente",
        help="Registra e reproduz o sorteio. Mesma base + mesma semente = mesma amostra, "
             "o que permite auditar o resultado. Troque para gerar outra amostra.",
    )

if tamanho > plano.tamanho_amostra:
    folga = tamanho - plano.tamanho_amostra
    st.caption(
        f"➕ Folga de {folga} pontos sobre a norma "
        f"({folga / plano.tamanho_amostra:.0%}) — margem para pontos inexistentes, "
        "inacessíveis ou com coordenada errada."
    )
elif tamanho < plano.tamanho_amostra:
    st.warning(
        f"A amostra ({tamanho}) está **abaixo** do mínimo da NBR 5426 "
        f"({plano.tamanho_amostra}) para os parâmetros escolhidos. O critério Ac/Re "
        "deixa de valer e o plano perde o lastro normativo."
    )

col_o1, col_o2, col_o3 = st.columns(3)
with col_o1:
    cobertura_classe = st.checkbox(
        "≥1 ponto de cada classe em cada planilha", value=True, key="am_cob_classe"
    )
with col_o2:
    cobertura_vias = st.checkbox(
        "≥1 ponto em cada via principal", value=True, key="am_cob_vias"
    )
with col_o3:
    dispersao = st.checkbox(
        "Dispersão geográfica (k-means)", value=tem_coordenadas, key="am_dispersao",
        disabled=not tem_coordenadas,
        help="Sem coordenadas válidas no cadastro esta opção fica indisponível.",
    )


# ── Passo 4: vias principais ──────────────────────────────────────────────────
_passo(
    "Passo 4 — Vias principais",
    "Detectadas pelo tipo do logradouro (avenida, rodovia, estrada, anel viário, "
    "marginal) e pela classe de iluminação mais exigente. Desmarque as que não quiser "
    "obrigar, ou marque outras.",
)

teto = st.slider(
    "Máximo de vias com cobertura obrigatória", min_value=0, max_value=60, value=20, step=1,
    key="am_teto",
    help="Cada via obrigatória consome 2 pontos da amostra (um por planilha). Em amostra "
         "pequena, teto alto engole a alocação proporcional.",
)

candidatas = identificar_vias_principais(base, col_chave="_chave_via", teto=teto)
tabela_vias = vias_para_dataframe(candidatas)
vias_obrigatorias: list[str] | None = None

if tabela_vias.empty:
    st.info("Nenhuma via principal identificada — verifique a coluna de logradouro.")
else:
    tabela_vias.insert(0, "Obrigatória", True)
    editada = st.data_editor(
        tabela_vias,
        hide_index=True,
        use_container_width=True,
        height=min(420, 40 + 35 * len(tabela_vias)),
        column_config={
            "Obrigatória": st.column_config.CheckboxColumn(width="small"),
            "_chave": None,
        },
        disabled=["Via", "Tipo", "Classes", "Pontos no cadastro", "Motivo"],
        key="am_editor_vias",
    )
    vias_obrigatorias = editada.loc[editada["Obrigatória"], "_chave"].tolist()
    custo = 2 * len(vias_obrigatorias)
    st.caption(
        f"{len(vias_obrigatorias)} vias obrigatórias consomem {custo} dos {tamanho} pontos "
        f"da amostra ({custo / max(tamanho, 1):.0%}); o restante é alocado "
        "proporcionalmente às classes."
    )

if ressalvas_preparacao:
    with st.expander(f"⚠️ {len(ressalvas_preparacao)} ressalva(s) na leitura do cadastro"):
        for ressalva in ressalvas_preparacao:
            st.markdown(f"- {ressalva}")


# ── Mapa do parque (antes do sorteio) ─────────────────────────────────────────
CORES = {
    "Parque (não sorteado)": "#334155",
    "Medição estrutural": "#00A9E0",
    "Medição de qualidade": "#F59E0B",
}
LIMITE_PONTOS_MAPA = 12000


def _mapa(df: pd.DataFrame, coluna_cor: str, titulo: str, cores=None) -> None:
    """Desenha o mapa dos pontos. Amostra o fundo quando o parque é grande demais."""
    plotavel = df[df["_tem_coord"]]
    if plotavel.empty:
        st.info("Sem coordenadas válidas — mapa indisponível.")
        return
    if len(plotavel) > LIMITE_PONTOS_MAPA:
        sorteados = plotavel[plotavel["_grupo"] != ""] if "_grupo" in plotavel else plotavel.iloc[0:0]
        fundo = plotavel.drop(index=sorteados.index).sample(
            n=max(LIMITE_PONTOS_MAPA - len(sorteados), 0), random_state=0
        )
        plotavel = pd.concat([fundo, sorteados])
        st.caption(
            f"Mapa com {len(plotavel):,} pontos — o parque foi subamostrado para não "
            "travar o navegador; todos os pontos sorteados estão presentes.".replace(",", ".")
        )
    figura = px.scatter_map(
        plotavel,
        lat="_lat",
        lon="_lon",
        color=coluna_cor,
        color_discrete_map=cores,
        category_orders={coluna_cor: list(cores)} if cores else None,
        hover_name="_id",
        hover_data={"_logradouro": True, "_bairro": True, "_classe": True,
                    "_lat": False, "_lon": False},
        zoom=11,
        height=560,
        map_style="open-street-map",
        title=titulo,
    )
    figura.update_traces(marker={"size": 7})
    figura.update_layout(
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(figura, use_container_width=True)


resultado = st.session_state[SS["resultado"]]

if tem_coordenadas and resultado is None:
    with st.expander("🗺️ Mapa do parque cadastrado (antes do sorteio)", expanded=False):
        _mapa(base, "_classe", "Parque de IP por classe de iluminação")


# ── Passo 5: sorteio ──────────────────────────────────────────────────────────
_passo(
    "Passo 5 — Sortear a amostra",
    "O sorteio é aleatório dentro de cada estrato e reprodutível pela semente.",
)

if st.button("🎲 Sortear amostra", type="primary", use_container_width=True):
    with st.spinner("Sorteando..."):
        try:
            config = amostrador.ConfigAmostragem(
                tamanho_amostra=int(tamanho),
                proporcao_estrutural=percentual_estrutural / 100,
                cobertura_por_classe=cobertura_classe,
                cobertura_vias_principais=cobertura_vias,
                vias_obrigatorias=vias_obrigatorias,
                teto_vias_principais=int(teto),
                dispersao_espacial=dispersao,
                semente=int(semente),
            )
            resultado = amostrador.sortear(
                base, config, plano=plano,
                municipio=municipio or "", uf=(uf or "").upper(),
                ressalvas_iniciais=ressalvas_preparacao,
            )
            st.session_state[SS["resultado"]] = resultado
            st.session_state[SS["xlsx_estrutural"]] = planilha_amostra.gerar(resultado, GRUPO_ESTRUTURAL)
            st.session_state[SS["xlsx_qualidade"]] = planilha_amostra.gerar(resultado, GRUPO_QUALIDADE)
            st.session_state[SS["relatorio"]] = relatorio.gerar(resultado)
            registrar_acao(
                "amostra_sorteada",
                alvo=f"{municipio}/{uf}".strip("/"),
                detalhe=f"{resultado.total_amostra} de {resultado.total_parque} pontos",
            )
            st.success(
                f"✅ Amostra sorteada: {len(resultado.estrutural)} pontos estruturais + "
                f"{len(resultado.qualidade)} de qualidade."
            )
        except Exception as exc:
            import traceback
            st.error(f"❌ Erro no sorteio: {exc}")
            with st.expander("Detalhes do erro"):
                st.code(traceback.format_exc())

resultado = st.session_state[SS["resultado"]]


# ── Passo 6: resultado ────────────────────────────────────────────────────────
if resultado is not None:
    _passo(
        "Passo 6 — Conferência e download",
        "Confira no mapa se a amostra varreu o município antes de mandar a equipe a campo.",
    )

    abrangencia = resultado.abrangencia
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Medição estrutural", len(resultado.estrutural))
    met2.metric("Medição de qualidade", len(resultado.qualidade))
    met3.metric(
        "Bairros cobertos",
        f"{abrangencia.get('bairros_amostra', 0)}/{abrangencia.get('bairros_parque', 0)}",
    )
    if abrangencia.get("cobertura_grid") is not None:
        met4.metric(
            "Cobertura territorial",
            f"{abrangencia['cobertura_grid']:.0%}",
            help="Fração das células da malha 12×12 com pontos de IP que receberam "
                 "ao menos um ponto sorteado.",
        )
    else:
        met4.metric("Cobertura territorial", "—")

    mapa_df = resultado.base.copy()
    mapa_df["Camada"] = mapa_df["_grupo"].map(
        {GRUPO_ESTRUTURAL: "Medição estrutural", GRUPO_QUALIDADE: "Medição de qualidade"}
    ).fillna("Parque (não sorteado)")
    _mapa(mapa_df, "Camada", "Amostra sorteada sobre o parque", cores=CORES)

    aba_classes, aba_vias, aba_ressalvas, aba_relatorio = st.tabs(
        ["Cobertura por classe", "Vias principais", "Ressalvas", "Relatório"]
    )
    with aba_classes:
        tabela = resultado.cobertura_classes.copy()
        tabela["% do parque"] = tabela["% do parque"].map(lambda v: f"{v:.1%}")
        tabela["Peso p/ extrapolação"] = tabela["Peso p/ extrapolação"].map(
            lambda v: f"{v:.1f}" if pd.notna(v) else "—"
        )
        st.dataframe(tabela, hide_index=True, use_container_width=True)
        st.caption(
            "A amostra é deliberadamente **não auto-ponderada** — as cotas de cobertura "
            "sobre-representam as classes exigentes. Ao extrapolar o resultado da inspeção "
            "para o parque, use a média ponderada por estrato com o peso `w_h = N_h / n_h`, "
            "nunca a média simples da amostra."
        )
    with aba_vias:
        if resultado.cobertura_vias.empty:
            st.info("Nenhuma via principal com cobertura obrigatória.")
        else:
            st.dataframe(resultado.cobertura_vias, hide_index=True, use_container_width=True)
    with aba_ressalvas:
        if resultado.ressalvas:
            for ressalva in resultado.ressalvas:
                st.markdown(f"- {ressalva}")
        else:
            st.success("Nenhuma ressalva — cobertura integral de classes e vias principais.")
    with aba_relatorio:
        st.markdown(st.session_state[SS["relatorio"]])

    slug = (municipio or "municipio").replace(" ", "_")
    baixa1, baixa2, baixa3 = st.columns(3)
    with baixa1:
        st.download_button(
            "🏗️ Medição Estrutural.xlsx",
            data=st.session_state[SS["xlsx_estrutural"]],
            file_name=f"{slug} - Amostra Medição Estrutural.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with baixa2:
        st.download_button(
            "💡 Medição de Qualidade.xlsx",
            data=st.session_state[SS["xlsx_qualidade"]],
            file_name=f"{slug} - Amostra Medição de Qualidade.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with baixa3:
        st.download_button(
            "📝 Plano de amostragem (.md)",
            data=st.session_state[SS["relatorio"]].encode("utf-8"),
            file_name=f"{slug} - Plano de Amostragem.md",
            mime="text/markdown",
            use_container_width=True,
        )
