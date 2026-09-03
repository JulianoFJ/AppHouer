"""
Amostragem para Inspeção de Campo — recebe o cadastro do município e sorteia as duas
amostras (medição estrutural e medição de qualidade) dimensionadas pela NBR 5426.

Esta página é apenas a UI; a lógica vive em `amostragem_ip/`.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from acesso import registrar_acao
from amostragem_ip import amostrador, nbr5426, relatorio
from amostragem_ip.amostrador import GRUPO_ESTRUTURAL, GRUPO_QUALIDADE
from amostragem_ip.leitura import ler_planilha
from amostragem_ip.saidas import planilha_amostra
from amostragem_ip.vias import identificar_vias_principais, vias_para_dataframe
from cadastro_ip import municipio as municipio_ibge
from cadastro_ip.normalizacao import detectar_colunas


# ── Estado da sessão ──────────────────────────────────────────────────────────
SS = {
    "cadastro": "am_cadastro",
    "nome_arquivo": "am_nome_arquivo",
    "resultado": "am_resultado",
    "xlsx_estrutural": "am_xlsx_estrutural",
    "xlsx_qualidade": "am_xlsx_qualidade",
    "relatorio": "am_relatorio",
    "mensagem": "am_mensagem",
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


# ── Etapas caras, memoizadas ──────────────────────────────────────────────────
# O Streamlit reexecuta a página inteira a cada clique em qualquer widget. Sem cache,
# mexer no slider do NQA refazia a preparação da base e a identificação de vias: 0,6 s
# num cadastro de 5 mil pontos e 6,5 s num de 100 mil — a cada clique, para um
# resultado idêntico. O cache é o que faz a página responder como formulário em vez de
# como lote. `max_entries` baixo porque cada entrada guarda uma cópia do cadastro.
@st.cache_data(show_spinner="Preparando a base…", max_entries=3)
def _preparar_base(cadastro: pd.DataFrame, colunas: dict, uf: str, zona_utm):
    return amostrador.preparar_base(cadastro, colunas, uf=uf or None, zona_utm=zona_utm)


@st.cache_data(show_spinner=False, max_entries=8)
def _vias_principais(chaves: pd.DataFrame, teto: int):
    """Recebe só as 4 colunas que o ranking usa — hashear a base inteira custaria mais
    que recalcular em cadastro grande."""
    return identificar_vias_principais(chaves, col_chave="_chave_via", teto=teto)


# ── Identificação do município ────────────────────────────────────────────────
# Município e UF alimentam o cabeçalho do relatório, o das duas planilhas, o nome do
# arquivo baixado e a trilha de uso — e a UF ainda **entra no cálculo**, estreitando as
# zonas UTM candidatas de um cadastro projetado. Por isso a escolha sai de lista
# fechada com a UF derivada, em vez de dois campos de texto livre que aceitam "mg",
# "Minas" ou um erro de digitação. A regra vive em `cadastro_ip.municipio`; aqui só a UI.
@st.cache_data(show_spinner=False)
def _lista_municipios() -> pd.DataFrame:
    return municipio_ibge.listar_municipios()


def _identificacao(cadastro: pd.DataFrame | None) -> tuple[str, str]:
    """Devolve (município, UF). Lista fechada quando há base do IBGE; texto livre senão."""
    lista = _lista_municipios()

    if lista.empty:   # deploy sem o parquet de entes — degrada, não trava
        nome = st.text_input("Município", key="am_municipio", placeholder="Ex.: Matozinhos")
        sigla = st.text_input("UF", key="am_uf", max_chars=2, placeholder="MG").strip().upper()
        if sigla and sigla not in municipio_ibge.UFS:
            st.error(f"“{sigla}” não é sigla de UF. A zona UTM de um cadastro projetado "
                     "depende dela.")
            sigla = ""
        return nome.strip(), sigla

    rotulos = lista["rotulo"].tolist()
    sugerido = municipio_ibge.sugerir(cadastro, lista)

    # O `index` só vale na primeira vez que o widget existe: depois disso o Streamlit
    # restaura o valor guardado na sessão e ignora o índice — e como o campo é desenhado
    # uma vez antes do upload (sem cadastro, logo sem sugestão), o `None` guardado ali
    # anularia a detecção. Escrever na chave do widget resolve, e a comparação com a
    # última sugestão aplicada é o que impede sobrescrever uma escolha manual do
    # operador: só reaplica quando o cadastro muda e a sugestão passa a ser outra.
    if sugerido and st.session_state.get("am_municipio_sugerido") != sugerido:
        st.session_state["am_municipio_sugerido"] = sugerido
        st.session_state["am_municipio_ibge"] = sugerido

    escolha = st.selectbox(
        "Município",
        rotulos,
        index=rotulos.index(sugerido) if sugerido in rotulos else None,
        placeholder="Digite para buscar — ex.: Matozinhos",
        accept_new_options=True,   # município fora da lista ainda pode ser digitado
        key="am_municipio_ibge",
        help="Os 5.570 municípios do IBGE. A UF sai da escolha — não precisa digitar e "
             "não dá para errar a sigla. Vai para o cabeçalho do relatório e das "
             "planilhas, para o nome do arquivo baixado e, quando o cadastro vem em "
             "UTM, para a definição da zona.",
    )
    if not escolha:
        return "", ""

    linha = lista[lista["rotulo"] == escolha]
    if not linha.empty:
        origem = " · detectado no cadastro" if sugerido == escolha else ""
        st.caption(f"UF **{linha.iloc[0]['uf']}**{origem}")
        return str(linha.iloc[0]["ente"]), str(linha.iloc[0]["uf"])

    nome, sigla = municipio_ibge.separar_rotulo(str(escolha))
    if sigla:
        st.caption(f"UF **{sigla}** · fora da lista do IBGE")
    else:
        st.caption("Sem UF — escreva `Município/UF` se o cadastro estiver em UTM.")
    return nome, sigla


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

# A leitura vem ANTES de desenhar o campo de município, embora o campo apareça ao lado
# do upload: `st.columns` guarda o lugar, então a ordem visual não precisa ser a ordem
# do código. É o que permite pré-selecionar o município já na passada do upload, em vez
# de só na interação seguinte.
if arquivo is not None and st.session_state[SS["nome_arquivo"]] != arquivo.name:
    try:
        st.session_state[SS["cadastro"]] = ler_planilha(arquivo)
        st.session_state[SS["nome_arquivo"]] = arquivo.name
        st.session_state[SS["resultado"]] = None
    except Exception as exc:
        st.error(f"Erro ao ler **{arquivo.name}**: {exc}")

cadastro = st.session_state[SS["cadastro"]]

with col_ident:
    municipio, uf = _identificacao(cadastro)

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
    ("coordenadas", "Coordenada única (lat, lon)", False),
]

deteccao = detectar_colunas(
    cadastro,
    obrigatorios=["id_ponto"],
    recomendados=["classe_via", "logradouro", "bairro", "latitude", "longitude",
                  "coordenadas"],
)
opcoes = ["— não usar —"] + list(cadastro.columns)
colunas: dict[str, str] = {}

AJUDA_CONCEITO = {
    "coordenadas": "Para cadastro que traz o par numa célula só (`-19,54, -44,08`). "
                   "Quando preenchida, prevalece sobre Latitude/Longitude.",
    "latitude": "Aceita ponto ou vírgula decimal, grau-minuto-segundo (19°32'45\"S) e "
                "coordenada UTM — a conversão é automática.",
    "longitude": "Aceita ponto ou vírgula decimal, grau-minuto-segundo (44°05'03\"W) e "
                 "coordenada UTM — a conversão é automática.",
}

grade = st.columns(3)
for indice, (conceito, rotulo, destaque) in enumerate(CONCEITOS):
    detectada = deteccao.mapeados.get(conceito)
    padrao = opcoes.index(detectada) if detectada in opcoes else 0
    ajuda = AJUDA_CONCEITO.get(conceito, "")
    if not ajuda:
        ajuda = "Detectada automaticamente." if detectada else "Não detectada — selecione."
    with grade[indice % 3]:
        escolha = st.selectbox(
            f"{'**' if destaque else ''}{rotulo}{'**' if destaque else ''}",
            opcoes,
            index=padrao,
            key=f"am_col_{conceito}",
            help=ajuda,
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

# Duas passadas quando o cadastro está em UTM: a primeira descobre que está, a segunda
# usa a zona que o operador confirmou. Não dá para pular a primeira — a zona só é
# oferecida depois de o formato ser reconhecido, e (E, N) sozinho não a determina.
zona_utm = st.session_state.get("am_zona_utm_valor")
base, ressalvas_preparacao = _preparar_base(cadastro, colunas, (uf or "").upper(), zona_utm)

# Zona UTM: só aparece quando o cadastro veio projetado E o dado não decide sozinho.
# O padrão do seletor é a zona que o módulo adotou, para o primeiro render não disparar
# um rerun só por divergir de si mesmo.
candidatas = base.attrs.get("zonas_utm_candidatas") or []
if len(candidatas) > 1:
    adotada = base.attrs.get("zona_utm_adotada") or candidatas[0]
    # Sem `key`: a lista de zonas candidatas encolhe quando o operador preenche a UF, e
    # um valor guardado na sessão que sumiu das opções quebra o selectbox. Quem guarda a
    # escolha é `am_zona_utm_valor`, e o índice reflete a zona efetivamente adotada.
    escolha_zona = st.selectbox(
        "Zona UTM do cadastro", candidatas, index=candidatas.index(adotada),
        help="O cadastro veio projetado em UTM, e o par (E, N) é compatível com mais de "
             "uma zona. Preencha a UF acima para reduzir as opções, escolha aqui e "
             "confira no mapa: zona errada desloca o município no sentido leste-oeste.",
    )
    if escolha_zona != adotada:
        st.session_state["am_zona_utm_valor"] = escolha_zona
        st.rerun()

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

vias_candidatas = _vias_principais(
    base[["_chave_via", "_logradouro", "_tipo_via", "_classe"]], int(teto))
tabela_vias = vias_para_dataframe(vias_candidatas)
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
# Teto de marcadores por mapa. O mapa de conferência da amostra precisa de todos os
# pontos sorteados e de um fundo denso; a prévia do parque, que é redesenhada a cada
# clique em qualquer widget da página, vale menos que o tempo de resposta — daí o
# teto menor nela (cada marcador vira JSON trafegado do servidor ao navegador).
LIMITE_PONTOS_MAPA = 12000
LIMITE_PONTOS_PREVIA = 4000


def _enquadrar(lat: pd.Series, lon: pd.Series) -> tuple[dict, float, float]:
    """
    Centro e zoom que enquadram a nuvem de pontos. Devolve (centro, zoom, span em graus).

    Existe porque o `zoom=11` fixo que estava aqui só servia para um município do
    tamanho de Matozinhos: em capital ele corta metade do parque, e em cadastro de
    consórcio intermunicipal mostra um quarteirão. E o centro não é a média — a média
    é puxada por um punhado de pontos com coordenada errada, e o mapa acaba num lugar
    onde não há ponto nenhum. O recorte de 2% a 98% descarta esses extremos antes de
    medir; o mapa continua desenhando todos os pontos, inclusive os de fora.
    """
    p_lat = lat.quantile([0.02, 0.98]).to_numpy()
    p_lon = lon.quantile([0.02, 0.98]).to_numpy()
    centro = {"lat": float((p_lat[0] + p_lat[1]) / 2), "lon": float((p_lon[0] + p_lon[1]) / 2)}

    # Mínimo de 0,004° (~450 m) evita zoom absurdo quando todos os pontos coincidem.
    span_lat = max(float(p_lat[1] - p_lat[0]), 0.004)
    span_lon = max(float(p_lon[1] - p_lon[0]), 0.004)

    # Web Mercator: no zoom z cada tile de 512 px cobre 360/2^z graus de longitude. O
    # enquadramento é o menor zoom que faz os dois eixos caberem na viewport, com uma
    # margem de 20% para o ponto da borda não encostar na moldura.
    largura, altura = 1100.0, 520.0
    z_lon = np.log2(largura * 360.0 / (512.0 * span_lon))
    z_lat = np.log2(altura * 360.0 / (512.0 * span_lat / np.cos(np.radians(centro["lat"]))))
    zoom = float(np.clip(min(z_lon, z_lat) - 0.35, 3.0, 16.0))
    return centro, zoom, max(span_lat, span_lon)


def _mapa(df: pd.DataFrame, coluna_cor: str, titulo: str, cores=None,
          limite: int = LIMITE_PONTOS_MAPA) -> None:
    """Desenha o mapa dos pontos. Amostra o fundo quando o parque é grande demais."""
    plotavel = df[df["_tem_coord"]]
    if plotavel.empty:
        formato = df.attrs.get("formato_coordenadas", "ausente")
        st.warning(
            "**Mapa indisponível: nenhuma coordenada válida.** "
            + ("Nenhuma coluna de coordenada foi indicada no Passo 2."
               if formato == "ausente" else
               f"As coordenadas foram lidas como *{formato}*, mas nenhuma caiu dentro "
               "do território brasileiro. Confira no Passo 2 se as colunas escolhidas "
               "são mesmo latitude e longitude.")
        )
        return
    if len(plotavel) > limite:
        sorteados = plotavel[plotavel["_grupo"] != ""] if "_grupo" in plotavel else plotavel.iloc[0:0]
        fundo = plotavel.drop(index=sorteados.index).sample(
            n=max(limite - len(sorteados), 0), random_state=0
        )
        plotavel = pd.concat([fundo, sorteados])
        st.caption(
            f"Mapa com {len(plotavel):,} pontos — o parque foi subamostrado para não "
            "travar o navegador; todos os pontos sorteados estão presentes.".replace(",", ".")
        )

    centro, zoom, span = _enquadrar(plotavel["_lat"], plotavel["_lon"])
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
        center=centro,
        zoom=zoom,
        height=560,
        map_style="open-street-map",
    )
    figura.update_traces(marker={"size": 7})
    # O título fica fora da figura (é `st.markdown`, acima): dentro dela ele dividia a
    # faixa superior com a legenda horizontal e os dois se sobrepunham. A legenda passa
    # a flutuar sobre o mapa, num cartão translúcido — não rouba altura do mapa nem
    # some no fundo claro do OpenStreetMap.
    figura.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "top", "y": 0.99, "xanchor": "left",
                "x": 0.01, "title_text": "", "bgcolor": "rgba(11,17,30,.82)",
                "bordercolor": "#2f3f5c", "borderwidth": 1},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    st.markdown(f"**{titulo}**")
    st.plotly_chart(figura, use_container_width=True)

    # Um município cabe em ~0,5°. Espalhamento maior é sinal de coordenada trocada ou
    # de mistura de bases — e é justamente o caso em que o mapa "abre no nada" e
    # parece quebrado. Dizer o motivo é mais útil que deixar o operador adivinhar.
    if span > 1.0:
        st.warning(
            f"Os pontos se espalham por {span:.1f}° (~{span * 111:.0f} km) — muito mais "
            "que um município. Há coordenada errada na base, ou a coluna escolhida no "
            "Passo 2 não é a de coordenada. O mapa abriu afastado para mostrar tudo."
        )


resultado = st.session_state[SS["resultado"]]

if resultado is None:
    # Fora de expander: o mapa é a única conferência visual de que as coordenadas foram
    # lidas certo, e escondido atrás de um clique ele só era visto depois do sorteio —
    # tarde demais. Quando não há coordenada válida, `_mapa` explica o motivo em vez de
    # sumir, que era o que fazia a página parecer quebrada.
    formato = base.attrs.get("formato_coordenadas", "ausente")
    if tem_coordenadas and formato != "graus decimais":
        st.caption(f"🧭 Coordenadas reconhecidas como **{formato}**.")
    _mapa(base, "_classe", "Parque de IP por classe de iluminação",
          limite=LIMITE_PONTOS_PREVIA)


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
            # Rerun em vez de seguir o script: a prévia do parque já foi desenhada
            # acima nesta passada (quando `resultado` ainda era None), e sem o rerun a
            # tela ficaria com dois mapas pesados até o próximo clique. A mensagem
            # atravessa o rerun pela sessão.
            st.session_state[SS["mensagem"]] = (
                f"✅ Amostra sorteada: {len(resultado.estrutural)} pontos estruturais + "
                f"{len(resultado.qualidade)} de qualidade."
            )
            st.rerun()
        except Exception as exc:
            import traceback
            st.error(f"❌ Erro no sorteio: {exc}")
            with st.expander("Detalhes do erro"):
                st.code(traceback.format_exc())

resultado = st.session_state[SS["resultado"]]
mensagem = st.session_state.pop(SS["mensagem"], None)
if mensagem:
    st.success(mensagem)


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
