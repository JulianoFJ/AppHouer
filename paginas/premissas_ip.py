"""
Premissas & Inputs IP — wizard neutro de coleta de premissas por município.

Coleta as premissas do projeto, as informações do parque (entrada manual) e as
proposições de engenharia (IAE, ID, demanda reprimida, marcos, prazo de concessão,
expansão...) e gera duas saídas: a planilha de inputs parametrizada por fórmulas
(layout idêntico ao modelo) e os blocos de dados do relatório de engenharia.

Esta página é só UI; o catálogo vive em `premissas_ip/schema.py`, o contrato de
dados em `premissas_ip/coleta.py`, e os geradores em `premissas_ip/saidas/`.
"""

from __future__ import annotations

import streamlit as st

from cadastro_ip import aneel_2590
from premissas_ip import coleta, schema

SS = "pip_estado"  # chave única do estado desta página


# ── Estilo local ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .pip-hero-title { font-size: 2.6rem; font-weight: 800;
            background: linear-gradient(90deg, #ffffff, #00A9E0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -1px; margin-bottom: 0.2rem; }
        .pip-hero-sub { font-size: 1rem; color: #94a3b8; margin-bottom: 1.2rem; }
        .pip-step { background: rgba(18,25,43,0.6); border: 1px solid #1f2937;
            border-left: 4px solid #00A9E0; border-radius: 12px;
            padding: 1rem 1.2rem; margin-bottom: 1rem; }
        .pip-step-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; }
        .pip-grp { color: #00A9E0; font-weight: 700; font-size: 0.95rem;
            margin: 0.6rem 0 0.2rem 0; }
        .pip-fonte { color: #64748b; font-size: 0.72rem; font-style: italic; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="pip-hero-title">🧮 Planilha de Engenharia IP</div>'
    '<div class="pip-hero-sub">Coleta neutra das premissas do município e proposições de '
    'engenharia → gera a planilha de inputs parametrizada e os blocos do relatório.</div>',
    unsafe_allow_html=True,
)


# ── Helpers de widget por tipo de parâmetro ───────────────────────────────────
def _render_param(p: schema.Parametro) -> None:
    """Desenha o widget de um parâmetro. Campos do município sem dado começam VAZIOS
    (não '0,00'), para deixar claro o que falta. Os valores são colhidos depois por
    `_coletar_valores()` a partir do estado — assim, mesmo campos não exibidos (filtrados
    no modo 'só pendentes') mantêm seu default de catálogo na geração das saídas.
    """
    if p.eh_grupo:
        st.markdown(f'<div class="pip-grp">{p.label}</div>', unsafe_allow_html=True)
        return

    key = f"pip::{p.id}"
    tem = key in st.session_state
    auto = p.id in st.session_state.get("pip_auto_ids", set())
    rotulo = ("📄 " if auto else ("📍 " if p.coleta else "")) + p.label_unidade
    ajuda = f"Fonte: {p.fonte}" if p.fonte else None
    num = isinstance(p.default, (int, float))
    vazio = None if p.coleta else 0.0   # município sem default → campo vazio

    if p.tipo == "percentual":
        if tem:
            st.number_input(rotulo, step=0.1, format="%.4f", key=key, help=ajuda)
        else:
            st.number_input(rotulo, value=(float(p.default) * 100 if num else vazio),
                            step=0.1, format="%.4f", key=key, help=ajuda)
    elif p.tipo in ("moeda", "numero"):
        if tem:
            st.number_input(rotulo, step=1.0, format="%.2f", key=key, help=ajuda)
        else:
            st.number_input(rotulo, value=(float(p.default) if num else vazio),
                            step=1.0, format="%.2f", key=key, help=ajuda)
    elif p.tipo == "sim_nao":
        opcoes = ["Sim", "Não"]
        if tem:
            st.selectbox(rotulo, opcoes, key=key, help=ajuda)
        else:
            idx = opcoes.index(p.default) if p.default in opcoes else 1
            st.selectbox(rotulo, opcoes, index=idx, key=key, help=ajuda)
    else:  # texto, data
        if tem:
            st.text_input(rotulo, key=key, help=ajuda)
        else:
            st.text_input(rotulo, value=("" if p.default is None else str(p.default)),
                          key=key, help=ajuda)

    if p.fonte:
        st.markdown(f'<div class="pip-fonte">{p.fonte}</div>', unsafe_allow_html=True)


def _filled(p: schema.Parametro) -> bool:
    """Um campo do município está 'preenchido' se foi auto-extraído (DTO/planilha) ou
    já tem valor no estado. Campos de catálogo (não-coleta) contam sempre como preenchidos."""
    if not p.coleta:
        return True
    if p.id in st.session_state.get("pip_auto_ids", set()):
        return True
    return st.session_state.get(f"pip::{p.id}", None) not in (None, "")


def _coletar_valores() -> dict:
    """Monta os valores de TODOS os inputs a partir do estado (ou default do catálogo),
    independente de terem sido exibidos — garante que as saídas usem tudo o que foi preenchido."""
    s = schema.carregar()
    out: dict = {}
    for p in s.todos_inputs():
        key = f"pip::{p.id}"
        if key in st.session_state:
            v = st.session_state[key]
            if p.tipo == "percentual" and isinstance(v, (int, float)):
                v = v / 100.0
        else:
            v = p.default
        out[p.id] = v
    return out


def _aplicar_valores(valores: dict) -> None:
    """Injeta valores auto-extraídos (DTO/planilhas) nas chaves dos widgets, antes de criá-los."""
    s = schema.carregar()
    ids = st.session_state.setdefault("pip_auto_ids", set())
    for pid, val in (valores or {}).items():
        p = s.parametro(pid)
        if p is None:
            continue
        key = f"pip::{pid}"
        st.session_state[key] = val * 100 if p.tipo == "percentual" else val
        ids.add(pid)


def _processar_upload(up, parser, chave: str) -> None:
    """Lê um arquivo enviado com `parser`, injeta valores e (se houver) a lista de IE."""
    if up is None:
        return
    fid = f"{up.name}:{up.size}"
    if st.session_state.get(f"pip_fid_{chave}") == fid:
        return
    try:
        with st.spinner(f"Lendo {up.name}..."):
            ex = parser(up)
        st.session_state[chave] = ex
        st.session_state[f"pip_fid_{chave}"] = fid
        _aplicar_valores(getattr(ex, "valores", {}))
        ie = getattr(ex, "iluminacao_especial", None)
        if ie:
            st.session_state["pip_ie_list"] = ie
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        import traceback
        st.error(f"❌ Erro ao ler {up.name}: {exc}")
        with st.expander("Detalhes"):
            st.code(traceback.format_exc())


def _painel_extracao(ex, titulo: str) -> None:
    """Mostra os itens extraídos de uma planilha para conferência."""
    itens = getattr(ex, "itens", [])
    if itens:
        with st.expander(f"{titulo} — {len(itens)} campo(s)", expanded=True):
            for it in itens:
                val = f"{it.valor:.4g}"
                st.markdown(f"- **{it.label_param}** = `{val}` "
                            f"<span class='pip-fonte'>({getattr(it,'secao','')} · {getattr(it,'origem', '')})</span>",
                            unsafe_allow_html=True)
    res_calc = getattr(ex, "resultados", [])
    if res_calc:
        with st.expander("📐 Resultados já calculados pela engenharia (referência)"):
            for linha in res_calc:
                st.markdown(f"- {linha}")
    for av in getattr(ex, "avisos", []):
        st.warning(av)


def _render_secao_dinamica(sec: schema.Secao, resp: coleta.Respostas) -> None:
    """Tabela editável para seções de itens definidos pelo usuário (Iluminação Especial)."""
    import pandas as pd

    st.caption(
        "Liste os elementos de iluminação especial deste município (monumentos, "
        "praças, prédios...). Estes itens são específicos de cada cidade."
    )
    cols = [c.id for c in sec.colunas_tabela]
    rotulos = {c.id: c.label for c in sec.colunas_tabela}
    ie_src = st.session_state.get("pip_ie_list", [])
    if ie_src:
        st.caption(f"📄 {len(ie_src)} item(ns) importado(s) do InvBens — edite à vontade.")
    base = pd.DataFrame(ie_src, columns=cols) if ie_src \
        else pd.DataFrame([{c: ("" if cc.tipo == "texto" else 0)
                            for c, cc in zip(cols, sec.colunas_tabela)}])
    edit = st.data_editor(base.rename(columns=rotulos), num_rows="dynamic",
                          use_container_width=True, key=f"pip_tbl::{sec.id}")
    inv = {v: k for k, v in rotulos.items()}
    resp.iluminacao_especial = edit.rename(columns=inv).to_dict("records")


# ── Passo 0: Importar DTO (opcional) ──────────────────────────────────────────
st.markdown(
    '<div class="pip-step"><div class="pip-step-title">Passo 0 — Importar fontes (opcional)</div>'
    '<div style="color:#94a3b8;font-size:.85rem">Envie o <b>DTO</b> (.docx) e/ou as planilhas de '
    '<b>Extrapolação</b>, <b>Proposição de IAE</b> e <b>InvBens/ID</b> para auto-preencher: parque, '
    'LED, expansão, demanda reprimida, vida útil, distribuição por classe viária, custo de estrutura '
    'e a lista de Iluminação Especial. Tudo permanece editável.</div></div>',
    unsafe_allow_html=True,
)
up_dto = st.file_uploader("DTO — Diagnóstico Técnico Operacional (.docx)", type=["docx"], key="pip_up_dto")
if up_dto is not None:
    from premissas_ip import dto as dto_mod
    _processar_upload(up_dto, dto_mod.extrair, "pip_dto_extra")

st.markdown("**Planilhas de proposição** (opcional) — alimentam Iluminação Viária, IAE e Iluminação Especial:")
cpl1, cpl2, cpl3 = st.columns(3)
with cpl1:
    up_ext = st.file_uploader("📐 Extrapolação (.xlsx)", type=["xlsx"], key="pip_up_ext")
with cpl2:
    up_iae = st.file_uploader("🏛️ Proposição de IAE (.xlsx)", type=["xlsx"], key="pip_up_iae")
with cpl3:
    up_inv = st.file_uploader("💡 InvBens / ID (.xlsx)", type=["xlsx"], key="pip_up_inv")
if any(u is not None for u in (up_ext, up_iae, up_inv)):
    from premissas_ip import planilhas as pl
    _processar_upload(up_ext, pl.extrair_extrapolacao, "pip_ext_extra")
    _processar_upload(up_iae, pl.extrair_proposicao_iae, "pip_iae_extra")
    _processar_upload(up_inv, pl.extrair_invbens, "pip_inv_extra")

# ── Painéis de conferência ────────────────────────────────────────────────────
_dx = st.session_state.get("pip_dto_extra")
if _dx is not None:
    _painel_extracao(_dx, "📄 Auto-preenchido do DTO")
    if getattr(_dx, "distribuicoes", None):
        with st.expander(f"📊 Distribuições detectadas no DTO ({len(_dx.distribuicoes)}) — revisar e aplicar manualmente"):
            for d in _dx.distribuicoes:
                st.markdown(f"**{d.titulo}**")
                st.table({"Item": [l[0] for l in d.linhas], "Valor": [l[1] for l in d.linhas]})

for _chave, _titulo in [("pip_ext_extra", "📐 Extrapolação"),
                        ("pip_iae_extra", "🏛️ Proposição de IAE"),
                        ("pip_inv_extra", "💡 InvBens / ID")]:
    _ex = st.session_state.get(_chave)
    if _ex is not None:
        _painel_extracao(_ex, _titulo)
        if getattr(_ex, "iluminacao_especial", None):
            st.caption(f"💡 {len(_ex.iluminacao_especial)} item(ns) de Iluminação Especial importado(s) "
                       "— veja a aba *Engenharia & Proposições*.")


# ── Passo 1: Identificação do município ───────────────────────────────────────
st.markdown(
    '<div class="pip-step"><div class="pip-step-title">Passo 1 — Identificação do município</div></div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns([3, 1, 2])
with c1:
    municipio = st.text_input("Município", key="pip_municipio", placeholder="Ex: Tramandaí")
with c2:
    ufs = ["", "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
           "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
    uf = st.selectbox("UF", ufs, key="pip_uf")
with c3:
    data_base = st.text_input("Data-base (AAAA-MM)", key="pip_data_base", placeholder="2026-06")

if municipio and uf:
    t = aneel_2590.buscar(municipio, uf)
    if t is not None:
        st.success(f"⏱️ Tempo de operação (ANEEL 2590/2019): **{t.formato_hhmm}** "
                   f"— premissa para consumo de energia.")

st.caption("📍 = campo específico do município (preencher). Demais campos trazem premissa/"
           "catálogo editável como ponto de partida.")


# ── Passo 2: Premissas por macro-bloco ────────────────────────────────────────
st.markdown(
    '<div class="pip-step"><div class="pip-step-title">Passo 2 — Preencher o que falta</div></div>',
    unsafe_allow_html=True,
)

_coleta = [p for p in coleta.inputs_no_escopo() if p.coleta]
_pend = [p for p in _coleta if not _filled(p)]

m1, m2, m3 = st.columns(3)
m1.metric("Campos do município (📍)", len(_coleta))
m2.metric("Preenchidos (auto + você)", len(_coleta) - len(_pend))
m3.metric("Faltam preencher", len(_pend))

so_pend = st.toggle("🔎 Mostrar apenas os campos pendentes", value=True, key="pip_so_pend",
                    help="Esconde o que já veio do catálogo, do DTO e das planilhas — você só vê o que falta.")
st.caption("📄 preenchido por DTO/planilha · 📍 específico do município · demais campos = catálogo editável "
           "(ficam ocultos no modo pendentes, mas entram nas saídas com o valor padrão).")

resp = coleta.Respostas(municipio=municipio or "", uf=uf or "", data_base=data_base or "")

blocos = coleta.secoes_por_bloco()
abas = st.tabs([nome for nome, _ in blocos])
for aba, (nome_bloco, secoes) in zip(abas, blocos):
    with aba:
        algum = False
        for sec in secoes:
            sec_pend = [p for p in sec.inputs() if p.coleta and not _filled(p)]
            if so_pend and not sec_pend and not sec.dinamica:
                continue
            algum = True
            etiqueta = sec.nome + (f"  · 📍{len(sec_pend)}" if sec_pend else "  · ✓")
            with st.expander(etiqueta, expanded=bool(sec_pend) if so_pend else False):
                if sec.dinamica:
                    _render_secao_dinamica(sec, resp)
                for p in sec.parametros:
                    if so_pend:
                        if not p.eh_grupo and p.coleta and not _filled(p):
                            _render_param(p)
                    else:
                        _render_param(p)
        if not algum:
            st.success("✓ Nada pendente neste bloco — tudo veio do catálogo/DTO/planilhas.")

resp.valores = _coletar_valores()
st.session_state[SS] = resp


# ── Passo 3: Geração das saídas ───────────────────────────────────────────────
st.markdown(
    '<div class="pip-step"><div class="pip-step-title">Passo 3 — Gerar saídas</div></div>',
    unsafe_allow_html=True,
)
pronto = bool(municipio and uf)
if not pronto:
    st.info("⏳ Informe município e UF para habilitar a geração.")

slug = (municipio or "municipio").strip().replace(" ", "_")
g1, g2 = st.columns(2)

with g1:
    if st.button("🧮 Gerar Planilha de Inputs (.xlsx)", type="primary",
                 disabled=not pronto, use_container_width=True):
        try:
            from premissas_ip.saidas import planilha_inputs
            data = planilha_inputs.gerar(resp)
            st.session_state["pip_xlsx_inputs"] = data
            st.success("✅ Planilha de inputs gerada.")
        except Exception as exc:  # noqa: BLE001
            import traceback
            st.error(f"❌ Erro ao gerar a planilha: {exc}")
            with st.expander("Detalhes"):
                st.code(traceback.format_exc())
    if st.session_state.get("pip_xlsx_inputs"):
        st.download_button("⬇️ Baixar Inputs IP.xlsx", st.session_state["pip_xlsx_inputs"],
                           file_name=f"{slug} - Inputs IP.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

with g2:
    if st.button("📑 Gerar Blocos do Relatório", disabled=not pronto, use_container_width=True):
        try:
            from premissas_ip.saidas import blocos_relatorio
            xlsx, md = blocos_relatorio.gerar(resp)
            st.session_state["pip_xlsx_rel"] = xlsx
            st.session_state["pip_md_rel"] = md
            st.success("✅ Blocos do relatório gerados.")
        except Exception as exc:  # noqa: BLE001
            import traceback
            st.error(f"❌ Erro ao gerar os blocos: {exc}")
            with st.expander("Detalhes"):
                st.code(traceback.format_exc())
    if st.session_state.get("pip_xlsx_rel"):
        st.download_button("⬇️ Baixar Blocos Relatório.xlsx", st.session_state["pip_xlsx_rel"],
                           file_name=f"{slug} - Blocos Relatorio.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    if st.session_state.get("pip_md_rel"):
        st.download_button("⬇️ Baixar Resumo (.md)", st.session_state["pip_md_rel"].encode("utf-8"),
                           file_name=f"{slug} - Resumo Engenharia.md", mime="text/markdown",
                           use_container_width=True)
