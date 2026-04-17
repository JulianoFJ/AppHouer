"""
03_app_previsao.py
-------------------
App Streamlit interativo — preve Fluxo Luminoso (lm) e Potencia (W)
para Ledstar, SX Lighting e Tecnowatt, tanto individualmente quanto em lote via planilha.

Execucao:
    streamlit run 03_app_previsao.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import joblib
import plotly.graph_objects as go

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Previsão de Iluminação — Houer',
    page_icon='💡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Estilo ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: #9ca3af; font-size: 0.95rem; margin-bottom: 1rem; }

    .forn-card {
        border-radius: 14px; padding: 1.2rem 1rem;
        text-align: center; margin-bottom: 0.5rem;
        border-top: 4px solid var(--cor);
        background: linear-gradient(135deg, #1a2035, #1e2540);
        border: 1px solid #2d4060;
        border-top: 4px solid var(--cor);
    }
    .forn-name  { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing:.06em; }
    .forn-val   { font-size: 1.9rem; font-weight: 700; }
    .forn-unit  { font-size: 0.75rem; color: #9ca3af; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #d1d5db; margin: 1rem 0 0.5rem; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0d1420,#111827);
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ── Caminhos ──────────────────────────────────────────────────────────────────
PASTA        = os.path.dirname(os.path.abspath(__file__))
MODELO_LM    = os.path.join(PASTA, 'modelo_lm.pkl')
MODELO_W     = os.path.join(PASTA, 'modelo_w.pkl')
FEATURES_PATH= os.path.join(PASTA, 'features.json')

# Compatibilidade com modelo antigo (modelo_potencia.pkl)
MODELO_LEGACY= os.path.join(PASTA, 'modelo_potencia.pkl')

FORNECEDORES = ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']
CORES = {'LEDSTAR': '#3B82F6', 'SX LIGHTING': '#22C55E', 'TECNOWATT': '#F59E0B'}

# ── Carrega modelos ───────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelos_v2():
    meta = {}
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, encoding='utf-8') as f:
            meta = json.load(f)

    # Modelo lm
    modelo_lm = None
    if os.path.exists(MODELO_LM):
        modelo_lm = joblib.load(MODELO_LM)
    elif os.path.exists(MODELO_LEGACY):
        modelo_lm = joblib.load(MODELO_LEGACY)

    # Modelo W
    modelo_w = None
    if os.path.exists(MODELO_W):
        modelo_w = joblib.load(MODELO_W)

    return modelo_lm, modelo_w, meta

modelo_lm, modelo_w, meta = carregar_modelos_v2()
num_ok = meta.get('features_numericas', [])
cat_ok = meta.get('features_categoricas', [])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">💡 Previsão de Iluminação Pública</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Compare Fluxo Luminoso (lm) e Potência (W) por fornecedor a partir das características físicas da instalação.</p>', unsafe_allow_html=True)
st.divider()

if modelo_lm is None:
    st.error('Modelo não encontrado! Execute primeiro o script **02_treinar_modelo.py**.')
    st.stop()

# ── Sidebar (Parâmetros da Simulação Individual) ──────────────────────────────
with st.sidebar:
    st.markdown('## ⚙️ Parâmetros (Apenas para Individual)')
    st.divider()

    st.markdown('### 🛣️ Geometria da Via')
    faixas         = st.slider('Faixas de Rodagem',            1, 6,    2, step=1)
    largura_via1   = st.slider('Largura Via 1 (m)',            4.0, 20.0, 7.0, step=0.5)
    largura_via2   = st.slider('Largura Via 2 (m)',            0.0, 20.0, 0.0, step=0.5)
    canteiro       = st.slider('Largura Canteiro Central (m)', 0.0, 10.0, 0.0, step=0.5)

    st.markdown('### 🏗️ Estrutura')
    altura_lum     = st.slider('Altura da Luminária (m)',      4.0, 16.0, 9.0,  step=0.5)
    projecao_braco = st.slider('Projeção do Braço (m)',        0.0,  4.0, 1.5,  step=0.25)
    dist_postes    = st.slider('Distância entre Postes (m)',  20.0, 60.0, 35.0, step=1.0)
    dist_poste_via = st.slider('Distância Poste à Via (m)',    0.0,  3.0,  0.5, step=0.25)
    altura_inst    = st.slider('Altura de Instalação (m)',     4.0, 16.0, 10.0, step=0.5)

    st.markdown('### 🔩 Configurações')

    tipo_estrutura_opts = ['Braço', 'Suporte']
    posteacao_opts      = ['Unilateral', 'Canteiro central', 'Bilateral alternada', 'Bilateral frontal']
    braco_opts          = ['Longo II', 'Longo I', 'Médio I', 'Médio II', 'Curto II', 'Curto I']

    tipo_estrutura = st.selectbox('Tipo de Estrutura', tipo_estrutura_opts)
    posteacao      = st.selectbox('Posteação',         posteacao_opts)
    braco_novo     = st.selectbox('Braço Novo',        braco_opts)

# ── Tabs Principais ───────────────────────────────────────────────────────────
tab_individual, tab_lote = st.tabs(['🎯 Simulação Individual', '📂 Simulação em Lote'])

# ==============================================================================
# TAB 1: SIMULAÇÃO INDIVIDUAL
# ==============================================================================
with tab_individual:
    
    # ── Função de previsão
    def montar_entrada(fornecedor):
        dados = {
            'Faixas de Rodagem':        faixas,
            'Largura Via 1':            largura_via1,
            'Largura Via 2':            largura_via2,
            'largura Canteiro Central': canteiro,
            'altura da luminaria':      altura_lum,
            'projecao do braço':        projecao_braco,
            'projecao do brao':         projecao_braco,
            'distancia entre postes':   dist_postes,
            'distancia Poste a via':    dist_poste_via,
            'Altura de Instalação':     altura_inst,
            'Altura de Instalao':      altura_inst,
            'Tipo de estrutura':        tipo_estrutura,
            'posteacao':                posteacao,
            'Braço Novo':               braco_novo,
            'Brao Novo':                braco_novo,
            'Fornecedor':               fornecedor,
        }
        colunas = num_ok + cat_ok
        return pd.DataFrame([{k: dados.get(k, np.nan) for k in colunas}])

    previsoes_lm = {}
    previsoes_w  = {}

    for forn in FORNECEDORES:
        X_in = montar_entrada(forn)
        try:
            previsoes_lm[forn] = max(modelo_lm.predict(X_in)[0], 0)
        except Exception as e:
            previsoes_lm[forn] = None
        try:
            previsoes_w[forn] = max(modelo_w.predict(X_in)[0], 0) if modelo_w else None
        except Exception as e:
            previsoes_w[forn] = None

    # ── Cards por fornecedor
    cols = st.columns(3)
    for i, forn in enumerate(FORNECEDORES):
        cor   = CORES[forn]
        val_lm = previsoes_lm.get(forn)
        val_w  = previsoes_w.get(forn)
        with cols[i]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a2035,#1e2540);border-radius:14px;
                        padding:1.3rem 1rem;text-align:center;border-top:4px solid {cor};
                        border:1px solid #2d4060;border-top:4px solid {cor};margin-bottom:.5rem;">
                <div style="font-size:.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem;">{forn}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{cor};">{f'{val_lm:,.0f}' if val_lm is not None else '—'}</div>
                <div style="font-size:.8rem;color:#9ca3af;">lúmens (lm)</div>
                <div style="margin:.6rem 0;border-top:1px solid #2d3a50;"></div>
                <div style="font-size:1.5rem;font-weight:700;color:#e2e8f0;">{f'{val_w:,.0f}' if val_w is not None else '—'}</div>
                <div style="font-size:.8rem;color:#9ca3af;">potência (W)</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Eficiência luminosa calculada (lm/W)
    st.markdown('<p class="section-title">⚡ Eficiência Luminosa Calculada (lm/W)</p>', unsafe_allow_html=True)
    ef_cols = st.columns(3)
    for i, forn in enumerate(FORNECEDORES):
        lm = previsoes_lm.get(forn)
        w  = previsoes_w.get(forn)
        with ef_cols[i]:
            if lm and w and w > 0:
                ef = lm / w
                cor = CORES[forn]
                st.markdown(f"""
                <div style="background:#111827;border-radius:10px;padding:.8rem;text-align:center;border:1px solid #1f2937;">
                    <div style="font-size:.7rem;color:#9ca3af;">{forn}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{cor};">{ef:.1f} lm/W</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#111827;border-radius:10px;padding:.8rem;text-align:center;border:1px solid #1f2937;">
                    <div style="font-size:.7rem;color:#9ca3af;">{forn}</div>
                    <div style="font-size:1.4rem;color:#6b7280;">— lm/W</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Gráfico comparativo
    tab_graf_lm, tab_graf_w, tab_graf_ef = st.tabs(['📊 Fluxo Luminoso (lm)', '⚡ Potência (W)', '🔀 Eficiência (lm/W)'])

    def bar_chart(valores, unidade, titulo):
        fig = go.Figure()
        for forn, val in valores.items():
            if val is not None:
                fig.add_trace(go.Bar(
                    x=[forn], y=[val], name=forn,
                    marker_color=CORES[forn],
                    text=[f'{val:,.0f} {unidade}'], textposition='outside',
                    textfont=dict(size=13, color='white'),
                ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d1d5db', family='Inter'),
            showlegend=False, bargap=0.35, height=360,
            margin=dict(t=30, b=10),
            yaxis=dict(title=titulo, gridcolor='#1f2937', zerolinecolor='#374151'),
            xaxis=dict(gridcolor='#1f2937'),
        )
        return fig

    with tab_graf_lm:
        st.plotly_chart(bar_chart(previsoes_lm, 'lm', 'Fluxo Luminoso (lm)'), use_container_width=True)

    with tab_graf_w:
        if modelo_w:
            st.plotly_chart(bar_chart(previsoes_w, 'W', 'Potência (W)'), use_container_width=True)
        else:
            st.warning('Modelo de potência não disponível. Execute o 02_treinar_modelo.py.')

    with tab_graf_ef:
        ef_vals = {}
        for forn in FORNECEDORES:
            lm = previsoes_lm.get(forn)
            w  = previsoes_w.get(forn)
            ef_vals[forn] = (lm / w) if lm and w and w > 0 else None
        if any(v is not None for v in ef_vals.values()):
            st.plotly_chart(bar_chart(ef_vals, 'lm/W', 'Eficiência Luminosa (lm/W)'), use_container_width=True)

    # ── Radar do perfil
    st.markdown('### 🕸️ Perfil da Instalação')
    categorias = ['Via 1 (m)', 'Altura Lum.', 'Projeção', 'Dist. Postes', 'Faixas']
    vals_norm  = [largura_via1/20, altura_lum/16, projecao_braco/4, dist_postes/60, faixas/6]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_norm + [vals_norm[0]],
        theta=categorias + [categorias[0]],
        fill='toself',
        fillcolor='rgba(255,165,0,0.15)',
        line=dict(color='#FFA500', width=2),
    ))
    fig_radar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d1d5db', family='Inter'),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], gridcolor='#2d3748', color='#9ca3af'),
            angularaxis=dict(gridcolor='#2d3748', color='#d1d5db'),
        ),
        showlegend=False, height=340, margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ==============================================================================
# TAB 2: SIMULAÇÃO EM LOTE
# ==============================================================================
with tab_lote:
    st.markdown('### 📥 1. Baixe a Planilha Padrão')
    st.markdown('Preencha as características geométricas de cada instalação em uma nova linha.')
    
    # Prepara o dataframe template (remove a coluna Fornecedor pois o próprio modelo vai iterar sobre os 3)
    colunas_template = num_ok.copy() + [c for c in cat_ok if c != 'Fornecedor']
    
    # Mapping in case internal names have issues
    mapeamento_encoding = {
        'projecao do brao': 'projecao do braço',
        'Altura de Instalao': 'Altura de Instalação',
        'Brao Novo': 'Braço Novo'
    }
    colunas_amigaveis = [mapeamento_encoding.get(c, c) for c in colunas_template]
    
    df_template = pd.DataFrame(columns=colunas_amigaveis)
    
    buffer_template = io.BytesIO()
    with pd.ExcelWriter(buffer_template, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False)
    
    st.download_button(
        label='⬇️ Baixar Planilha Padrão (.xlsx)',
        data=buffer_template.getvalue(),
        file_name='template_simulacao.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    
    st.markdown('---')
    st.markdown('### 📤 2. Envie a Planilha Preenchida')
    
    arquivo_up = st.file_uploader('Selecione o arquivo modificado', type=['xlsx', 'csv'])
    
    if arquivo_up is not None:
        try:
            if arquivo_up.name.endswith('.csv'):
                df_entrada = pd.read_csv(arquivo_up)
            else:
                df_entrada = pd.read_excel(arquivo_up)
            
            st.success(f'Arquivo lido com sucesso! ({len(df_entrada)} linhas identificadas)')
            
            with st.spinner('Realizando previsões...'):
                df_saida = df_entrada.copy()
                
                # Prepara dataframe espelho com os nomes corretos esperados pelo pipeline
                map_inverso = {v: k for k, v in mapeamento_encoding.items()}
                df_pipeline = df_entrada.rename(columns=map_inverso)
                
                for forn in FORNECEDORES:
                    df_run = df_pipeline.copy()
                    df_run['Fornecedor'] = forn
                    
                    # Garante que todas as colunas esperadas existem (mesmo que com NaN)
                    for col in num_ok + cat_ok:
                        if col not in df_run.columns:
                            df_run[col] = np.nan
                            
                    # Previsões
                    if modelo_lm:
                        preds_lm = modelo_lm.predict(df_run)
                        df_saida[f'Fluxo Luminoso (lm) - {forn}'] = [max(p, 0) for p in preds_lm]
                    
                    if modelo_w:
                        preds_w = modelo_w.predict(df_run)
                        df_saida[f'Potência (W) - {forn}'] = [max(p, 0) for p in preds_w]
                        
                        # Calcula a eficiêcia
                        if modelo_lm and modelo_w:
                            df_saida[f'Eficiência (lm/W) - {forn}'] = df_saida[f'Fluxo Luminoso (lm) - {forn}'] / df_saida[f'Potência (W) - {forn}']

                st.markdown('### ✨ Pré-visualização dos Resultados')
                st.dataframe(df_saida)
                
                # Download dos resultados
                buffer_resultado = io.BytesIO()
                with pd.ExcelWriter(buffer_resultado, engine='openpyxl') as writer:
                    df_saida.to_excel(writer, index=False)
                
                st.download_button(
                    label='✅ Baixar Resultados (.xlsx)',
                    data=buffer_resultado.getvalue(),
                    file_name='resultados_simulacao.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary'
                )

        except Exception as e:
            st.error(f'Oops! Erro ao processar o arquivo. Verifique se o formato das colunas está correto. Erro detalhado: {str(e)}')


# ── Info do modelo ─────────────────────────────────────────────────────────────
with st.expander('ℹ️ Métricas dos Modelos Treinados'):
    meta_lm_info = meta.get('modelo_lm', meta)
    meta_w_info  = meta.get('modelo_w', {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Modelo — Fluxo Luminoso (lm)**')
        st.metric('Algoritmo', meta_lm_info.get('modelo', '—'))
        st.metric('R² (teste)', f"{meta_lm_info.get('r2_teste','—'):.4f}" if meta_lm_info.get('r2_teste') else '—')
        st.metric('MAE (teste)', f"{meta_lm_info.get('mae_teste','—'):.0f} lm" if meta_lm_info.get('mae_teste') else '—')
    with c2:
        st.markdown('**Modelo — Potência (W)**')
        st.metric('Algoritmo', meta_w_info.get('modelo', '—') if meta_w_info else '—')
        st.metric('R² (teste)', f"{meta_w_info.get('r2_teste','—'):.4f}" if meta_w_info.get('r2_teste') else '—')
        st.metric('MAE (teste)', f"{meta_w_info.get('mae_teste','—'):.1f} W" if meta_w_info.get('mae_teste') else '—')

st.markdown('---')
st.markdown(
    '<p style="text-align:center;color:#4b5563;font-size:.8rem;">'
    '💡 Houer | Previsão de Iluminação Pública via Machine Learning</p>',
    unsafe_allow_html=True,
)
