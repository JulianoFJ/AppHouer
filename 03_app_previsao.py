import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import joblib
import json
from geopy.geocoders import GoogleV3
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

# Chave API do Google Maps (recuperada do script de georreferenciamento)
GOOGLE_MAPS_API_KEY = "AIzaSyAN_M7Zet2BfygkMmGD10E1jrR5d_akrw8"

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Previsão de Iluminação — Houer',
    page_icon='💡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Inicializa estado para Simulação em Lote
if 'df_lote' not in st.session_state:
    st.session_state.df_lote = None

# ── Estilo ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Paleta Houer: Navy (#1B3664), Teal (#00A9E0) */
    :root {
        --houer-navy: #1B3664;
        --houer-teal: #00A9E0;
        --bg-dark: #0b111e;
        --card-bg: #12192b;
        --card-border: #1f2937;
    }
    
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1B366433, transparent), 
                    radial-gradient(circle at 100% 100%, #00A9E011, transparent),
                    #0b111e;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, var(--houer-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .forn-card {
        background: rgba(18, 25, 43, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--houer-teal);
        border-radius: 20px;
        padding: 1.8rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .forn-card:hover {
        transform: translateY(-8px);
        border-color: var(--houer-teal);
        background: rgba(27, 54, 100, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    .forn-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff; /* Valor sempre branco para destaque */
    }
    .metric-unit {
        font-size: 0.9rem;
        font-weight: 400;
        color: #64748b;
    }
    
    .section-title { 
        font-size: 1.5rem; 
        font-weight: 700; 
        color: #f8fafc; 
        margin: 3rem 0 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-title::before {
        content: "";
        display: block;
        width: 6px;
        height: 28px;
        background: var(--houer-teal);
        border-radius: 3px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--houer-navy), #0b111e);
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ── Caminhos ──────────────────────────────────────────────────────────────────
PASTA = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(PASTA, 'features.json')

# ── Configurações ─────────────────────────────────────────────────────────────
FORNECEDORES = ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']
CORES = {'LEDSTAR': '#00A9E0', 'SX LIGHTING': '#1B3664', 'TECNOWATT': '#64748b'}

TARGETS_MAP = {
    'lmed': 'Luminância Média',
    'uo': 'Fator de Uniformidade',
    'ul': 'Uniformidade Longitudinal',
    'emed': 'Iluminância Média',
    'emin': 'Iluminância mínima horizontal E (lux)',
    'w': 'Potência (W)'
}
UNITS_MAP = {
    'lmed': 'cd/m²', 'uo': '', 'ul': '', 'emed': 'lux', 'emin': 'lux', 'w': 'W'
}

# ── Template de Exportação (Colunas D a DM das planilhas originais) ─────────────
TEMPLATE_COLUMNS = [
    'ID', 'Padrão', 'Logradouro', 'latitude', 'longitude', 'Classificação viária',
    'Classificação ciclovia', 'Classificação pedonal', 'Tipo de lâmpada',
    'Potencia da lâmpada', 'Potência Reator', 'Potencia 2o nivel', 'POT TOTAL MALHA',
    'Faixas de Rodagem', 'Largura Passeio 1', 'Largura Via 1', 'largura Passeio Central 1',
    'largura Passeio Central 2', 'Largura Via 2', 'largura Passeio 2',
    'largura Canteiro Central', 'largura Ciclovia 1', 'largura Ciclovia 2',
    'estacionamento 1', 'estacionamento 2', 'posteacao', 'Tipo de estrutura',
    'distancia entre postes', 'altura da luminaria', 'qtd de Lampadas IP Princ',
    'distancia Poste a via', 'projecao do braço', 'Pendor', 'Altura de Instalação',
    'Projeção Vertical', 'Exclusivo', 'Quantidade de pontos inspecionados IP Veic',
    'qtd de Lampadas IP 2o nivel', 'Tipo de posteação 2o nivel', 'Distanciamento 2o nivel',
    'Altura da luminaria 2o nivel', 'Projecao 2o nivel', 'Distancia poste-via 2o nivel',
    'Quantidade de pontos inspecionados IP Sec', 'informacoes Adicionais', 'Emed - Norma',
    'U - Norma', 'Luminância Média Exigida', 'Uo Uniformidade Global Exigida',
    'Uniformidade Longitudinal Exigida', 'Incremento Linear Exigido',
    'Atendimento pleno à norma', 'Excesso de iluminância média IV',
    'Excesso de Uniformidade IV', 'Excesso de luminância média IV',
    'Iluminância Média', 'Fator de Uniformidade', 'Luminância Média',
    'Uniformidade Longitudinal', 'Incremento Linear (TI)', 'EIR',
    'ATENDE TUDO - RUA SIMU', 'Atende à Iluminância Média',
    'Atende à Uniformidade Global Miníma', 'Atende à Luminância Média',
    'Classe IV', 'Classe IP', 'Excesso de iluminância média P1', 'Iluminância Média.1',
    'Iluminância mínima horizontal E (lux)', 'Iluminância Média (Exigida)',
    'Iluminância mínima horizontal E (lux).1', 'Atende à NBR 5101 - TUDO SIMU',
    'Atende à Iluminância Média.1', 'Atende à Iluminância mínima horizontal',
    'Classe IP.1', 'Excesso de iluminância média P2', 'Iluminância Média.2',
    'Iluminância mínima horizontal E (lux).2', 'Iluminância Média (Exigida).1',
    'Iluminância mínima horizontal E (lux) exigida', 'Atende à NBR 5101 - TUDO SIMU P2',
    'Atende à Iluminância Média.2', 'Atende à Iluminância mínima horizontal.1',
    'Fornecedor', 'Código Luminária 1', 'Luminária Simulada (IP Principal)',
    'Código Luminária 2', 'Luminária Simulada (IP Secundário)',
    'Pontos por poste (IP Principal)', ' Potência simulada - IP Principal (W)',
    ' Potência simulada - IP Secundário (W)', 'Fluxo Luminoso - IP Principal (lm)',
    'Fluxo Luminoso - IP Secundário (lm)', 'Ângulo antigo', 'Ângulo Simulado',
    'Braço Antigo', 'Braço Novo', 'Projeção com alteração',
    'Altura de luminária com alteração', 'Correção de Ponto Escuro (CPE)',
    'Quantidade de pontos adicionados para via de veículo',
    'Quantidade de pontos adicionados para via de pedestres',
    'Observação CPE  (Reduçao entre postes e/ou Tipo de Posteação)',
    'obs conferência', 'rev conferência', 'IP PRINC SIM', 'IP SEC SIM',
    'TOTAL SIMULADO', 'Eficientização', 'Simulação', 'Conferência ',
    'Considerar na Extrapolação', 'Inspeção'
]

# Mapeamento de inputs da planilha para os nomes internos do modelo
MAPEAMENTO_COLS = {
    'Classificacao (M/C/P)': 'Classificação viária',
    'Classificacao': 'Classificação viária',
    'Classificao viria': 'Classificação viária',
    'Altura de Instalação': 'Altura de Instalação',
    'Altura de Instalao': 'Altura de Instalação',
    'altura da luminaria': 'altura da luminaria',
    'distancia entre poste': 'distancia entre postes',
    'distancia entre postes': 'distancia entre postes',
    'Largura da Via 1': 'Largura Via 1',
    'Largura Via 1': 'Largura Via 1',
    'projecao do braço': 'projecao do braço',
    'projecao do brao': 'projecao do braço',
    'Braço Novo': 'Braço Novo',
    'Brao Novo': 'Braço Novo',
    'Posteação': 'posteacao',
    'posteacao': 'posteacao',
    'Potencia Atual (W)': 'Potencia da lâmpada',
    'Potencia da lmpada': 'Potencia da lâmpada',
    'Tipo de lmpada': 'Tipo de lâmpada',
    'tipo de lampada': 'Tipo de lâmpada',
    'Tipo de lampada atual': 'Tipo de lâmpada',
    'Potencia Atual': 'Potencia da lâmpada',
    'Potência Atual (W)': 'Potencia da lâmpada',
    'Altura de luminária com alteração': 'Altura de luminária com alteração',
    'Projeção com alteração': 'Projeção com alteração',
    'Faixas de Rodagem': 'Faixas de Rodagem',
    'Largura Via 1': 'Largura Via 1',
    'Largura Via 2': 'Largura Via 2',
    'Largura Passeio 1': 'Largura Passeio 1',
    'largura Passeio 2': 'largura Passeio 2',
    'largura Canteiro Central': 'largura Canteiro Central',
    'distancia Poste a via': 'distancia Poste a via',
    'Tipo de estrutura': 'Tipo de estrutura'
}

# ── Tabela NBR 5101 – Requisitos Mínimos por Subclasse ─────────────────────────
# Fonte: ABNT NBR 5101:2024
# M = Luminância (cd/m²) | C/P = Iluminância (lux)
NBR5101 = {
    # Vias Motorizadas (Lmed em cd/m², Uo, Ul)
    'M1': {'metricas': ['lmed','uo','ul','w'], 'lmed': 2.0, 'uo': 0.40, 'ul': 0.70},
    'M2': {'metricas': ['lmed','uo','ul','w'], 'lmed': 1.5, 'uo': 0.40, 'ul': 0.70},
    'M3': {'metricas': ['lmed','uo','ul','w'], 'lmed': 1.0, 'uo': 0.40, 'ul': 0.60},
    'M4': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.75,'uo': 0.40, 'ul': 0.60},
    'M5': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.50,'uo': 0.35, 'ul': 0.40},
    'M6': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.30,'uo': 0.35, 'ul': 0.40},
    # Áreas de Conflito (Emed em lux, Uo)
    'C0': {'metricas': ['emed','uo','w'], 'emed': 50.0, 'uo': 0.40},
    'C1': {'metricas': ['emed','uo','w'], 'emed': 30.0, 'uo': 0.40},
    'C2': {'metricas': ['emed','uo','w'], 'emed': 20.0, 'uo': 0.40},
    'C3': {'metricas': ['emed','uo','w'], 'emed': 15.0, 'uo': 0.35},
    'C4': {'metricas': ['emed','uo','w'], 'emed': 10.0, 'uo': 0.35},
    'C5': {'metricas': ['emed','uo','w'], 'emed':  5.0, 'uo': 0.35},
    # Vias Pedonais/Ciclovias (Emed e Emin em lux)
    'P1': {'metricas': ['emed','emin','w'], 'emed': 20.0, 'emin': 7.5},
    'P2': {'metricas': ['emed','emin','w'], 'emed': 15.0, 'emin': 5.0},
    'P3': {'metricas': ['emed','emin','w'], 'emed': 10.0, 'emin': 3.0},
    'P4': {'metricas': ['emed','emin','w'], 'emed':  7.5, 'emin': 1.5},
    'P5': {'metricas': ['emed','emin','w'], 'emed':  5.0, 'emin': 1.0},
    'P6': {'metricas': ['emed','emin','w'], 'emed':  3.0, 'emin': 0.6},
}

# ── Carrega modelos ───────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelos(suffix=""):
    meta_path = os.path.join(PASTA, f'features{suffix}.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

    modelos = {}
    for key in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']:
        path = os.path.join(PASTA, f'modelo_{key}{suffix}.pkl')
        if os.path.exists(path):
            modelos[key] = joblib.load(path)
    return modelos, meta

# ── Carrega banco de dados de luminarias ────────────────────────────────────
@st.cache_data
def carregar_banco_luminarias():
    """Lê a aba 'Banco de dados' das planilhas e retorna DataFrame com Fornecedor, Potência e Valor."""
    dfs = []
    for arq in os.listdir(PASTA):
        if not arq.endswith('.xlsx'):
            continue
        try:
            xl = pd.ExcelFile(os.path.join(PASTA, arq), engine='openpyxl')
            aba = next((n for n in xl.sheet_names if 'banco' in n.lower()), None)
            if not aba:
                continue
            df_raw = pd.read_excel(os.path.join(PASTA, arq), sheet_name=aba, header=1)
            df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')
            # A primeira linha contém os nomes reais
            df_raw.columns = df_raw.iloc[0]
            df_raw = df_raw[1:].reset_index(drop=True)
            # Padroniza nomes das colunas relevantes
            col_forn = next((c for c in df_raw.columns if 'forn' in str(c).lower()), None)
            col_pot  = next((c for c in df_raw.columns if 'pot' in str(c).lower() and '[w]' in str(c).lower()), None)
            col_lum  = next((c for c in df_raw.columns if 'lumin' in str(c).lower() and 'cod' not in str(c).lower() and 'consider' not in str(c).lower()), None)
            col_val  = next((c for c in df_raw.columns if str(c).strip().lower() == 'valor'), None)
            if not all([col_forn, col_pot, col_val]):
                continue
            df_sel = df_raw[[col_forn, col_pot, col_val]].copy()
            if col_lum:
                df_sel['Luminaria'] = df_raw[col_lum]
            df_sel.columns = ['Fornecedor', 'Potencia_W', 'Valor_R$'] + (['Luminaria'] if col_lum else [])
            df_sel = df_sel[df_sel['Fornecedor'].isin(['LEDSTAR', 'SX LIGHTING', 'TECNOWATT'])]
            df_sel['Potencia_W'] = pd.to_numeric(df_sel['Potencia_W'], errors='coerce')
            df_sel['Valor_R$']   = pd.to_numeric(df_sel['Valor_R$'],   errors='coerce')
            df_sel = df_sel.dropna(subset=['Potencia_W', 'Valor_R$'])
            dfs.append(df_sel)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame(columns=['Fornecedor', 'Potencia_W', 'Valor_R$'])

def buscar_custo(banco: pd.DataFrame, fornecedor: str, potencia_w: float):
    """Retorna (luminaria, potencia_real, valor) da luminaria mais próxima em potência."""
    sub = banco[banco['Fornecedor'] == fornecedor].copy()
    if sub.empty or potencia_w is None:
        return None, None, None
    idx_min = (sub['Potencia_W'] - potencia_w).abs().idxmin()
    row = sub.loc[idx_min]
    lum = row.get('Luminaria', '') if 'Luminaria' in sub.columns else ''
    return lum, row['Potencia_W'], row['Valor_R$']

# ── Carrega banco de dados de luminarias ────────────────────────────────────
banco_luminarias = carregar_banco_luminarias()

# ── Header ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 3])
with col_l:
    logo_path = os.path.join(PASTA, 'images.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=180)
    else:
        st.markdown('<div style="background:white; border-radius:12px; padding:10px; display:flex; justify-content:center; align-items:center; width:80px; height:80px;"><span style="color:#1B3664; font-size:24px; font-weight:900;">H<span style="color:#00A9E0;">O</span></span></div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<p class="hero-title" style="margin-top:10px;">Houer</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="margin-top:-15px; font-weight:600; color:var(--houer-teal);">impactando gerações</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🏷️ Classificação da Via')
    st.markdown('**Selecione a subclasse NBR 5101:**')
    
    opcoes_via = (
        ['M1','M2','M3','M4','M5','M6'] +
        ['C0','C1','C2','C3','C4','C5'] +
        ['P1','P2','P3','P4','P5','P6']
    )
    subclasse = st.selectbox(
        'Subclasse da Via',
        opcoes_via,
        index=3,  # default C3
        format_func=lambda x: f"{x} — {'Via Motorizada' if x.startswith('M') else ('Área de Conflito' if x.startswith('C') else 'Via Pedonal/Ciclovia')}"
    )
    info_nbr = NBR5101.get(subclasse, {})
    
    # Mostra requisitos mínimos da subclasse selecionada
    st.markdown('**Requisitos NBR 5101:**')
    req_html = ""
    for k, v in info_nbr.items():
        if k == 'metricas': continue
        label = TARGETS_MAP.get(k, k)
        unit  = UNITS_MAP.get(k, '')
        req_html += f"<div style='font-size:0.8rem;color:#9ca3af;'>{label}: <b style='color:#FFD700;'>≥ {v} {unit}</b></div>"
    st.markdown(req_html, unsafe_allow_html=True)
    
    st.divider()
    st.markdown('## 🧹 Inteligência de Dados')
    modo_dados = st.radio(
        'Base de Treinamento:',
        ['Padrão (Com outliers)', 'Otimizada (Sem outliers)'],
        index=1,
        help="A opção Otimizada remove valores fisicamente impossíveis das planilhas originais para melhorar a precisão."
    )
    sufixo_modelo = "_limpo" if modo_dados == 'Otimizada (Sem outliers)' else ""
    
    modelos, meta = carregar_modelos(sufixo_modelo)
    num_ok = meta.get('features_numericas', [])
    cat_ok = meta.get('features_categoricas', [])

    # Métricas cujos modelos são confiáveis o suficiente para verificar conformidade NBR
    # Threshold: R² >= 0.5. Modelos abaixo disso (ex: uo R²=-57.5, ul R²=-0.05) predizem
    # valores fisicamente impossíveis e causariam falsos 'Não Atende'.
    R2_MIN_CONFORMIDADE = 0.5
    metricas_confiaveis = {
        m for m in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
        if meta.get(f'modelo_{m}', {}).get('r2', 0) >= R2_MIN_CONFORMIDADE
    }

    if not modelos:
        st.error(f'Modelos ({modo_dados}) não encontrados!')
        st.stop()

    st.divider()
    st.markdown('## ⚙️ Parâmetros (Individual)')
    
    st.markdown('### 🛣️ Geometria da Via')
    faixas         = st.slider('Faixas de Rodagem',            1, 6,    2, step=1)
    largura_via1   = st.slider('Largura Via 1 (m)',            4.0, 20.0, 7.0, step=0.5)
    largura_via2   = st.slider('Largura Via 2 (m)',            0.0, 20.0, 0.0, step=0.5)
    largura_passeio1 = st.slider('Largura Passeio 1 (m)',      0.0, 10.0, 2.0, step=0.5)
    largura_passeio2 = st.slider('Largura Passeio 2 (m)',      0.0, 10.0, 2.0, step=0.5)
    canteiro       = st.slider('Largura Canteiro Central (m)', 0.0, 10.0, 0.0, step=0.5)

    st.markdown('### 🏗️ Estrutura')
    altura_lum     = st.slider('Altura da Luminária (m)',      4.0, 16.0, 9.0,  step=0.5)
    projecao_braco = st.slider('Projeção do Braço (m)',        0.0,  4.0, 1.5,  step=0.25)
    dist_postes    = st.slider('Distância entre Postes (m)',  10.0, 60.0, 35.0, step=1.0)
    dist_poste_via = st.slider('Distância Poste à Via (m)',    0.0,  3.0,  0.5, step=0.25)
    altura_inst    = st.slider('Altura de Instalação (m)',     4.0, 16.0, 10.0, step=0.5)

    st.markdown('### 🔩 Configurações')
    tipo_estrutura = st.selectbox('Tipo de Estrutura', ['Braço', 'Suporte'])
    posteacao      = st.selectbox('Posteação', ['Unilateral', 'Canteiro central', 'Bilateral alternada', 'Bilateral frontal'])
    braco_novo     = st.selectbox('Braço Novo', ['Longo II', 'Longo I', 'Médio I', 'Médio II', 'Curto II', 'Curto I'])

    st.markdown('---')
    st.markdown('### 📍 Localização do Projeto')
    endereco_busca = st.text_input('Endereço ou Rua', placeholder='Ex: Rua Joaquim Murtinho, Cuiabá')
    
    if st.button('🔍 Localizar no Mapa'):
        if GOOGLE_MAPS_API_KEY and endereco_busca:
            try:
                geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY)
                location = geolocator.geocode(endereco_busca, timeout=10)
                if location:
                    st.session_state.lat = location.latitude
                    st.session_state.lon = location.longitude
                    st.session_state.address = location.address
                    st.success(f"📍 Localizado: {location.address}")
                else:
                    st.warning("Endereço não encontrado.")
            except Exception as e:
                st.error(f"Erro na geocodificação: {e}")
        else:
            st.info("Insira um endereço para localizar.")

    st.markdown('### ⚡ Eficientização')
    potencia_atual = st.number_input('Potência Atual (W)', min_value=0.0, value=250.0, step=10.0, help="Potência da luminária instalada atualmente (ex: Sódio 250W, 400W)")

def gerar_pdf(fornecedores, resultados, info_nbr, inputs, banco_luminarias, sugestoes, endereco="Não informado"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Cabeçalho
    pdf.set_text_color(27, 54, 100) # Azul Houer
    pdf.cell(0, 10, "Relatório de Simulação de Iluminação Pública", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    pdf.ln(5)

    # Endereço
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Localização do Projeto:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 7, endereco)
    pdf.ln(5)

    # Parâmetros de Entrada
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "1. Parâmetros da Via:", ln=True)
    pdf.set_font("Arial", "", 9)
    
    col_width = 45
    for k, v in inputs.items():
        if k == 'Fornecedor': continue
        pdf.cell(col_width, 7, f"{k}: {v}", border=1)
        if pdf.get_x() > 140: pdf.ln()
    
    if pdf.get_x() > 10: pdf.ln() # Garante que quebrou a linha no final do loop
    pdf.ln(5)

    # Tabela de Resultados
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "2. Resultados Técnicos por Fornecedor:", ln=True)
    pdf.set_font("Arial", "B", 9)
    
    # Header da Tabela
    pdf.cell(35, 8, "Fornecedor", 1, 0, 'C')
    pdf.cell(25, 8, "Pot. (W)", 1, 0, 'C')
    pdf.cell(30, 8, "Ilum./Lum.", 1, 0, 'C')
    pdf.cell(30, 8, "Uniform.", 1, 0, 'C')
    pdf.cell(40, 8, "Status NBR", 1, 1, 'C')

    pdf.set_font("Arial", "", 9)
    for forn in fornecedores:
        pot = resultados['w'].get(forn, 0)
        m_v = resultados['lmed'].get(forn) if 'lmed' in resultados else resultados['emed'].get(forn)
        u_v = resultados['uo'].get(forn) if 'uo' in resultados else 0
        
        status = "OK"
        for m in info_nbr.get('metricas', []):
            if m == 'w': continue
            if m not in metricas_confiaveis: continue
            val = resultados[m].get(forn)
            req = info_nbr.get(m)
            if val is not None and req is not None and val < req:
                status = "Não Atende"
                break

        pdf.cell(35, 8, forn, 1)
        pdf.cell(25, 8, f"{pot:.1f}", 1, 0, 'C')
        pdf.cell(30, 8, f"{m_v:.2f}" if m_v else "-", 1, 0, 'C')
        pdf.cell(30, 8, f"{u_v:.2f}" if u_v else "-", 1, 0, 'C')
        pdf.cell(40, 8, status, 1, 1, 'C')
    
    pdf.ln(5)

    # Custos e Eficiência
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "3. Viabilidade Econômica:", ln=True)
    pdf.set_font("Arial", "", 9)
    for forn in fornecedores:
        pot_prev = resultados['w'].get(forn)
        lum_nome, pot_real, custo = buscar_custo(banco_luminarias, forn, pot_prev)
        if custo:
            # Garante que começa na margem esquerda (X=10)
            pdf.set_x(10)
            pdf.multi_cell(0, 7, f"- {forn}: Sugerida {lum_nome} ({pot_real}W) | Custo Unitário: R$ {custo:,.2f}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(0, 5, "Este relatório foi gerado por Inteligência Artificial baseado em dados históricos de simulações. Os resultados são estimativas e devem ser validados por projeto luminotécnico definitivo.")

    return bytes(pdf.output())

def gerar_pdf_lote(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ── PÁGINA 1: DASHBOARD EXECUTIVO
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(27, 54, 100)
    pdf.cell(0, 10, "Relatório Executivo de Simulação em Lote", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # KPIs Globais
    pdf.set_fill_color(240, 244, 248)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Resumo Consolidado (Performance Global)", ln=True)
    pdf.set_font("Arial", "B", 9)
    
    # Cabeçalho da tabela de KPIs
    pdf.cell(35, 8, "Fornecedor", 1, 0, 'C', True)
    pdf.cell(50, 8, "CAPEX Total Est.", 1, 0, 'C', True)
    pdf.cell(50, 8, "Economia Total", 1, 0, 'C', True)
    pdf.cell(50, 8, "Conformidade NBR", 1, 1, 'C', True)

    pdf.set_font("Arial", "", 9)
    for forn in FORNECEDORES:
        c_col = f'Custo Unitario (R$) - {forn}'
        e_col = f'Economia (W) - {forn}'
        s_col = f'Status NBR - {forn}'
        
        capex = df[c_col].sum() if c_col in df.columns else 0
        eco_kw = (df[e_col].sum() / 1000) if e_col in df.columns else 0
        conf = (df[s_col] == '✔ Atende').mean() * 100 if s_col in df.columns else 0
        
        pdf.cell(40, 8, forn, 1)
        pdf.cell(50, 8, f"R$ {capex:,.2f}", 1, 0, 'C')
        pdf.cell(50, 8, f"{eco_kw:,.1f} kW", 1, 0, 'C')
        pdf.cell(45, 8, f"{conf:.1f}%", 1, 1, 'C')

    pdf.ln(10)

    # Mix de Vias
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Perfil do Inventário Simulado", ln=True)
    pdf.set_font("Arial", "", 10)
    if 'Classificacao (M/C/P)' in df.columns:
        counts = df['Classificacao (M/C/P)'].value_counts()
        for classe, qtd in counts.items():
            pdf.cell(0, 7, f"- Classe {classe}: {qtd} pontos ({qtd/len(df)*100:.1f}%)", ln=True)
    
    pdf.ln(10)

    # ── PÁGINAS SEGUINTES: LISTA COMPACTA
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Inventário Detalhado de Simulações", ln=True)
    pdf.ln(5)
    
    # Cabeçalho da lista compacta (font reduzida)
    pdf.set_font("Arial", "B", 7)
    pdf.set_fill_color(230, 230, 230)
    cols_lista = [
        (10, "ID"), (15, "Classe"), (15, "Alt."), (15, "Dist."), 
        (45, "LEDSTAR (W/Std)"), (45, "SX LIGHT. (W/Std)"), (45, "TECNOWATT (W/Std)")
    ]
    for w, t in cols_lista:
        pdf.cell(w, 6, t, 1, 0, 'C', True)
    pdf.ln()

    pdf.set_font("Arial", "", 5.5) # Fonte um pouco menor para caber os 3
    
    # Identifica colunas de geometria de forma robusta
    def find_col(df, text):
        for col in df.columns:
            if text.lower() in col.lower(): return col
        return None

    col_alt = find_col(df, 'Altura')
    col_dist = find_col(df, 'distancia')

    for idx, row in df.iterrows():
        pdf.cell(10, 5, str(idx+1), 1, 0, 'C')
        pdf.cell(15, 5, str(row.get('Classificacao (M/C/P)', '-')), 1, 0, 'C')
        pdf.cell(15, 5, f"{row.get(col_alt, 0):.1f}" if col_alt else "0.0", 1, 0, 'C')
        pdf.cell(15, 5, f"{row.get(col_dist, 0):.1f}" if col_dist else "0.0", 1, 0, 'C')
        
        for forn in FORNECEDORES:
            # Busca a coluna de potência de forma flexível (evitando problemas de acento)
            pot_col = None
            for c in df.columns:
                c_low = c.lower()
                # Procura por 'prevista' ou 'nominal' + nome do fornecedor
                if ('prevista' in c_low or 'nominal' in c_low) and forn.lower() in c_low:
                    pot_col = c
                    break
            
            std_col = f'Status NBR - {forn}'
            p_val = row.get(pot_col, 0) if pot_col else 0
            s_val = "OK" if row.get(std_col) == '✔ Atende' else "NA"
            pdf.cell(45, 5, f"{p_val:.1f}W [{s_val}]", 1, 0, 'C')
        pdf.ln()

    return bytes(pdf.output())

def gerar_template_lote():
    cols = [
        'ID', 'Classificacao (M/C/P)', 'Altura de Instalação', 'distancia entre poste',
        'Largura da Via 1', 'projecao do braço', 'Braço Novo', 'Posteação', 'Potencia Atual (W)'
    ]
    df_temp = pd.DataFrame(columns=cols)
    # Linha de exemplo
    df_temp.loc[0] = [1, 'M3', 10.0, 35.0, 7.0, 1.5, 'Longo II', 'Unilateral', 250.0]
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_temp.to_excel(writer, index=False)
    return buffer.getvalue()

def analisar_melhorias(forn, modelos, metricas_ativas, info_nbr, config_base, num_ok, cat_ok):
    """Testa variações estruturais para tentar atingir a conformidade."""
    sugestoes = []
    reqs = {m: info_nbr.get(m) for m in metricas_ativas if m != 'w' and info_nbr.get(m) is not None}
    if not reqs: return []

    def verifica_atende(conf):
        X_test = pd.DataFrame([{k: conf.get(k, np.nan) for k in num_ok + cat_ok}])
        for m, req in reqs.items():
            if m in modelos:
                val = modelos[m].predict(X_test)[0]
                if val < req: return False
        return True

    # 1. Tentar aumentar altura
    for h_add in [1.0, 2.0]:
        c = config_base.copy()
        c['altura da luminaria'] += h_add
        if 'Altura de Instalação' in c: c['Altura de Instalação'] += h_add
        if 'Altura de Instalao' in c: c['Altura de Instalao'] += h_add
        if verifica_atende(c):
            sugestoes.append(f"📐 **Alteração Estrutural**: Aumentar a altura para **{c['altura da luminaria']:.1f}m**")
            break
            
    # 2. Tentar aumentar projeção
    for p_add in [0.5, 1.0]:
        c = config_base.copy()
        c['projecao do braço'] += p_add
        if 'projecao do brao' in c: c['projecao do brao'] += p_add
        if verifica_atende(c):
            sugestoes.append(f"🏗️ **Ajuste de Braço**: Aumentar a projeção para **{c['projecao do braço']:.1f}m**")
            break

    # 3. Tentar reduzir distância
    dist_atual = config_base['distancia entre postes']
    for d_sub in [5.0, 10.0]:
        if dist_atual - d_sub >= 10:
            c = config_base.copy()
            c['distancia entre postes'] -= d_sub
            if verifica_atende(c):
                sugestoes.append(f"📍 **Ajuste de Vão**: Reduzir a distância para **{c['distancia entre postes']:.1f}m**")
                break
                
    # 4. Solução Drástica: Novo poste no meio (Ponto Escuro)
    if not sugestoes and dist_atual >= 20:
        c = config_base.copy()
        c['distancia entre postes'] /= 2
        if verifica_atende(c):
            sugestoes.append(f"🔦 **Correção de Ponto Escuro**: Instalar poste intermediário (nova distância: **{c['distancia entre postes']:.1f}m**)")

    return sugestoes

def formatar_resultado_template(df_saida):
    """
    Transforma o DataFrame de saída (largo) em um formato longo (1 linha por fornecedor)
    e mapeia para as colunas do template original.
    """
    rows = []
    for idx, row in df_saida.iterrows():
        for forn in FORNECEDORES:
            new_row = {col: row.get(col, np.nan) for col in TEMPLATE_COLUMNS}
            
            # Identifica ID
            new_row['ID'] = row.get('ID', idx + 1)
            
            # Mapeia inputs originais se existirem no df_saida
            for original, interno in MAPEAMENTO_COLS.items():
                if original in row:
                    new_row[interno] = row[original]
            
            # Adiciona rastreabilidade e parâmetros extras pedida pelo usuário
            for extra in [
                'Padrão', 'Logradouro', 'latitude', 'longitude', 'Tipo de lâmpada', 
                'qtd de Lampadas IP Princ', 'Faixas de Rodagem', 'Largura Via 1', 
                'Largura Via 2', 'Largura Passeio 1', 'largura Passeio 2', 
                'largura Canteiro Central', 'distancia Poste a via', 'Tipo de estrutura',
                'Altura de Instalação'
            ]:
                if extra in row:
                    new_row[extra] = row[extra]
            
            # Dados do Fornecedor e Predições
            new_row['Fornecedor'] = forn
            
            # Mapeia métricas preditas
            map_targets = {
                'lmed': 'Luminância Média',
                'uo': 'Fator de Uniformidade',
                'ul': 'Uniformidade Longitudinal',
                'emed': 'Iluminância Média',
                'emin': 'Iluminância mínima horizontal E (lux)',
                'w': ' Potência simulada - IP Principal (W)'
            }
            
            for key, template_name in map_targets.items():
                col_name = f'{TARGETS_MAP[key]} - {forn}'
                if col_name in row:
                    new_row[template_name] = row[col_name]
            
            # NBR Status e Requisitos
            classe = str(row.get('Classificação viária', 'M3')).upper()
            info_v = NBR5101.get(classe, {})
            
            new_row['Emed - Norma'] = info_v.get('emed', np.nan)
            new_row['Luminância Média Exigida'] = info_v.get('lmed', np.nan)
            new_row['Uo Uniformidade Global Exigida'] = info_v.get('uo', np.nan)
            new_row['Uniformidade Longitudinal Exigida'] = info_v.get('ul', np.nan)
            
            status_col = f'Status NBR - {forn}'
            if status_col in row:
                status_txt = row[status_col]
                new_row['Atendimento pleno à norma'] = 'Sim' if 'Atende' in status_txt and 'Não' not in status_txt else 'Não'
                # Preenchimento redundante para outras colunas de status no template
                new_row['ATENDE TUDO - RUA SIMU'] = new_row['Atendimento pleno à norma']
                new_row['Atende à Iluminância Média'] = new_row['Atendimento pleno à norma']
            
            # Modelo Sugerido e Custos
            new_row['Luminária Simulada (IP Principal)'] = row.get(f'Modelo Sugerido - {forn}', '')
            new_row['Eficientização'] = row.get(f'Reducao (%) - {forn}', 0)
            
            rows.append(new_row)
            
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)

# Métricas ativas baseadas na subclasse NBR 5101
metricas_ativas = NBR5101.get(subclasse, {}).get('metricas', ['emed', 'w'])

# ── Tabs Principais ───────────────────────────────────────────────────────────
tab_individual, tab_lote, tab_dash = st.tabs(['🎯 Simulação Individual', '📂 Simulação em Lote', '📊 Dashboard de Lote'])

# ==============================================================================
# TAB 1: SIMULAÇÃO INDIVIDUAL
# ==============================================================================
with tab_individual:
    # ── Exibição do Mapa (se localizado)
    if 'lat' in st.session_state and 'lon' in st.session_state:
        fig_map = px.scatter_mapbox(
            lat=[st.session_state.lat], 
            lon=[st.session_state.lon],
            zoom=15, 
            height=300
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(f"📌 {st.session_state.address}")

    def montar_entrada(fornecedor):
        dados = {
            'Faixas de Rodagem':        faixas,
            'Largura Via 1':            largura_via1,
            'Largura Via 2':            largura_via2,
            'Largura Passeio 1':        largura_passeio1,
            'largura Passeio 2':        largura_passeio2,
            'largura Canteiro Central': canteiro,
            'altura da luminaria':      altura_lum,
            'projecao do braço':        projecao_braco,
            'projecao do brao':         projecao_braco,
            'distancia entre postes':   dist_postes,
            'distancia Poste a via':    dist_poste_via,
            'Altura de Instalação':     altura_inst,
            'Altura de Instalao':       altura_inst,
            'Classificação viária':    subclasse,   # subclasse NBR 5101
            'Tipo de estrutura':        tipo_estrutura,
            'posteacao':                posteacao,
            'Braço Novo':               braco_novo,
            'Brao Novo':                braco_novo,
            'Fornecedor':               fornecedor,
        }
        colunas = num_ok + cat_ok
        return pd.DataFrame([{k: dados.get(k, np.nan) for k in colunas}])

    # Roda as predições e busca melhorias
    resultados = {m: {} for m in metricas_ativas}
    sugestoes_por_forn = {}

    for forn in FORNECEDORES:
        X_in = montar_entrada(forn)
        config_dict = X_in.iloc[0].to_dict()
        
        # Predição base
        for m in metricas_ativas:
            if m in modelos:
                try:
                    val = modelos[m].predict(X_in)[0]
                    resultados[m][forn] = max(val, 0)
                except Exception as e:
                    resultados[m][forn] = None
            else:
                resultados[m][forn] = None
        
        # Verifica se atende tudo — usa apenas métricas cujos modelos são confiáveis
        falhou = False
        for m in metricas_ativas:
            if m == 'w': continue
            if m not in metricas_confiaveis: continue
            val = resultados[m].get(forn)
            req_min = info_nbr.get(m)
            if val is not None and req_min is not None:
                if val < req_min:
                    falhou = True
                    break
        
        if falhou:
            sugestoes_por_forn[forn] = analisar_melhorias(forn, modelos, metricas_ativas, info_nbr, config_dict, num_ok, cat_ok)

    # Exibe os Cards Dinâmicos
    st.markdown('<p class="section-title">📊 Resultados por Fornecedor</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, forn in enumerate(FORNECEDORES):
        cor = CORES[forn]
        with cols[i]:
            html_content = f'<div class="forn-card" style="border-top-color:{cor};"><div class="forn-name">{forn}</div>'
            for m in metricas_ativas:
                val = resultados[m].get(forn)
                val_str = f"{val:,.2f}" if val is not None else "—"
                unit = UNITS_MAP[m]
                label = TARGETS_MAP[m]
                req_min = info_nbr.get(m)
                badge_html = ""
                if val is not None and req_min is not None and m != 'w':
                    if m in metricas_confiaveis:
                        # Modelo confiável: badge verde/vermelho de conformidade
                        atende = val >= req_min
                        b_color = '#22c55e' if atende else '#ef4444'
                        b_txt   = '✔ Atende' if atende else '✘ Não Atende'
                        badge_html = f'<span style="font-size:.65rem;padding:2px 6px;border-radius:99px;background:{b_color};color:#fff;margin-left:6px;">{b_txt}</span>'
                    else:
                        # Modelo com baixa acurácia: apenas informa que é estimativa
                        badge_html = '<span style="font-size:.65rem;padding:2px 6px;border-radius:99px;background:#6b7280;color:#fff;margin-left:6px;">Estimado</span>'
                
                html_content += f'<div class="metric-label"><span>{label}</span>{badge_html}</div>'
                html_content += f'<div class="metric-value">{val_str} <span class="metric-unit">{unit}</span></div>'
            
            # Adiciona métricas de eficientização se 'w' foi previsto
            if 'w' in resultados and resultados['w'].get(forn) is not None and potencia_atual > 0:
                pot_prev = resultados['w'].get(forn)
                economia = potencia_atual - pot_prev
                reducao  = (economia / potencia_atual) * 100
                if economia > 0:
                    html_content += f'<div style="margin-top:15px; padding-top:15px; border-top:1px dashed #374151;">'
                    html_content += f'<div style="font-size:0.75rem; color:#22c55e; font-weight:700; text-transform:uppercase; margin-bottom:5px;">🍃 Economia de Energia</div>'
                    html_content += f'<div style="display:flex; justify-content:space-between; align-items:baseline;">'
                    html_content += f'<span style="font-size:1.2rem; font-weight:800; color:#22c55e;">{economia:,.1f}W</span>'
                    html_content += f'<span style="font-size:0.9rem; font-weight:600; color:#4ade80;">-{reducao:.1f}%</span>'
                    html_content += f'</div></div>'

            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)
    
    # ── Botão de Exportação PDF
    st.markdown("---")
    addr = st.session_state.get('address', 'Não informado')
    inputs_dict = {
        'Subclasse': subclasse,
        'Posteação': posteacao,
        'Altura (m)': altura_lum,
        'Distância (m)': dist_postes,
        'Largura (m)': largura_via1,
        'Projeção (m)': projecao_braco,
    }
    
    pdf_bytes = gerar_pdf(FORNECEDORES, resultados, info_nbr, inputs_dict, banco_luminarias, sugestoes_por_forn, addr)
    
    st.download_button(
        label="📥 Baixar Relatório Técnico (PDF)",
        data=pdf_bytes,
        file_name=f"Relatorio_Simulacao_{subclasse}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

    # ── Cards de Custo por Fornecedor
    if not banco_luminarias.empty and 'w' in resultados:
        st.markdown('<p class="section-title">💰 Luminária Mais Próxima e Custo Estimado</p>', unsafe_allow_html=True)
        custo_cols = st.columns(3)
        for i, forn in enumerate(FORNECEDORES):
            pot_prev = resultados['w'].get(forn)
            lum_nome, pot_real, custo = buscar_custo(banco_luminarias, forn, pot_prev)
            cor = CORES[forn]
            with custo_cols[i]:
                if custo is not None:
                    delta_str = f"+{pot_real - pot_prev:.0f}W" if pot_real > pot_prev else f"{pot_real - pot_prev:.0f}W"
                    c_html = f'<div style="background:rgba(27,54,100,0.4); backdrop-filter:blur(10px); border-radius:20px; padding:1.5rem; text-align:center; border:1px solid #2d4060; border-top:4px solid {cor}; height:100%;">'
                    c_html += f'<div style="font-size:.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.1em; margin-bottom:1rem;">{forn}</div>'
                    c_html += f'<div style="font-size:.85rem; color:#f8fafc; font-weight:600; margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{lum_nome}">{lum_nome}</div>'
                    c_html += f'<div style="font-size:.9rem; color:#94a3b8; margin-bottom:1rem;">{pot_real:.0f}W <span style="font-size:.7rem; color:#64748b;">({delta_str})</span></div>'
                    c_html += f'<div style="font-size:2.2rem; font-weight:800; color:white;">R$ {custo:,.2f}</div></div>'
                    st.markdown(c_html, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size:.75rem; color:#94a3b8; text-transform:uppercase;">{forn}</div><div style="font-size:1.1rem; color:#64748b; margin-top:1rem;">Sem dados no banco</div></div>', unsafe_allow_html=True)

    # ── Seção de Sugestões (Compliance Assistant)
    if sugestoes_por_forn:
        st.markdown('<p class="section-title">💡 Assistente de Conformidade (Sugestões)</p>', unsafe_allow_html=True)
        sug_cols = st.columns(3)
        for i, forn in enumerate(FORNECEDORES):
            with sug_cols[i]:
                sugs = sugestoes_por_forn.get(forn, [])
                if sugs:
                    cor = CORES[forn]
                    st.markdown(f"""
                    <div style="background:rgba(255, 215, 0, 0.05); border-left:4px solid {cor}; border-radius:10px; padding:15px; height:100%;">
                        <div style="font-size:0.8rem; font-weight:700; color:{cor}; margin-bottom:8px;">{forn} - RECOMENDAÇÕES:</div>
                        {"".join([f'<div style="font-size:0.85rem; color:#f8fafc; margin-bottom:5px;">{s}</div>' for s in sugs])}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="padding:15px; color:#22c55e; font-size:0.85rem; font-weight:600;">✔ Configuração atende os requisitos para {forn}.</div>', unsafe_allow_html=True)

    # Gráficos
    st.markdown('<p class="section-title">📈 Comparativo Gráfico</p>', unsafe_allow_html=True)
    tabs_graf = st.tabs([TARGETS_MAP[m] for m in metricas_ativas])
    
    def bar_chart(valores, unidade, titulo):
        fig = go.Figure()
        for forn, val in valores.items():
            if val is not None:
                fig.add_trace(go.Bar(
                    x=[forn], y=[val], name=forn,
                    marker_color=CORES[forn],
                    text=[f'{val:,.2f} {unidade}'], textposition='outside',
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

    for idx, m in enumerate(metricas_ativas):
        with tabs_graf[idx]:
            st.plotly_chart(bar_chart(resultados[m], UNITS_MAP[m], TARGETS_MAP[m]), use_container_width=True)

# ==============================================================================
# TAB 2: SIMULAÇÃO EM LOTE
# ==============================================================================
with tab_lote:
    st.markdown('### 📥 1. Baixe a Planilha Padrão')
    st.markdown('Preencha as características geométricas de cada instalação.')
    
    # Colunas necessárias para o modelo + rastreabilidade (em ordem do template)
    cols_template = [
        'ID', 'Padrão', 'Logradouro', 'latitude', 'longitude', 
        'Classificação viária', 'Tipo de lâmpada', 'Potencia da lâmpada',
        'Faixas de Rodagem', 'Largura Passeio 1', 'Largura Via 1', 'Largura Via 2',
        'largura Passeio 2', 'largura Canteiro Central', 'posteacao', 'Tipo de estrutura',
        'distancia entre postes', 'altura da luminaria', 'qtd de Lampadas IP Princ',
        'distancia Poste a via', 'projecao do braço', 'Altura de Instalação', 'Braço Novo'
    ]
    
    df_template = pd.DataFrame(columns=cols_template)
    # Linha de exemplo
    exemplo = {
        'ID': 1,
        'Padrão': 'V4',
        'Logradouro': 'Rua Exemplo, 100',
        'latitude': -23.5505,
        'longitude': -46.6333,
        'Classificação viária': 'M3',
        'Tipo de lâmpada': 'Sódio',
        'Potencia da lâmpada': 250.0,
        'Faixas de Rodagem': 2,
        'Largura Passeio 1': 2.0,
        'Largura Via 1': 7.0,
        'Largura Via 2': 0.0,
        'largura Passeio 2': 2.0,
        'largura Canteiro Central': 0.0,
        'posteacao': 'Unilateral',
        'Tipo de estrutura': 'Braço',
        'distancia entre postes': 35.0,
        'altura da luminaria': 10.0,
        'qtd de Lampadas IP Princ': 1,
        'distancia Poste a via': 0.5,
        'projecao do braço': 1.5,
        'Altura de Instalação': 10.0,
        'Braço Novo': 'Longo II'
    }
    df_template = pd.concat([df_template, pd.DataFrame([exemplo])], ignore_index=True)
    
    buffer_template = io.BytesIO()
    with pd.ExcelWriter(buffer_template, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False)
    
    st.download_button(
        label='⬇️ Baixar Planilha Padrão (.xlsx)',
        data=buffer_template.getvalue(),
        file_name='template_simulacao_HOUER.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    
    st.markdown('---')
    st.markdown('### 📤 2. Envie a Planilha Preenchida')
    arquivo_up = st.file_uploader('Selecione o arquivo modificado', type=['xlsx', 'csv'])
    
    if arquivo_up is not None:
        try:
            df_entrada = pd.read_csv(arquivo_up) if arquivo_up.name.endswith('.csv') else pd.read_excel(arquivo_up)
            
            # Tratamento robusto de decimais: aplica vírgula→ponto célula a célula antes de tentar converter
            def normalizar_coluna(serie):
                serie_norm = serie.apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
                convertida = pd.to_numeric(serie_norm, errors='coerce')
                # Se pelo menos 50% dos valores converteram bem, usa a versão numérica
                # Caso contrário (coluna de texto como Fornecedor), mantém original
                if convertida.notna().mean() >= 0.5:
                    return convertida
                return serie
            
            for col in df_entrada.columns:
                df_entrada[col] = normalizar_coluna(df_entrada[col])
            
            st.success(f'Arquivo lido com sucesso! ({len(df_entrada)} linhas)')
            
            with st.spinner('Realizando previsões...'):
                # Prepara DF de saída (preservando o ID se existir de forma case-insensitive)
                df_saida = df_entrada.copy()
                cols_upper = [c.upper() for c in df_saida.columns]
                if 'ID' not in cols_upper:
                    df_saida.insert(0, 'ID', range(1, len(df_saida) + 1))
                
                # Mapeia as colunas do arquivo para os nomes internos
                df_pipeline = df_entrada.rename(columns=MAPEAMENTO_COLS)
                
                # Se a planilha não tiver Classificacao, usamos a do sidebar
                # Procura por qualquer variação de 'Classificacao'
                col_classe = next((c for c in df_pipeline.columns if 'Classificação viária' == c), None)
                tem_classe = col_classe is not None
                
                for forn in FORNECEDORES:
                    df_run = df_pipeline.copy()
                    df_run['Fornecedor'] = forn
                    for col in num_ok + cat_ok:
                        if col not in df_run.columns:
                            df_run[col] = np.nan
                            
                    # Prever todas as métricas para a facilidade de extração, 
                    # ou apenas as métricas dependendo da linha. Para simplificar no Lote, calculamos tudo.
                    for m in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']:
                        if m in modelos:
                            preds = modelos[m].predict(df_run)
                            
                            # Filtro dinâmico por linha baseada na classificação
                            if tem_classe:
                                classes = df_pipeline['Classificação viária'].fillna('M').astype(str).str.upper().str[0]
                                # Máscara de validade
                                mask_valid = [False]*len(preds)
                                for i, c in enumerate(classes):
                                    if c == 'M' and m in ['lmed', 'uo', 'ul', 'w']: mask_valid[i] = True
                                    elif c == 'C' and m in ['emed', 'uo', 'w']: mask_valid[i] = True
                                    elif c == 'P' and m in ['emed', 'emin', 'w']: mask_valid[i] = True
                                    
                                preds = [p if valid else np.nan for p, valid in zip(preds, mask_valid)]
                            
                            df_saida[f'{TARGETS_MAP[m]} - {forn}'] = [max(p, 0) if pd.notna(p) else p for p in preds]

                # ── Pós-processamento: Ajuste de Potência pela Hierarquia NBR ──────────────
                # Se a iluminância prevista está abaixo do mínimo exigido pela classe NBR,
                # a potência é escalada proporcionalmente (assume P ∝ Iluminância para a
                # mesma geometria). Garante que vias mais exigentes (ex: C0) recebam
                # potência condizente com os requisitos da norma.
                if tem_classe:
                    classes_serie = df_pipeline['Classificação viária'].fillna('').astype(str).str.upper()
                    for forn in FORNECEDORES:
                        pot_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_col not in df_saida.columns:
                            continue
                        for idx_loc, classe in zip(df_saida.index, classes_serie):
                            info_v = NBR5101.get(classe, {})
                            if not info_v:
                                continue
                            # Métrica de iluminância principal da classe
                            metric_ref = 'lmed' if classe.startswith('M') else 'emed'
                            ilum_col   = f'{TARGETS_MAP[metric_ref]} - {forn}'
                            req        = info_v.get(metric_ref)
                            if req is None or ilum_col not in df_saida.columns:
                                continue
                            pred_ilum = df_saida.loc[idx_loc, ilum_col]
                            pred_pot  = df_saida.loc[idx_loc, pot_col]
                            if (pd.notna(pred_ilum) and pred_ilum > 0
                                    and pd.notna(pred_pot) and pred_ilum < req):
                                # Escala a potência proporcionalmente ao déficit de iluminância
                                df_saida.loc[idx_loc, pot_col] = pred_pot * (req / pred_ilum)

                # Cálculo de Eficientização e Conformidade por linha no lote
                # Detecta a coluna de potência atual de forma robusta (suporta template novo e arquivos legados)
                col_potencia_atual = next(
                    (c for c in df_entrada.columns if c in ['Potencia da lâmpada', 'Potencia Atual (W)', 'Potência Atual (W)']),
                    None
                )
                if col_potencia_atual:
                    for forn in FORNECEDORES:
                        pot_prev_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_prev_col in df_saida.columns:
                            df_saida[f'Economia (W) - {forn}'] = df_saida[col_potencia_atual] - df_saida[pot_prev_col]
                            df_saida[f'Reducao (%) - {forn}'] = (df_saida[f'Economia (W) - {forn}'] / df_saida[col_potencia_atual]) * 100
                
                # Validação NBR 5101 por linha (Status)
                if tem_classe:
                    for forn in FORNECEDORES:
                        status_list = []
                        for idx, row in df_saida.iterrows():
                            # Busca classificação de forma flexível
                            classe_val = row.get('Classificação viária', 'M3')
                            classe = str(classe_val).upper()
                            info_v = NBR5101.get(classe, {})
                            atende_linha = True
                            for m in info_v.get('metricas', []):
                                if m == 'w': continue
                                # Ignora métricas cujo modelo não é confiável (R² < 0.5)
                                if m not in metricas_confiaveis: continue
                                col_res = f'{TARGETS_MAP[m]} - {forn}'
                                if col_res in df_saida.columns:
                                    val = row[col_res]
                                    req = info_v.get(m)
                                    if pd.notna(val) and req is not None and val < req:
                                        atende_linha = False
                                        break
                            status_list.append('✔ Atende' if atende_linha else '✘ Não Atende')
                        df_saida[f'Status NBR - {forn}'] = status_list
                
                # Busca de Custo e Modelo no Banco de Dados para o Lote
                if not banco_luminarias.empty:
                    for forn in FORNECEDORES:
                        pot_prev_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_prev_col in df_saida.columns:
                            modelos_sug = []
                            pots_reais = []
                            custos_unit = []
                            for p_prev in df_saida[pot_prev_col]:
                                lum_n, p_real, v_unit = buscar_custo(banco_luminarias, forn, p_prev)
                                modelos_sug.append(lum_n)
                                pots_reais.append(p_real)
                                custos_unit.append(v_unit)
                            
                            df_saida[f'Modelo Sugerido - {forn}'] = modelos_sug
                            df_saida[f'Potencia Real (W) - {forn}'] = pots_reais
                            df_saida[f'Custo Unitario (R$) - {forn}'] = custos_unit
                
                # Formata para o template HOUER (Columns D-DM)
                df_export = formatar_resultado_template(df_saida)
                st.session_state.df_lote = df_saida
                st.session_state.df_export = df_export

                st.markdown('### ✨ Resultados (Preview)')
                st.dataframe(df_saida.head(10)) # Mostra preview compacto
                
                buffer_resultado = io.BytesIO()
                with pd.ExcelWriter(buffer_resultado, engine='openpyxl') as writer:
                    df_export.to_excel(writer, index=False)
                st.download_button('✅ Baixar Resultados no Template (.xlsx)', data=buffer_resultado.getvalue(), file_name='resultados_simulacao_HOUER.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type='primary')

        except Exception as e:
            st.error(f'Erro ao processar: {str(e)}')

# ==============================================================================
# TAB 3: DASHBOARD DE LOTE
# ==============================================================================
with tab_dash:
    if st.session_state.df_lote is None:
        st.info('👉 Realize uma simulação na aba "Simulação em Lote" para habilitar o Dashboard.')
    else:
        df = st.session_state.df_lote
        st.markdown('<p class="hero-title" style="font-size:2rem;">Dashboard Executivo</p>', unsafe_allow_html=True)
        
        # Filtro de fornecedor principal para o Dash
        forn_dash = st.selectbox('Selecione o Fornecedor para análise detalhada:', FORNECEDORES)
        
        st.divider()
        
        # ── KPIs Superiores
        m1, m2, m3, m4 = st.columns(4)
        
        col_custo = f'Custo Unitario (R$) - {forn_dash}'
        col_eco   = f'Economia (W) - {forn_dash}'
        col_red   = f'Reducao (%) - {forn_dash}'
        col_status = f'Status NBR - {forn_dash}'
        
        total_capex = df[col_custo].sum() if col_custo in df.columns else 0
        total_eco_kw = (df[col_eco].sum() / 1000) if col_eco in df.columns else 0
        media_red = df[col_red].mean() if col_red in df.columns else 0
        
        # % que atende
        if col_status in df.columns:
            atende_count = (df[col_status] == '✔ Atende').sum()
            perc_atende = (atende_count / len(df)) * 100
        else:
            perc_atende = 0

        with m1:
            st.metric("CAPEX Total Est.", f"R$ {total_capex:,.2f}")
        with m2:
            st.metric("Economia Total", f"{total_eco_kw:,.1f} kW")
        with m3:
            st.metric("Eficiência Média", f"{media_red:.1f}%")
        with m4:
            st.metric("Conformidade NBR", f"{perc_atende:.1f}%")

        st.markdown('---')
        
        # ── Gráficos
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown('### 💰 CAPEX Total por Fornecedor')
            custos_forn = {}
            for f in FORNECEDORES:
                c_col = f'Custo Unitario (R$) - {f}'
                if c_col in df.columns:
                    custos_forn[f] = df[c_col].sum()
            
            fig_custo = go.Figure(go.Bar(
                x=list(custos_forn.keys()),
                y=list(custos_forn.values()),
                marker_color=[CORES[f] for f in custos_forn.keys()],
                text=[f'R$ {v:,.0f}' for v in custos_forn.values()],
                textposition='auto'
            ))
            fig_custo.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_custo, use_container_width=True)

        with g2:
            st.markdown('### ✅ Conformidade por Fornecedor')
            conf_forn = {}
            for f in FORNECEDORES:
                s_col = f'Status NBR - {f}'
                if s_col in df.columns:
                    conf_forn[f] = (df[s_col] == '✔ Atende').mean() * 100
            
            fig_conf = go.Figure(go.Bar(
                x=list(conf_forn.keys()),
                y=list(conf_forn.values()),
                marker_color=[CORES[f] for f in conf_forn.keys()],
                text=[f'{v:.1f}%' for v in conf_forn.values()],
                textposition='auto'
            ))
            fig_conf.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis=dict(range=[0, 105]))
            st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown('---')
        st.markdown(f'### 🛣️ Análise por Classificação de Via ({forn_dash})')
        
        col_classe_dash = next((c for c in df.columns if c in ['Classificação viária', 'Classificacao (M/C/P)']), None)
        if col_classe_dash:
            df['Classe_Resumo'] = df[col_classe_dash].fillna('N/A').astype(str).str.upper()
            
            # Agrupa dados
            col_pot_real = f'Potencia Real (W) - {forn_dash}'
            
            if col_pot_real in df.columns:
                # Prepara o dicionário de agregação de forma dinâmica
                agg_dict = {col_pot_real: ['mean', 'count']}
                if col_red in df.columns:
                    agg_dict[col_red] = 'mean'
                
                analise_via = df.groupby('Classe_Resumo').agg(agg_dict).reset_index()
                
                # Ajusta nomes das colunas após o agg
                novas_cols = ['Classe', 'Potência Média (W)', 'Quantidade']
                if col_red in df.columns:
                    novas_cols.append('Economia Média (%)')
                analise_via.columns = novas_cols
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_via = go.Figure()
                    fig_via.add_trace(go.Bar(
                        x=analise_via['Classe'], y=analise_via['Potência Média (W)'],
                        name='Potência Média (W)', marker_color=CORES[forn_dash]
                    ))
                    fig_via.update_layout(title="Potência Média por Tipo de Via", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_via, use_container_width=True)
                
                with c2:
                    fig_pie = go.Figure(go.Pie(
                        labels=analise_via['Classe'], values=analise_via['Quantidade'],
                        hole=.4, marker=dict(colors=['#00A9E0', '#1B3664', '#64748b', '#334155'])
                    ))
                    fig_pie.update_layout(title="Mix de Vias", height=350, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_pie, use_container_width=True)

                formato = {'Potência Média (W)': '{:.1f} W'}
                if 'Economia Média (%)' in analise_via.columns:
                    formato['Economia Média (%)'] = '{:.1f} %'
                
                st.table(analise_via.style.format(formato))
            else:
                st.warning(f'Dados de potência para {forn_dash} não encontrados na simulação.')
        else:
            st.warning('Coluna "Classificação viária" não encontrada para análise por via.')

        # ── Botão de Exportação PDF de Lote (desabilitado temporariamente)
        # st.markdown("---")
        # pdf_lote_bytes = gerar_pdf_lote(df)
        # st.download_button(
        #     label="📥 Baixar Relatório Executivo de Lote (PDF)",
        #     data=pdf_lote_bytes,
        #     file_name=f"Relatorio_Executivo_Lote.pdf",
        #     mime="application/pdf",
        #     type="primary",
        #     use_container_width=True
        # )

# ── Info do modelo ─────────────────────────────────────────────────────────────
with st.expander('ℹ️ Métricas dos Modelos Treinados'):
    cols = st.columns(3)
    targets_disp = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
    for i, m in enumerate(targets_disp):
        with cols[i % 3]:
            st.markdown(f"**{TARGETS_MAP[m]}**")
            info = meta.get(f'modelo_{m}')
            if info:
                r2_val = info.get('r2')
                mae_val = info.get('mae')
                st.metric('R² (teste)', f"{r2_val:.4f}" if r2_val is not None else "—")
                st.caption(f"Erro Médio (MAE): {mae_val:.2f}" if mae_val is not None else "")
                st.caption(f"Algoritmo: {info.get('type','—')}")
            else:
                st.info("Modelo não treinado para este alvo.")
