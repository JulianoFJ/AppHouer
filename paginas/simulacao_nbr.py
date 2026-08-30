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

def get_google_maps_api_key():
    """Lê a chave da API via Streamlit Secrets ou variável de ambiente."""
    try:
        if "GOOGLE_MAPS_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_MAPS_API_KEY")

GOOGLE_MAPS_API_KEY = get_google_maps_api_key()

# ── Página renderizada via app.py (hub) ───────────────────────────────────────
# st.set_page_config é chamado pelo entry point `app.py`. Não duplicar aqui.

# Inicializa estado para Simulação em Lote
if 'df_lote' not in st.session_state:
    st.session_state.df_lote = None

# ── Estilo ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Paleta: Navy (#1B3664), Teal (#00A9E0) */
    :root {
        --marca-navy: #1B3664;
        --marca-teal: #00A9E0;
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
        background: linear-gradient(90deg, #ffffff, var(--marca-teal));
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
        border-top: 4px solid var(--marca-teal);
        border-radius: 20px;
        padding: 1.8rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .forn-card:hover {
        transform: translateY(-8px);
        border-color: var(--marca-teal);
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
        background: var(--marca-teal);
        border-radius: 3px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--marca-navy), #0b111e);
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ── Caminhos ──────────────────────────────────────────────────────────────────
# Esta página vive em `paginas/`, mas os .pkl / .csv / .json originais estão
# na raiz do projeto (junto com app.py). Por isso subimos um nível.
PASTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# Projeção física de cada classificação de braço (metros) — tabela padrão interna
BRACOS_PROJECAO = {
    'Curto I':  1.2,
    'Curto II': 1.4,
    'Médio I':  1.8,
    'Médio II': 2.4,
    'Longo I':  2.8,
    'Longo II': 3.5,
}
BRACOS_ORDENADOS = sorted(BRACOS_PROJECAO.items(), key=lambda x: x[1])

def projecao_para_braco(proj_m: float) -> str:
    """Retorna a classificação de braço mais próxima da projeção em metros."""
    return min(BRACOS_PROJECAO, key=lambda b: abs(BRACOS_PROJECAO[b] - float(proj_m)))

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

@st.cache_resource
def carregar_classificadores(suffix=""):
    """Carrega modelo_cpe e modelo_braco se existirem."""
    clf_cpe   = None
    clf_braco = None
    path_cpe   = os.path.join(PASTA, f'modelo_cpe{suffix}.pkl')
    path_braco = os.path.join(PASTA, f'modelo_braco{suffix}.pkl')
    if os.path.exists(path_cpe):
        clf_cpe = joblib.load(path_cpe)
    if os.path.exists(path_braco):
        clf_braco = joblib.load(path_braco)
    return clf_cpe, clf_braco

def prever_metricas_com_dependencia_w(df_base: pd.DataFrame, modelos: dict, metricas: list, meta: dict):
    """Prevê métricas respeitando dependência de W (emed/emin treinados com coluna de potência)."""
    preds = {}
    w_col = meta.get('feature_w_col', 'Potencia simulada - IP Principal (W)')
    dependem_w = set(meta.get('modelos_dependem_de_w', []))

    # 1) Prevê W primeiro quando necessário
    if 'w' in metricas and 'w' in modelos:
        preds_w = modelos['w'].predict(df_base)
        preds['w'] = np.maximum(preds_w, 0)
    elif 'w' in metricas:
        preds['w'] = np.array([np.nan] * len(df_base))

    # 2) Prevê demais métricas
    for m in metricas:
        if m == 'w':
            continue
        if m not in modelos:
            preds[m] = np.array([np.nan] * len(df_base))
            continue
        try:
            if m in dependem_w:
                df_m = df_base.copy()
                if w_col not in df_m.columns:
                    df_m[w_col] = preds.get('w', np.array([np.nan] * len(df_base)))
                p = modelos[m].predict(df_m)
            else:
                p = modelos[m].predict(df_base)
            preds[m] = np.maximum(p, 0)
        except Exception:
            preds[m] = np.array([np.nan] * len(df_base))

    return preds

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

@st.cache_data
def carregar_media_historica():
    """Lê o dataset.csv original e calcula a média de potência por classe para comparação."""
    caminho = os.path.join(PASTA, 'dataset.csv')
    if not os.path.exists(caminho):
        return pd.DataFrame(columns=['Classe_Resumo', 'Média Histórica (W)'])
    try:
        df_hist = pd.read_csv(caminho)
        # Tenta achar colunas
        col_classe = next((c for c in df_hist.columns if 'classifica' in c.lower() and 'vi' in c.lower()), None)
        col_pot = next((c for c in df_hist.columns if 'potencia' in c.lower() and '(w)' in c.lower()), None)
        if not col_classe or not col_pot:
            return pd.DataFrame(columns=['Classe_Resumo', 'Média Histórica (W)'])
        
        df_hist['Classe_Resumo'] = df_hist[col_classe].fillna('N/A').astype(str).str.upper()
        df_hist[col_pot] = pd.to_numeric(df_hist[col_pot], errors='coerce')
        medias = df_hist.groupby('Classe_Resumo')[col_pot].mean().reset_index()
        medias.rename(columns={col_pot: 'Média Histórica (W)'}, inplace=True)
        return medias
    except Exception:
        return pd.DataFrame(columns=['Classe_Resumo', 'Média Histórica (W)'])

medias_historicas = carregar_media_historica()

@st.cache_data
def carregar_regras_braco_novo():
    """Aprende padrão de Braço Novo com base histórica (classe, largura e altura)."""
    caminho = os.path.join(PASTA, 'dataset.csv')
    vazio = pd.DataFrame(columns=['Classe_Prefixo', 'Largura_R', 'Altura_R', 'Dist_R', 'Braco_Recomendado'])
    vazio_classe = pd.DataFrame(columns=['Classe_Prefixo', 'Braco_Recomendado'])
    if not os.path.exists(caminho):
        return vazio, vazio_classe
    try:
        df = pd.read_csv(caminho)
        cols_req = ['Classificação viária', 'Largura Via 1', 'altura da luminaria', 'distancia entre postes', 'Braço Novo']
        if any(c not in df.columns for c in cols_req):
            return vazio, vazio_classe
        d = df[cols_req].copy()
        d = d.dropna(subset=['Classificação viária', 'Largura Via 1', 'altura da luminaria', 'Braço Novo'])
        if d.empty:
            return vazio, vazio_classe
        d['Classe_Prefixo'] = d['Classificação viária'].astype(str).str.upper().str[:1]
        d = d[d['Classe_Prefixo'].isin(['M', 'C', 'P'])]
        d['Largura_R'] = pd.to_numeric(d['Largura Via 1'], errors='coerce').round(1)
        d['Altura_R'] = pd.to_numeric(d['altura da luminaria'], errors='coerce').round(1)
        d['Dist_R'] = pd.to_numeric(d['distancia entre postes'], errors='coerce').round(1)
        d = d.dropna(subset=['Largura_R', 'Altura_R', 'Dist_R'])
        if d.empty:
            return vazio, vazio_classe
        agg = (d.groupby(['Classe_Prefixo', 'Largura_R', 'Altura_R', 'Dist_R', 'Braço Novo'])
                 .size().reset_index(name='n')
                 .sort_values('n', ascending=False))
        regras_finas = (agg.drop_duplicates(['Classe_Prefixo', 'Largura_R', 'Altura_R', 'Dist_R'])
                          .rename(columns={'Braço Novo': 'Braco_Recomendado'})
                          [['Classe_Prefixo', 'Largura_R', 'Altura_R', 'Dist_R', 'Braco_Recomendado']])
        agg_classe = (d.groupby(['Classe_Prefixo', 'Braço Novo'])
                        .size().reset_index(name='n')
                        .sort_values('n', ascending=False))
        regras_classe = (agg_classe.drop_duplicates(['Classe_Prefixo'])
                           .rename(columns={'Braço Novo': 'Braco_Recomendado'})
                           [['Classe_Prefixo', 'Braco_Recomendado']])
        return regras_finas, regras_classe
    except Exception:
        return vazio, vazio_classe

def sugerir_braco_novo(subclasse, largura_via1, altura_lum, dist_postes,
                       regras_finas, regras_classe, clf_braco=None,
                       tipo_estrutura='Braço', posteacao='Unilateral',
                       dist_poste_via=0.5, faixas=2, largura_via2=0.0,
                       passeio1=0.0, passeio2=0.0, canteiro=0.0):
    """Sugere Braço Novo: ML (se disponível) → vizinhança histórica → fallback por classe."""

    # 1) Classificador ML
    if clf_braco is not None:
        try:
            row = pd.DataFrame([{
                'Faixas de Rodagem':      faixas,
                'Largura Via 1':          largura_via1,
                'Largura Via 2':          largura_via2,
                'Largura Passeio 1':      passeio1,
                'largura Passeio 2':      passeio2,
                'largura Canteiro Central': canteiro,
                'altura da luminaria':    altura_lum,
                'projecao do braço':      1.5,   # valor neutro — não afeta classificação
                'distancia entre postes': dist_postes,
                'distancia Poste a via':  dist_poste_via,
                'Classificação viária':   subclasse,
                'Tipo de estrutura':      tipo_estrutura,
                'posteacao':              posteacao,
                'Fornecedor':             'LEDSTAR',  # neutro — modelo generaliza
            }])
            return clf_braco.predict(row)[0], 'ml'
        except Exception:
            pass

    # 2) Vizinhança histórica
    if not regras_finas.empty:
        pref = str(subclasse).upper()[:1]
        larg = round(float(largura_via1), 1)
        alt  = round(float(altura_lum), 1)
        dist = round(float(dist_postes), 1)
        cand = regras_finas[regras_finas['Classe_Prefixo'] == pref].copy()
        if not cand.empty:
            cand['dist'] = (
                (cand['Largura_R'] - larg).abs()
                + (cand['Altura_R'] - alt).abs()
                + 0.25 * (cand['Dist_R'] - dist).abs()
            )
            best = cand.sort_values('dist').iloc[0]
            if best['dist'] <= 2.5:
                return best['Braco_Recomendado'], 'historico'

    # 3) Fallback por classe
    if not regras_classe.empty:
        pref = str(subclasse).upper()[:1]
        cand_c = regras_classe[regras_classe['Classe_Prefixo'] == pref]
        if not cand_c.empty:
            return cand_c.iloc[0]['Braco_Recomendado'], 'historico'

    return None, None

regras_braco_finas, regras_braco_classe = carregar_regras_braco_novo()


# ── Header ────────────────────────────────────────────────────────────────────
# Sem logotipo: a identidade do portal é textual e vive na sidebar do `app.py`, para
# que exista um único lugar a mexer se o nome mudar de novo.
#
# Estilo inline, e não as classes `hero-title`/`hero-sub`: aquelas foram calibradas
# para uma palavra curta (o antigo logotipo textual) e usam margem negativa para colar
# o subtítulo — com um título longo as duas linhas se sobrepõem.
st.markdown(
    """
    <div style="margin: 0.4rem 0 0.2rem 0;">
      <div style="font-size:2.6rem; font-weight:800; line-height:1.15;
                  background: linear-gradient(90deg,#ffffff,#00A9E0);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  letter-spacing:-1px;">Simulação NBR 5101</div>
      <div style="font-size:1.02rem; color:#94a3b8; margin-top:0.35rem;">
        Dimensionamento luminotécnico assistido por aprendizado de máquina
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
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
    clf_cpe, clf_braco = carregar_classificadores(sufixo_modelo)
    num_ok = meta.get('features_numericas', [])
    cat_ok = meta.get('features_categoricas', [])
    feature_w_col = meta.get('feature_w_col', 'Potencia simulada - IP Principal (W)')

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
    st.caption("ℹ️ Informacional — não integra o modelo preditivo atual.")

    st.markdown('### 🔩 Configurações')
    tipo_estrutura = st.selectbox('Tipo de Estrutura', ['Braço', 'Suporte'])
    posteacao      = st.selectbox('Posteação', ['Unilateral', 'Canteiro central', 'Bilateral alternada', 'Bilateral frontal'])
    # Braço atual derivado automaticamente da projeção informada
    braco_novo = projecao_para_braco(projecao_braco)
    st.caption(f"📏 Braço atual identificado: **{braco_novo}** ({projecao_braco:.2f}m) — sugestão de troca gerada pelo modelo ML de geometria histórica.")

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
            if not GOOGLE_MAPS_API_KEY:
                st.warning("Defina `GOOGLE_MAPS_API_KEY` em `st.secrets` (deploy) ou variável de ambiente (local).")
            else:
                st.info("Insira um endereço para localizar.")

    st.markdown('### ⚡ Eficientização')
    potencia_atual = st.number_input('Potência Atual (W)', min_value=0.0, value=250.0, step=10.0, help="Potência da luminária instalada atualmente (ex: Sódio 250W, 400W)")

def gerar_pdf(fornecedores, resultados, info_nbr, inputs, banco_luminarias, sugestoes, endereco="Não informado"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Cabeçalho
    pdf.set_text_color(27, 54, 100) # Azul da paleta
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
        'Largura da Via 1', 'projecao do braço', 'Posteação', 'Potencia Atual (W)'
    ]
    df_temp = pd.DataFrame(columns=cols)
    # Linha de exemplo
    df_temp.loc[0] = [1, 'M3', 10.0, 35.0, 7.0, 1.5, 'Unilateral', 250.0]
    
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
            
    # 2. Tentar braços discretos em ordem crescente de projeção
    proj_atual = config_base.get('projecao do braço', 1.8)
    for arm_name, arm_proj in BRACOS_ORDENADOS:
        if arm_proj <= proj_atual:
            continue  # só testa braços maiores que o atual
        c = config_base.copy()
        c['Braço Novo'] = arm_name
        c['projecao do braço'] = arm_proj
        if 'projecao do brao' in c: c['projecao do brao'] = arm_proj
        if verifica_atende(c):
            sugestoes.append(f"🏗️ **Ajuste de Braço**: Trocar para **{arm_name}** (projeção {arm_proj:.1f}m)")
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


def formatar_tabela_resultado(df_saida):
    """
    Formata o resultado no estilo da aba 'tabela dinamica' do BRDE04:
    uma linha por (ponto × fornecedor), com as colunas operacionais principais.
    """
    col_pot_w = TARGETS_MAP['w']   # 'Potência (W)'
    rows = []
    for _, row in df_saida.iterrows():
        for forn in FORNECEDORES:
            rows.append({
                'ID':                                    row.get('ID', ''),
                'Logradouro':                            row.get('Logradouro', ''),
                'latitude':                              row.get('latitude', np.nan),
                'longitude':                             row.get('longitude', np.nan),
                'Classificação viária':                  row.get('Classificação viária', ''),
                'Potência Atual (W)':                    row.get('Potencia da lâmpada', np.nan),
                'Fornecedor':                            forn,
                'Código de Luminária':                   row.get(f'Modelo Sugerido - {forn}', ''),
                'Potência Proposta (W)':                 row.get(f'{col_pot_w} - {forn}', np.nan),
                'Braço Antigo':                          row.get('Braço Antigo', row.get('Braço Atual', '')),
                'Braço Novo':                            row.get('Sugestão Braço Novo', ''),
                'Tipo de CPE':                           row.get('Correção de Ponto Escuro (CPE)', 'Não'),
                'Observação CPE':                        row.get('Observação CPE  (Reduçao entre postes e/ou Tipo de Posteação)', ''),
                'Qtd. Pontos IP Veic':                   row.get('Quantidade de pontos inspecionados IP Veic', np.nan),
                'Qtd. Pontos IP Sec':                    row.get('Quantidade de pontos inspecionados IP Sec', np.nan),
                'Aumento de Pontos Proposto (Veículos)': row.get('Quantidade de pontos adicionados para via de veículo', 0),
                'Status NBR':                            row.get(f'Status NBR - {forn}', ''),
                'Economia (W)':                          row.get(f'Economia (W) - {forn}', np.nan),
                'Redução (%)':                           row.get(f'Reducao (%) - {forn}', np.nan),
            })
    return pd.DataFrame(rows)


def formatar_tabela_dinamica(df_saida):
    """
    Gera aba 'Tabela Dinâmica' no estilo do BRDE04:
    agrupa por (Potência Atual, Classe de Iluminação, Fornecedor, Código da Luminária,
    Potência Proposta, Braço Antigo, Braço Novo, Tipo de CPE,
    Qtd IP Veic, Qtd IP Sec, Aumento Veículos, Aumento 2º Nível)
    e conta o número de logradouros por combinação.
    """
    col_pot_w = TARGETS_MAP['w']
    rows = []
    for _, row in df_saida.iterrows():
        # pd.isna necessário pois np.nan é truthy — 'or' simples não funciona com NaN
        _ba  = row.get('Braço Antigo')
        _ba2 = row.get('Braço Atual')
        braco_antigo = (str(_ba).strip()  if pd.notna(_ba)  and str(_ba).strip()  else
                        str(_ba2).strip() if pd.notna(_ba2) and str(_ba2).strip() else '')
        _bn = row.get('Sugestão Braço Novo')
        braco_novo_raw = str(_bn).strip() if pd.notna(_bn) else ''
        braco_novo = braco_novo_raw if braco_novo_raw and braco_novo_raw != braco_antigo else ''

        aum_veic = row.get('Quantidade de pontos adicionados para via de veículo', '')
        aum_sec  = row.get('Quantidade de pontos adicionados para via de pedestres', '')
        aum_veic = '' if pd.isna(aum_veic) or aum_veic == 0 else aum_veic
        aum_sec  = '' if pd.isna(aum_sec)  or aum_sec  == 0 else aum_sec

        cpe = str(row.get('Correção de Ponto Escuro (CPE)', '') or '')
        cpe = '' if cpe.lower() in ('não', 'nao', 'no', '', 'nan', 'none') else cpe

        pot_atual_raw = row.get('Potencia da lâmpada', np.nan)
        pot_atual = int(round(pot_atual_raw)) if pd.notna(pot_atual_raw) and pot_atual_raw >= 1 else ''

        qtd_veic = row.get('Quantidade de pontos inspecionados IP Veic', 0)
        qtd_sec  = row.get('Quantidade de pontos inspecionados IP Sec', 0)
        qtd_veic = 0 if pd.isna(qtd_veic) else int(qtd_veic)
        qtd_sec  = 0 if pd.isna(qtd_sec)  else int(qtd_sec)

        for forn in FORNECEDORES:
            pot_prop_raw = row.get(f'{col_pot_w} - {forn}', np.nan)
            pot_prop = int(round(pot_prop_raw)) if pd.notna(pot_prop_raw) else ''
            codigo = str(row.get(f'Modelo Sugerido - {forn}', '') or '').strip()
            rows.append({
                'Potência Atual':                                pot_atual,
                'Classe de Iluminação':                          row.get('Classificação viária', ''),
                'Fornecedor':                                    forn,
                'Código da Luminária':                           codigo,
                'Potência Proposta':                             pot_prop,
                'Braço Antigo':                                  braco_antigo,
                'Braço Novo':                                    braco_novo,
                'Tipo de CPE':                                   cpe,
                'Quantidade de pontos inspecionados IP Veic':    qtd_veic,
                'Quantidade de pontos inspecionados IP Sec':     qtd_sec,
                'Aumento de pontos proposto (via de veículos)':  aum_veic,
                'Aumento de pontos proposto (segundo nível)':    aum_sec,
            })

    df_flat = pd.DataFrame(rows)
    group_cols = [
        'Potência Atual', 'Classe de Iluminação', 'Fornecedor', 'Código da Luminária',
        'Potência Proposta', 'Braço Antigo', 'Braço Novo', 'Tipo de CPE',
        'Quantidade de pontos inspecionados IP Veic',
        'Quantidade de pontos inspecionados IP Sec',
        'Aumento de pontos proposto (via de veículos)',
        'Aumento de pontos proposto (segundo nível)',
    ]
    df_flat[group_cols] = df_flat[group_cols].fillna('').astype(str)
    df_grouped = (
        df_flat.groupby(group_cols, sort=True)
               .size()
               .reset_index(name='Sum of Número de logradouros')
    )
    # Converte de volta colunas numéricas que foram stringificadas
    for col in ['Potência Atual', 'Potência Proposta',
                'Quantidade de pontos inspecionados IP Veic',
                'Quantidade de pontos inspecionados IP Sec']:
        df_grouped[col] = pd.to_numeric(df_grouped[col], errors='coerce')
    return df_grouped


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

    def montar_entrada(fornecedor, dist_override=None):
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
            'distancia entre postes':   dist_override if dist_override is not None else dist_postes,
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
        colunas = list(dict.fromkeys(num_ok + cat_ok + [feature_w_col]))
        return pd.DataFrame([{k: dados.get(k, np.nan) for k in colunas}])

    # Roda as predições para todos os fornecedores
    resultados = {m: {} for m in metricas_ativas}
    config_dicts = {}

    for forn in FORNECEDORES:
        X_in = montar_entrada(forn)
        config_dicts[forn] = X_in.iloc[0].to_dict()
        preds_in = prever_metricas_com_dependencia_w(X_in, modelos, metricas_ativas, meta)
        for m in metricas_ativas:
            v = preds_in.get(m, np.array([np.nan]))[0]
            resultados[m][forn] = None if pd.isna(v) else float(v)

    # ── Ajuste de Potência pela Hierarquia NBR ───────────────────────────────
    # Classes M: o modelo lmed tem R²=0.28 e superestima luminância, então
    # o gatilho "pred < req" nunca dispara. Usa-se fator proporcional direto
    # em relação ao M3 (baseline), garantindo M1 > M3 > M6 em potência.
    # Classes C e P: o modelo emed é confiável (R²>0.8) — escalonamento
    # condicional apenas quando pred_ilum < req (mantém comportamento atual).
    fator_m = None
    metric_ref = 'emed'
    req_ilum_ref = None

    if 'w' in metricas_ativas:
        if subclasse.startswith('M'):
            req_m3_base = NBR5101['M3']['lmed']   # 1.0 cd/m²
            req_classe  = info_nbr.get('lmed', req_m3_base)
            fator_m     = req_classe / req_m3_base
            metric_ref  = 'lmed'
            for forn in FORNECEDORES:
                if resultados['w'].get(forn) is not None:
                    resultados['w'][forn] *= fator_m
        else:
            metric_ref   = 'emed'
            req_ilum_ref = info_nbr.get(metric_ref)
            if metric_ref in metricas_ativas and req_ilum_ref:
                for forn in FORNECEDORES:
                    pred_ilum = resultados[metric_ref].get(forn)
                    pred_pot  = resultados['w'].get(forn)
                    if (pred_ilum is not None and pred_ilum > 0
                            and pred_pot is not None and pred_ilum < req_ilum_ref):
                        resultados['w'][forn] = pred_pot * (req_ilum_ref / pred_ilum)

    # Verifica conformidade e gera sugestões estruturais
    sugestoes_por_forn = {}
    for forn in FORNECEDORES:
        falhou = any(
            resultados[m].get(forn) is not None
            and info_nbr.get(m) is not None
            and resultados[m][forn] < info_nbr[m]
            for m in metricas_ativas
            if m != 'w' and m in metricas_confiaveis
        )
        if falhou:
            sugestoes_por_forn[forn] = analisar_melhorias(
                forn, modelos, metricas_ativas, info_nbr, config_dicts[forn], num_ok, cat_ok)

    # ── Sugestão ML de braço (individual) ────────────────────────────────────
    BRACO_ML_CONF_MIN_IND = 0.50  # só exibe sugestão se confiança >= 50%
    braco_ml_sugestao = None
    braco_ml_prob = None
    if clf_braco is not None:
        try:
            row_braco_ind = pd.DataFrame([{
                'Faixas de Rodagem':        faixas,
                'Largura Via 1':            largura_via1,
                'Largura Via 2':            largura_via2,
                'Largura Passeio 1':        largura_passeio1,
                'largura Passeio 2':        largura_passeio2,
                'largura Canteiro Central': canteiro,
                'altura da luminaria':      altura_lum,
                'projecao do braço':        projecao_braco,
                'distancia entre postes':   dist_postes,
                'distancia Poste a via':    dist_poste_via,
                'Classificação viária':     subclasse,
                'Tipo de estrutura':        tipo_estrutura,
                'posteacao':                posteacao,
                'Fornecedor':               'LEDSTAR',
            }])
            _pred = clf_braco.predict(row_braco_ind)[0]
            _probs = clf_braco.predict_proba(row_braco_ind)[0]
            _prob = float(_probs[list(clf_braco.classes_).index(_pred)])
            if _prob >= BRACO_ML_CONF_MIN_IND:
                braco_ml_sugestao = _pred
                braco_ml_prob = _prob
        except Exception:
            pass

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
    
    # ── Sugestão ML de Braço ─────────────────────────────────────────────────
    if braco_ml_sugestao is not None:
        proj_ml = BRACOS_PROJECAO.get(braco_ml_sugestao, None)
        proj_atual_str = f"{projecao_braco:.2f}m"
        if str(braco_ml_sugestao) != str(braco_novo):
            cor_badge = '#f59e0b'
            icone = '🔄'
            msg = f"O modelo ML recomenda **{braco_ml_sugestao}** ({proj_ml:.1f}m) em vez do atual **{braco_novo}** ({proj_atual_str}) — baseado no padrão histórico desta geometria."
        else:
            cor_badge = '#22c55e'
            icone = '✔'
            msg = f"O modelo ML confirma o braço atual **{braco_novo}** ({proj_atual_str}) como adequado para esta geometria."
        conf_str = f"{braco_ml_prob*100:.0f}%" if braco_ml_prob is not None else "—"
        st.markdown(
            f'<div style="background:rgba(18,25,43,0.7); border:1px solid #1f2937; border-left:4px solid {cor_badge}; '
            f'border-radius:12px; padding:14px 18px; margin:12px 0;">'
            f'<span style="font-size:0.8rem; font-weight:700; color:{cor_badge}; text-transform:uppercase; letter-spacing:.05em;">'
            f'{icone} Braço — Recomendação ML</span>'
            f'<div style="font-size:0.9rem; color:#e2e8f0; margin-top:6px;">{msg}</div>'
            f'<div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Confiança do modelo: {conf_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Detecção Proativa de Ponto Escuro (CPE) ──────────────────────────────
    DIST_CPE_MIN = 45.0
    POT_DESVIO_FATOR = 1.40

    pot_previstas = [resultados['w'].get(f) for f in FORNECEDORES if 'w' in resultados and resultados['w'].get(f) is not None]
    pot_media_prev = np.mean(pot_previstas) if pot_previstas else None

    row_hist = medias_historicas[medias_historicas['Classe_Resumo'] == subclasse.upper()] if not medias_historicas.empty else pd.DataFrame()
    media_hist_w = row_hist['Média Histórica (W)'].iloc[0] if not row_hist.empty else None

    cpe_por_distancia = dist_postes >= DIST_CPE_MIN
    cpe_por_desvio = (
        media_hist_w is not None and pot_media_prev is not None
        and pot_media_prev > media_hist_w * POT_DESVIO_FATOR
    )
    cpe_acionado = cpe_por_distancia and (cpe_por_desvio or media_hist_w is None)

    # Classificador ML de CPE — sinal adicional ao conjunto de regras
    cpe_ml_pred = None
    cpe_ml_prob = None
    if clf_cpe is not None:
        try:
            row_cpe = pd.DataFrame([{
                'Faixas de Rodagem':        faixas,
                'Largura Via 1':            largura_via1,
                'Largura Via 2':            largura_via2,
                'Largura Passeio 1':        largura_passeio1,
                'largura Passeio 2':        largura_passeio2,
                'largura Canteiro Central': canteiro,
                'altura da luminaria':      altura_lum,
                'projecao do braço':        projecao_braco,
                'distancia entre postes':   dist_postes,
                'distancia Poste a via':    dist_poste_via,
                'Classificação viária':     subclasse,
                'Tipo de estrutura':        tipo_estrutura,
                'posteacao':                posteacao,
                'Fornecedor':               'LEDSTAR',
            }])
            cpe_ml_pred = int(clf_cpe.predict(row_cpe)[0])
            cpe_ml_prob = float(clf_cpe.predict_proba(row_cpe)[0][1])
        except Exception:
            pass

    # CPE acionado se as regras OR o modelo ML apontarem risco
    cpe_acionado = cpe_acionado or (cpe_ml_pred == 1)

    if 'w' in metricas_ativas and cpe_acionado:
        dist_cpe = dist_postes / 2
        resultados_cpe = {m: {} for m in metricas_ativas}

        for forn in FORNECEDORES:
            X_cpe = montar_entrada(forn, dist_override=dist_cpe)
            preds_cpe = prever_metricas_com_dependencia_w(X_cpe, modelos, metricas_ativas, meta)
            for m in metricas_ativas:
                v = preds_cpe.get(m, np.array([np.nan]))[0]
                resultados_cpe[m][forn] = None if pd.isna(v) else float(v)

        # Aplica ajuste NBR no cenário CPE — mesma lógica do cenário base
        if subclasse.startswith('M') and fator_m is not None:
            for forn in FORNECEDORES:
                if resultados_cpe['w'].get(forn) is not None:
                    resultados_cpe['w'][forn] *= fator_m
        elif metric_ref in metricas_ativas and req_ilum_ref:
            for forn in FORNECEDORES:
                p_ilum = resultados_cpe[metric_ref].get(forn)
                p_pot  = resultados_cpe['w'].get(forn)
                if (p_ilum is not None and p_ilum > 0
                        and p_pot is not None and p_ilum < req_ilum_ref):
                    resultados_cpe['w'][forn] = p_pot * (req_ilum_ref / p_ilum)

        motivo_parts = []
        if cpe_por_distancia:
            motivo_parts.append(f"distância de **{dist_postes:.0f}m** entre postes (limite recomendado: {DIST_CPE_MIN:.0f}m)")
        if cpe_por_desvio:
            motivo_parts.append(f"potência prevista **{pot_media_prev:.0f}W** acima da média histórica **{media_hist_w:.0f}W** (+{((pot_media_prev/media_hist_w)-1)*100:.0f}%)")
        if media_hist_w is None and not cpe_por_distancia:
            motivo_parts.append("classe sem média histórica disponível para validação de desvio")
        if cpe_ml_pred == 1 and not cpe_por_distancia and not cpe_por_desvio:
            motivo_parts.append(f"modelo ML detectou padrão geométrico de risco (probabilidade {cpe_ml_prob*100:.0f}%)")

        sinal_ml = ""
        if cpe_ml_prob is not None:
            cor_ml = '#22c55e' if cpe_ml_pred == 0 else '#f59e0b'
            sinal_ml = (
                f'\n\n🤖 **Classificador ML:** probabilidade de CPE = **{cpe_ml_prob*100:.0f}%** '
                f'<span style="color:{cor_ml};">{"⚠ Risco" if cpe_ml_pred == 1 else "✔ Baixo risco"}</span>'
            )

        st.markdown('<p class="section-title">⚠️ Correção de Ponto Escuro (CPE)</p>', unsafe_allow_html=True)
        st.warning(
            "**Risco de ponto escuro detectado** — " + " e ".join(motivo_parts) + ".\n\n"
            f"Cenário proposto: inserção de estrutura intermediária reduz a distância de **{dist_postes:.0f}m → {dist_cpe:.0f}m**."
            + sinal_ml
        )

        cpe_cols = st.columns(3)
        for i, forn in enumerate(FORNECEDORES):
            cor = CORES[forn]
            with cpe_cols[i]:
                cpe_html = f'<div class="forn-card" style="border-top-color:#f59e0b;">'
                cpe_html += f'<div class="forn-name" style="color:#f59e0b;">CPE — {forn}</div>'
                cpe_html += f'<div style="font-size:0.75rem;color:#94a3b8;margin-bottom:10px;">Distância: {dist_postes:.0f}m → <b style="color:#f59e0b;">{dist_cpe:.0f}m</b></div>'

                for m in metricas_ativas:
                    val_orig = resultados[m].get(forn)
                    val_cpe  = resultados_cpe[m].get(forn)
                    if val_cpe is None:
                        continue
                    delta = val_cpe - val_orig if val_orig is not None else None
                    if delta is not None:
                        delta_color = '#22c55e' if delta >= 0 else '#ef4444'
                        delta_str = f' <span style="color:{delta_color};font-size:0.7rem;">({("+" if delta >= 0 else "")}{delta:.2f})</span>'
                    else:
                        delta_str = ''
                    cpe_html += f'<div class="metric-label"><span>{TARGETS_MAP[m]}</span></div>'
                    cpe_html += f'<div class="metric-value">{val_cpe:,.2f} <span class="metric-unit">{UNITS_MAP[m]}</span>{delta_str}</div>'

                cpe_html += '</div>'
                st.markdown(cpe_html, unsafe_allow_html=True)

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
        'distancia Poste a via', 'projecao do braço', 'Altura de Instalação',
        'Braço Antigo',                    # referência — tipo de braço atual instalado
        'Quantidade de pontos inspecionados IP Veic',
        'Quantidade de pontos inspecionados IP Sec',
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
        'Braço Antigo': 'Médio II',
        'Quantidade de pontos inspecionados IP Veic': 2,
        'Quantidade de pontos inspecionados IP Sec': 0,
    }
    df_template = pd.concat([df_template, pd.DataFrame([exemplo])], ignore_index=True)
    
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

                metricas_lote = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']

                # ── Sugestão de Braço via Classificador ML ───────────────────────
                # Opção A: clf_braco aprende o padrão histórico de qual braço é
                # usado para cada geometria, independentemente da conformidade NBR.
                # Isso resolve o bloqueio das classes M (lmed R²<0.5) e o efeito
                # quase nulo da projeção sobre emed nas classes C/P.
                # Fallback: conformidade NBR (preservado para quando clf_braco=None).

                # 1. Identifica braço atual por projeção
                proj_series = pd.to_numeric(
                    df_pipeline.get('projecao do braço', pd.Series([1.8]*len(df_pipeline))),
                    errors='coerce'
                ).fillna(1.8).values
                bracos_atuais = [projecao_para_braco(p) for p in proj_series]

                classes_s = (
                    df_pipeline['Classificação viária'].values
                    if 'Classificação viária' in df_pipeline.columns
                    else [''] * len(df_pipeline)
                )

                bracos_pred = []

                BRACO_ML_CONF_MIN = 0.50  # só sugere troca se a confiança do modelo for >= 50%

                if clf_braco is not None:
                    # 2a. Predição vetorizada pelo classificador ML de braço
                    rows_ml = []
                    for i in range(len(df_pipeline)):
                        r = df_pipeline.iloc[i]
                        rows_ml.append({
                            'Faixas de Rodagem':        r.get('Faixas de Rodagem', 2),
                            'Largura Via 1':            r.get('Largura Via 1', 7),
                            'Largura Via 2':            r.get('Largura Via 2', 0),
                            'Largura Passeio 1':        r.get('Largura Passeio 1', 2),
                            'largura Passeio 2':        r.get('largura Passeio 2', 2),
                            'largura Canteiro Central': r.get('largura Canteiro Central', 0),
                            'altura da luminaria':      r.get('altura da luminaria', 9),
                            'projecao do braço':        r.get('projecao do braço', 1.5),
                            'distancia entre postes':   r.get('distancia entre postes', 35),
                            'distancia Poste a via':    r.get('distancia Poste a via', 0.5),
                            'Classificação viária':     r.get('Classificação viária', 'M3'),
                            'Tipo de estrutura':        r.get('Tipo de estrutura', 'Braço'),
                            'posteacao':                r.get('posteacao', 'Unilateral'),
                            'Fornecedor':               'LEDSTAR',
                        })
                    df_ml_input = pd.DataFrame(rows_ml)
                    try:
                        ml_preds = clf_braco.predict(df_ml_input)
                        ml_probs = clf_braco.predict_proba(df_ml_input).max(axis=1)
                        for ml_pred, ml_prob, atual in zip(ml_preds, ml_probs, bracos_atuais):
                            if str(ml_pred) != str(atual) and ml_prob >= BRACO_ML_CONF_MIN:
                                bracos_pred.append(ml_pred)
                            else:
                                bracos_pred.append(None)
                    except Exception:
                        bracos_pred = [None] * len(df_pipeline)

                else:
                    # 2b. Fallback: conformidade NBR (funciona apenas para C/P com emed confiável)
                    arm_preds_lote = {}
                    for arm_name, arm_proj in BRACOS_ORDENADOS:
                        df_arm = df_pipeline.copy()
                        df_arm['Braço Novo'] = arm_name
                        df_arm['projecao do braço'] = arm_proj
                        arm_preds_lote[arm_name] = {}
                        for forn in FORNECEDORES:
                            df_ra = df_arm.copy()
                            df_ra['Fornecedor'] = forn
                            for col in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col])):
                                if col not in df_ra.columns:
                                    df_ra[col] = np.nan
                            arm_preds_lote[arm_name][forn] = prever_metricas_com_dependencia_w(
                                df_ra, modelos, metricas_lote, meta
                            )

                    def arm_atende_linha(arm_name, i, classe_i):
                        info_i = NBR5101.get(str(classe_i).upper(), {})
                        for m in info_i.get('metricas', []):
                            req = info_i.get(m)
                            if m == 'w' or req is None or m not in metricas_confiaveis or m not in modelos:
                                continue
                            for forn in FORNECEDORES:
                                arr = arm_preds_lote.get(arm_name, {}).get(forn, {}).get(m, np.array([]))
                                val = float(arr[i]) if i < len(arr) else np.nan
                                if pd.notna(val) and val < req:
                                    return False
                        return True

                    for i in range(len(df_pipeline)):
                        cl_i   = classes_s[i]
                        atual  = bracos_atuais[i]
                        info_i = NBR5101.get(str(cl_i).upper(), {})
                        tem_req = any(
                            info_i.get(m) is not None and m != 'w'
                            and m in metricas_confiaveis and m in modelos
                            for m in info_i.get('metricas', [])
                        )
                        if not tem_req or arm_atende_linha(atual, i, cl_i):
                            bracos_pred.append(None)
                        else:
                            rec = None
                            for arm_name, _ in BRACOS_ORDENADOS:
                                if arm_name == atual:
                                    continue
                                if arm_atende_linha(arm_name, i, cl_i):
                                    rec = arm_name
                                    break
                            bracos_pred.append(rec)

                # 3. Injeta braço final na pipeline (atual ou recomendado)
                bracos_final = [rec if rec else atual for rec, atual in zip(bracos_pred, bracos_atuais)]
                df_pipeline['Braço Novo'] = bracos_final
                df_saida['Braço Atual']         = bracos_atuais
                df_saida['Sugestão Braço Novo'] = [rec or '' for rec in bracos_pred]

                for forn in FORNECEDORES:
                    df_run = df_pipeline.copy()
                    df_run['Fornecedor'] = forn
                    for col in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col])):
                        if col not in df_run.columns:
                            df_run[col] = np.nan

                    preds_all = prever_metricas_com_dependencia_w(df_run, modelos, metricas_lote, meta)

                    # Filtro dinâmico por linha baseada na classificação
                    classes = None
                    if tem_classe:
                        classes = df_pipeline['Classificação viária'].fillna('M').astype(str).str.upper().str[0]

                    for m in metricas_lote:
                        if m not in preds_all:
                            continue
                        preds = preds_all[m]
                        if tem_classe:
                            mask_valid = [False] * len(preds)
                            for i, c in enumerate(classes):
                                if c == 'M' and m in ['lmed', 'uo', 'ul', 'w']: mask_valid[i] = True
                                elif c == 'C' and m in ['emed', 'uo', 'w']: mask_valid[i] = True
                                elif c == 'P' and m in ['emed', 'emin', 'w']: mask_valid[i] = True
                            preds = [p if valid else np.nan for p, valid in zip(preds, mask_valid)]

                        df_saida[f'{TARGETS_MAP[m]} - {forn}'] = [max(p, 0) if pd.notna(p) else p for p in preds]

                # ── Pós-processamento: Ajuste de Potência pela Hierarquia NBR ──────────────
                # M: fator proporcional direto (lmed R²=0.28 não é confiável).
                # C/P: escalonamento condicional por iluminância (emed R²>0.8).
                if tem_classe:
                    classes_serie = df_pipeline['Classificação viária'].fillna('').astype(str).str.upper()
                    req_m3_lote = NBR5101['M3']['lmed']
                    for forn in FORNECEDORES:
                        pot_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_col not in df_saida.columns:
                            continue
                        for idx_loc, classe in zip(df_saida.index, classes_serie):
                            info_v = NBR5101.get(classe, {})
                            if not info_v:
                                continue
                            pred_pot = df_saida.loc[idx_loc, pot_col]
                            if pd.isna(pred_pot):
                                continue
                            if classe.startswith('M'):
                                fator_lote = info_v.get('lmed', req_m3_lote) / req_m3_lote
                                df_saida.loc[idx_loc, pot_col] = pred_pot * fator_lote
                            else:
                                ilum_col = f'{TARGETS_MAP["emed"]} - {forn}'
                                req = info_v.get('emed')
                                if req is None or ilum_col not in df_saida.columns:
                                    continue
                                pred_ilum = df_saida.loc[idx_loc, ilum_col]
                                if (pd.notna(pred_ilum) and pred_ilum > 0 and pred_ilum < req):
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

                # Sugestão Braço Novo já calculada e injetada antes do loop de regressão

                # ── CPE: Detecção, Cálculo e Preenchimento do Template ─────────────────
                # Regra revisada:
                # distância >= 45m + desvio de potência histórica relevante.
                # Sem média histórica da classe: fallback por distância.
                # Para cada linha flagada, roda predições com distância/2 e aplica
                # o mesmo ajuste NBR proporcional, garantindo consistência total.
                col_dist_lote = next(
                    (c for c in df_saida.columns if 'distancia entre' in c.lower()), None
                )
                if col_dist_lote:
                    cpe_sim, cpe_qtd_veic, cpe_obs_list = [], [], []
                    DIST_CPE_MIN_LOTE = 45.0
                    POT_DESVIO_FATOR_LOTE = 1.40
                    for _, row_c in df_saida.iterrows():
                        dist_v = pd.to_numeric(row_c.get(col_dist_lote, 0), errors='coerce') or 0
                        classe_v = str(row_c.get('Classificação viária', '')).upper()
                        row_hist_v = medias_historicas[medias_historicas['Classe_Resumo'] == classe_v] if not medias_historicas.empty else pd.DataFrame()
                        media_hist_v = row_hist_v['Média Histórica (W)'].iloc[0] if not row_hist_v.empty else None
                        pot_vals = [
                            row_c.get(f'{TARGETS_MAP["w"]} - {forn}')
                            for forn in FORNECEDORES
                            if pd.notna(row_c.get(f'{TARGETS_MAP["w"]} - {forn}'))
                        ]
                        pot_prev_row = np.mean(pot_vals) if pot_vals else None
                        desvio_ok = (
                            media_hist_v is not None and pot_prev_row is not None
                            and pot_prev_row > media_hist_v * POT_DESVIO_FATOR_LOTE
                        )
                        aciona_regra = dist_v >= DIST_CPE_MIN_LOTE and (desvio_ok or media_hist_v is None)

                        # Classificador ML como sinal adicional por linha
                        cpe_ml_linha = False
                        if clf_cpe is not None:
                            try:
                                row_ml = pd.DataFrame([{
                                    'Faixas de Rodagem':        pd.to_numeric(row_c.get('Faixas de Rodagem', 2), errors='coerce') or 2,
                                    'Largura Via 1':            pd.to_numeric(row_c.get('Largura Via 1', 7), errors='coerce') or 7,
                                    'Largura Via 2':            pd.to_numeric(row_c.get('Largura Via 2', 0), errors='coerce') or 0,
                                    'Largura Passeio 1':        pd.to_numeric(row_c.get('Largura Passeio 1', 2), errors='coerce') or 2,
                                    'largura Passeio 2':        pd.to_numeric(row_c.get('largura Passeio 2', 2), errors='coerce') or 2,
                                    'largura Canteiro Central': pd.to_numeric(row_c.get('largura Canteiro Central', 0), errors='coerce') or 0,
                                    'altura da luminaria':      pd.to_numeric(row_c.get('altura da luminaria', 9), errors='coerce') or 9,
                                    'projecao do braço':        pd.to_numeric(row_c.get('projecao do braço', 1.5), errors='coerce') or 1.5,
                                    'distancia entre postes':   dist_v,
                                    'distancia Poste a via':    pd.to_numeric(row_c.get('distancia Poste a via', 0.5), errors='coerce') or 0.5,
                                    'Classificação viária':     classe_v,
                                    'Tipo de estrutura':        row_c.get('Tipo de estrutura', 'Braço'),
                                    'posteacao':                row_c.get('posteacao', 'Unilateral'),
                                    'Fornecedor':               'LEDSTAR',
                                }])
                                cpe_ml_linha = int(clf_cpe.predict(row_ml)[0]) == 1
                            except Exception:
                                pass

                        aciona_cpe = aciona_regra or cpe_ml_linha
                        if aciona_cpe:
                            cpe_sim.append('Sim')
                            cpe_qtd_veic.append(1)
                            obs = f"Redução {dist_v:.0f}m → {dist_v/2:.0f}m"
                            if desvio_ok and media_hist_v is not None and pot_prev_row is not None:
                                obs += f" | Pot {pot_prev_row:.0f}W > hist {media_hist_v:.0f}W"
                            if cpe_ml_linha and not aciona_regra:
                                obs += " | Detectado por ML"
                            cpe_obs_list.append(obs)
                        else:
                            cpe_sim.append('Não')
                            cpe_qtd_veic.append(0)
                            cpe_obs_list.append('')
                    df_saida['Correção de Ponto Escuro (CPE)'] = cpe_sim
                    df_saida['Quantidade de pontos adicionados para via de veículo'] = cpe_qtd_veic
                    df_saida['Observação CPE  (Reduçao entre postes e/ou Tipo de Posteação)'] = cpe_obs_list

                    # Predições CPE para linhas flagadas
                    cpe_indices = df_saida[df_saida['Correção de Ponto Escuro (CPE)'] == 'Sim'].index
                    if len(cpe_indices) > 0:
                        df_pipeline_cpe = df_pipeline.loc[cpe_indices].copy()
                        if 'distancia entre postes' in df_pipeline_cpe.columns:
                            df_pipeline_cpe['distancia entre postes'] = (
                                df_pipeline_cpe['distancia entre postes'] / 2
                            )

                        for forn in FORNECEDORES:
                            df_run_cpe = df_pipeline_cpe.copy()
                            df_run_cpe['Fornecedor'] = forn
                            for col in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col])):
                                if col not in df_run_cpe.columns:
                                    df_run_cpe[col] = np.nan

                            metricas_lote = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
                            preds_c_all = prever_metricas_com_dependencia_w(df_run_cpe, modelos, metricas_lote, meta)
                            for m in metricas_lote:
                                if m in preds_c_all:
                                    df_saida.loc[cpe_indices, f'CPE {TARGETS_MAP[m]} - {forn}'] = [
                                        max(p, 0) if pd.notna(p) else np.nan for p in preds_c_all[m]
                                    ]

                        # Ajuste NBR nas predições CPE — idêntico ao ajuste do cenário base
                        if tem_classe and 'Classificação viária' in df_pipeline_cpe.columns:
                            req_m3_cpe = NBR5101['M3']['lmed']
                            for forn in FORNECEDORES:
                                cpe_pot_col = f'CPE {TARGETS_MAP["w"]} - {forn}'
                                if cpe_pot_col not in df_saida.columns:
                                    continue
                                for idx_loc in cpe_indices:
                                    cl = str(df_pipeline.loc[idx_loc, 'Classificação viária']).upper()
                                    info_cpe = NBR5101.get(cl, {})
                                    if not info_cpe:
                                        continue
                                    p_pot = df_saida.loc[idx_loc, cpe_pot_col]
                                    if pd.isna(p_pot):
                                        continue
                                    if cl.startswith('M'):
                                        fator_cpe = info_cpe.get('lmed', req_m3_cpe) / req_m3_cpe
                                        df_saida.loc[idx_loc, cpe_pot_col] = p_pot * fator_cpe
                                    else:
                                        cpe_ilum_col = f'CPE {TARGETS_MAP["emed"]} - {forn}'
                                        req_cpe = info_cpe.get('emed')
                                        if req_cpe is None or cpe_ilum_col not in df_saida.columns:
                                            continue
                                        p_ilum = df_saida.loc[idx_loc, cpe_ilum_col]
                                        if (pd.notna(p_ilum) and p_ilum > 0 and p_ilum < req_cpe):
                                            df_saida.loc[idx_loc, cpe_pot_col] = p_pot * (req_cpe / p_ilum)

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
                
                # Formata para o template de exportação (Columns D-DM)
                df_export = formatar_resultado_template(df_saida)
                st.session_state.df_lote = df_saida
                st.session_state.df_export = df_export

                st.markdown('### ✨ Resultados (Preview)')
                # Exibe preview sem as colunas CPE intermediárias (ficaria muito largo)
                cols_preview = [c for c in df_saida.columns if not c.startswith('CPE ')]
                st.dataframe(df_saida[cols_preview].head(10))

                # ── Seção CPE visual — mesma lógica do modo individual ───────────────
                if 'Correção de Ponto Escuro (CPE)' in df_saida.columns:
                    df_cpe_vis = df_saida[df_saida['Correção de Ponto Escuro (CPE)'] == 'Sim']
                    if not df_cpe_vis.empty:
                        st.markdown('<p class="section-title">⚠️ Correção de Ponto Escuro (CPE)</p>', unsafe_allow_html=True)
                        st.warning(
                            f"**{len(df_cpe_vis)} instalação(ões)** com risco de ponto escuro detectado(s) "
                            f"por regra de distância/potência ou pelo classificador ML 🤖. "
                            "Recomenda-se inserção de estrutura intermediária. "
                            "Veja a coluna **Observação CPE** para o motivo por linha."
                        )
                        cpe_rows_display = []
                        for _, row in df_cpe_vis.iterrows():
                            dist_orig = pd.to_numeric(row.get(col_dist_lote, 0), errors='coerce') or 0
                            entry = {
                                'ID': row.get('ID', ''),
                                'Logradouro': str(row.get('Logradouro', ''))[:35],
                                'Classe': row.get('Classificação viária', ''),
                                'Dist. Atual (m)': f"{dist_orig:.0f}",
                                'Dist. CPE (m)': f"{dist_orig/2:.0f}",
                            }
                            for forn in FORNECEDORES:
                                pot_orig = row.get(f'{TARGETS_MAP["w"]} - {forn}')
                                pot_cpe  = row.get(f'CPE {TARGETS_MAP["w"]} - {forn}')
                                entry[f'{forn[:8]} Orig (W)'] = f"{pot_orig:.0f}" if pd.notna(pot_orig) else '-'
                                entry[f'{forn[:8]} CPE (W)']  = f"{pot_cpe:.0f}"  if pd.notna(pot_cpe)  else '-'
                            cpe_rows_display.append(entry)
                        st.dataframe(
                            pd.DataFrame(cpe_rows_display),
                            use_container_width=True,
                            hide_index=True
                        )

                # Gera o resultado no formato da tabela dinâmica (linha por ponto × fornecedor)
                df_tabela = formatar_tabela_resultado(df_saida)
                df_tabela_din = formatar_tabela_dinamica(df_saida)
                df_preview_export = df_saida[cols_preview].copy()

                buffer_resultado = io.BytesIO()
                with pd.ExcelWriter(buffer_resultado, engine='openpyxl') as writer:
                    df_tabela.to_excel(writer, sheet_name='Resultado', index=False)
                    df_tabela_din.to_excel(writer, sheet_name='Tabela Dinâmica', index=False)
                    df_preview_export.to_excel(writer, sheet_name='Dados Completos', index=False)
                    df_export.to_excel(writer, sheet_name='Template', index=False)
                    writer.sheets['Template'].sheet_state = 'hidden'
                st.download_button(
                    '✅ Baixar Resultados (.xlsx)',
                    data=buffer_resultado.getvalue(),
                    file_name='resultados_simulacao.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary',
                )

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
                if col_custo in df.columns:
                    agg_dict[col_custo] = 'sum'
                
                analise_via = df.groupby('Classe_Resumo').agg(agg_dict).reset_index()
                
                # Ajusta nomes das colunas após o agg
                novas_cols = ['Classe', 'Potência Média (W)', 'Quantidade']
                if col_red in df.columns:
                    novas_cols.append('Economia Média (%)')
                if col_custo in df.columns:
                    novas_cols.append('CAPEX Total (R$)')
                analise_via.columns = novas_cols
                
                # Mescla a média histórica de treinamento
                medias_historicas = carregar_media_historica()
                if not medias_historicas.empty:
                    analise_via = analise_via.merge(
                        medias_historicas.rename(columns={'Classe_Resumo': 'Classe'}),
                        on='Classe', how='left'
                    )
                    # Reordena para ficar 'Média Histórica (W)' logo após 'Potência Média (W)'
                    cols = list(analise_via.columns)
                    if 'Média Histórica (W)' in cols:
                        cols.remove('Média Histórica (W)')
                        idx_pot = cols.index('Potência Média (W)')
                        cols.insert(idx_pot + 1, 'Média Histórica (W)')
                        analise_via = analise_via[cols]
                
                c1, c2, c3 = st.columns([1.5, 1.5, 1])
                with c1:
                    fig_via = go.Figure()
                    fig_via.add_trace(go.Bar(
                        x=analise_via['Classe'], y=analise_via['Potência Média (W)'],
                        name='Potência Média (W)', marker_color=CORES[forn_dash]
                    ))
                    fig_via.update_layout(title="Potência Média por Tipo de Via", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_via, use_container_width=True)
                
                with c2:
                    if 'CAPEX Total (R$)' in analise_via.columns:
                        fig_capex = go.Figure()
                        fig_capex.add_trace(go.Bar(
                            x=analise_via['Classe'], y=analise_via['CAPEX Total (R$)'],
                            name='CAPEX Total (R$)', marker_color='#00A9E0'
                        ))
                        fig_capex.update_layout(title="CAPEX Total por Tipo de Via", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_capex, use_container_width=True)
                
                with c3:
                    fig_pie = go.Figure(go.Pie(
                        labels=analise_via['Classe'], values=analise_via['Quantidade'],
                        hole=.4, marker=dict(colors=['#00A9E0', '#1B3664', '#64748b', '#334155'])
                    ))
                    fig_pie.update_layout(title="Mix de Vias", height=350, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_pie, use_container_width=True)

                formato = {'Potência Média (W)': '{:.1f} W'}
                if 'Média Histórica (W)' in analise_via.columns:
                    formato['Média Histórica (W)'] = '{:.1f} W'
                if 'Economia Média (%)' in analise_via.columns:
                    formato['Economia Média (%)'] = '{:.1f} %'
                if 'CAPEX Total (R$)' in analise_via.columns:
                    formato['CAPEX Total (R$)'] = 'R$ {:,.2f}'
                
                st.dataframe(analise_via.style.format(formato, na_rep='-'), use_container_width=True, hide_index=True)
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
