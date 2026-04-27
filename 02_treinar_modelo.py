"""
02_treinar_modelo.py
---------------------
Treina modelos para o dataset bruto (dataset.csv) e para o dataset sem outliers (dataset_limpo.csv).
Salva modelos com prefixos correspondentes:
  - modelo_{key}.pkl        (Bruto)
  - modelo_{key}_limpo.pkl  (Limpo)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Configurações ─────────────────────────────────────────────────────────────
PASTA = r'c:\Users\julia\OneDrive\Área de Trabalho\Houer\ML - Simulação'

TARGETS_INFO = {
    'lmed': ('Luminância Média', 'cd/m²'),
    'uo':   ('Fator de Uniformidade', ''),
    'ul':   ('Uniformidade Longitudinal', ''),
    'emed': ('Iluminância Média', 'lux'),
    'emin': ('Iluminância mínima horizontal E (lux)', 'lux'),
    'w':    ('Potencia simulada - IP Principal (W)', 'W')
}

FEATURES_NUMERICAS = [
    'Faixas de Rodagem',
    'Largura Via 1',
    'Largura Via 2',
    'Largura Passeio 1',
    'largura Passeio 2',
    'largura Canteiro Central',
    'altura da luminaria',
    'projecao do braço',
    'distancia entre postes',
    'distancia Poste a via',
    # 'Altura de Instalação'  # REMOVIDA
]

FEATURES_CATEGORICAS = [
    'Classificação viária',
    'Tipo de estrutura',
    'posteacao',
    'Braço Novo',
    'Fornecedor',
]

def treinar_e_salvar(dataset_name, suffix=""):
    DATASET = os.path.join(PASTA, dataset_name)
    print(f'\n\n{"#"*80}')
    print(f'### TREINANDO PARA: {dataset_name} (Sufixo: {suffix})')
    print(f'{"#"*80}')

    if not os.path.exists(DATASET):
        print(f'[ERRO] Arquivo {DATASET} não encontrado!')
        return

    df = pd.read_csv(DATASET, encoding='utf-8-sig')
    print(f'[INFO] {len(df)} linhas carregadas')

    num_ok = [c for c in FEATURES_NUMERICAS  if c in df.columns]
    cat_ok = [c for c in FEATURES_CATEGORICAS if c in df.columns]
    all_features = num_ok + cat_ok

    # Pré-processamento
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_ok),
        ('cat', cat_transformer, cat_ok),
    ])

    modelos_treinados = {}
    meta_geral = {'features_numericas': num_ok, 'features_categoricas': cat_ok}

    for key, (coluna, unidade) in TARGETS_INFO.items():
        if coluna not in df.columns: continue
        
        # Filtra apenas linhas que possuem o alvo específico
        df_t = df.dropna(subset=[coluna]).copy()
        df_t = df_t[df_t[coluna] > 0]
        if len(df_t) < 10: 
            print(f"  [AVISO] Poucas amostras para {key}. Pulando...")
            continue

        print(f'\n[TREINO] {key.upper()} | {len(df_t)} amostras')
        X = df_t[all_features]
        y = df_t[coluna]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Seleciona o melhor estimador para cada alvo
        base_est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
        if key in ['lmed', 'emed']: # RandomForest costuma ser melhor para médias
            base_est = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)

        pipe = Pipeline([('prep', preprocessor), ('model', base_est)])
        pipe.fit(X_train, y_train)

        # Avaliação
        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'  >> R2 = {r2:.4f} | MAE = {mae:.2f}')

        # Salva o modelo individual
        path_mod = os.path.join(PASTA, f'modelo_{key}{suffix}.pkl')
        joblib.dump(pipe, path_mod)
        meta_geral[f'modelo_{key}'] = {'r2': round(r2, 4), 'mae': round(mae, 2), 'type': type(base_est).__name__}

    meta_path = os.path.join(PASTA, f'features{suffix}.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_geral, f, ensure_ascii=False, indent=2)

# ── Execução ──────────────────────────────────────────────────────────────────
treinar_e_salvar('dataset.csv', suffix="")
treinar_e_salvar('dataset_limpo.csv', suffix="_limpo")

print('\n[DONE] Treinamento completo!')
