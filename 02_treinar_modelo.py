"""
02_treinar_modelo.py
---------------------
Le o dataset.csv, treina dois modelos de Random Forest:
  - modelo_lm.pkl   → preve Fluxo Luminoso (lm)
  - modelo_w.pkl    → preve Potencia simulada (W)
Salva tambem features.json com metadados de ambos.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Configurações ─────────────────────────────────────────────────────────────
PASTA   = r'c:\Users\julia\OneDrive\Área de Trabalho\Houer\ML - Simulação'
DATASET = os.path.join(PASTA, 'dataset.csv')

TARGET_LM = 'Fluxo Luminoso - IP Principal (lm)'
TARGET_W  = 'Potencia simulada - IP Principal (W)'

FEATURES_NUMERICAS = [
    'Faixas de Rodagem',
    'Largura Via 1',
    'Largura Via 2',
    'largura Canteiro Central',
    'altura da luminaria',
    'projecao do braço',
    'distancia entre postes',
    'distancia Poste a via',
    'Altura de Instalação',
]

FEATURES_CATEGORICAS = [
    'Tipo de estrutura',
    'posteacao',
    'Braço Novo',
    'Fornecedor',
]

# ── Carrega dados ─────────────────────────────────────────────────────────────
print('[INFO] Carregando dataset...')
df = pd.read_csv(DATASET, encoding='utf-8-sig')
print(f'   {len(df)} linhas carregadas')

num_ok = [c for c in FEATURES_NUMERICAS  if c in df.columns]
cat_ok = [c for c in FEATURES_CATEGORICAS if c in df.columns]
all_features = num_ok + cat_ok

print(f'\n[INFO] Features numericas  ({len(num_ok)}): {num_ok}')
print(f'[INFO] Features categoricas ({len(cat_ok)}): {cat_ok}')
print(f'\n[INFO] Distribuicao de Fornecedor:')
if 'Fornecedor' in df.columns:
    print(df['Fornecedor'].value_counts().to_string())

# ── Pré-processamento (compartilhado) ─────────────────────────────────────────
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

# ── Modelos candidatos ────────────────────────────────────────────────────────
def get_modelos():
    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
        ),
        'Ridge': Ridge(alpha=10.0),
    }

# ── Função de treino/avaliação ────────────────────────────────────────────────
def treinar(df_base, target_col, label):
    """Treina e avalia modelos para um target. Retorna o melhor pipeline."""
    if target_col not in df_base.columns:
        print(f'\n[AVISO] Coluna "{target_col}" nao encontrada no dataset. Pulando.')
        return None, {}

    df_t = df_base.dropna(subset=[target_col]).copy()
    df_t = df_t[df_t[target_col] > 0]
    print(f'\n{"="*60}')
    print(f'[TREINO] Target: {label} ({target_col})')
    print(f'         Amostras: {len(df_t)}')
    print(f'{"="*60}')

    X = df_t[all_features].copy()
    y = df_t[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print('\n[CV] Avaliacao Cross-Validation (5-fold):\n')
    resultados = {}
    for nome, estimador in get_modelos().items():
        pipe = Pipeline([('prep', preprocessor), ('model', estimador)])
        scores_r2  = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        scores_mae = cross_val_score(pipe, X_train, y_train, cv=5,
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
        r2_m  = scores_r2.mean()
        mae_m = -scores_mae.mean()
        resultados[nome] = {'R2': r2_m, 'MAE': mae_m, 'pipe': pipe}
        unidade = 'lm' if 'lm' in target_col else 'W'
        print(f'  {nome:30s} | R2 = {r2_m:.4f} | MAE = {mae_m:.1f} {unidade}')

    melhor_nome = max(resultados, key=lambda k: resultados[k]['R2'])
    melhor_pipe  = resultados[melhor_nome]['pipe']
    print(f'\n[RESULTADO] Melhor modelo: {melhor_nome} (R2 CV = {resultados[melhor_nome]["R2"]:.4f})')

    melhor_pipe.fit(X_train, y_train)
    y_pred = melhor_pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    unidade = 'lm' if 'lm' in target_col else 'W'
    print(f'\n[METRICAS] Conjunto de teste ({len(X_test)} amostras):')
    print(f'   MAE  = {mae:.1f} {unidade}')
    print(f'   RMSE = {rmse:.1f} {unidade}')
    print(f'   R2   = {r2:.4f}')

    # Importância
    if hasattr(melhor_pipe['model'], 'feature_importances_'):
        try:
            feat_names   = melhor_pipe['prep'].get_feature_names_out()
            importancias = melhor_pipe['model'].feature_importances_
            imp_df = pd.DataFrame({'feature': feat_names, 'importancia': importancias})
            imp_df = imp_df.sort_values('importancia', ascending=False).head(10)
            print('\n[TOP-10] Features mais importantes:')
            print(imp_df.to_string(index=False))
        except Exception as e:
            print(f'   (erro ao extrair importancias: {e})')

    meta = {
        'modelo': melhor_nome,
        'target': target_col,
        'features_numericas':  num_ok,
        'features_categoricas': cat_ok,
        'r2_cv':     round(resultados[melhor_nome]['R2'],  4),
        'mae_cv':    round(resultados[melhor_nome]['MAE'], 2),
        'r2_teste':  round(r2,   4),
        'mae_teste': round(mae,  2),
        'rmse_teste':round(rmse, 2),
        'n_amostras_treino': len(X_train),
        'n_amostras_teste':  len(X_test),
    }
    return melhor_pipe, meta

# ── Treina modelo de Fluxo Luminoso (lm) ─────────────────────────────────────
pipe_lm, meta_lm = treinar(df, TARGET_LM, 'Fluxo Luminoso')

# ── Treina modelo de Potencia (W) ─────────────────────────────────────────────
pipe_w, meta_w = treinar(df, TARGET_W, 'Potencia')

# ── Salva modelos e metadados ─────────────────────────────────────────────────
print(f'\n{"="*60}')
print('[SALVANDO]')

if pipe_lm:
    path_lm = os.path.join(PASTA, 'modelo_lm.pkl')
    joblib.dump(pipe_lm, path_lm)
    print(f'[OK] Modelo lm salvo:  {path_lm}')

if pipe_w:
    path_w = os.path.join(PASTA, 'modelo_w.pkl')
    joblib.dump(pipe_w, path_w)
    print(f'[OK] Modelo W  salvo:  {path_w}')

features_path = os.path.join(PASTA, 'features.json')
meta_geral = {
    'features_numericas':  num_ok,
    'features_categoricas': cat_ok,
    'modelo_lm': meta_lm,
    'modelo_w':  meta_w,
}
with open(features_path, 'w', encoding='utf-8') as f:
    json.dump(meta_geral, f, ensure_ascii=False, indent=2)
print(f'[OK] Metadados salvos: {features_path}')

print('\n[DONE] Treinamento concluido!')
