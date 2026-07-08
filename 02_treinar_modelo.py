"""
02_treinar_modelo.py
---------------------
Treina modelos para o dataset bruto (dataset.csv) e para o dataset sem outliers (dataset_limpo.csv).
Salva modelos com prefixos correspondentes:
  - modelo_{key}.pkl        (Bruto)
  - modelo_{key}_limpo.pkl  (Limpo)

Arquitetura de features:
  - Modelos independentes (lmed, uo, ul, w): geometry + class + Fornecedor
  - Modelos dependentes de W (emed, emin):   geometry + class + Fornecedor + W_simulado
    Isso permite que o app rode emed/emin DEPOIS de ajustar W pela hierarquia NBR,
    criando um loop de convergência onde W proposto → emed prevista → W corrigido.
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

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import joblib

# ── Configurações ─────────────────────────────────────────────────────────────
PASTA = os.path.dirname(os.path.abspath(__file__))

TARGETS_INFO = {
    'lmed': ('Luminância Média', 'cd/m²'),
    'uo':   ('Fator de Uniformidade', ''),
    'ul':   ('Uniformidade Longitudinal', ''),
    'emed': ('Iluminância Média', 'lux'),
    'emin': ('Iluminância mínima horizontal E (lux)', 'lux'),
    'w':    ('Potencia simulada - IP Principal (W)', 'W'),
}

# W como feature adicional para modelos de iluminância
W_COL = TARGETS_INFO['w'][0]
MODELOS_COM_W = ['emed', 'emin']

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
    # 'Altura de Instalação'  # REMOVIDA — baixa correlação com targets
]

FEATURES_CATEGORICAS = [
    'Classificação viária',
    'Tipo de estrutura',
    'posteacao',
    'Braço Novo',
    'Fornecedor',
]


def make_preprocessor(num_feats, cat_feats):
    """Cria ColumnTransformer para o conjunto de features especificado."""
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ]), num_feats),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_feats),
    ])


def treinar_e_salvar(dataset_name, suffix=""):
    DATASET = os.path.join(PASTA, dataset_name)
    print(f'\n\n{"#"*80}')
    print(f'### TREINANDO PARA: {dataset_name} (Sufixo: {suffix or "bruto"})')
    print(f'{"#"*80}')

    if not os.path.exists(DATASET):
        print(f'[ERRO] Arquivo {DATASET} não encontrado!')
        return

    df = pd.read_csv(DATASET, encoding='utf-8-sig')
    print(f'[INFO] {len(df)} linhas carregadas')

    num_ok = [c for c in FEATURES_NUMERICAS  if c in df.columns]
    cat_ok = [c for c in FEATURES_CATEGORICAS if c in df.columns]

    meta_geral = {
        'features_numericas':    num_ok,
        'features_categoricas':  cat_ok,
        'feature_w_col':         W_COL,
        'modelos_dependem_de_w': MODELOS_COM_W,
    }

    for key, (coluna, unidade) in TARGETS_INFO.items():
        if coluna not in df.columns:
            continue

        # emed e emin usam W como feature adicional.
        # Requer que ambos (target e W) sejam válidos na linha de treino.
        if key in MODELOS_COM_W and W_COL in df.columns:
            num_feats = num_ok + [W_COL]
            df_t = df.dropna(subset=[coluna, W_COL]).copy()
            df_t = df_t[(df_t[coluna] > 0) & (df_t[W_COL] > 0)]
        else:
            num_feats = num_ok
            df_t = df.dropna(subset=[coluna]).copy()
            df_t = df_t[df_t[coluna] > 0]

        if len(df_t) < 10:
            print(f"  [AVISO] Poucas amostras para {key} ({len(df_t)}). Pulando...")
            continue

        print(f'\n[TREINO] {key.upper()} | {len(df_t)} amostras | '
              f'{len(num_feats)} num + {len(cat_ok)} cat features')

        all_feats = num_feats + cat_ok
        X = df_t[all_feats]
        y = df_t[coluna]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        base_est = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
        )
        if key in ['lmed', 'emed']:
            base_est = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
            )

        prep = make_preprocessor(num_feats, cat_ok)
        pipe = Pipeline([('prep', prep), ('model', base_est)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'  >> R2 = {r2:.4f} | MAE = {mae:.2f}')

        path_mod = os.path.join(PASTA, f'modelo_{key}{suffix}.pkl')
        joblib.dump(pipe, path_mod)

        meta_geral[f'modelo_{key}'] = {
            'r2':                round(r2, 4),
            'mae':               round(mae, 2),
            'type':              type(base_est).__name__,
            'features_numericas': num_feats,
        }

    # ── Classificador CPE (binário: precisa inserir ponto?) ───────────────────
    if 'tem_cpe' in df.columns:
        print(f'\n[TREINO] CLASSIFICADOR CPE | features: geométricas + classe')
        # Features: apenas geometria e classificação — sem Braço Novo (não disponível na previsão)
        cat_cpe = [c for c in cat_ok if c != 'Braço Novo']
        df_cpe = df.dropna(subset=num_ok + cat_cpe).copy()
        X_cpe = df_cpe[num_ok + cat_cpe]
        y_cpe = df_cpe['tem_cpe'].astype(int)
        n_pos = int(y_cpe.sum())
        print(f'  Amostras: {len(df_cpe)} | Positivos (CPE=1): {n_pos} ({n_pos/len(df_cpe)*100:.1f}%)')
        if n_pos >= 5:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_cpe, y_cpe, test_size=0.2, random_state=42, stratify=y_cpe
            )
            clf_cpe = Pipeline([
                ('prep', make_preprocessor(num_ok, cat_cpe)),
                ('model', RandomForestClassifier(
                    n_estimators=400, class_weight='balanced', random_state=42, n_jobs=-1
                )),
            ])
            clf_cpe.fit(X_tr, y_tr)
            y_pred_cpe = clf_cpe.predict(X_te)
            print(classification_report(y_te, y_pred_cpe, target_names=['sem_cpe', 'com_cpe'], zero_division=0))
            joblib.dump(clf_cpe, os.path.join(PASTA, f'modelo_cpe{suffix}.pkl'))
            meta_geral['modelo_cpe'] = {
                'features_numericas':   num_ok,
                'features_categoricas': cat_cpe,
                'type': 'RandomForestClassifier',
            }
            print(f'  [OK] modelo_cpe{suffix}.pkl salvo')
        else:
            print(f'  [AVISO] Poucos positivos ({n_pos}) — classificador CPE não treinado')

    # ── Classificador Braço Novo (multiclasse: qual braço usar?) ──────────────
    if 'Braço Novo' in df.columns:
        print(f'\n[TREINO] CLASSIFICADOR BRAÇO NOVO | features: geométricas + classe')
        # Remove Braço Novo das features — é o target aqui
        cat_braco = [c for c in cat_ok if c != 'Braço Novo']
        df_braco = df[df['Braço Novo'].notna()].dropna(subset=num_ok + cat_braco).copy()
        X_braco = df_braco[num_ok + cat_braco]
        y_braco = df_braco['Braço Novo'].astype(str)
        print(f'  Amostras com troca de braço: {len(df_braco)} | Classes: {sorted(y_braco.unique())}')
        if len(df_braco) >= 20:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_braco, y_braco, test_size=0.2, random_state=42, stratify=y_braco
            )
            clf_braco = Pipeline([
                ('prep', make_preprocessor(num_ok, cat_braco)),
                ('model', RandomForestClassifier(
                    n_estimators=400, random_state=42, n_jobs=-1
                )),
            ])
            clf_braco.fit(X_tr, y_tr)
            y_pred_braco = clf_braco.predict(X_te)
            print(classification_report(y_te, y_pred_braco, zero_division=0))
            joblib.dump(clf_braco, os.path.join(PASTA, f'modelo_braco{suffix}.pkl'))
            meta_geral['modelo_braco'] = {
                'features_numericas':   num_ok,
                'features_categoricas': cat_braco,
                'classes':              sorted(y_braco.unique().tolist()),
                'type': 'RandomForestClassifier',
            }
            print(f'  [OK] modelo_braco{suffix}.pkl salvo')
        else:
            print(f'  [AVISO] Poucas amostras ({len(df_braco)}) — classificador Braço não treinado')

    meta_path = os.path.join(PASTA, f'features{suffix}.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_geral, f, ensure_ascii=False, indent=2)
    print(f'\n[OK] Metadados salvos em {meta_path}')


# ── Execução ──────────────────────────────────────────────────────────────────
treinar_e_salvar('dataset.csv',       suffix="")
treinar_e_salvar('dataset_limpo.csv', suffix="_limpo")

print('\n[DONE] Treinamento completo!')
