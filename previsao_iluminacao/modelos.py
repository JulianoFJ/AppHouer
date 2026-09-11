"""Carregamento dos modelos de ML e inferência de métricas luminotécnicas.

Sem decorador de cache aqui de propósito — `st.cache_resource`/`st.cache_data` ficam
na página (`paginas/simulacao_nbr.py`), que é quem tem contexto Streamlit. Mesma
convenção dos demais pacotes de domínio (`amostragem_ip`, `cadastro_ip`), nenhum dos
quais importa `streamlit`.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from .constantes import BRACOS_ORDENADOS, BRACOS_PROJECAO, PASTA


def projecao_para_braco(proj_m: float) -> str:
    """Retorna a classificação de braço mais próxima da projeção em metros."""
    return min(BRACOS_PROJECAO, key=lambda b: abs(BRACOS_PROJECAO[b] - float(proj_m)))


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
