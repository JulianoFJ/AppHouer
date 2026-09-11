"""Dados históricos/de referência (custo de luminárias e médias) usados para custo e
comparações — não os modelos de ML em si (ver `modelos.py`).

`carregar_regras_braco_novo` e `sugerir_braco_novo` (que consumia o retorno dela)
existiam aqui até 10/09/2026: nenhum call site no app — a página reimplementa a mesma
cascata ML→histórico→fallback inline no lote — removidas por serem código morto, não
só movidas."""

from __future__ import annotations

import os

import pandas as pd

from .constantes import PASTA


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
