"""
01_extrair_dados.py
--------------------
Lê as abas 'Simulações' das planilhas de simulação, filtra pelos fornecedores
Ledstar, SX Lighting e Tecnowatt, seleciona as features físicas e salva dataset.csv.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os

# ── Configurações ────────────────────────────────────────────────────────────
PASTA = r'c:\Users\julia\OneDrive\Área de Trabalho\Houer\ML - Simulação'

FORNECEDORES_ALVO = ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']

# Features físicas/geométricas (SEM classificação viária/pedonal/ciclovia)
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
    'Fornecedor',   # feature — o modelo aprende o padrão de cada fornecedor
]

TARGET_LM = 'Fluxo Luminoso - IP Principal (lm)'
TARGET_W  = ' Potência simulada - IP Principal (W)'   # espaço no início é do Excel
TARGETS = [TARGET_LM, TARGET_W]

# ── Extração ─────────────────────────────────────────────────────────────────
arquivos = [f for f in os.listdir(PASTA) if f.endswith('.xlsx')]
print(f'[INFO] Arquivos encontrados: {arquivos}\n')

dfs = []

for arquivo in arquivos:
    caminho = os.path.join(PASTA, arquivo)
    print(f'[LENDO] {arquivo}')

    xl = pd.ExcelFile(caminho, engine='openpyxl')

    # Encontra aba de simulações (flexível)
    aba = next((n for n in xl.sheet_names if 'simul' in n.lower()), None)
    if not aba:
        print(f'   [AVISO] Aba "Simulações" não encontrada. Abas disponíveis: {xl.sheet_names}')
        continue

    df = pd.read_excel(caminho, sheet_name=aba, header=0, engine='openpyxl')
    print(f'   [OK] Aba: "{aba}" | {len(df)} linhas x {len(df.columns)} colunas')

    # Diagnóstico: colunas esperadas x disponíveis
    todas_esperadas = FEATURES_NUMERICAS + FEATURES_CATEGORICAS + TARGETS
    faltando = [c for c in todas_esperadas if c not in df.columns]
    if faltando:
        print(f'   [AVISO] Colunas nao encontradas: {faltando}')
        # Tenta sugerir alternativas
        for c_falta in faltando:
            sugestoes = [c for c in df.columns if c_falta[:6].lower() in c.lower()]
            if sugestoes:
                print(f'      Sugestão para "{c_falta}": {sugestoes[:3]}')

    # Seleciona apenas colunas disponíveis
    cols_ok = [c for c in todas_esperadas if c in df.columns]
    df_sel = df[cols_ok].copy()
    df_sel['arquivo_origem'] = arquivo

    dfs.append(df_sel)
    print(f'   [INFO] {len(df_sel)} linhas selecionadas')

if not dfs:
    print('\n❌ Nenhum dado extraído. Verifique os nomes das planilhas.')
    raise SystemExit(1)

# ── Combinação e Limpeza ──────────────────────────────────────────────────────
df_total = pd.concat(dfs, ignore_index=True)
print(f'\n[INFO] Total combinado (bruto): {len(df_total)} linhas')

# Normaliza Fornecedor
if 'Fornecedor' in df_total.columns:
    df_total['Fornecedor'] = df_total['Fornecedor'].astype(str).str.strip().str.upper()
    print(f'\nFornecedores unicos (antes do filtro):')
    print(df_total['Fornecedor'].value_counts().to_string())

    df_total = df_total[df_total['Fornecedor'].isin(FORNECEDORES_ALVO)].copy()
    print(f'\nApós filtro de fornecedores: {len(df_total)} linhas')
    print(df_total['Fornecedor'].value_counts().to_string())

# Limpa TARGET lm
if TARGET_LM in df_total.columns:
    df_total[TARGET_LM] = pd.to_numeric(df_total[TARGET_LM], errors='coerce')
    antes = len(df_total)
    df_total = df_total.dropna(subset=[TARGET_LM])
    df_total = df_total[df_total[TARGET_LM] > 0]
    print(f'\nLinhas removidas por target invalido: {antes - len(df_total)}')
    print(f'Dataset final: {len(df_total)} linhas')

# Limpa TARGET W (potencia) — renomeia para coluna sem espaco
if TARGET_W in df_total.columns:
    df_total[TARGET_W] = pd.to_numeric(df_total[TARGET_W], errors='coerce')
    # Renomeia para nome limpo
    df_total.rename(columns={TARGET_W: 'Potencia simulada - IP Principal (W)'}, inplace=True)
    print(f'Amostras com potencia valida: {df_total["Potencia simulada - IP Principal (W)"].notna().sum()}')

# Limpa features numéricas disponíveis
for col in FEATURES_NUMERICAS:
    if col in df_total.columns:
        df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

# ── Salva ──────────────────────────────────────────────────────────────────────
output = os.path.join(PASTA, 'dataset.csv')
df_total.to_csv(output, index=False, encoding='utf-8-sig')

print(f'\n[OK] Dataset salvo em: {output}')
print(f'\nEstatisticas do alvo - Lux (lm):')
print(df_total[TARGET_LM].describe().to_string())

if 'Potencia simulada - IP Principal (W)' in df_total.columns:
    print(f'\nEstatisticas do alvo - Potencia (W):')
    print(df_total['Potencia simulada - IP Principal (W)'].describe().to_string())

print(f'\nAmostra por fornecedor:')
for forn in FORNECEDORES_ALVO:
    sub = df_total[df_total['Fornecedor'] == forn]
    if len(sub) > 0:
        lm_med = sub[TARGET_LM].mean()
        w_med  = sub['Potencia simulada - IP Principal (W)'].mean() if 'Potencia simulada - IP Principal (W)' in sub.columns else float('nan')
        print(f'   {forn}: {len(sub)} linhas | media {lm_med:.0f} lm | {w_med:.0f} W')
    else:
        print(f'   {forn}: sem dados')
