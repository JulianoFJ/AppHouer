"""
01_extrair_dados.py
--------------------
Lê as abas 'Simulações' das planilhas de simulação, filtra pelos fornecedores
Ledstar, SX Lighting e Tecnowatt, seleciona as features físicas e salva dataset.csv.

Além das features e targets originais, extrai:
  - tem_cpe   : 1 se o ponto teve Correção de Ponto Escuro, 0 caso contrário
  - Braço Novo: mantido como feature (usado tb como target de classificação no treino)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os

# ── Configurações ────────────────────────────────────────────────────────────
PASTA = os.path.dirname(os.path.abspath(__file__))

FORNECEDORES_ALVO = ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']

# Features físicas/geométricas
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
    # 'Altura de Instalação'  <- REMOVIDA: 100% vazia no dataset
]

FEATURES_CATEGORICAS = [
    'Classificação viária',   # C0-C5, M1-M6, P4-P6 — essencial para previsão por subclasse
    'Tipo de estrutura',
    'posteacao',
    'Braço Novo',
    'Fornecedor',   # feature — o modelo aprende o padrão de cada fornecedor
]

# Apelidos de colunas — cada planilha de origem pode grafar o mesmo campo de forma
# diferente (acentos, sufixos, sinônimos). O rename só é aplicado se o nome canônico
# ainda não existir na planilha.
ALIAS_COLUNAS = {
    # ARMBH01.2 - Base de Simulações
    'Faixas de Rodagem via C':                     'Faixas de Rodagem',
    'Altura da luminária':                         'altura da luminaria',
    'Distância entre postes':                      'distancia entre postes',
    'Classificação viária (Análise do Cadastro)':  'Classificação viária',
    # CEF06 (Recife) — Fator de Uniformidade grafado como Uniformidade Global Mínima
    'Uniformidade Global Mínima':                  'Fator de Uniformidade',
}

def _normalizar_colunas(df):
    """Renomeia apelidos conhecidos para o nome canônico (sem sobrescrever existentes)."""
    renames = {orig: novo for orig, novo in ALIAS_COLUNAS.items()
               if orig in df.columns and novo not in df.columns}
    if renames:
        df = df.rename(columns=renames)
    return df, renames

# Coluna de CPE — pode ter variação de encoding entre planilhas
CPE_KEYWORDS = ['cpe', 'ponto escuro', 'corre']   # substrings para localizar a coluna

def _achar_col_cpe(colunas):
    """Retorna o nome da coluna de CPE na planilha, ou None se não encontrada."""
    for c in colunas:
        cl = c.lower()
        if 'cpe' in cl or 'ponto escuro' in cl:
            return c
    return None

TARGET_LMED = 'Luminância Média'
TARGET_UO   = 'Fator de Uniformidade'
TARGET_UL   = 'Uniformidade Longitudinal'
TARGET_EMED = 'Iluminância Média'
TARGET_EMIN = 'Iluminância mínima horizontal E (lux)'
TARGET_W    = ' Potência simulada - IP Principal (W)'   # espaço no início é do Excel

TARGETS = [TARGET_LMED, TARGET_UO, TARGET_UL, TARGET_EMED, TARGET_EMIN, TARGET_W]

# ── Extração ─────────────────────────────────────────────────────────────────
arquivos = [f for f in os.listdir(PASTA) if f.endswith('.xlsx')]
print(f'[INFO] Arquivos encontrados: {arquivos}\n')

dfs = []

for arquivo in arquivos:
    caminho = os.path.join(PASTA, arquivo)
    print(f'[LENDO] {arquivo}')

    try:
        xl = pd.ExcelFile(caminho, engine='openpyxl')
    except PermissionError:
        print(f'   [AVISO] Arquivo bloqueado (aberto no Excel?) — pulando: {arquivo}')
        continue
    except Exception as e:
        print(f'   [AVISO] Não foi possível abrir {arquivo}: {e}')
        continue

    # Encontra aba de simulações (flexível)
    aba = next((n for n in xl.sheet_names if 'simul' in n.lower()), None)
    if not aba:
        print(f'   [AVISO] Aba "Simulações" não encontrada. Abas disponíveis: {xl.sheet_names}')
        continue

    df = pd.read_excel(caminho, sheet_name=aba, header=0, engine='openpyxl')
    print(f'   [OK] Aba: "{aba}" | {len(df)} linhas x {len(df.columns)} colunas')

    df, renames = _normalizar_colunas(df)
    if renames:
        print(f'   [ALIAS] Colunas renomeadas: {renames}')

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

    # ── Extrai CPE como label binário ─────────────────────────────────────────
    col_cpe = _achar_col_cpe(df.columns)
    if col_cpe:
        cpe_raw = df[col_cpe]
        # Positivo: valor não-nulo E não-vazio (evita converter NaN→'nan' antes de checar)
        cpe_valida = cpe_raw.notna() & (cpe_raw.astype(str).str.strip() != '')
        df_sel['tem_cpe'] = cpe_valida.values.astype(int)
        print(f'   [CPE] Coluna encontrada: "{col_cpe}" | positivos: {int(df_sel["tem_cpe"].sum())} / {len(df_sel)}')
    else:
        df_sel['tem_cpe'] = 0
        print(f'   [CPE] Coluna não encontrada — tem_cpe = 0 para todas as linhas')

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

# Converte os alvos para numérico
for tgt in TARGETS:
    if tgt in df_total.columns:
        df_total[tgt] = pd.to_numeric(df_total[tgt], errors='coerce')
        print(f'Valores válidos para {tgt}: {df_total[tgt].notna().sum()}')

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

# ── Salva dataset bruto (com outliers) ─────────────────────────────────────────────
output_bruto = os.path.join(PASTA, 'dataset.csv')
df_total.to_csv(output_bruto, index=False, encoding='utf-8-sig')
print(f'\n[OK] Dataset BRUTO salvo em: {output_bruto} ({len(df_total)} linhas)')

# ── Aplica filtro de outliers e salva dataset limpo ───────────────────────────────
df_limpo = df_total.copy()

def filtrar(df, col_pattern, vmin=None, vmax=None):
    """Remove linhas onde col < vmin ou col > vmax (apenas se col existir)."""
    c = next((x for x in df.columns if col_pattern.lower() in x.lower()), None)
    if c is None:
        return df
    serie = pd.to_numeric(df[c], errors='coerce')
    mask = pd.Series([True] * len(df), index=df.index)
    if vmin is not None:
        mask &= (serie >= vmin) | serie.isna()
    if vmax is not None:
        mask &= (serie <= vmax) | serie.isna()
    removidas = (~mask).sum()
    if removidas > 0:
        print(f'  [FILTRO] {c}: {removidas} linhas removidas (fora de [{vmin}, {vmax}])')
    return df[mask]

antes = len(df_limpo)
df_limpo = filtrar(df_limpo, 'Largura Passeio 1',     vmin=0,   vmax=50)
df_limpo = filtrar(df_limpo, 'largura Passeio 2',     vmin=0,   vmax=50)
df_limpo = filtrar(df_limpo, 'altura da luminaria',   vmin=3,   vmax=25)
df_limpo = filtrar(df_limpo, 'distancia entre poste', vmin=5,   vmax=85)
# Uniformidades são razões (min/média): valores acima de 1 são erro de digitação
df_limpo = filtrar(df_limpo, 'Fator de Uniformidade',      vmin=0, vmax=1)
df_limpo = filtrar(df_limpo, 'Uniformidade Longitudinal',  vmin=0, vmax=1)
print(f'  Outliers removidos no total: {antes - len(df_limpo)} linhas')

output_limpo = os.path.join(PASTA, 'dataset_limpo.csv')
try:
    df_limpo.to_csv(output_limpo, index=False, encoding='utf-8-sig')
    print(f'[OK] Dataset LIMPO salvo em: {output_limpo} ({len(df_limpo)} linhas)')
except PermissionError:
    print(f'[AVISO] dataset_limpo.csv está bloqueado (feche o arquivo e re-execute). Dataset bruto salvo com sucesso.')
print(f'\nResumo Estatístico dos Alvos:')
for tgt in TARGETS:
    if tgt in df_total.columns:
        col_clean = 'Potencia simulada - IP Principal (W)' if tgt == TARGET_W else tgt
        print(f'--- {col_clean} ---')
        print(df_total[col_clean].describe().to_string())
        print()

if 'Potencia simulada - IP Principal (W)' in df_total.columns:
    print(f'\nEstatisticas do alvo - Potencia (W):')
    print(df_total['Potencia simulada - IP Principal (W)'].describe().to_string())

if 'tem_cpe' in df_total.columns:
    n_cpe = int(df_total['tem_cpe'].sum())
    print(f'\n[CPE] Positivos no dataset final: {n_cpe} / {len(df_total)} ({n_cpe/len(df_total)*100:.1f}%)')
if 'Braço Novo' in df_total.columns:
    n_braco = int(df_total['Braço Novo'].notna().sum())
    print(f'[Braço Novo] Linhas com troca registrada: {n_braco} / {len(df_total)}')

print(f'\nAmostra por fornecedor:')
for forn in FORNECEDORES_ALVO:
    sub = df_total[df_total['Fornecedor'] == forn]
    if len(sub) > 0:
        print(f'   {forn}: {len(sub)} linhas')
    else:
        print(f'   {forn}: sem dados')
