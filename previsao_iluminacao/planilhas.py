"""Formatação das saídas da simulação em lote para os formatos de planilha esperados."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constantes import FORNECEDORES, MAPEAMENTO_COLS, NBR5101, TARGETS_MAP, TEMPLATE_COLUMNS


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
