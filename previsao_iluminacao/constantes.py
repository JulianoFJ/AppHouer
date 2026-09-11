"""Constantes de referência da previsão luminotécnica — sem lógica."""

from __future__ import annotations

import os

# Este pacote vive na raiz do projeto, no mesmo nível de `paginas/` — mesma distância
# até `ml/` (irmã de ambos) que a página tinha antes da extração: dois `dirname` para
# sair de `previsao_iluminacao/constantes.py` até a raiz, depois entrar em `ml/`.
PASTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml')

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
