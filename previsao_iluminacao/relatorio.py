"""Geração de relatório PDF da simulação."""

from __future__ import annotations

import pandas as pd
from fpdf import FPDF

from .historico import buscar_custo


def gerar_pdf(fornecedores, resultados, info_nbr, inputs, banco_luminarias, sugestoes,
             metricas_confiaveis, endereco="Não informado"):
    """
    `metricas_confiaveis` é parâmetro explícito (não era, na página original): antes,
    esta função lia uma variável de módulo atribuída dentro de um bloco `with
    st.sidebar:` — funcionava só porque blocos `with` não criam escopo próprio em
    Python, então virava global do módulo antes desta função ser chamada. Fora do
    módulo da página essa premissa não existe mais.
    """
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
