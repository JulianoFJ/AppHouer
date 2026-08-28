"""
Hub de Municípios — pesquisa de arrecadação de COSIP e parque de iluminação pública
para qualquer município do Brasil, com os indicadores que cruzam as duas bases.

Fontes:
  - SICONFI / DCA Anexo I-C (Tesouro Nacional) — receita de COSIP declarada.
  - BDGD / entidade PIP (ANEEL) — parque de IP: pontos, carga, consumo, tecnologia.

Entry points:
    from hub_municipios import siconfi, bdgd, indicadores
    cosip = siconfi.consultar_com_cache(["3106200"], [2023, 2024])
    painel = indicadores.cruzar(cosip)

O ETL da BDGD roda offline (`py -m hub_municipios.etl_bdgd`), fora do Streamlit — ver
`README.md` do pacote.
"""

__version__ = "1.0.0"
