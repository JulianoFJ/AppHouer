"""Previsão de métricas luminotécnicas por ML — carregamento de modelos, regras de
engenharia (braço/CPE), formatação de saídas e geração de relatório PDF.

Extraído de `paginas/simulacao_nbr.py` (a página ficava com toda a lógica inline,
único caso no app sem pacote de domínio próprio). Zero dependência de Streamlit,
como os demais pacotes de domínio — cache (`st.cache_data`/`st.cache_resource`) fica
só na página."""
