"""
Cadastro ponto a ponto de iluminação pública a partir da BDGD da ANEEL.

O Hub de Municípios lê o agregado da BDGD — totais por município — e responde
"quantos pontos e quanta carga". Este pacote responde outra pergunta: **quais são os
pontos**, um a um, com coordenada, para que o sorteio de amostra de inspeção funcione
em município que não tem cadastro próprio.

Módulos:
    extracao    ETL offline (exige GDAL): .gdb -> dados/bdgd/cadastros/<ibge>.parquet
    vias_osm    logradouro e classe viária pelo OpenStreetMap, que a BDGD não tem
    classe_nbr  classe de iluminação M pela Tabela 1 da ABNT NBR 5101:2024
    montagem    junta tudo no formato que a página de amostragem consome
    caminhos    onde os artefatos ficam em disco

Nada aqui altera `hub_municipios` nem `amostragem_ip`: o pacote lê o agregado e reusa
a inferência de tecnologia do Hub, e entrega um DataFrame comum para a amostragem.
"""

__all__ = ["caminhos", "classe_nbr", "extracao", "montagem", "vias_osm"]
