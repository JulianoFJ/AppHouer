"""
Amostragem para inspeção de campo de iluminação pública.

Recebe o cadastro de um município e devolve duas amostras disjuntas — medição
estrutural e medição de qualidade — dimensionadas pela ABNT NBR 5426 e sorteadas de
forma aleatória, porém com cobertura garantida de todas as classes de iluminação, das
vias estruturantes e da extensão territorial do município.

Uso típico (a página `paginas/amostragem_campo.py` é só a UI disso):

    from amostragem_ip import nbr5426, amostrador, relatorio
    from amostragem_ip.saidas import planilha_amostra

    plano = nbr5426.plano(len(cadastro), nivel="II", nqa=2.5)
    base, ressalvas = amostrador.preparar_base(cadastro, colunas)
    config = amostrador.ConfigAmostragem(tamanho_amostra=plano.tamanho_amostra)
    resultado = amostrador.sortear(base, config, plano=plano,
                                   municipio="Matozinhos", uf="MG",
                                   ressalvas_iniciais=ressalvas)

    xlsx_estrutural = planilha_amostra.gerar(resultado, amostrador.GRUPO_ESTRUTURAL)
    xlsx_qualidade = planilha_amostra.gerar(resultado, amostrador.GRUPO_QUALIDADE)
    texto = relatorio.gerar(resultado)
"""

from . import amostrador, nbr5426, relatorio, vias
from .amostrador import (
    GRUPO_ESTRUTURAL,
    GRUPO_QUALIDADE,
    ConfigAmostragem,
    ResultadoAmostragem,
    preparar_base,
    sortear,
)
from .nbr5426 import PlanoAmostragem

__all__ = [
    "amostrador", "nbr5426", "relatorio", "vias",
    "ConfigAmostragem", "ResultadoAmostragem", "PlanoAmostragem",
    "GRUPO_ESTRUTURAL", "GRUPO_QUALIDADE", "preparar_base", "sortear",
]
