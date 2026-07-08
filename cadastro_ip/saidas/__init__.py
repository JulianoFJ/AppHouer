"""
Geradores dos 3 arquivos .xlsx de saída (seções 12.1, 12.2, 12.3 das instruções).

Cada gerador exporta `gerar(resultado)` que recebe o `ResultadoPipeline` e
retorna bytes prontos para download.
"""

from . import classificacao_pontos, analise_cadastro, quantitativo_uso_final  # noqa: F401
