"""
Pacote `premissas_ip` — coleta neutra de premissas de IP por município e geração
das saídas (planilha de inputs parametrizada + blocos do relatório de engenharia).

- `schema`   : catálogo neutro das 32 seções de premissas (estrutura fixa, valores por município).
- `modelo`   : relações de dimensionamento e construtores de fórmula Excel.
- `saidas`   : geradores de .xlsx/.md (planilha de inputs, blocos do relatório).

A UI (wizard) vive em `paginas/premissas_ip.py`; este pacote contém só a lógica.
"""

from . import schema  # noqa: F401

__all__ = ["schema"]
