"""
Sem `__init__.py` em `tests/`, o pytest usa import "rootless" e só põe a pasta de cada
arquivo de teste no `sys.path` — não a raiz do repositório. Sem isto, `import db` e
`from acesso import ...` (que assumem a raiz como root de import, mesmo padrão de
`scratch/test_full.py`) não resolveriam.

Os testes rodam contra o Postgres real do docker-compose (`docker compose exec app
pytest`), não um mock — mesmo padrão de verificação usado no resto do projeto. Cada
teste que grava dado usa um identificador único (`pytest_<hex>`) e limpa depois de si.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
