"""
Onde os cadastros ficam em disco — e por que em dois lugares.

Há duas raízes, procuradas nesta ordem:

1. **`cadastro_bdgd/data/`, dentro do pacote e versionada.** É a única que existe no
   Streamlit Cloud, que clona o repositório e não tem o `.gdb` nem o GDAL. Município
   que precisa estar no ar entra aqui.
2. **`dados/bdgd/cadastros/`, fora do repositório.** É onde o ETL grava por padrão, e
   serve para trabalho local: município em estudo, teste, base recém-processada que
   ainda não se decidiu publicar.

O ETL grava em (2); `publicar()` move para (1) quando o município deve ir ao ar. A
separação é deliberada: versionar tudo que se extrai encheria o repositório de
município que ninguém vai abrir, e gravar direto no pacote faria o `git status` acusar
arquivo novo a cada teste.

Quanto cabe
-----------
Com o esquema enxuto (`extracao.COLUNAS_PUBLICADAS`, dtypes reduzidos, compressão
zstd) o cadastro custa **~13 bytes por ponto**: 139 KB para os 10.590 pontos de Ponta
Porã, 47 KB no município mediano. Daí MG inteiro dar ~33 MB e SP ~50 MB — ambos cabem
num repositório que hoje tem 73 MB. O Brasil inteiro daria ~253 MB, o que já pesaria em
cada clone do Cloud: para essa escala o caminho é hospedagem externa, não git.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hub_municipios.config import DADOS

PACOTE = Path(__file__).resolve().parent

# (1) Versionada — vai para o Cloud.
CADASTROS_PUBLICADOS = PACOTE / "data" / "cadastros"

# (2) Local — fora do repositório, como as bases brutas da BDGD.
CADASTROS_LOCAIS = DADOS / "bdgd" / "cadastros"

# Resposta bruta do Overpass. Fica sempre fora do repositório: no Cloud o disco é
# efêmero e o cache se refaz em ~5 s por município, o que não justifica versionar.
OSM_CACHE = DADOS / "bdgd" / "osm"


def garantir_pastas() -> None:
    for pasta in (CADASTROS_PUBLICADOS, CADASTROS_LOCAIS, OSM_CACHE):
        pasta.mkdir(parents=True, exist_ok=True)


def caminho_cadastro(codigo_ibge: str) -> Path | None:
    """O cadastro deste município, publicado ou local. None se não houver."""
    for raiz in (CADASTROS_PUBLICADOS, CADASTROS_LOCAIS):
        candidato = raiz / f"{codigo_ibge}.parquet"
        if candidato.exists():
            return candidato
    return None


def caminho_para_gravar(codigo_ibge: str, publicar: bool = False) -> Path:
    """Onde o ETL grava. Por padrão na raiz local, fora do repositório."""
    raiz = CADASTROS_PUBLICADOS if publicar else CADASTROS_LOCAIS
    raiz.mkdir(parents=True, exist_ok=True)
    return raiz / f"{codigo_ibge}.parquet"


def publicar(codigo_ibge: str) -> Path:
    """
    Move o cadastro local para a pasta versionada, para o município ir ao ar.

    Levanta `FileNotFoundError` se o município não foi extraído — publicar o que não
    existe deixaria a página oferecendo um item que não abre.
    """
    origem = CADASTROS_LOCAIS / f"{codigo_ibge}.parquet"
    if not origem.exists():
        raise FileNotFoundError(
            f"{codigo_ibge} não foi extraído ainda. Rode "
            f"`py -m cadastro_bdgd.extracao {codigo_ibge}` antes de publicar."
        )
    CADASTROS_PUBLICADOS.mkdir(parents=True, exist_ok=True)
    destino = CADASTROS_PUBLICADOS / f"{codigo_ibge}.parquet"
    shutil.copy2(origem, destino)
    return destino


def municipios_disponiveis() -> list[str]:
    """Códigos IBGE com cadastro em qualquer uma das duas raízes, sem repetir."""
    codigos: set[str] = set()
    for raiz in (CADASTROS_PUBLICADOS, CADASTROS_LOCAIS):
        if raiz.exists():
            codigos.update(p.stem for p in raiz.glob("*.parquet"))
    return sorted(codigos)


def esta_publicado(codigo_ibge: str) -> bool:
    return (CADASTROS_PUBLICADOS / f"{codigo_ibge}.parquet").exists()


__all__ = [
    "CADASTROS_PUBLICADOS", "CADASTROS_LOCAIS", "OSM_CACHE", "garantir_pastas",
    "caminho_cadastro", "caminho_para_gravar", "publicar", "municipios_disponiveis",
    "esta_publicado",
]
