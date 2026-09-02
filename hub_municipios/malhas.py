"""
Malhas municipais do IBGE, para o mapa do Hub.

Fonte: API de malhas territoriais do IBGE (servicodados.ibge.gov.br/api/v3/malhas).
Pede-se a malha de UMA UF por vez, com a subdivisão municipal e qualidade mínima —
são ~100 a 800 polígonos e algumas centenas de KB, contra 3,6 MB da malha nacional
inteira. Carregar o país todo num choropleth de 5.570 polígonos trava o navegador
sem entregar nada que o recorte por UF não entregue melhor.

A propriedade `codarea` de cada feição é o código IBGE de 7 dígitos, que casa direto
com `codigo_municipio` do agregado da BDGD e com o `cod_ibge` do SICONFI.

As malhas são cacheadas em disco (`dados/malhas/`), então a rede é usada uma vez por UF.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from . import config

URL = ("https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}"
       "?formato=application/vnd.geo+json&intrarregiao=municipio&qualidade=minima")
# Mesma API sem `intrarregiao`: devolve o contorno do estado como UM polígono, em vez
# das centenas de polígonos municipais. MG sai com 372 vértices em 8 KB — é o que
# permite desenhar o mapa em vetor no PowerPoint sem empilhar 800 formas no slide.
URL_CONTORNO = ("https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}"
                "?formato=application/vnd.geo+json&qualidade=minima")
TIMEOUT = 60

_MEMORIA: Dict[str, Dict[str, Any]] = {}


def _caminho(uf: str):
    return config.DADOS / "malhas" / f"municipios_{uf.upper()}.geojson"


def carregar(uf: str) -> Optional[Dict[str, Any]]:
    """
    GeoJSON dos municípios da UF. Devolve None se não houver malha nem rede — o mapa
    é um recurso adicional, e a falta dele não pode derrubar o resto da página.
    """
    uf = uf.strip().upper()
    if uf in _MEMORIA:
        return _MEMORIA[uf]

    destino = _caminho(uf)
    if destino.exists():
        try:
            with open(destino, "r", encoding="utf-8") as fh:
                malha = json.load(fh)
            _MEMORIA[uf] = malha
            return malha
        except Exception:
            pass

    try:
        resp = requests.get(URL.format(uf=uf), timeout=TIMEOUT)
        resp.raise_for_status()
        malha = resp.json()
    except Exception:
        return None

    if not malha.get("features"):
        return None

    # o id da feição é o que o plotly casa com `locations`
    for feicao in malha["features"]:
        feicao["id"] = feicao.get("properties", {}).get("codarea")

    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "w", encoding="utf-8") as fh:
            json.dump(malha, fh)
    except Exception:
        pass

    _MEMORIA[uf] = malha
    return malha


def _caminho_contorno(uf: str):
    return config.DADOS / "malhas" / f"contorno_{uf.upper()}.geojson"


def carregar_contorno_uf(uf: str) -> Optional[Dict[str, Any]]:
    """
    Contorno do estado como um único polígono, cacheado em disco.

    Usado pelo gerador de apresentação para desenhar o mapa de localização em vetor.
    Como toda a família de malhas, devolve None em vez de levantar: mapa é adorno,
    e a falta de rede não pode impedir a geração do arquivo.
    """
    uf = uf.strip().upper()
    chave = f"_contorno_{uf}"
    if chave in _MEMORIA:
        return _MEMORIA[chave]

    destino = _caminho_contorno(uf)
    if destino.exists():
        try:
            with open(destino, "r", encoding="utf-8") as fh:
                malha = json.load(fh)
            _MEMORIA[chave] = malha
            return malha
        except Exception:
            pass

    try:
        resp = requests.get(URL_CONTORNO.format(uf=uf), timeout=TIMEOUT)
        resp.raise_for_status()
        malha = resp.json()
    except Exception:
        return None

    if not malha.get("features"):
        return None

    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "w", encoding="utf-8") as fh:
            json.dump(malha, fh)
    except Exception:
        pass

    _MEMORIA[chave] = malha
    return malha


def geometria_do_municipio(uf: str, cod_ibge: str) -> Optional[Dict[str, Any]]:
    """Geometria GeoJSON de um município dentro da malha da sua UF, ou None."""
    malha = carregar(uf)
    if not malha:
        return None
    alvo = str(cod_ibge).strip()
    for feicao in malha.get("features", []):
        if str(feicao.get("id") or "").strip() == alvo:
            return feicao.get("geometry")
    return None


def codigos_da_malha(malha: Dict[str, Any]) -> set:
    return {f.get("id") for f in malha.get("features", []) if f.get("id")}


def _extremos(malha: Dict[str, Any]):
    """Bounding box da malha, percorrendo as coordenadas do GeoJSON."""
    lons, lats = [], []

    def percorrer(coords):
        if not coords:
            return
        if isinstance(coords[0], (int, float)) and len(coords) >= 2:
            lons.append(coords[0])
            lats.append(coords[1])
            return
        for c in coords:
            percorrer(c)

    for feicao in malha.get("features", []):
        percorrer((feicao.get("geometry") or {}).get("coordinates"))

    if not lons or not lats:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def centro_aproximado(malha: Dict[str, Any]) -> Dict[str, float]:
    """Centro do bounding box, para posicionar o mapa na UF."""
    caixa = _extremos(malha)
    if not caixa:
        return {"lat": -15.8, "lon": -47.9}      # Brasília, fallback
    lon_min, lat_min, lon_max, lat_max = caixa
    return {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}


def zoom_aproximado(malha: Dict[str, Any]) -> float:
    """
    Zoom que enquadra a UF. O `fitbounds` não existe no mapa baseado em tiles, então o
    zoom é derivado da maior dimensão do bounding box — de São Paulo (~9°) a Amazonas
    (~20°) a diferença de enquadramento é grande o bastante para importar.
    """
    caixa = _extremos(malha)
    if not caixa:
        return 4.0
    lon_min, lat_min, lon_max, lat_max = caixa
    extensao = max(lon_max - lon_min, lat_max - lat_min)
    for limite, zoom in ((2, 8.0), (4, 7.0), (6, 6.5), (9, 6.0), (13, 5.5), (18, 5.0)):
        if extensao <= limite:
            return zoom
    return 4.5
