"""
Cruza os pontos de um cadastro de IP com áreas do OpenStreetMap que costumam pedir
tratamento à parte na amostragem de campo: parque/praça, cemitério e campo/quadra/
ginásio/estádio. Também busca o contorno administrativo do município (uso só visual,
no mapa) — mesma fonte, mesmo cache, por isso vive no mesmo módulo.

Por que existe
---------------
Ponto de IP dentro de um parque ou de um campo de futebol pode ter acesso restrito
(portão fechado, horário de cemitério, propriedade de clube) e nem sempre representa a
iluminação viária que a NBR 5426 quer amostrar. O agente não decide sozinho o que
excluir — só identifica candidatos e devolve a lista para revisão humana antes do
sorteio (decisão de 08/09/2026: exclusão automática herdaria direto qualquer erro de
traçado do OSM para dentro do N usado na norma).

Fonte dos polígonos
--------------------
OpenStreetMap via Overpass API, mesmo serviço e mesmo padrão de cache/retry de
`cadastro_bdgd.vias_osm`. Precisão do traçado é a do voluntário que mapeou — testado em
São José da Lapa/MG em 08/09/2026: os polígonos batem com deslocamento de alguns metros
em relação ao cadastro real (fontes independentes, cada uma com seu próprio erro), o
que é esperado e não é bug de projeção.

`relation` multipolygon (contorno partido em mais de uma `way`, papel comum em
cemitério maior) é remontada por costura de vértice compartilhado — sem isso o teste
inicial ignorava exatamente essas geometrias e subestimava cemitério.
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from caminhos_dados import DADOS

CACHE_DIR = DADOS / "amostragem" / "areas_especiais"

OVERPASS_ESPELHOS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "Plataforma-IP/1.0 (amostragem de iluminacao publica - areas especiais)"

# Folga sobre o bbox dos pontos, para não cortar polígono que cruza a borda do recorte.
FOLGA_GRAUS = 0.003

# Filtros Overpass. Chave interna -> (categoria exibida ao usuário, filtro OverpassQL).
# Seis filtros de tag dobram em três categorias de UI porque é isso que o operador
# decide excluir ou não — a tag exata só importa para o link de conferência no OSM.
_FILTROS: dict[str, tuple[str, str]] = {
    "parque_praca": ("Parque/praça", "leisure~'^(park|garden|common|recreation_ground)$'"),
    "praca_place": ("Parque/praça", "place=square"),
    "cemiterio_landuse": ("Cemitério", "landuse=cemetery"),
    "cemiterio_amenity": ("Cemitério", "amenity=grave_yard"),
    "esporte_leisure": ("Campo/quadra/ginásio",
                         "leisure~'^(pitch|sports_centre|stadium|track|fitness_station)$'"),
    "esporte_building": ("Campo/quadra/ginásio", "building=sports_hall"),
}

CATEGORIAS = sorted({rotulo for rotulo, _ in _FILTROS.values()})


@dataclass
class Poligono:
    osm_id: int
    osm_tipo: str          # "way" ou "relation"
    categoria: str          # rótulo de UI (uma das CATEGORIAS)
    nome: str
    aneis_externos: list[list[tuple[float, float]]] = field(default_factory=list)
    aneis_internos: list[list[tuple[float, float]]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)   # min_lon, min_lat, max_lon, max_lat

    @property
    def url_osm(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_tipo}/{self.osm_id}"


# ── Download ─────────────────────────────────────────────────────────────────
def _consulta(bbox: tuple[float, float, float, float]) -> str:
    sul, oeste, norte, leste = bbox
    partes = [
        f"{elemento}[{filtro}]({sul},{oeste},{norte},{leste});"
        for _, filtro in _FILTROS.values()
        for elemento in ("way", "relation")
    ]
    return f"[out:json][timeout:90];({''.join(partes)});out geom;"


def _consultar_overpass(bbox: tuple[float, float, float, float],
                        usar_cache: bool = True) -> list[dict]:
    """
    Baixa os elementos do OSM dentro do bbox. Cache em disco por hash da consulta —
    IBGE nem sempre está disponível nesta página (cadastro pode vir de upload livre,
    sem o código do município), então a chave é o próprio bbox arredondado.
    """
    chave = hashlib.sha1(_consulta(bbox).encode("utf-8")).hexdigest()[:16]
    destino = CACHE_DIR / f"{chave}.json"
    if usar_cache and destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))

    consulta = _consulta(bbox)
    erro_final: Optional[Exception] = None
    # 3 rodadas pelos dois espelhos (6 tentativas), com espera crescente: o Overpass
    # público devolve 504 sob carga com frequência — testado ao vivo em 08/09/2026, só
    # respondeu na terceira rodada com essa mesma cadência. Uma tentativa por espelho,
    # como era antes, falhava rápido demais para um serviço de cortesia que às vezes só
    # precisa de mais alguns segundos de fôlego.
    rodadas = [(url, 15.0 * (r + 1)) for r in range(3) for url in OVERPASS_ESPELHOS]
    for tentativa, (url, espera) in enumerate(rodadas):
        try:
            resposta = requests.post(url, data={"data": consulta}, timeout=100,
                                     headers={"User-Agent": USER_AGENT})
            resposta.raise_for_status()
            elementos = resposta.json().get("elements", [])
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps(elementos), encoding="utf-8")
            return elementos
        except Exception as exc:                      # noqa: BLE001
            erro_final = exc
            if tentativa < len(rodadas) - 1:
                time.sleep(espera)
    # Mensagem técnica e só isso: quem chama (a página) é quem decide o tom de "está
    # tudo bem, é opcional" — duplicar essa frase aqui e na página produzia um aviso
    # repetido ("não foi possível... não foi possível... é opcional... é opcional...").
    raise RuntimeError(str(erro_final))


# ── Geometria ────────────────────────────────────────────────────────────────
def _categoria_de(tags: dict) -> Optional[str]:
    leisure = tags.get("leisure", "")
    if leisure in ("park", "garden", "common", "recreation_ground"):
        return "Parque/praça"
    if tags.get("place") == "square":
        return "Parque/praça"
    if tags.get("landuse") == "cemetery" or tags.get("amenity") == "grave_yard":
        return "Cemitério"
    if leisure in ("pitch", "sports_centre", "stadium", "track", "fitness_station"):
        return "Campo/quadra/ginásio"
    if tags.get("building") == "sports_hall":
        return "Campo/quadra/ginásio"
    return None


def _anel_de_geometria(geom: list[dict]) -> Optional[list[tuple[float, float]]]:
    if len(geom) < 4:
        return None
    if geom[0]["lat"] != geom[-1]["lat"] or geom[0]["lon"] != geom[-1]["lon"]:
        return None
    return [(g["lon"], g["lat"]) for g in geom]


def _pt(g: dict) -> tuple[float, float]:
    return (round(g["lat"], 9), round(g["lon"], 9))


def _montar_aneis_relation(members: list[dict], papel: str) -> list[list[tuple[float, float]]]:
    """
    Costura as `way` de um dado papel (outer/inner) pelos vértices que se tocam, até
    fechar anel(is). Uma `relation` de cemitério maior chega com o contorno partido em
    mais de uma `way` — sem essa costura, cada pedaço solto não fecha e é descartado.
    """
    segmentos = [[_pt(g) for g in (m.get("geometry") or [])]
                 for m in members if m.get("role") == papel and len(m.get("geometry") or []) >= 2]

    aneis: list[list[tuple[float, float]]] = []
    usados = [False] * len(segmentos)
    for i in range(len(segmentos)):
        if usados[i]:
            continue
        usados[i] = True
        anel = list(segmentos[i])
        mudou = True
        while anel[0] != anel[-1] and mudou:
            mudou = False
            for j, seg in enumerate(segmentos):
                if usados[j]:
                    continue
                if seg[0] == anel[-1]:
                    anel += seg[1:]
                    usados[j] = mudou = True
                    break
                if seg[-1] == anel[-1]:
                    anel += list(reversed(seg))[1:]
                    usados[j] = mudou = True
                    break
        if anel[0] == anel[-1] and len(anel) >= 4:
            aneis.append([(lon, lat) for lat, lon in anel])
    return aneis


def _bbox_de(aneis: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = [p[0] for anel in aneis for p in anel]
    ys = [p[1] for anel in aneis for p in anel]
    return min(xs), min(ys), max(xs), max(ys)


def montar_poligonos(elementos: list[dict]) -> list[Poligono]:
    poligonos: list[Poligono] = []
    for el in elementos:
        tags = el.get("tags", {}) or {}
        categoria = _categoria_de(tags)
        if categoria is None:
            continue
        nome = tags.get("name", "")

        if el["type"] == "way":
            anel = _anel_de_geometria(el.get("geometry") or [])
            if anel is None:
                continue
            aneis_externos = [anel]
            aneis_internos: list[list[tuple[float, float]]] = []
        elif el["type"] == "relation":
            aneis_externos = _montar_aneis_relation(el.get("members") or [], "outer")
            aneis_internos = _montar_aneis_relation(el.get("members") or [], "inner")
            if not aneis_externos:
                continue
        else:
            continue

        poligonos.append(Poligono(
            osm_id=el.get("id"), osm_tipo=el["type"], categoria=categoria, nome=nome,
            aneis_externos=aneis_externos, aneis_internos=aneis_internos,
            bbox=_bbox_de(aneis_externos),
        ))
    return poligonos


# ── Ponto-em-polígono (vetorizado) ──────────────────────────────────────────
def _dentro_do_anel(lon: np.ndarray, lat: np.ndarray,
                    anel: list[tuple[float, float]]) -> np.ndarray:
    """Ray casting padrão, vetorizado sobre todos os pontos para um anel por vez."""
    xs = np.array([p[0] for p in anel])
    ys = np.array([p[1] for p in anel])
    n = len(anel)
    dentro = np.zeros(len(lon), dtype=bool)
    x1, y1 = xs[0], ys[0]
    for i in range(1, n + 1):
        x2, y2 = xs[i % n], ys[i % n]
        cruza = (y1 > lat) != (y2 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersecao = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
        dentro ^= cruza & (lon < x_intersecao)
        x1, y1 = x2, y2
    return dentro


def _dentro_do_poligono(lon: np.ndarray, lat: np.ndarray, poligono: Poligono) -> np.ndarray:
    dentro = np.zeros(len(lon), dtype=bool)
    for anel in poligono.aneis_externos:
        dentro |= _dentro_do_anel(lon, lat, anel)
    for anel in poligono.aneis_internos:
        dentro &= ~_dentro_do_anel(lon, lat, anel)
    return dentro


def classificar_pontos(lat: pd.Series, lon: pd.Series,
                       poligonos: list[Poligono]) -> pd.DataFrame:
    """
    Para cada ponto (por posição, alinhado a `lat`/`lon`), devolve a categoria, o nome e
    a URL do primeiro polígono em que ele cai — ou tudo vazio se não cair em nenhum.
    Primeiro polígono que casa, não todos: dois polígonos sobrepostos (achado no teste
    de São José da Lapa — dois cemitérios mapeados quase no mesmo lugar) não devem
    contar o mesmo ponto duas vezes.
    """
    lon_arr = lon.to_numpy(dtype=float)
    lat_arr = lat.to_numpy(dtype=float)
    n = len(lon_arr)
    categoria = np.full(n, "", dtype=object)
    nome = np.full(n, "", dtype=object)
    url = np.full(n, "", dtype=object)
    resolvido = np.zeros(n, dtype=bool)

    valido = ~(np.isnan(lon_arr) | np.isnan(lat_arr))

    for poligono in poligonos:
        pendente = valido & ~resolvido
        if not pendente.any():
            break
        bx0, by0, bx1, by1 = poligono.bbox
        candidato = pendente & (lon_arr >= bx0) & (lon_arr <= bx1) & \
                    (lat_arr >= by0) & (lat_arr <= by1)
        if not candidato.any():
            continue
        achou = np.zeros(n, dtype=bool)
        achou[candidato] = _dentro_do_poligono(lon_arr[candidato], lat_arr[candidato], poligono)
        categoria[achou] = poligono.categoria
        nome[achou] = poligono.nome or f"(sem nome, {poligono.osm_tipo} {poligono.osm_id})"
        url[achou] = poligono.url_osm
        resolvido |= achou

    return pd.DataFrame({"_area_categoria": categoria, "_area_nome": nome, "_area_url": url})


# ── Entrada de alto nível ────────────────────────────────────────────────────
def buscar(lat: pd.Series, lon: pd.Series, usar_cache: bool = True) -> pd.DataFrame:
    """
    Ponto de entrada da página: bbox a partir das coordenadas válidas, consulta o OSM,
    classifica cada ponto. Levanta em caso de falha de rede — quem chama decide se isso
    é bloqueante (é opcional por design, então normalmente não deveria ser).
    """
    validos = lat.notna() & lon.notna()
    if not validos.any():
        return pd.DataFrame({"_area_categoria": [""] * len(lat), "_area_nome": [""] * len(lat),
                             "_area_url": [""] * len(lat)}, index=lat.index)

    bbox = (
        float(lat[validos].min()) - FOLGA_GRAUS, float(lon[validos].min()) - FOLGA_GRAUS,
        float(lat[validos].max()) + FOLGA_GRAUS, float(lon[validos].max()) + FOLGA_GRAUS,
    )
    elementos = _consultar_overpass(bbox, usar_cache=usar_cache)
    poligonos = montar_poligonos(elementos)
    resultado = classificar_pontos(lat, lon, poligonos)
    resultado.index = lat.index
    resultado.attrs["n_poligonos"] = len(poligonos)
    return resultado


# ── Contorno administrativo do município (uso visual, no mapa) ─────────────────
def _normalizar_nome(nome: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return sem_acento.strip().lower()


def buscar_contorno_municipio(lat: pd.Series, lon: pd.Series, nome_municipio: str = "",
                              usar_cache: bool = True) -> list[list[tuple[float, float]]]:
    """
    Contorno administrativo (`admin_level=8`, nível de município no Brasil no OSM) que
    contém o centro de massa do cadastro. Puramente visual — ajuda a enxergar no mapa
    se a amostra varreu o município inteiro ou ficou concentrada num canto. Não tem
    efeito nenhum sobre dimensionamento, sorteio ou exclusão: falha de rede, nenhum
    candidato ou nome que não bate devolvem lista vazia, nunca erro — o mapa funciona
    igual sem o contorno.
    """
    validos = lat.notna() & lon.notna()
    if not validos.any():
        return []
    centro_lat = float(lat[validos].median())
    centro_lon = float(lon[validos].median())

    # Folga de 0,6° de cada lado: município grande (interior de MG, por exemplo) passa
    # de 1° de ponta a ponta, e o candidato certo precisa estar inteiro no recorte para
    # o teste de ponto-dentro-do-anel funcionar.
    bbox = (centro_lat - 0.6, centro_lon - 0.6, centro_lat + 0.6, centro_lon + 0.6)
    consulta = (
        f"[out:json][timeout:60];"
        f"relation[boundary=administrative][admin_level=8]"
        f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out geom;"
    )
    chave = hashlib.sha1(consulta.encode("utf-8")).hexdigest()[:16]
    destino = CACHE_DIR / f"contorno_{chave}.json"

    try:
        if usar_cache and destino.exists():
            elementos = json.loads(destino.read_text(encoding="utf-8"))
        else:
            elementos = None
            for tentativa, url in enumerate(OVERPASS_ESPELHOS):
                try:
                    resposta = requests.post(url, data={"data": consulta}, timeout=90,
                                             headers={"User-Agent": USER_AGENT})
                    resposta.raise_for_status()
                    elementos = resposta.json().get("elements", [])
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    destino.write_text(json.dumps(elementos), encoding="utf-8")
                    break
                except Exception:                      # noqa: BLE001
                    if tentativa < len(OVERPASS_ESPELHOS) - 1:
                        time.sleep(2.0)
            if elementos is None:
                return []
    except Exception:                                  # noqa: BLE001
        return []

    candidatos: list[tuple[str, list[list[tuple[float, float]]]]] = []
    for el in elementos:
        if el.get("type") != "relation":
            continue
        aneis = _montar_aneis_relation(el.get("members") or [], "outer")
        if not aneis:
            continue
        candidatos.append(((el.get("tags") or {}).get("name", ""), aneis))

    if not candidatos:
        return []

    # Prioridade 1: nome bate com o que o operador escolheu no Passo 1 — evita pegar o
    # contorno do município vizinho quando o cadastro está perto da divisa.
    if nome_municipio:
        alvo = _normalizar_nome(nome_municipio)
        for nome_osm, aneis in candidatos:
            if _normalizar_nome(nome_osm) == alvo:
                return aneis

    # Prioridade 2: o candidato cujo interior realmente contém o centro do cadastro
    # (nome pode não bater por sigla/abreviação/divergência de grafia).
    for _, aneis in candidatos:
        if any(_dentro_do_anel(np.array([centro_lon]), np.array([centro_lat]), anel)[0]
               for anel in aneis):
            return aneis

    return []
