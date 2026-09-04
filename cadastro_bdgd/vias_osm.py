"""
Logradouro e classe viária a partir do OpenStreetMap, para cadastro que não os tem.

O problema
----------
A entidade PIP da BDGD é uma tabela sem geometria e sem endereço: tem município,
carga, tipo e potência de lâmpada, e nada mais que localize o ponto na cidade. A
coordenada é recuperável pelo ponto notável da rede (ver `extracao.py`), mas nome de
rua e classe viária não existem em lugar nenhum da base — e são exatamente as duas
colunas de que o sorteio da amostra depende para garantir cobertura de avenida e
rodovia.

A solução aqui é casar cada ponto com a via mais próxima do OSM. Medido em Ponta
Porã/MS (10.590 pontos) em 04/09/2026: 4,8 s para baixar a malha, 0,4 s para casar,
79% dos pontos ganharam nome de logradouro e 100% ganharam hierarquia funcional.

O que isso é e o que não é
--------------------------
É uma **atribuição por proximidade**, não um cadastro. A distância mediana entre o
ponto e o eixo da via ficou em 24,7 m (p90 = 61 m), o que basta para dizer "este ponto
pertence à Avenida Brasil" e não basta para dizer "fica no número 250". Em esquina, em
canteiro central e em ponto de praça o vizinho mais próximo pode ser a via errada.

Por isso toda coluna produzida aqui sai marcada como inferida, e a distância do
casamento vai junto: é ela que permite descartar o que ficou longe demais.

Densificação em vez de projeção
-------------------------------
A distância correta é do ponto ao SEGMENTO, não ao vértice, e o vizinho mais próximo
sobre vértices superestima quando a via é desenhada com poucos nós. Em vez de projetar
ponto em segmento (que não vetoriza bem), os segmentos são densificados a cada
`PASSO_DENSIFICACAO_M`; o erro que sobra é de metade do passo, conhecido e pequeno
perto da incerteza de 25 m do próprio casamento.

Sem dependência nova: `BallTree` com métrica haversine vem do scikit-learn, que já
está fixado no `requirements.txt` por causa dos modelos de simulação.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from . import caminhos
from .classe_nbr import ClasseEstimada, e_pedonal, estimar_classe_m

RAIO_TERRA_M = 6_371_000.0


def _milhar(n: int) -> str:
    """1234567 -> '1.234.567', sem tocar no resto da frase."""
    return f"{n:,}".replace(",", ".")

# Hierarquias baixadas do OSM. `service` entra porque em muitos municípios a via
# interna de conjunto habitacional é mapeada assim e tem ponto de IP; `footway` e
# `cycleway` ficam de fora do casamento de via motorizada — poste de praça casaria com
# a trilha em vez da rua.
HIERARQUIAS_VIARIAS = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
]

PASSO_DENSIFICACAO_M = 10.0
# Acima disto o casamento é considerado duvidoso e o logradouro não é atribuído. 150 m
# é largo de propósito: em área rural o poste fica mesmo longe do eixo mapeado, e
# descartar cedo demais devolveria menos cobertura que o sorteio precisa.
DISTANCIA_MAXIMA_M = 150.0

OVERPASS = "https://overpass-api.de/api/interpreter"
# A instância principal do Overpass devolve 429/504 sob carga; a de Kumi Systems é o
# espelho recomendado pela própria wiki do projeto.
OVERPASS_ESPELHOS = [OVERPASS, "https://overpass.kumi.systems/api/interpreter"]
USER_AGENT = "Plataforma-IP/1.0 (amostragem de iluminacao publica)"


@dataclass
class MalhaViaria:
    """Vias do município, já densificadas e indexadas para busca espacial."""

    vertices: np.ndarray            # (n, 2) em RADIANOS, [lat, lon]
    via_do_vertice: np.ndarray      # (n,) índice da via de cada vértice
    tags: list[dict]                # tags OSM de cada via
    intersecoes_por_km: list[float]
    arvore: BallTree
    de_cache: bool = False

    @property
    def total_vias(self) -> int:
        return len(self.tags)


# ── Download ─────────────────────────────────────────────────────────────────

def _consulta(bbox: tuple[float, float, float, float]) -> str:
    sul, oeste, norte, leste = bbox
    filtro = "|".join(HIERARQUIAS_VIARIAS)
    return (f"[out:json][timeout:180];"
            f"way[highway~'^({filtro})$']({sul},{oeste},{norte},{leste});"
            f"out tags geom;")


def baixar_vias(bbox: tuple[float, float, float, float],
                codigo_ibge: str,
                usar_cache: bool = True) -> tuple[list[dict], bool]:
    """
    Baixa as vias do OSM dentro do bbox. Devolve (ways, veio_do_cache).

    O cache é por município e fica em disco porque a malha não muda de uma sessão para
    outra e o Overpass é um serviço público de cortesia — repetir a mesma consulta a
    cada rerun do Streamlit seria abuso, além de lento.
    """
    destino = caminhos.OSM_CACHE / f"{codigo_ibge}.json"
    if usar_cache and destino.exists():
        with destino.open(encoding="utf-8") as f:
            return json.load(f), True

    consulta = _consulta(bbox)
    erro_final: Optional[Exception] = None
    for tentativa, url in enumerate(OVERPASS_ESPELHOS):
        try:
            resposta = requests.post(url, data={"data": consulta}, timeout=240,
                                     headers={"User-Agent": USER_AGENT})
            resposta.raise_for_status()
            ways = resposta.json().get("elements", [])
            destino.parent.mkdir(parents=True, exist_ok=True)
            with destino.open("w", encoding="utf-8") as f:
                json.dump(ways, f)
            return ways, False
        except Exception as exc:                       # noqa: BLE001
            erro_final = exc
            if tentativa < len(OVERPASS_ESPELHOS) - 1:
                time.sleep(2.0)
    raise RuntimeError(
        f"Não foi possível consultar o OpenStreetMap: {erro_final}. "
        "O enriquecimento é opcional — o cadastro pode ser gerado sem logradouro."
    )


# ── Geometria ────────────────────────────────────────────────────────────────

def _densificar(pontos: list[dict], passo_m: float) -> list[tuple[float, float]]:
    """Insere vértices ao longo de cada segmento, a cada `passo_m`."""
    saida: list[tuple[float, float]] = []
    for anterior, atual in zip(pontos, pontos[1:]):
        lat1, lon1 = anterior["lat"], anterior["lon"]
        lat2, lon2 = atual["lat"], atual["lon"]
        saida.append((lat1, lon1))
        # Aproximação plana: em escala de rua o erro é irrelevante e evita trigonometria
        # esférica dentro do laço, que domina o custo em município de 100 mil pontos.
        dy = (lat2 - lat1) * 111_320.0
        dx = (lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
        comprimento = math.hypot(dx, dy)
        n = int(comprimento // passo_m)
        for k in range(1, n + 1):
            t = k / (n + 1)
            saida.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
    if pontos:
        saida.append((pontos[-1]["lat"], pontos[-1]["lon"]))
    return saida


def _comprimento_km(pontos: list[dict]) -> float:
    total = 0.0
    for a, b in zip(pontos, pontos[1:]):
        dy = (b["lat"] - a["lat"]) * 111.32
        dx = (b["lon"] - a["lon"]) * 111.32 * math.cos(math.radians(a["lat"]))
        total += math.hypot(dx, dy)
    return total


def _densidade_intersecoes(ways: list[dict]) -> list[float]:
    """
    Interseções por km de cada via — parâmetro da Tabela 1 da NBR 5101:2024.

    Duas vias se cruzam quando compartilham uma coordenada. O OSM não devolve o id dos
    nós em `out geom`, então a contagem usa a própria coordenada arredondada como
    chave: um cruzamento real é o mesmo nó nas duas ways, com lat/lon idênticos.
    """
    contagem: Counter = Counter()
    chaves_por_via: list[list[tuple]] = []
    for w in ways:
        chaves = [(round(g["lat"], 7), round(g["lon"], 7)) for g in w.get("geometry", [])]
        chaves_por_via.append(chaves)
        for c in set(chaves):
            contagem[c] += 1

    densidades = []
    for w, chaves in zip(ways, chaves_por_via):
        km = _comprimento_km(w.get("geometry", []))
        cruzamentos = sum(1 for c in set(chaves) if contagem[c] > 1)
        densidades.append(cruzamentos / km if km > 0.05 else 0.0)
    return densidades


def montar_malha(ways: list[dict], de_cache: bool = False) -> MalhaViaria:
    """Densifica as vias e monta o índice espacial."""
    lat, lon, indice = [], [], []
    tags, uteis = [], []
    for w in ways:
        geom = w.get("geometry") or []
        if len(geom) < 2:
            continue
        i = len(tags)
        tags.append(w.get("tags", {}) or {})
        uteis.append(w)
        for plat, plon in _densificar(geom, PASSO_DENSIFICACAO_M):
            lat.append(plat)
            lon.append(plon)
            indice.append(i)

    if not lat:
        raise ValueError("O OpenStreetMap não devolveu nenhuma via utilizável para "
                         "este município.")

    vertices = np.radians(np.c_[np.array(lat), np.array(lon)])
    return MalhaViaria(
        vertices=vertices,
        via_do_vertice=np.array(indice, dtype=np.int32),
        tags=tags,
        intersecoes_por_km=_densidade_intersecoes(uteis),
        arvore=BallTree(vertices, metric="haversine"),
        de_cache=de_cache,
    )


# ── Casamento ────────────────────────────────────────────────────────────────

def casar(df: pd.DataFrame, malha: MalhaViaria,
          col_lat: str = "latitude", col_lon: str = "longitude",
          col_urbano: str = "area_urbana") -> pd.DataFrame:
    """
    Anexa logradouro, hierarquia e classe NBR estimada a cada ponto do cadastro.

    Colunas acrescentadas:
        `logradouro`            nome OSM da via mais próxima ("" quando a via não tem
                                nome ou o casamento ficou além de DISTANCIA_MAXIMA_M)
        `hierarquia_osm`        motorway/trunk/primary/… da via casada
        `classe_via`            classe M da NBR 5101:2024 estimada (ver `classe_nbr`)
        `dist_via_m`            distância ao eixo, para conferência e descarte
        `metodo_classe`         como a classe foi obtida, para o relatório

    Pontos sem coordenada saem com tudo vazio, sem exceção: já é assim que o resto do
    pipeline trata coordenada irrecuperável (`amostragem_ip.amostrador`).
    """
    resultado = df.copy()
    for coluna in ("logradouro", "hierarquia_osm", "classe_via", "metodo_classe"):
        resultado[coluna] = ""
    resultado["dist_via_m"] = np.nan

    tem_coord = resultado[col_lat].notna() & resultado[col_lon].notna()
    if not tem_coord.any():
        return resultado

    pontos = np.radians(resultado.loc[tem_coord, [col_lat, col_lon]].to_numpy(float))
    distancia, vizinho = malha.arvore.query(pontos, k=1)
    distancia_m = distancia[:, 0] * RAIO_TERRA_M
    via = malha.via_do_vertice[vizinho[:, 0]]

    urbano = (resultado.loc[tem_coord, col_urbano].tolist()
              if col_urbano in resultado.columns
              else [None] * int(tem_coord.sum()))

    # A classe depende das tags da via e do urbano/rural do PONTO, então o cache é por
    # (via, urbano) — em município de 100 mil pontos isso troca 100 mil avaliações da
    # Tabela 1 por algumas milhares.
    cache: dict[tuple[int, Optional[bool]], ClasseEstimada] = {}
    logradouros, hierarquias, classes, origens = [], [], [], []

    for k, indice_via in enumerate(via):
        tags = malha.tags[indice_via]
        longe = distancia_m[k] > DISTANCIA_MAXIMA_M

        if longe or e_pedonal(tags):
            logradouros.append("")
            hierarquias.append("")
            classes.append("")
            origens.append("fora do alcance do casamento" if longe else "via pedonal")
            continue

        logradouros.append(str(tags.get("name", "")).strip())
        hierarquias.append(str(tags.get("highway", "")).strip())

        chave = (int(indice_via), urbano[k])
        if chave not in cache:
            cache[chave] = estimar_classe_m(
                tags,
                area_urbana=urbano[k],
                intersecoes_por_km=malha.intersecoes_por_km[indice_via],
            )
        estimada = cache[chave]
        classes.append(estimada.classe)
        origens.append(f"NBR 5101:2024 Tabela 1 (V_PS = {estimada.soma_ponderacao:+.1f})")

    resultado.loc[tem_coord, "logradouro"] = logradouros
    resultado.loc[tem_coord, "hierarquia_osm"] = hierarquias
    resultado.loc[tem_coord, "classe_via"] = classes
    resultado.loc[tem_coord, "metodo_classe"] = origens
    resultado.loc[tem_coord, "dist_via_m"] = distancia_m.round(1)
    return resultado


def enriquecer(df: pd.DataFrame, codigo_ibge: str, usar_cache: bool = True,
               col_lat: str = "latitude", col_lon: str = "longitude") -> tuple[pd.DataFrame, list[str]]:
    """
    Caminho completo: bbox do cadastro -> OSM -> casamento. Devolve (df, ressalvas).
    """
    ressalvas: list[str] = []
    validos = df[[col_lat, col_lon]].dropna()
    if validos.empty:
        return df, ["Nenhum ponto com coordenada: o enriquecimento pelo OSM foi pulado."]

    # Margem de 0,01° (~1,1 km) para que ponto na borda ainda encontre a via dele.
    bbox = (validos[col_lat].min() - 0.01, validos[col_lon].min() - 0.01,
            validos[col_lat].max() + 0.01, validos[col_lon].max() + 0.01)

    ways, do_cache = baixar_vias(bbox, codigo_ibge, usar_cache=usar_cache)
    malha = montar_malha(ways, de_cache=do_cache)
    enriquecido = casar(df, malha, col_lat=col_lat, col_lon=col_lon)

    com_nome = int((enriquecido["logradouro"] != "").sum())
    total = len(enriquecido)
    distancias = enriquecido["dist_via_m"].dropna()

    # O milhar é formatado número a número. Aplicar `.replace(",", ".")` na frase
    # inteira — atalho comum no resto do projeto — trocaria também a vírgula do texto.
    ressalvas.append(
        f"Logradouro e classe viária vieram do OpenStreetMap por proximidade, não do "
        f"cadastro: {_milhar(com_nome)} de {_milhar(total)} pontos "
        f"({com_nome / total:.0%}) receberam nome de via."
    )
    if not distancias.empty:
        ressalvas.append(
            f"Distância do ponto ao eixo da via casada: mediana {distancias.median():.0f} m, "
            f"p90 {distancias.quantile(0.90):.0f} m. Serve para atribuir a via, não o "
            f"endereço — confira antes de mandar a equipe a um ponto específico."
        )
    longe = int((enriquecido["metodo_classe"] == "fora do alcance do casamento").sum())
    if longe:
        ressalvas.append(
            f"{_milhar(longe)} {'ponto ficou' if longe == 1 else 'pontos ficaram'} a mais "
            f"de {DISTANCIA_MAXIMA_M:.0f} m de qualquer via mapeada e "
            f"{'continuou' if longe == 1 else 'continuaram'} sem logradouro e sem classe."
        )
    ressalvas.append(
        "A classe segue o método da Tabela 1 da NBR 5101:2024, mas volume de tráfego e "
        "qualidade da sinalização não existem em base pública: o volume foi inferido da "
        "hierarquia viária e a sinalização entrou no valor neutro. É estimativa para "
        "estratificar amostra, não enquadramento normativo."
    )
    if do_cache:
        ressalvas.append("Malha viária lida do cache local, sem consulta ao OpenStreetMap.")
    return enriquecido, ressalvas


__all__ = [
    "HIERARQUIAS_VIARIAS", "DISTANCIA_MAXIMA_M", "PASSO_DENSIFICACAO_M",
    "MalhaViaria", "baixar_vias", "montar_malha", "casar", "enriquecer",
]
