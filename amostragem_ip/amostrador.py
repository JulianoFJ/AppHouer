"""
Sorteio da amostra de inspeção de campo — aleatório, mas com abrangência garantida.

O problema com amostra puramente aleatória em cadastro de IP é conhecido: o parque é
dominado por via local (tipicamente 70–85% dos pontos), então um sorteio simples
devolve quase só rua residencial e a inspeção volta sem nenhuma medição em avenida,
rodovia ou área de conflito — justamente onde a NBR 5101 é mais exigente e onde o
risco de não conformidade tem consequência contratual. Por outro lado, amostra
escolhida "a dedo" perde a aleatoriedade e não sustenta inferência estatística sobre
o parque nem defesa perante banca.

A solução implementada aqui é amostragem **estratificada com cotas de cobertura e
dispersão espacial**, em três camadas:

  1. **Cotas obrigatórias** — pelo menos 1 ponto de cada classe de iluminação e pelo
     menos 1 ponto em cada via estruturante (avenida/rodovia/anel/marginal/estrada ou
     via de classe exigente). Ao sortear a cota de uma classe, prefere-se um ponto que
     também cubra uma via principal ainda descoberta, para não gastar amostra à toa.
  2. **Alocação proporcional** — o restante é distribuído entre as classes na proporção
     do parque (método do maior resto), preservando a estrutura do município.
  3. **Dispersão espacial** — dentro de cada estrato o sorteio é espacialmente
     balanceado: k-means sobre as coordenadas com k = número de pontos a sortear e um
     ponto aleatório por cluster. Isso espalha a amostra pela mancha urbana em vez de
     concentrá-la onde há mais pontos, sem tirar a aleatoriedade — quem é sorteado
     dentro de cada cluster continua sendo decidido pelo gerador.

As duas amostras (estrutural e qualidade) são **disjuntas** e cada uma cumpre a
cobertura de forma independente, conforme decidido em 01/09/2026: a frente de medição
luminotécnica precisa ter uma via de cada classe para confrontar com a NBR 5101, e a
frente de levantamento estrutural precisa representar o parque inteiro.

Todo o sorteio é reprodutível pela semente — requisito de auditoria: o mesmo cadastro
com a mesma semente devolve exatamente a mesma amostra, e o número fica registrado no
relatório e na planilha.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .nbr5426 import PlanoAmostragem
from .vias import (
    ROTULO_SEM_CLASSE,
    ViaPrincipal,
    identificar_vias_principais,
    normalizar_classe,
    rank_exigencia,
    tipo_via,
)


# Faixa plausível de coordenadas em território brasileiro. Cadastro municipal traz com
# frequência lat/long trocados, zerados ou em projeção UTM — coordenada fora daqui é
# tratada como ausente em vez de arrastar o k-means para o meio do oceano.
LAT_MIN, LAT_MAX = -34.0, 6.0
LON_MIN, LON_MAX = -74.5, -33.0

GRUPO_ESTRUTURAL = "estrutural"
GRUPO_QUALIDADE = "qualidade"

COLUNAS_AUXILIARES = [
    "_id", "_logradouro", "_bairro", "_classe", "_lat", "_lon",
    "_chave_logradouro", "_chave_via", "_tipo_via", "_tem_coord", "_grupo",
]


@dataclass
class ConfigAmostragem:
    """Parâmetros do sorteio. Todos expostos na UI."""

    tamanho_amostra: int
    proporcao_estrutural: float = 0.60
    cobertura_por_classe: bool = True
    cobertura_vias_principais: bool = True
    vias_obrigatorias: list[str] | None = None   # chaves de logradouro; None = automático
    teto_vias_principais: int = 20
    dispersao_espacial: bool = True
    semente: int = 2026

    @property
    def proporcao_qualidade(self) -> float:
        return 1.0 - self.proporcao_estrutural


@dataclass
class ResultadoAmostragem:
    """Amostra sorteada + tudo que o relatório e o mapa precisam para prestar contas."""

    base: pd.DataFrame                    # cadastro preparado, com a coluna `_grupo`
    estrutural: pd.DataFrame
    qualidade: pd.DataFrame
    config: ConfigAmostragem
    vias_principais: list[ViaPrincipal]
    cobertura_classes: pd.DataFrame
    cobertura_vias: pd.DataFrame
    abrangencia: dict
    ressalvas: list[str] = field(default_factory=list)
    plano: PlanoAmostragem | None = None
    municipio: str = ""
    uf: str = ""

    @property
    def total_amostra(self) -> int:
        return len(self.estrutural) + len(self.qualidade)

    @property
    def total_parque(self) -> int:
        return len(self.base)


# ── Preparação da base ────────────────────────────────────────────────────────
def preparar_base(df: pd.DataFrame, colunas: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Normaliza o cadastro para o sorteio, criando as colunas auxiliares com prefixo `_`.

    Args:
        df: cadastro do município, como veio da planilha.
        colunas: conceito → nome real da coluna. Conceitos usados: `id_ponto`,
            `logradouro`, `bairro`, `classe_via`, `latitude`, `longitude`.
            Todos são opcionais exceto na prática `classe_via` (sem ela não há
            estratificação por classe) e `logradouro` (sem ele não há via principal).

    Returns:
        (base preparada, lista de ressalvas encontradas na preparação)
    """
    from cadastro_ip.normalizacao import chave_logradouro, limpar_id_serie

    base = df.copy().reset_index(drop=True)
    ressalvas: list[str] = []

    col_id = colunas.get("id_ponto")
    if col_id and col_id in base.columns:
        base["_id"] = limpar_id_serie(base[col_id])
        duplicados = int(base["_id"].duplicated().sum())
        if duplicados:
            ressalvas.append(
                f"{duplicados} identificadores repetidos na coluna `{col_id}` — o sorteio "
                "trata cada linha como um ponto, mas o cadastro deveria ter ID único."
            )
    else:
        base["_id"] = (base.index + 1).astype(str)
        ressalvas.append(
            "Nenhuma coluna de identificador foi indicada — a amostra usa o número da "
            "linha do cadastro como ID, o que dificulta o retorno da equipe de campo."
        )

    col_log = colunas.get("logradouro")
    base["_logradouro"] = base[col_log].astype(str).str.strip() if col_log in base.columns else ""

    col_bairro = colunas.get("bairro")
    base["_bairro"] = base[col_bairro].astype(str).str.strip() if col_bairro in base.columns else ""

    if col_log in base.columns and col_bairro in base.columns:
        base["_chave_logradouro"] = [
            chave_logradouro(log, bai) for log, bai in zip(base["_logradouro"], base["_bairro"])
        ]
    elif col_log in base.columns:
        base["_chave_logradouro"] = base["_logradouro"].map(chave_logradouro)
    else:
        base["_chave_logradouro"] = ""
        ressalvas.append(
            "Sem coluna de logradouro: não é possível garantir cobertura das avenidas e "
            "rodovias — a amostra fica com abrangência apenas por classe e geográfica."
        )

    # Chave da VIA, deliberadamente sem o bairro. `_chave_logradouro` inclui o bairro
    # porque "Rua A" do Centro e "Rua A" do distrito são ruas diferentes — mas uma
    # avenida ou rodovia atravessa vários bairros e continua sendo UMA via. Sem esta
    # segunda chave, a mesma rodovia vira N vias principais e consome N cotas da amostra.
    base["_chave_via"] = (
        base["_logradouro"].map(chave_logradouro) if col_log in base.columns else ""
    )

    base["_tipo_via"] = base["_logradouro"].map(tipo_via) if col_log in base.columns else "indefinido"

    col_classe = colunas.get("classe_via")
    if col_classe and col_classe in base.columns:
        base["_classe"] = base[col_classe].map(normalizar_classe)
    else:
        base["_classe"] = ROTULO_SEM_CLASSE
        ressalvas.append(
            "Sem coluna de classificação viária: a estratificação por classe fica "
            "desligada e a amostra passa a ser aleatória com dispersão geográfica apenas."
        )
    sem_classe = int((base["_classe"] == ROTULO_SEM_CLASSE).sum())
    if 0 < sem_classe < len(base):
        ressalvas.append(
            f"{sem_classe} pontos ({sem_classe / len(base):.1%}) sem classificação viária — "
            "eles formam um estrato próprio e continuam elegíveis ao sorteio."
        )

    col_lat, col_lon = colunas.get("latitude"), colunas.get("longitude")
    if col_lat in base.columns and col_lon in base.columns:
        base["_lat"] = pd.to_numeric(base[col_lat], errors="coerce")
        base["_lon"] = pd.to_numeric(base[col_lon], errors="coerce")
        dentro_faixa = (
            base["_lat"].between(LAT_MIN, LAT_MAX) & base["_lon"].between(LON_MIN, LON_MAX)
        )
        invalidos = int((~dentro_faixa).sum())
        base.loc[~dentro_faixa, ["_lat", "_lon"]] = np.nan
        if invalidos:
            ressalvas.append(
                f"{invalidos} pontos com coordenada ausente ou fora do território brasileiro — "
                "eles entram no sorteio, mas sem contribuir para o balanceamento espacial."
            )
    else:
        base["_lat"] = np.nan
        base["_lon"] = np.nan
        ressalvas.append(
            "Sem coordenadas no cadastro: a dispersão geográfica fica desligada e não há "
            "mapa de conferência da amostra."
        )

    base["_tem_coord"] = base["_lat"].notna() & base["_lon"].notna()
    base["_grupo"] = ""
    return base, ressalvas


# ── Sorteio espacialmente balanceado ──────────────────────────────────────────
def _coordenadas_metricas(sub: pd.DataFrame) -> np.ndarray:
    """Converte lat/long em um plano local aproximadamente métrico (km)."""
    lat = sub["_lat"].to_numpy(dtype=float)
    lon = sub["_lon"].to_numpy(dtype=float)
    lat_ref = float(np.nanmean(lat))
    return np.column_stack([lat * 111.32, lon * 111.32 * np.cos(np.radians(lat_ref))])


def _clusters_espaciais(coords: np.ndarray, k: int, semente: int) -> np.ndarray:
    """
    Rótulo de cluster para cada coordenada, com k clusters.

    Usa k-means (scikit-learn já é dependência fixa do portal por causa dos modelos de
    simulação). Se o scikit-learn não estiver disponível, cai para um grid regular —
    menos equilibrado, mas suficiente para espalhar a amostra.
    """
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans

        if len(coords) > 20000 or k > 400:
            modelo = MiniBatchKMeans(
                n_clusters=k, random_state=semente, n_init=3, batch_size=2048
            )
        else:
            modelo = KMeans(n_clusters=k, random_state=semente, n_init=3)
        return modelo.fit_predict(coords)
    except Exception:
        lado = max(int(np.ceil(np.sqrt(k))), 1)
        rotulos = np.zeros(len(coords), dtype=int)
        for eixo in (0, 1):
            valores = coords[:, eixo]
            faixa = valores.max() - valores.min()
            if faixa <= 0:
                continue
            celula = np.clip(((valores - valores.min()) / faixa * lado).astype(int), 0, lado - 1)
            rotulos = rotulos * lado + celula if eixo else celula
        return rotulos


def _sortear_disperso(
    pool: pd.DataFrame, k: int, rng: np.random.Generator, dispersao: bool
) -> list[int]:
    """
    Sorteia `k` índices de `pool` espalhando-os geograficamente.

    Pontos sem coordenada recebem uma fatia proporcional do k e são sorteados
    aleatoriamente; o restante é sorteado um por cluster espacial.
    """
    if k <= 0 or pool.empty:
        return []
    if k >= len(pool):
        return pool.index.tolist()

    com_coord = pool[pool["_tem_coord"]]
    sem_coord = pool[~pool["_tem_coord"]]

    if not dispersao or len(com_coord) < 2:
        return rng.choice(pool.index.to_numpy(), size=k, replace=False).tolist()

    k_sem = int(round(k * len(sem_coord) / len(pool)))
    k_sem = min(k_sem, len(sem_coord))
    k_com = min(k - k_sem, len(com_coord))
    k_sem = k - k_com   # devolve ao grupo sem coordenada o que sobrar

    escolhidos: list[int] = []
    if k_com > 0:
        coords = _coordenadas_metricas(com_coord)
        rotulos = _clusters_espaciais(coords, k_com, int(rng.integers(0, 2**31 - 1)))
        indices = com_coord.index.to_numpy()
        for rotulo in pd.unique(rotulos):
            candidatos = indices[rotulos == rotulo]
            escolhidos.append(int(rng.choice(candidatos)))
        # k-means pode devolver cluster vazio; completa o que faltar aleatoriamente.
        if len(escolhidos) < k_com:
            restantes = np.setdiff1d(indices, np.array(escolhidos))
            faltam = min(k_com - len(escolhidos), len(restantes))
            if faltam > 0:
                escolhidos += rng.choice(restantes, size=faltam, replace=False).tolist()
    if k_sem > 0 and len(sem_coord):
        faltam = min(k_sem, len(sem_coord))
        escolhidos += rng.choice(
            sem_coord.index.to_numpy(), size=faltam, replace=False
        ).tolist()

    return [int(i) for i in escolhidos[:k]]


def _alocar_maior_resto(total: int, pesos: dict[str, int]) -> dict[str, int]:
    """Distribui `total` unidades entre os estratos proporcionalmente, pelo maior resto."""
    soma = sum(pesos.values())
    if total <= 0 or soma <= 0:
        return {chave: 0 for chave in pesos}
    exatos = {chave: total * peso / soma for chave, peso in pesos.items()}
    alocado = {chave: int(np.floor(valor)) for chave, valor in exatos.items()}
    sobra = total - sum(alocado.values())
    ordem = sorted(pesos, key=lambda c: (-(exatos[c] - alocado[c]), -pesos[c], c))
    for chave in ordem[:sobra]:
        alocado[chave] += 1
    return alocado


# ── Sorteio principal ─────────────────────────────────────────────────────────
def sortear(
    base: pd.DataFrame,
    config: ConfigAmostragem,
    plano: PlanoAmostragem | None = None,
    municipio: str = "",
    uf: str = "",
    ressalvas_iniciais: list[str] | None = None,
) -> ResultadoAmostragem:
    """
    Executa o sorteio das duas amostras disjuntas.

    Args:
        base: cadastro já passado por `preparar_base`.
        config: parâmetros do sorteio.
        plano: plano NBR 5426 que originou o tamanho (só para registro no relatório).

    Returns:
        ResultadoAmostragem com as duas amostras, as tabelas de cobertura, o
        diagnóstico de abrangência e as ressalvas.
    """
    ressalvas = list(ressalvas_iniciais or [])
    rng = np.random.default_rng(config.semente)
    base = base.copy()
    base["_grupo"] = ""

    n_total = int(min(config.tamanho_amostra, len(base)))
    if config.tamanho_amostra > len(base):
        ressalvas.append(
            f"A amostra pedida ({config.tamanho_amostra}) é maior que o parque "
            f"({len(base)}) — sorteados todos os pontos do cadastro."
        )
    n_estrutural = int(round(n_total * config.proporcao_estrutural))
    n_qualidade = n_total - n_estrutural

    # ── Estratos e vias que precisam de cobertura ────────────────────────────
    vias = identificar_vias_principais(
        base, col_chave="_chave_via", teto=config.teto_vias_principais
    )
    if config.vias_obrigatorias is not None:
        selecionadas = set(config.vias_obrigatorias)
        vias = [v for v in vias if v.chave in selecionadas]

    indices_por_classe = {
        classe: grupo.index.to_numpy() for classe, grupo in base.groupby("_classe")
    }
    indices_por_via = {
        via.chave: base.index[base["_chave_via"] == via.chave].to_numpy() for via in vias
    }

    sel = {GRUPO_ESTRUTURAL: set(), GRUPO_QUALIDADE: set()}
    capacidade = {GRUPO_ESTRUTURAL: n_estrutural, GRUPO_QUALIDADE: n_qualidade}
    usados: set[int] = set()

    def _cobrir(rotulo: str, candidatos: np.ndarray, grupo: str, preferidos: set[int] | None = None) -> bool:
        """Garante 1 ponto de `candidatos` no `grupo`, preferindo os índices `preferidos`."""
        if len(sel[grupo]) >= capacidade[grupo]:
            return False
        if set(candidatos) & sel[grupo]:
            return True                      # estrato já coberto por sorteio anterior
        disponiveis = np.array([i for i in candidatos if i not in usados], dtype=int)
        if len(disponiveis) == 0:
            return False
        if preferidos:
            interseccao = np.array([i for i in disponiveis if i in preferidos], dtype=int)
            if len(interseccao):
                disponiveis = interseccao
        escolhido = int(rng.choice(disponiveis))
        sel[grupo].add(escolhido)
        usados.add(escolhido)
        return True

    # Camada 1a — cobertura por classe (prioridade normativa), preferindo pontos que
    # também resolvam uma via principal ainda descoberta.
    if config.cobertura_por_classe:
        pontos_de_vias = {int(i) for indices in indices_por_via.values() for i in indices}
        ordem_classes = sorted(indices_por_classe, key=rank_exigencia)
        for grupo in (GRUPO_QUALIDADE, GRUPO_ESTRUTURAL):
            for classe in ordem_classes:
                candidatos = indices_por_classe[classe]
                if not _cobrir(classe, candidatos, grupo, preferidos=pontos_de_vias):
                    if len(candidatos) < 2:
                        ressalvas.append(
                            f"A classe {classe} tem apenas {len(candidatos)} ponto(s) no "
                            "cadastro — não há como cobri-la nas duas planilhas ao mesmo "
                            "tempo mantendo amostras disjuntas."
                        )

    # Camada 1b — cobertura das vias principais.
    if config.cobertura_vias_principais:
        for grupo in (GRUPO_QUALIDADE, GRUPO_ESTRUTURAL):
            for via in vias:
                candidatos = indices_por_via.get(via.chave, np.array([], dtype=int))
                if len(candidatos) == 0:
                    continue
                if not _cobrir(via.chave, candidatos, grupo):
                    if len(candidatos) < 2:
                        ressalvas.append(
                            f"A via {via.nome} tem apenas {len(candidatos)} ponto(s) — "
                            "coberta em apenas uma das duas planilhas."
                        )

    for grupo, alvo in capacidade.items():
        if len(sel[grupo]) > alvo:
            ressalvas.append(
                f"As cotas obrigatórias ({len(sel[grupo])}) excedem o tamanho da amostra "
                f"{grupo} ({alvo}). Aumente o tamanho da amostra ou reduza o teto de vias "
                "principais — a planilha saiu maior que o pedido para não perder cobertura."
            )

    # Camada 2 — preenchimento proporcional por classe, com dispersão espacial.
    for grupo in (GRUPO_QUALIDADE, GRUPO_ESTRUTURAL):
        faltam = capacidade[grupo] - len(sel[grupo])
        if faltam <= 0:
            continue
        disponivel = base.loc[~base.index.isin(usados)]
        if disponivel.empty:
            break
        pesos = disponivel.groupby("_classe").size().to_dict()
        alocacao = _alocar_maior_resto(faltam, pesos)
        for classe, quantidade in alocacao.items():
            if quantidade <= 0:
                continue
            pool = disponivel[disponivel["_classe"] == classe]
            escolhidos = _sortear_disperso(pool, quantidade, rng, config.dispersao_espacial)
            sel[grupo].update(escolhidos)
            usados.update(escolhidos)
        # Sobra por arredondamento ou estrato esgotado: completa do pool geral.
        faltam = capacidade[grupo] - len(sel[grupo])
        if faltam > 0:
            pool = base.loc[~base.index.isin(usados)]
            escolhidos = _sortear_disperso(pool, faltam, rng, config.dispersao_espacial)
            sel[grupo].update(escolhidos)
            usados.update(escolhidos)

    base.loc[sorted(sel[GRUPO_ESTRUTURAL]), "_grupo"] = GRUPO_ESTRUTURAL
    base.loc[sorted(sel[GRUPO_QUALIDADE]), "_grupo"] = GRUPO_QUALIDADE

    estrutural = base[base["_grupo"] == GRUPO_ESTRUTURAL].copy()
    qualidade = base[base["_grupo"] == GRUPO_QUALIDADE].copy()

    return ResultadoAmostragem(
        base=base,
        estrutural=estrutural,
        qualidade=qualidade,
        config=config,
        vias_principais=vias,
        cobertura_classes=_tabela_cobertura_classes(base),
        cobertura_vias=_tabela_cobertura_vias(base, vias),
        abrangencia=diagnostico_abrangencia(base),
        ressalvas=ressalvas,
        plano=plano,
        municipio=municipio,
        uf=uf,
    )


# ── Prestação de contas ───────────────────────────────────────────────────────
def _tabela_cobertura_classes(base: pd.DataFrame) -> pd.DataFrame:
    """Classe × parque × amostra estrutural × amostra qualidade, com o desvio proporcional."""
    linhas = []
    total = len(base)
    for classe, grupo in base.groupby("_classe"):
        n_est = int((grupo["_grupo"] == GRUPO_ESTRUTURAL).sum())
        n_qua = int((grupo["_grupo"] == GRUPO_QUALIDADE).sum())
        linhas.append(
            {
                "Classe": classe,
                "Pontos no parque": len(grupo),
                "% do parque": len(grupo) / total if total else 0.0,
                "Estrutural": n_est,
                "Qualidade": n_qua,
                "Amostra total": n_est + n_qua,
                "Coberta nas duas": "Sim" if n_est and n_qua else "Não",
                # Peso de extrapolação w_h = N_h / n_h. A amostra é deliberadamente
                # NÃO auto-ponderada: as cotas de cobertura sobre-representam as vias
                # exigentes. Extrapolar a inspeção para o parque somando pontos daria
                # resultado enviesado — tem que ser pela média ponderada por estrato.
                "Peso p/ extrapolação": (len(grupo) / (n_est + n_qua)) if (n_est + n_qua) else None,
            }
        )
    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    df["_rank"] = df["Classe"].map(rank_exigencia)
    return df.sort_values(["_rank", "Classe"]).drop(columns=["_rank"]).reset_index(drop=True)


def _tabela_cobertura_vias(base: pd.DataFrame, vias: list[ViaPrincipal]) -> pd.DataFrame:
    """Vias principais × pontos sorteados em cada planilha."""
    linhas = []
    for via in vias:
        grupo = base[base["_chave_via"] == via.chave]
        n_est = int((grupo["_grupo"] == GRUPO_ESTRUTURAL).sum())
        n_qua = int((grupo["_grupo"] == GRUPO_QUALIDADE).sum())
        linhas.append(
            {
                "Via": via.nome,
                "Tipo": via.tipo,
                "Classes": ", ".join(via.classes) if via.classes else "—",
                "Pontos no parque": via.pontos,
                "Estrutural": n_est,
                "Qualidade": n_qua,
                "Motivo da obrigatoriedade": via.motivo_texto,
            }
        )
    return pd.DataFrame(linhas)


def diagnostico_abrangencia(base: pd.DataFrame, celulas_por_eixo: int = 12) -> dict:
    """
    Mede se a amostra realmente varreu o município.

    Três indicadores independentes, todos comparando amostra contra parque:

      - **grid**: divide a mancha do cadastro em uma malha `celulas_por_eixo`² e mede
        quantas células com pontos de IP receberam ao menos um ponto sorteado. É o
        indicador que responde "a amostra pegou a cidade toda ou só o centro?".
      - **bairros / logradouros**: fração de bairros e de ruas do cadastro tocados.
      - **distância de representatividade**: para cada ponto do parque, a distância até
        o ponto inspecionado mais próximo. A mediana diz, na prática, quão longe está a
        evidência de campo mais próxima de um ponto qualquer do município.
    """
    amostra = base[base["_grupo"] != ""]
    diagnostico: dict = {
        "pontos_parque": int(len(base)),
        "pontos_amostra": int(len(amostra)),
        "bairros_parque": int(base["_bairro"].replace("", np.nan).nunique()),
        "bairros_amostra": int(amostra["_bairro"].replace("", np.nan).nunique()),
        "logradouros_parque": int(base["_chave_logradouro"].replace("", np.nan).nunique()),
        "logradouros_amostra": int(amostra["_chave_logradouro"].replace("", np.nan).nunique()),
    }

    com_coord = base[base["_tem_coord"]]
    amostra_coord = amostra[amostra["_tem_coord"]]
    if len(com_coord) < 2 or amostra_coord.empty:
        diagnostico.update(
            celulas_com_parque=0, celulas_cobertas=0, cobertura_grid=None,
            distancia_mediana_km=None, distancia_p90_km=None,
        )
        return diagnostico

    lat, lon = com_coord["_lat"].to_numpy(), com_coord["_lon"].to_numpy()
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    span_lat = max(lat_max - lat_min, 1e-9)
    span_lon = max(lon_max - lon_min, 1e-9)

    def _celula(sub: pd.DataFrame) -> set[tuple[int, int]]:
        i = np.clip(((sub["_lat"] - lat_min) / span_lat * celulas_por_eixo).astype(int), 0, celulas_por_eixo - 1)
        j = np.clip(((sub["_lon"] - lon_min) / span_lon * celulas_por_eixo).astype(int), 0, celulas_por_eixo - 1)
        return set(zip(i.tolist(), j.tolist()))

    celulas_parque = _celula(com_coord)
    celulas_amostra = _celula(amostra_coord) & celulas_parque
    diagnostico["celulas_com_parque"] = len(celulas_parque)
    diagnostico["celulas_cobertas"] = len(celulas_amostra)
    diagnostico["cobertura_grid"] = len(celulas_amostra) / len(celulas_parque) if celulas_parque else None

    # Distância de cada ponto do parque ao ponto amostrado mais próximo (em km),
    # calculada em blocos para não materializar uma matriz N×n gigante.
    alvo = _coordenadas_metricas(amostra_coord)
    origem = _coordenadas_metricas(com_coord)
    menores = np.empty(len(origem), dtype=float)
    bloco = max(1, int(5_000_000 / max(len(alvo), 1)))
    for inicio in range(0, len(origem), bloco):
        fatia = origem[inicio: inicio + bloco]
        distancias = np.sqrt(
            ((fatia[:, None, :] - alvo[None, :, :]) ** 2).sum(axis=2)
        )
        menores[inicio: inicio + len(fatia)] = distancias.min(axis=1)
    diagnostico["distancia_mediana_km"] = float(np.median(menores))
    diagnostico["distancia_p90_km"] = float(np.percentile(menores, 90))
    return diagnostico


__all__ = [
    "COLUNAS_AUXILIARES", "GRUPO_ESTRUTURAL", "GRUPO_QUALIDADE",
    "ConfigAmostragem", "ResultadoAmostragem",
    "preparar_base", "sortear", "diagnostico_abrangencia",
]
