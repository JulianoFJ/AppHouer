"""
Leitura de coordenadas geográficas vindas de cadastro municipal.

Por que este módulo existe: `pd.to_numeric` resolve exatamente **um** dos formatos que
chegam na prática (ponto decimal, tipo `-19.546047`) e transforma todos os outros em
`NaN` silenciosamente. Cadastro de prefeitura não tem um formato — tem um por origem:

* planilha exportada em locale pt-BR: `-19,546047`;
* planilha que passou por edição manual: `-19.546.047` (o Excel "corrigiu" o decimal);
* coleta de campo em GMS: `19°32'45,8"S`, `19° 32.763' S`, `S 19 32 45.8`;
* célula única copiada do Google Maps: `-19,546047, -44,084346`;
* GIS da prefeitura em UTM/SIRGAS 2000: `604321,55` / `7838412,10`;
* coluna de latitude e de longitude **trocadas** entre si;
* coordenada em micrograus inteiros: `-19546047`.

Todos esses casos são recuperáveis sem perguntar nada ao usuário, e cada recuperação
gera uma ressalva — o operador precisa saber que a base dele estava torta, senão a
mesma planilha volta torta na próxima rodada.

O que **não** é recuperável fica `NaN` de propósito: coordenada fora do território
brasileiro é erro de cadastro, não formato, e inventar um valor plausível para ela
colocaria um ponto inexistente no mapa de conferência da amostra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Envelope continental do Brasil, com folga. Serve de juiz para todas as heurísticas
# deste módulo: uma conversão só é aceita se colocar a maioria dos pontos aqui dentro.
LAT_MIN, LAT_MAX = -34.0, 6.0
LON_MIN, LON_MAX = -74.5, -33.0

# Fração dos pontos que precisa cair dentro do envelope para uma recuperação
# (reescala, troca de eixos, projeção UTM) ser considerada correta. Abaixo disso é
# mais provável que a heurística tenha acertado por acaso do que que a base esteja
# nesse formato.
_LIMIAR_ACEITE = 0.80

# Zonas UTM que cobrem o Brasil (meridianos centrais de 75°W a 33°W).
_ZONAS_UTM_BRASIL = range(18, 26)

# Ordem de desempate entre zonas UTM, quando não há UF para decidir. Não é preferência
# estética: (E, N) **não** determina a zona — cada zona reprojeta o mesmo par 6° a
# oeste, e zonas 18 a 24 jogam um município de Minas dentro do envelope do Brasil do
# mesmo jeito. Sem a UF o módulo escolhe a zona mais usada no país e diz, na ressalva,
# que a escolha é um chute — o mapa de conferência existe exatamente para isso.
_ZONAS_POR_USO = (23, 22, 24, 21, 25, 20, 19, 18)

# Envelope por UF (lat_min, lat_max, lon_min, lon_max), com folga. Serve só para
# desempatar zona UTM: uma zona errada desloca o resultado 6° em longitude, o que é
# uma ordem de grandeza maior que a imprecisão desta tabela.
_ENVELOPE_UF: dict[str, tuple[float, float, float, float]] = {
    "AC": (-11.2, -7.1, -74.0, -66.6),   "AL": (-10.6, -8.8, -38.3, -35.1),
    "AM": (-9.9, 2.3, -73.9, -56.0),     "AP": (-1.3, 4.5, -54.9, -49.8),
    "BA": (-18.4, -8.5, -46.7, -37.3),   "CE": (-7.9, -2.7, -41.5, -37.2),
    "DF": (-16.1, -15.5, -48.3, -47.3),  "ES": (-21.4, -17.8, -41.9, -39.6),
    "GO": (-19.6, -12.3, -53.3, -45.9),  "MA": (-10.3, -1.0, -48.8, -41.7),
    "MG": (-23.0, -14.2, -51.1, -39.8),  "MS": (-24.1, -17.1, -58.2, -50.9),
    "MT": (-18.1, -7.3, -61.7, -50.2),   "PA": (-9.9, 2.6, -59.0, -46.0),
    "PB": (-8.4, -6.0, -38.8, -34.7),    "PE": (-9.5, -7.2, -41.4, -32.3),
    "PI": (-11.0, -2.7, -46.0, -40.3),   "PR": (-26.8, -22.5, -54.7, -48.0),
    "RJ": (-23.4, -20.7, -44.9, -40.9),  "RN": (-7.0, -4.8, -38.6, -34.9),
    "RO": (-13.7, -7.9, -66.9, -59.7),   "RR": (-1.6, 5.3, -64.9, -58.8),
    "RS": (-33.8, -27.0, -57.7, -49.6),  "SC": (-29.4, -25.9, -53.9, -48.3),
    "SE": (-11.6, -9.5, -38.3, -36.3),   "SP": (-25.4, -19.7, -53.2, -44.1),
    "TO": (-13.5, -5.1, -50.8, -45.6),
}

_SINAL_CARDEAL = {"N": 1, "S": -1, "E": 1, "L": 1, "W": -1, "O": -1}

# Qualquer coisa que um teclado ou um GIS use no lugar do hífen ASCII.
_MENOS = {"−": "-", "–": "-", "—": "-", "‒": "-"}

# Símbolos de grau/minuto/segundo, incluindo o ordinal masculino que o Word troca.
_SIMBOLOS_GMS = "°º'′´`\"″”"

_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")
_NAO_COORDENADA = {"", "nan", "none", "null", "na", "-", "--", "s/coordenada",
                   "sem coordenada", "0", "#n/d", "#n/a"}


# ── Conversão de um valor isolado ─────────────────────────────────────────────
def _decimal(texto: str) -> float:
    """
    Interpreta o separador decimal de um número escrito por humano ou por Excel.

    A regra é posicional, não de locale: **o último separador é o decimal**, e o outro
    é milhar. Resolve `1.234,56` (pt-BR) e `1,234.56` (en-US) com a mesma linha. O caso
    de vários pontos e nenhuma vírgula (`-19.546.047`) é tratado à parte: em coordenada
    não existe milhar — `-19.546.047` é `-19.546047` com o decimal quebrado pelo Excel.
    """
    texto = texto.replace(" ", "")
    ponto, virgula = texto.rfind("."), texto.rfind(",")
    if ponto >= 0 and virgula >= 0:
        if virgula > ponto:
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif virgula >= 0:
        texto = texto.replace(",", ".")
    elif texto.count(".") > 1:
        cabeca, _, cauda = texto.partition(".")
        texto = f"{cabeca}.{cauda.replace('.', '')}"
    return float(texto)


def _gms_sem_simbolo(texto: str, partes: list[str]) -> bool:
    """
    Reconhece `19 32 45.8` (GMS que perdeu os símbolos ° ' ") sem confundir com um par.

    As três guardas são o que separa um caso do outro: grau e minuto têm de ser
    inteiros (`-19.546047 -44.08` cai fora pelo decimal do primeiro), minuto e segundo
    têm de ser menores que 60, e não pode sobrar sinal no meio da string — dois números
    negativos são um par `lat lon`, não um ângulo.
    """
    if not (2 <= len(partes) <= 3) or "-" in texto:
        return False
    if any("." in p or "," in p for p in partes[:-1]):
        return False
    try:
        valores = [float(_decimal(p)) for p in partes]
    except ValueError:
        return False
    return valores[0] <= 180 and all(v < 60 for v in valores[1:])


def para_graus(valor) -> float:
    """
    Converte um valor de coordenada em graus decimais. Devolve `NaN` se não der.

    Aceita número, decimal com vírgula ou ponto, grau-minuto-segundo com qualquer
    combinação de símbolos, grau com minuto decimal, e sufixo/prefixo cardeal
    (N/S/E/W, além de L/O em português).
    """
    if valor is None:
        return float("nan")
    if isinstance(valor, (int, float, np.integer, np.floating)) and not isinstance(valor, bool):
        return float(valor)
    if valor is pd.NaT or (isinstance(valor, float) and np.isnan(valor)):
        return float("nan")

    texto = str(valor).strip()
    if texto.lower().replace(" ", "") in _NAO_COORDENADA:
        return float("nan")
    for origem, destino in _MENOS.items():
        texto = texto.replace(origem, destino)

    # Cardeal: pode vir antes ou depois do número. Só conta na borda da string —
    # letra no meio é sujeira de cadastro, não hemisfério.
    sinal = 1
    if texto[-1:].upper() in _SINAL_CARDEAL:
        sinal, texto = _SINAL_CARDEAL[texto[-1].upper()], texto[:-1].strip()
    elif texto[:1].upper() in _SINAL_CARDEAL:
        sinal, texto = _SINAL_CARDEAL[texto[0].upper()], texto[1:].strip()

    negativo = texto.startswith("-")
    if negativo:
        texto = texto[1:].strip()

    partes = _NUMERO.findall(texto)
    if not partes:
        return float("nan")

    if (any(s in texto for s in _SIMBOLOS_GMS) or _gms_sem_simbolo(texto, partes)) \
            and len(partes) >= 2:
        graus = _decimal(partes[0])
        minutos = _decimal(partes[1])
        segundos = _decimal(partes[2]) if len(partes) >= 3 else 0.0
        magnitude = graus + minutos / 60.0 + segundos / 3600.0
    else:
        try:
            magnitude = abs(_decimal(texto))
        except ValueError:
            return float("nan")

    return -magnitude if negativo else sinal * magnitude


def serie_para_graus(serie) -> pd.Series:
    """
    Vetoriza `para_graus`, evitando o caminho lento quando a coluna já é numérica.

    A esmagadora maioria dos cadastros chega com a coluna já em `float64` — pagar um
    `.map` por linha em base de 100 mil pontos para nada é o tipo de custo que aparece
    na UI como meio segundo a cada clique.
    """
    if serie is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").astype("float64")
    direto = pd.to_numeric(serie, errors="coerce")
    if direto.notna().all():
        return direto.astype("float64")
    return pd.Series([para_graus(v) for v in serie], index=serie.index, dtype="float64")


# ── Célula única com o par (lat, lon) ─────────────────────────────────────────
def dividir_par(valor) -> tuple[float, float]:
    """
    Separa `-19,546047, -44,084346` (ou com `;`, `/`, ou só espaço) em (lat, lon).

    A ambiguidade real é a vírgula, que serve de separador do par **e** de decimal.
    Resolve pela contagem: quatro pedaços significam decimal com vírgula, então os
    pedaços são remontados dois a dois.
    """
    vazio = (float("nan"), float("nan"))
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return vazio
    texto = str(valor).strip()
    if not texto:
        return vazio
    for origem, destino in _MENOS.items():
        texto = texto.replace(origem, destino)

    for separador in (";", "/", "|"):
        if separador in texto:
            pedacos = [p.strip() for p in texto.split(separador) if p.strip()]
            if len(pedacos) == 2:
                return para_graus(pedacos[0]), para_graus(pedacos[1])

    if "," in texto:
        pedacos = [p.strip() for p in texto.split(",")]
        if len(pedacos) == 4:   # "-19,546047, -44,084346" → vírgula é decimal
            return (para_graus(f"{pedacos[0]},{pedacos[1]}"),
                    para_graus(f"{pedacos[2]},{pedacos[3]}"))
        if len(pedacos) == 2 and all("." in p or p.strip("-+ ").isdigit() for p in pedacos):
            return para_graus(pedacos[0]), para_graus(pedacos[1])

    pedacos = texto.split()
    if len(pedacos) == 2:
        return para_graus(pedacos[0]), para_graus(pedacos[1])
    return vazio


def parece_par(serie, amostra: int = 200) -> bool:
    """
    True quando a coluna carrega o par inteiro em vez de um eixo só.

    Público de propósito: quem monta a UI precisa decidir, **olhando o dado**, se a
    coluna que o detector chamou de "coordenadas" é mesmo um par — nome de coluna não
    basta, e tratar um eixo solto como par produziria latitude igual à longitude.
    """
    valores = serie.dropna().astype(str).head(amostra)
    if valores.empty:
        return False
    acertos = sum(
        1 for texto in valores
        if not any(np.isnan(v) for v in dividir_par(texto))
    )
    return acertos / len(valores) >= _LIMIAR_ACEITE


# ── UTM → graus (Snyder, elipsoide WGS84 ≈ SIRGAS 2000) ──────────────────────
_A_WGS84 = 6378137.0
_F_WGS84 = 1 / 298.257223563
_K0 = 0.9996


def utm_para_graus(este, norte, zona: int, sul: bool = True):
    """
    Inverte a projeção UTM (série de Snyder) sem depender de GDAL/pyproj.

    SIRGAS 2000 e WGS84 diferem por menos de 1 m em território brasileiro — irrelevante
    para dispersão espacial de amostra e para mapa de conferência, que é o uso aqui.
    """
    e2 = _F_WGS84 * (2 - _F_WGS84)
    el2 = e2 / (1 - e2)                     # e'² de Snyder
    x = np.asarray(este, dtype=float) - 500000.0
    y = np.asarray(norte, dtype=float) - (10000000.0 if sul else 0.0)

    m = y / _K0
    mu = m / (_A_WGS84 * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
            + (151 * e1**3 / 96) * np.sin(6 * mu)
            + (1097 * e1**4 / 512) * np.sin(8 * mu))

    sen, cos, tan = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    c1 = el2 * cos**2
    t1 = tan**2
    n1 = _A_WGS84 / np.sqrt(1 - e2 * sen**2)
    r1 = _A_WGS84 * (1 - e2) / (1 - e2 * sen**2) ** 1.5
    d = x / (n1 * _K0)

    lat = phi1 - (n1 * tan / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * el2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * el2 - 3 * c1**2) * d**6 / 720
    )
    lon = np.radians(zona * 6 - 183) + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * el2 + 24 * t1**2) * d**5 / 120
    ) / cos
    return np.degrees(lat), np.degrees(lon)


# ── Orquestração ──────────────────────────────────────────────────────────────
@dataclass
class ResultadoCoordenadas:
    """Coordenadas em graus decimais + o que foi preciso fazer para chegar nelas."""
    latitude: pd.Series
    longitude: pd.Series
    ressalvas: list[str] = field(default_factory=list)
    formato: str = "graus decimais"
    # Zonas UTM que o dado não permite distinguir. Vazio quando não houve UTM ou
    # quando sobrou uma só; com mais de um elemento, a UI precisa perguntar.
    zonas_utm_candidatas: list[int] = field(default_factory=list)
    zona_utm_adotada: int | None = None

    @property
    def validas(self) -> int:
        return int((self.latitude.notna() & self.longitude.notna()).sum())


def envelope(uf: str | None = None) -> tuple[float, float, float, float]:
    """Retângulo de plausibilidade: o da UF quando ela é conhecida, o do país senão."""
    if uf:
        limites = _ENVELOPE_UF.get(str(uf).strip().upper())
        if limites:
            folga = 1.0   # cadastro tem ponto na divisa, e a tabela é aproximada
            lat_min, lat_max, lon_min, lon_max = limites
            return (lat_min - folga, lat_max + folga, lon_min - folga, lon_max + folga)
    return (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)


def _dentro(lat, lon, limites=None) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = limites or (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def _fracao_valida(lat, lon, limites=None) -> float:
    """Fração dos pares **preenchidos** que cai dentro do retângulo de plausibilidade."""
    arr_lat = np.asarray(lat, dtype=float)
    arr_lon = np.asarray(lon, dtype=float)
    preenchidos = ~np.isnan(arr_lat) & ~np.isnan(arr_lon)
    if not preenchidos.any():
        return 0.0
    return float(_dentro(arr_lat, arr_lon, limites).sum() / preenchidos.sum())


def _tentar_utm(lat: pd.Series, lon: pd.Series, uf: str | None = None,
                zona_utm: int | None = None):
    """
    Reconhece coordenada projetada e devolve `(lat, lon, zona, ambigua)`, ou None.

    O ponto delicado é que **(E, N) não determina a zona**: a mesma dupla reprojetada
    na zona vizinha cai 6° a oeste, e para um município de Minas as zonas 18 a 24 caem
    todas dentro do envelope do Brasil. Por isso a decisão é, em ordem: a zona que o
    usuário informou; a UF, cujo envelope é estreito o bastante para sobrar uma só; e,
    em último caso, a zona mais usada no país — marcada como ambígua, para que a UI
    peça confirmação em vez de fingir certeza.
    """
    magnitudes = pd.concat([lat.abs(), lon.abs()]).dropna()
    if magnitudes.empty or float(magnitudes.median()) < 1000:
        return None   # magnitude de grau, não de metro

    limites = envelope(uf)

    # Norte (7 dígitos) e Este (6 dígitos) podem vir em qualquer ordem de coluna.
    for eixo_e, eixo_n in ((lon, lat), (lat, lon)):
        if eixo_n.dropna().empty or float(eixo_n.abs().median()) < 1e6:
            continue
        este = eixo_e.to_numpy(dtype=float)
        norte = eixo_n.to_numpy(dtype=float)
        candidatos = []
        for zona in _ZONAS_UTM_BRASIL:
            for sul in (True, False):
                nova_lat, nova_lon = utm_para_graus(este, norte, zona, sul)
                candidatos.append(
                    (_fracao_valida(nova_lat, nova_lon, limites), zona, nova_lat, nova_lon))
        melhor_fracao = max(c[0] for c in candidatos)
        if melhor_fracao < _LIMIAR_ACEITE:
            continue
        # Empate técnico = zonas indistinguíveis pelo envelope disponível. A UF corta
        # boa parte delas (em MG sobram 22 e 23; em SE sobra só a 24), mas estado largo
        # continua abrigando mais de uma — é aí que a UI tem de perguntar.
        #
        # A lista de empatados é sempre a completa, mesmo quando o usuário já fixou a
        # zona: é ela que mantém o seletor na tela para ele poder trocar de novo. Fixar
        # a zona só decide qual sai adotada, não apaga as alternativas.
        empatados = sorted({c[1] for c in candidatos if c[0] >= melhor_fracao - 0.02})
        if zona_utm and any(c[1] == int(zona_utm) for c in candidatos):
            escolhida = int(zona_utm)   # decisão do usuário vence a heurística
        else:
            escolhida = next(z for z in _ZONAS_POR_USO if z in empatados)
        _, _, nova_lat, nova_lon = max(
            (c for c in candidatos if c[1] == escolhida), key=lambda c: c[0])
        return (pd.Series(nova_lat, index=lat.index),
                pd.Series(nova_lon, index=lon.index),
                escolhida, empatados)
    return None


def normalizar(serie_lat, serie_lon=None, uf: str | None = None,
               zona_utm: int | None = None) -> ResultadoCoordenadas:
    """
    Converte um par de colunas de cadastro em latitude/longitude em graus decimais.

    Args:
        serie_lat: coluna de latitude — ou a coluna única com o par `lat, lon`.
        serie_lon: coluna de longitude. Pode ser `None`, ou a própria coluna de
            `serie_lat`, quando o cadastro traz o par em uma célula só.
        uf: sigla do estado, quando conhecida. Só é usada para desempatar a zona UTM
            de um cadastro projetado — mas aí é decisiva, porque sem ela a zona é
            indeterminável a partir de (E, N).
        zona_utm: zona informada pelo usuário. Prevalece sobre tudo.

    Returns:
        `ResultadoCoordenadas`. Coordenada que não pôde ser lida, ou que caiu fora do
        território brasileiro, vira `NaN` — nunca um valor inventado.
    """
    ressalvas: list[str] = []
    formato = "graus decimais"
    zonas_utm_candidatas: list[int] = []
    zona_adotada: int | None = None

    if serie_lat is None:
        vazia = pd.Series(dtype="float64")
        return ResultadoCoordenadas(vazia, vazia, ["Sem coluna de coordenada."], "ausente")

    indice = serie_lat.index
    coluna_unica = (
        serie_lon is None
        or (serie_lon.name is not None and serie_lon.name == serie_lat.name)
        or serie_lon.equals(serie_lat)
    )

    if coluna_unica and parece_par(serie_lat):
        divididos = [dividir_par(v) for v in serie_lat]
        lat = pd.Series([p[0] for p in divididos], index=indice, dtype="float64")
        lon = pd.Series([p[1] for p in divididos], index=indice, dtype="float64")
        formato = "par em coluna única"
        ressalvas.append(
            "A coordenada vinha em uma coluna só, no formato `latitude, longitude` — "
            "foi separada automaticamente em dois eixos."
        )
    else:
        lat = serie_para_graus(serie_lat)
        lon = (serie_para_graus(serie_lon) if serie_lon is not None
               else pd.Series(np.nan, index=indice, dtype="float64"))
        recuperadas = 0
        for original, convertida in ((serie_lat, lat), (serie_lon, lon)):
            if original is None or pd.api.types.is_numeric_dtype(original):
                continue
            cru = pd.to_numeric(original, errors="coerce")
            recuperadas += max(int(cru.isna().sum() - convertida.isna().sum()), 0)
        if recuperadas:
            formato = "texto (vírgula decimal ou grau-minuto-segundo)"
            ressalvas.append(
                f"{recuperadas} coordenadas estavam em texto (vírgula decimal, "
                "grau-minuto-segundo ou sufixo N/S/L/O) e foram convertidas para graus "
                "decimais — `pd.to_numeric` sozinho as descartaria."
            )

    fracao = _fracao_valida(lat, lon)

    # 1) Projetada em metros — é como o GIS de prefeitura costuma exportar.
    if fracao < _LIMIAR_ACEITE:
        utm = _tentar_utm(lat, lon, uf=uf, zona_utm=zona_utm)
        if utm is not None:
            lat, lon, zona, candidatas = utm
            formato = f"UTM zona {zona} (convertida)"
            zonas_utm_candidatas, zona_adotada = list(candidatas), zona
            ressalvas.append(
                f"As coordenadas estavam projetadas em UTM (zona {zona}) e foram "
                "convertidas para graus decimais WGS84/SIRGAS 2000."
            )
            if len(candidatas) > 1:
                lista = ", ".join(str(z) for z in candidatas)
                ressalvas.append(
                    f"⚠️ A zona UTM é indeterminável a partir do par (E, N): as zonas "
                    f"{lista} são todas compatíveis"
                    + (f" com o território de {uf}" if uf else " com o território brasileiro")
                    + f", e foi adotada a {zona}. **Confira o município no mapa** — se "
                    "estiver deslocado no sentido leste-oeste, escolha a zona correta."
                )
            fracao = _fracao_valida(lat, lon)

    # 2) Inteiro sem separador decimal (micrograus).
    if fracao < _LIMIAR_ACEITE:
        for expoente in range(1, 8):
            escala = 10.0**expoente
            if _fracao_valida(lat / escala, lon / escala) >= _LIMIAR_ACEITE:
                lat, lon = lat / escala, lon / escala
                formato = f"inteiro sem separador decimal (÷10^{expoente})"
                ressalvas.append(
                    f"As coordenadas vieram sem separador decimal e foram divididas por "
                    f"10^{expoente}. Confira alguns pontos no mapa antes de ir a campo."
                )
                fracao = _fracao_valida(lat, lon)
                break

    # 3) Eixos trocados. Por último porque só faz sentido sobre valores já em graus,
    #    e porque é a hipótese mais fácil de acertar por acaso.
    if fracao < _LIMIAR_ACEITE and _fracao_valida(lon, lat) >= max(fracao + 0.2, _LIMIAR_ACEITE):
        lat, lon = lon.copy(), lat.copy()
        ressalvas.append(
            "As colunas de latitude e longitude estavam trocadas entre si — a ordem foi "
            "corrigida. Confira o mapa: se o município aparecer no lugar errado, "
            "escolha as colunas manualmente."
        )

    lat = lat.astype("float64")
    lon = lon.astype("float64")
    lat.name, lon.name = "_lat", "_lon"
    return ResultadoCoordenadas(lat, lon, ressalvas, formato,
                                zonas_utm_candidatas, zona_adotada)


__all__ = [
    "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX",
    "ResultadoCoordenadas", "dividir_par", "envelope", "normalizar", "parece_par",
    "para_graus", "serie_para_graus", "utm_para_graus",
]
