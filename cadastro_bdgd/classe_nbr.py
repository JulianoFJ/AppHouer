"""
Classe de iluminação da ABNT NBR 5101:2024 estimada a partir dos atributos da via.

Por que isto existe
-------------------
A BDGD não traz classificação viária, e o sorteio da amostra depende dela para as
cotas por classe. As tags do OpenStreetMap cobrem parte dos parâmetros que a norma
usa, então em vez de inventar um "de-para" (`primary` -> M2) a estimativa **replica o
método da norma**: soma os valores de ponderação da tabela e aplica a fórmula.

    Número de classe de iluminação M = 6 - V_PS          (NBR 5101:2024, 4.2, Tabela 1)
    Número de classe de iluminação C = 6 - V_PS          (NBR 5101:2024, 4.3, Tabela 3)

Isso muda o que se pode defender perante banca. Um de-para é opinião; a soma de
ponderações é o procedimento da norma aplicado com os parâmetros disponíveis, e cada
parcela fica registrada para conferência.

**Bug corrigido em 08/09/2026, achado ao vivo pelo usuário**: até essa data só existia
`estimar_classe_m` — TODO ponto saía M, nunca C, porque não havia decisão nenhuma entre
as duas (a função `classe_conflito_equivalente` existia, mas nada a chamava). Conferido
contra o dataset real da Simulação NBR (`dataset.csv`, 2.646 luminárias de projetos
entregues): a distribuição real é C4 (1.131), C5 (846), C3 (312), C1 (141), C2 (138),
C0 (18) — contra **M6: 3**. Ou seja, a esmagadora maioria da malha urbana brasileira é
área de conflito (C), e M é a exceção — exatamente o oposto do que o código fazia. A
norma é explícita (4.3): "a classe C compreende as áreas de conflito e as vias de
tráfego cuja composição seja principalmente motorizada [...] onde há interseção entre
fluxos de veículos e quando o fluxo de veículos se depara com áreas frequentadas por
pedestres, ciclistas ou outros usuários" — o que descreve a rua urbana brasileira
típica (calçada, esquina, estacionamento na via) muito mais que a Tabela 1 (via de
tráfego motorizado ininterrupto, sem esses conflitos). A decisão adotada aqui, por
`classificar_via`: só motorway/trunk (e os `_link`) — o que no Brasil corresponde a
rodovia/via expressa — usa a Tabela 1 (M); todo o resto usa a Tabela 3 (C).

O que a norma pede e o OSM não dá
---------------------------------
Da Tabela 1 (M), o OSM cobre três parâmetros com dado real (velocidade, separação das
faixas, densidade de interseções), um por proxy (luminância ambiente, pela área
urbana/rural da própria BDGD) e um raramente (veículos estacionados). Da Tabela 3 (C),
cobre dois com dado real (velocidade, separação das faixas), um por proxy (luminância
ambiente) e infere composição do tráfego da própria hierarquia OSM (ver
`_p_composicao_trafego`). **Volume de tráfego e qualidade da sinalização não existem em
base pública** em nenhuma das duas tabelas — a norma manda consultar o órgão de
trânsito local. Ambos entram com o valor NEUTRO da tabela (0) quando não dá para
inferir, que é o mais próximo de "não afirmar nada"; a alternativa seria empurrar a
classe para um lado sem evidência.

Consequência a declarar em qualquer entregável: a classe daqui é **estimada**, serve
para estratificar amostra e dimensionar inspeção, e não substitui o enquadramento
feito com contagem de tráfego e vistoria. É por isso que `estimar_classe_m` e
`estimar_classe_c` devolvem o detalhamento das parcelas junto com a classe, e não só
o rótulo.

Referência: ABNT NBR 5101:2024, Tabela 1 (p. 9), Tabela 2 (p. 10), Tabela 3 (p. 10),
Tabela 4 (p. 11) e Tabela 5 (p. 12). As tabelas numéricas do PDF oficial vêm com fonte
de proteção contra cópia — `pdftotext`/PyMuPDF devolvem texto embaralhado para elas.
Os valores abaixo foram conferidos visualmente na página renderizada, não copiados do
texto extraído.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── Tabela 2 — requisitos por classe M (p. 10) ───────────────────────────────
# Conferido no PDF da norma em docs/. Atenção: Ul de M1/M2 é 0,65 (não 0,70) e o de
# M5/M6 é 0,35 (não 0,40) — os valores 0,70/0,40 são da edição anterior.
REQUISITOS_M = {
    "M1": {"lmed": 2.00, "uo": 0.40, "ul": 0.65, "fTI": 14, "eir": 0.45},
    "M2": {"lmed": 1.50, "uo": 0.40, "ul": 0.65, "fTI": 14, "eir": 0.45},
    "M3": {"lmed": 1.00, "uo": 0.40, "ul": 0.60, "fTI": 15, "eir": 0.45},
    "M4": {"lmed": 0.75, "uo": 0.40, "ul": 0.60, "fTI": 16, "eir": 0.45},
    "M5": {"lmed": 0.50, "uo": 0.35, "ul": 0.35, "fTI": 16, "eir": 0.45},
    "M6": {"lmed": 0.30, "uo": 0.35, "ul": 0.35, "fTI": 16, "eir": 0.45},
}

# Tabela 4 (p. 11) — requisitos por classe C. Só documental/auditoria (a exemplo de
# REQUISITOS_M): não entra na fórmula de classificação.
REQUISITOS_C = {
    "C0": {"emed": 50, "u0": 0.38, "fTI": 14},
    "C1": {"emed": 30, "u0": 0.38, "fTI": 14},
    "C2": {"emed": 20, "u0": 0.28, "fTI": 14},
    "C3": {"emed": 15, "u0": 0.18, "fTI": 15},
    "C4": {"emed": 10, "u0": 0.18, "fTI": 16},
    "C5": {"emed": 7.5, "u0": 0.18, "fTI": 16},
}

# Tabela 5 (p. 12) — a área de conflito equivalente a cada classe M, usada quando duas
# vias M se cruzam (a interseção em si vira C). Não é o que decide se UMA via é M ou C
# — isso é `HIERARQUIAS_M` abaixo — fica para o dia em que a amostragem tratar
# interseção como um caso à parte da via.
EQUIVALENCIA_M_PARA_C = {
    "M1": "C0", "M2": "C1", "M3": "C2", "M4": "C3", "M5": "C4", "M6": "C5",
}

# Hierarquias OSM que a norma trata como via de tráfego motorizado ininterrupto — a
# Tabela 1 (classe M). No Brasil corresponde a rodovia/via expressa: trecho longo sem
# cruzamento em nível, sem pedestre nem estacionamento na via. Todo o resto (avenida,
# coletora, rua local) é área de conflito por definição da norma (4.3: interseção,
# pedestre cruzando o fluxo, estacionamento na via) e usa a Tabela 3 (classe C).
HIERARQUIAS_M = {"motorway", "motorway_link", "trunk", "trunk_link"}

# Velocidade padrão por hierarquia OSM, em km/h, para quando a via não declara
# `maxspeed`. São os limites usuais por tipo de via; servem só de fallback e ficam
# registrados como inferidos na parcela.
VELOCIDADE_PADRAO_OSM = {
    "motorway": 100, "motorway_link": 60,
    "trunk": 80, "trunk_link": 50,
    "primary": 60, "primary_link": 40,
    "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 30,
    "unclassified": 40,
    "residential": 30,
    "living_street": 20,
    "service": 20,
}

# Hierarquias que a norma trata como classe P (pedonal/ciclovia), não M.
HIERARQUIAS_PEDONAIS = {"footway", "path", "pedestrian", "steps", "cycleway", "track"}


@dataclass
class ParcelaPonderacao:
    """Uma linha da Tabela 1: o parâmetro, a opção reconhecida e o peso."""

    parametro: str
    opcao: str
    valor: float
    origem: str          # "osm", "bdgd", "padrão da hierarquia" ou "neutro (sem dado)"


@dataclass
class ClasseEstimada:
    classe: str
    soma_ponderacao: float
    parcelas: list[ParcelaPonderacao] = field(default_factory=list)

    @property
    def parametros_sem_dado(self) -> list[str]:
        return [p.parametro for p in self.parcelas if p.origem.startswith("neutro")]

    @property
    def requisitos(self) -> dict:
        if self.classe.startswith("M"):
            return REQUISITOS_M.get(self.classe, {})
        return REQUISITOS_C.get(self.classe, {})

    @property
    def classe_conflito_equivalente(self) -> str:
        return EQUIVALENCIA_M_PARA_C.get(self.classe, "")


# ── Parâmetros da Tabela 1, um a um ──────────────────────────────────────────

def _maxspeed_kmh(tags: dict) -> Optional[float]:
    """Lê `maxspeed` do OSM. Aceita '60', '60 km/h', 'BR:urban'; ignora o resto."""
    bruto = str(tags.get("maxspeed", "")).strip().lower()
    if not bruto:
        return None
    if bruto.startswith("br:"):
        # Zonas de velocidade implícitas do Brasil, conforme a wiki do OSM.
        return {"br:urban": 60.0, "br:rural": 80.0,
                "br:motorway": 110.0, "br:living_street": 30.0}.get(bruto)
    numero = "".join(c for c in bruto if c.isdigit() or c == ".")
    if not numero:
        return None
    try:
        v = float(numero)
    except ValueError:
        return None
    if "mph" in bruto:
        v *= 1.609
    return v if 5 <= v <= 150 else None


def _p_velocidade(velocidade_kmh: Optional[float], hierarquia: str) -> ParcelaPonderacao:
    """
    Muito alta > 80 km/h -> 2 · 60 < alta <= 80 -> 1 · 40 < moderada <= 60 -> 0.

    A tabela não define opção abaixo de 40 km/h: a classe M pressupõe tráfego
    motorizado. Via lenta cai em 0, o mesmo peso da faixa moderada — o que deveria sair
    da classe M inteira é tratado antes, por `e_pedonal`.
    """
    origem = "osm"
    if velocidade_kmh is None:
        velocidade_kmh = VELOCIDADE_PADRAO_OSM.get(hierarquia)
        origem = "padrão da hierarquia"
        if velocidade_kmh is None:
            return ParcelaPonderacao("Velocidade", "moderada (assumida)", 0.0,
                                     "neutro (sem dado)")

    if velocidade_kmh > 80:
        return ParcelaPonderacao("Velocidade", "muito alta (> 80 km/h)", 2.0, origem)
    if velocidade_kmh > 60:
        return ParcelaPonderacao("Velocidade", "alta (60-80 km/h)", 1.0, origem)
    return ParcelaPonderacao("Velocidade", "moderada (<= 60 km/h)", 0.0, origem)


def _p_volume_trafego(hierarquia: str) -> ParcelaPonderacao:
    """
    Alto > 1200/h -> 1 · moderado 600-1200/h -> 0 · baixo < 600/h -> -1.

    Não há base pública de volume por via, e a norma manda consultar o órgão de
    trânsito. **Mas deixar esta parcela no neutro destrói o poder de discriminação da
    estimativa**: medido em 04/09/2026, com volume neutro uma avenida, uma coletora e
    uma rua local urbanas caíam todas em M4, porque em via urbana a velocidade quase
    nunca passa de 60 km/h e o que sobra (separação de faixas, interseções) não separa
    uma da outra. Classe que não distingue avenida de rua local não serve para
    estratificar amostra — que é justamente o uso aqui.

    Então o volume é inferido da hierarquia funcional do OSM. Não é um chute
    disfarçado: a classificação funcional existe para refletir função e volume, e é o
    critério que o próprio município usa ao hierarquizar a malha. Fica registrado como
    inferido, e continua sendo o parâmetro a substituir primeiro quando houver contagem
    de tráfego real.
    """
    if hierarquia in ("motorway", "trunk", "primary", "motorway_link", "trunk_link",
                      "primary_link"):
        return ParcelaPonderacao("Volume de tráfego", "alto (via arterial)", 1.0,
                                 "inferido da hierarquia")
    if hierarquia in ("secondary", "tertiary", "secondary_link", "tertiary_link"):
        return ParcelaPonderacao("Volume de tráfego", "moderado (via coletora)", 0.0,
                                 "inferido da hierarquia")
    if hierarquia in ("residential", "living_street", "service", "unclassified"):
        return ParcelaPonderacao("Volume de tráfego", "baixo (via local)", -1.0,
                                 "inferido da hierarquia")
    return ParcelaPonderacao("Volume de tráfego", "moderado (assumido)", 0.0,
                             "neutro (sem dado)")


def _p_separacao_faixas(tags: dict) -> ParcelaPonderacao:
    """Não separadas -> 1 · separadas -> 0."""
    # `dual_carriageway` é a marcação explícita, mas na prática o OSM representa pista
    # dupla como duas ways `oneway=yes` paralelas — daí o oneway ser o sinal utilizável.
    separada = (tags.get("dual_carriageway") == "yes"
                or tags.get("oneway") == "yes"
                or tags.get("divider") not in (None, "no"))
    if separada:
        return ParcelaPonderacao("Separação das faixas", "sim", 0.0, "osm")
    return ParcelaPonderacao("Separação das faixas", "não", 1.0, "osm")


def _p_densidade_intersecoes(intersecoes_por_km: Optional[float]) -> ParcelaPonderacao:
    """Alta >= 3/km -> 1 · moderada < 3/km -> 0."""
    if intersecoes_por_km is None:
        return ParcelaPonderacao("Densidade de interseções", "moderada (assumida)", 0.0,
                                 "neutro (sem dado)")
    if intersecoes_por_km >= 3.0:
        return ParcelaPonderacao("Densidade de interseções",
                                 f"alta ({intersecoes_por_km:.1f}/km)", 1.0, "osm")
    return ParcelaPonderacao("Densidade de interseções",
                             f"moderada ({intersecoes_por_km:.1f}/km)", 0.0, "osm")


def _p_veiculos_estacionados(tags: dict) -> ParcelaPonderacao:
    """Presentes -> 0,5 · ausentes -> 0."""
    chaves = [k for k in tags if k.startswith("parking:")]
    if chaves:
        proibido = all(str(tags[k]).lower() in ("no", "separate") for k in chaves)
        if proibido:
            return ParcelaPonderacao("Veículos estacionados", "ausentes", 0.0, "osm")
        return ParcelaPonderacao("Veículos estacionados", "presentes", 0.5, "osm")
    # Sem tag de estacionamento não se sabe. O neutro da tabela é "ausentes" (0).
    return ParcelaPonderacao("Veículos estacionados", "ausentes (assumido)", 0.0,
                             "neutro (sem dado)")


def _p_luminancia_ambiente(area_urbana: Optional[bool]) -> ParcelaPonderacao:
    """
    Alta -> 1 · moderada -> 0 · baixa -> -1.

    A nota b da Tabela 1 define "baixa" como área rural e "moderada" como local com
    iluminação residencial e de outdoors. O campo ARE_LOC da BDGD separa exatamente
    urbano (UB) de não urbano (NU), então serve de proxy direto para esses dois níveis.
    "Alta" (centro de metrópole com fachada iluminada) não é afirmável a partir do dado
    disponível e nunca é atribuída aqui.
    """
    if area_urbana is None:
        return ParcelaPonderacao("Luminância ambiente", "moderada (assumida)", 0.0,
                                 "neutro (sem dado)")
    if area_urbana:
        return ParcelaPonderacao("Luminância ambiente", "moderada (área urbana)", 0.0,
                                 "bdgd")
    return ParcelaPonderacao("Luminância ambiente", "baixa (área não urbana)", -1.0,
                             "bdgd")


def _p_sinalizacao() -> ParcelaPonderacao:
    """Ruins -> 0,5 · moderados ou bons -> 0. Exige vistoria; fica no neutro."""
    return ParcelaPonderacao("Sinalização e controle de tráfego",
                             "moderados ou bons (assumido)", 0.0, "neutro (sem dado)")


# ── Parâmetros da Tabela 3 (classe C), os que não são compartilhados com a M ────
# Separação das faixas, luminância ambiente e sinalização têm os mesmos pesos nas duas
# tabelas (conferido nas duas imagens da norma) — reaproveitados como estão.

def _p_velocidade_c(velocidade_kmh: Optional[float], hierarquia: str) -> ParcelaPonderacao:
    """Muito alta > 60 -> 3 · alta 40-60 -> 2 · moderada 30-40 -> 1 · baixa <= 30 -> 0.

    Limiares mais baixos que os da Tabela 1 (M): área de conflito pressupõe velocidade
    de aproximação menor que a de um trecho corrido de rodovia."""
    origem = "osm"
    if velocidade_kmh is None:
        velocidade_kmh = VELOCIDADE_PADRAO_OSM.get(hierarquia)
        origem = "padrão da hierarquia"
        if velocidade_kmh is None:
            return ParcelaPonderacao("Velocidade", "moderada (assumida)", 1.0,
                                     "neutro (sem dado)")

    if velocidade_kmh > 60:
        return ParcelaPonderacao("Velocidade", "muito alta (> 60 km/h)", 3.0, origem)
    if velocidade_kmh > 40:
        return ParcelaPonderacao("Velocidade", "alta (40-60 km/h)", 2.0, origem)
    if velocidade_kmh > 30:
        return ParcelaPonderacao("Velocidade", "moderada (30-40 km/h)", 1.0, origem)
    return ParcelaPonderacao("Velocidade", "baixa (<= 30 km/h)", 0.0, origem)


def _p_volume_trafego_c(hierarquia: str) -> ParcelaPonderacao:
    """
    Muito alto > 1200/h -> 1 · alto 600-1200 -> 0,5 · moderado 300-600 -> 0 ·
    baixo 150-300 -> -0,5 · muito baixo < 150 -> -1.

    Mesma lógica de `_p_volume_trafego` (M): sem contagem pública, infere-se da
    hierarquia funcional do OSM — só que aqui a hierarquia já vem restrita ao que NÃO
    é motorway/trunk (isso é tratado como M, não chega aqui).
    """
    if hierarquia in ("primary", "primary_link"):
        return ParcelaPonderacao("Volume de tráfego", "alto (avenida principal)", 0.5,
                                 "inferido da hierarquia")
    if hierarquia in ("secondary", "secondary_link"):
        return ParcelaPonderacao("Volume de tráfego", "moderado (via coletora)", 0.0,
                                 "inferido da hierarquia")
    if hierarquia in ("tertiary", "tertiary_link", "unclassified"):
        return ParcelaPonderacao("Volume de tráfego", "baixo (via secundária)", -0.5,
                                 "inferido da hierarquia")
    if hierarquia in ("residential", "living_street", "service"):
        return ParcelaPonderacao("Volume de tráfego", "muito baixo (via local)", -1.0,
                                 "inferido da hierarquia")
    return ParcelaPonderacao("Volume de tráfego", "moderado (assumido)", 0.0,
                             "neutro (sem dado)")


def _p_composicao_trafego(hierarquia: str) -> ParcelaPonderacao:
    """
    Motorizado apenas -> 0 · misto -> 1 · misto com alto % não motorizado -> 2.

    Parâmetro exclusivo da Tabela 3 (a M não tem — tem densidade de interseções e
    veículos estacionados no lugar). Não há contagem de pedestre em base pública, então
    infere-se da hierarquia: `living_street` prioriza pedestre por definição da própria
    tag OSM (tráfego motorizado é convidado, não é o uso principal da via); o restante
    das hierarquias que chegam aqui (avenida, coletora, rua local) é tráfego misto
    comum — carro, pedestre na calçada cruzando esquina, às vezes ciclista.
    """
    if hierarquia == "living_street":
        return ParcelaPonderacao("Composição do tráfego",
                                 "misto com alto percentual de não motorizado", 2.0,
                                 "inferido da hierarquia")
    return ParcelaPonderacao("Composição do tráfego", "misto", 1.0,
                             "inferido da hierarquia")


# ── Estimativa ───────────────────────────────────────────────────────────────

def estimar_classe_m(tags: dict,
                     area_urbana: Optional[bool] = None,
                     intersecoes_por_km: Optional[float] = None) -> ClasseEstimada:
    """
    Aplica a Tabela 1 da NBR 5101:2024 e devolve a classe M com as parcelas.

    Args:
        tags: tags OSM da via (`highway`, `maxspeed`, `oneway`, `parking:*`…).
        area_urbana: True/False vindo do ARE_LOC da BDGD; None quando não se sabe.
        intersecoes_por_km: densidade calculada da malha OSM; None quando não se sabe.

    A fórmula da norma é `classe = 6 - V_PS`, e ela manda adotar o próximo inteiro
    inferior quando o resultado não é inteiro (4.2) — daí o `floor`. O resultado é
    limitado à faixa M1-M6: a norma faz o mesmo para as classes C e P ("menor que 0
    -> C0; maior que 5 -> C5") e não existe classe para representar o que cai fora.
    """
    hierarquia = str(tags.get("highway", "")).strip().lower()
    parcelas = [
        _p_velocidade(_maxspeed_kmh(tags), hierarquia),
        _p_volume_trafego(hierarquia),
        _p_separacao_faixas(tags),
        _p_densidade_intersecoes(intersecoes_por_km),
        _p_veiculos_estacionados(tags),
        _p_luminancia_ambiente(area_urbana),
        _p_sinalizacao(),
    ]
    soma = sum(p.valor for p in parcelas)
    numero = max(1, min(6, math.floor(6 - soma)))
    return ClasseEstimada(classe=f"M{numero}", soma_ponderacao=soma, parcelas=parcelas)


def estimar_classe_c(tags: dict,
                     area_urbana: Optional[bool] = None) -> ClasseEstimada:
    """
    Aplica a Tabela 3 da NBR 5101:2024 e devolve a classe C com as parcelas.

    Mesma fórmula da M (`classe = 6 - V_PS`, floor), mas faixa C0-C5 em vez de M1-M6 —
    a norma (4.3) manda C0 quando o resultado é negativo e C5 quando passa de 5, e não
    há "C0 a mais" nem "C6": o clamp já cobre isso (`max(0, min(5, ...))`).
    """
    hierarquia = str(tags.get("highway", "")).strip().lower()
    parcelas = [
        _p_velocidade_c(_maxspeed_kmh(tags), hierarquia),
        _p_volume_trafego_c(hierarquia),
        _p_composicao_trafego(hierarquia),
        _p_separacao_faixas(tags),
        _p_luminancia_ambiente(area_urbana),
        _p_sinalizacao(),
    ]
    soma = sum(p.valor for p in parcelas)
    numero = max(0, min(5, math.floor(6 - soma)))
    return ClasseEstimada(classe=f"C{numero}", soma_ponderacao=soma, parcelas=parcelas)


def classificar_via(tags: dict,
                    area_urbana: Optional[bool] = None,
                    intersecoes_por_km: Optional[float] = None) -> ClasseEstimada:
    """
    Decide entre a Tabela 1 (M) e a Tabela 3 (C) pela hierarquia OSM e aplica a que
    corresponde. É o ponto de entrada que `vias_osm` deve chamar — não
    `estimar_classe_m` direto, que sempre devolve M e foi o bug corrigido em
    08/09/2026 (ver docstring do módulo).

    Só motorway/trunk (e os `_link`) vão para M; todo o resto — avenida, coletora, rua
    local, ou hierarquia ausente/não reconhecida — vai para C, por ser o caso comum na
    malha urbana brasileira (ver a distribuição real citada no topo do arquivo).
    """
    hierarquia = str(tags.get("highway", "")).strip().lower()
    if hierarquia in HIERARQUIAS_M:
        return estimar_classe_m(tags, area_urbana=area_urbana,
                                intersecoes_por_km=intersecoes_por_km)
    return estimar_classe_c(tags, area_urbana=area_urbana)


def e_pedonal(tags: dict) -> bool:
    """A via é de pedestre/ciclista? Nesse caso a classe é P, não M nem C."""
    return str(tags.get("highway", "")).strip().lower() in HIERARQUIAS_PEDONAIS


__all__ = [
    "REQUISITOS_M", "REQUISITOS_C", "EQUIVALENCIA_M_PARA_C", "HIERARQUIAS_M",
    "VELOCIDADE_PADRAO_OSM", "HIERARQUIAS_PEDONAIS", "ParcelaPonderacao",
    "ClasseEstimada", "estimar_classe_m", "estimar_classe_c", "classificar_via",
    "e_pedonal",
]
