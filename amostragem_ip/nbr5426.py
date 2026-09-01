"""
Planos de amostragem por atributos — ABNT NBR 5426:1985.

Implementa a Tabela 1 (códigos literais de tamanho de amostra por tamanho de lote
× nível de inspeção) e a Tabela 2 (amostragem simples, números de aceitação e
rejeição por NQA), incluindo o mecanismo das **setas**: quando a célula da tabela
não tem plano definido, a norma manda usar o primeiro plano acima ou abaixo — e o
tamanho da amostra muda junto, porque muda a letra-código.

A norma foi cancelada pela ABNT em 2018 sem substituta nacional (a referência
internacional equivalente é a ISO 2859-1), mas segue sendo a citada nos termos de
referência de concessões e PPPs de iluminação pública — que é o uso aqui: dimensionar
a amostra de verificação do cadastro que a inspeção de campo vai percorrer.

Convenções desta implementação, todas visíveis no resultado:

  - `regime` altera o tamanho da amostra apenas em "atenuado"; em "severo" o tamanho
    é o mesmo do normal e o que muda é o critério de aceitação (a norma tem tabela
    própria de Ac/Re para o regime severo — aqui o Ac severo é o do plano imediatamente
    mais rigoroso, comportamento declarado em `PlanoAmostragem.observacoes`).
  - Se o tamanho da amostra alcançar o lote, a norma manda inspecionar 100% do lote —
    o plano devolvido reflete isso em `inspecao_total`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Letras-código, na ordem da norma (não existem "I" nem "O", para não confundir) ──
LETRAS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R"]

NIVEIS_ESPECIAIS = ["S-1", "S-2", "S-3", "S-4"]
NIVEIS_GERAIS = ["I", "II", "III"]
NIVEIS = NIVEIS_ESPECIAIS + NIVEIS_GERAIS
NIVEL_PADRAO = "II"   # nível geral II é o default da própria norma

# ── Tabela 1 — código literal por faixa de lote × nível de inspeção ───────────
# (limite_superior_do_lote, {nível: letra}). O último item cobre lotes acima de 500.000.
_TABELA_1: list[tuple[int, dict[str, str]]] = [
    (8,      {"S-1": "A", "S-2": "A", "S-3": "A", "S-4": "A", "I": "A", "II": "A", "III": "B"}),
    (15,     {"S-1": "A", "S-2": "A", "S-3": "A", "S-4": "A", "I": "A", "II": "B", "III": "C"}),
    (25,     {"S-1": "A", "S-2": "A", "S-3": "B", "S-4": "B", "I": "B", "II": "C", "III": "D"}),
    (50,     {"S-1": "A", "S-2": "B", "S-3": "B", "S-4": "C", "I": "C", "II": "D", "III": "E"}),
    (90,     {"S-1": "B", "S-2": "B", "S-3": "C", "S-4": "C", "I": "C", "II": "E", "III": "F"}),
    (150,    {"S-1": "B", "S-2": "B", "S-3": "C", "S-4": "D", "I": "D", "II": "F", "III": "G"}),
    (280,    {"S-1": "B", "S-2": "C", "S-3": "D", "S-4": "E", "I": "E", "II": "G", "III": "H"}),
    (500,    {"S-1": "B", "S-2": "C", "S-3": "D", "S-4": "E", "I": "F", "II": "H", "III": "J"}),
    (1200,   {"S-1": "C", "S-2": "C", "S-3": "E", "S-4": "F", "I": "G", "II": "J", "III": "K"}),
    (3200,   {"S-1": "C", "S-2": "D", "S-3": "E", "S-4": "G", "I": "H", "II": "K", "III": "L"}),
    (10000,  {"S-1": "C", "S-2": "D", "S-3": "F", "S-4": "G", "I": "J", "II": "L", "III": "M"}),
    (35000,  {"S-1": "C", "S-2": "D", "S-3": "F", "S-4": "H", "I": "K", "II": "M", "III": "N"}),
    (150000, {"S-1": "D", "S-2": "E", "S-3": "G", "S-4": "J", "I": "L", "II": "N", "III": "P"}),
    (500000, {"S-1": "D", "S-2": "E", "S-3": "G", "S-4": "J", "I": "M", "II": "P", "III": "Q"}),
    (10**12, {"S-1": "D", "S-2": "E", "S-3": "H", "S-4": "K", "I": "N", "II": "Q", "III": "R"}),
]

# ── Tamanho da amostra por letra-código ───────────────────────────────────────
TAMANHO_NORMAL = {
    "A": 2, "B": 3, "C": 5, "D": 8, "E": 13, "F": 20, "G": 32, "H": 50,
    "J": 80, "K": 125, "L": 200, "M": 315, "N": 500, "P": 800, "Q": 1250, "R": 2000,
}
# Regime atenuado (inspeção reduzida): a norma usa amostras menores.
TAMANHO_ATENUADO = {
    "A": 2, "B": 2, "C": 2, "D": 3, "E": 5, "F": 8, "G": 13, "H": 20,
    "J": 32, "K": 50, "L": 80, "M": 125, "N": 200, "P": 315, "Q": 500, "R": 800,
}
REGIMES = ["normal", "severo", "atenuado"]

# ── Tabela 2 — números de aceitação (amostragem simples) ──────────────────────
# A tabela mestra é uma diagonal: para cada NQA existe uma letra a partir da qual o
# plano passa a ser definido, e daí para baixo o número de aceitação percorre sempre
# a mesma sequência. Acima dessa letra a célula é seta "para baixo" (usar o primeiro
# plano abaixo); depois do fim da sequência, seta "para cima".
_SEQUENCIA_ACEITACAO = [0, 1, 2, 3, 5, 7, 10, 14, 21]

# NQA (%) → letra em que a sequência começa
_LETRA_INICIAL_POR_NQA: dict[float, str] = {
    0.65: "H",
    1.0:  "G",
    1.5:  "F",
    2.5:  "E",
    4.0:  "D",
    6.5:  "C",
    10.0: "B",
}
NQAS = sorted(_LETRA_INICIAL_POR_NQA)
NQA_PADRAO = 2.5


class NBR5426Error(ValueError):
    """Parâmetro fora do domínio da norma (nível, NQA ou regime desconhecido)."""


def _indice(letra: str) -> int:
    return LETRAS.index(letra)


def letra_codigo(tamanho_lote: int, nivel: str = NIVEL_PADRAO) -> str:
    """Tabela 1: código literal do tamanho de amostra para o lote e o nível de inspeção."""
    if nivel not in NIVEIS:
        raise NBR5426Error(f"Nível de inspeção desconhecido: {nivel!r}. Use um de {NIVEIS}.")
    if tamanho_lote < 2:
        raise NBR5426Error("A norma cobre lotes a partir de 2 unidades.")
    for limite, mapa in _TABELA_1:
        if tamanho_lote <= limite:
            return mapa[nivel]
    return _TABELA_1[-1][1][nivel]   # inalcançável — guarda defensiva


def aceitacao(letra: str, nqa: float) -> int | str:
    """
    Tabela 2: número de aceitação para a letra-código e o NQA.

    Devolve um inteiro quando existe plano, ou a string "baixo"/"cima" indicando
    que a norma manda usar o primeiro plano naquela direção.
    """
    if nqa not in _LETRA_INICIAL_POR_NQA:
        raise NBR5426Error(f"NQA fora dos tabelados aqui: {nqa}. Use um de {NQAS}.")
    base = _indice(_LETRA_INICIAL_POR_NQA[nqa])
    pos = _indice(letra) - base
    if pos < 0:
        return "baixo"
    if pos >= len(_SEQUENCIA_ACEITACAO):
        return "cima"
    return _SEQUENCIA_ACEITACAO[pos]


@dataclass
class PlanoAmostragem:
    """Plano de amostragem resolvido, com a memória de cálculo para o relatório."""

    tamanho_lote: int
    nivel: str
    nqa: float
    regime: str
    letra_original: str          # letra da Tabela 1, antes de resolver seta
    letra_codigo: str            # letra efetivamente usada
    tamanho_amostra: int
    numero_aceitacao: int
    numero_rejeicao: int
    seta_aplicada: str | None = None     # "baixo" | "cima" | None
    inspecao_total: bool = False
    observacoes: list[str] = field(default_factory=list)

    @property
    def fracao_do_lote(self) -> float:
        return self.tamanho_amostra / self.tamanho_lote if self.tamanho_lote else 0.0

    def resumo(self) -> str:
        if self.inspecao_total:
            return (f"Lote {self.tamanho_lote} · nível {self.nivel} · NQA {self.nqa}% → "
                    f"inspeção 100% ({self.tamanho_amostra} pontos)")
        return (f"Lote {self.tamanho_lote} · nível {self.nivel} · NQA {self.nqa}% · "
                f"regime {self.regime} → letra {self.letra_codigo}, n = {self.tamanho_amostra}, "
                f"Ac = {self.numero_aceitacao} / Re = {self.numero_rejeicao}")


def plano(
    tamanho_lote: int,
    nivel: str = NIVEL_PADRAO,
    nqa: float = NQA_PADRAO,
    regime: str = "normal",
) -> PlanoAmostragem:
    """
    Resolve o plano de amostragem simples da NBR 5426 para um lote.

    Args:
        tamanho_lote: número de pontos do parque de IP (o "lote" a inspecionar).
        nivel: nível de inspeção — "I", "II" (padrão), "III" ou os especiais "S-1".."S-4".
        nqa: nível de qualidade aceitável em % (0,65 a 10).
        regime: "normal", "severo" ou "atenuado".

    Returns:
        PlanoAmostragem com letra-código, tamanho de amostra, Ac/Re e a memória
        de cálculo (seta aplicada, inspeção total, observações).
    """
    if regime not in REGIMES:
        raise NBR5426Error(f"Regime desconhecido: {regime!r}. Use um de {REGIMES}.")

    letra_orig = letra_codigo(tamanho_lote, nivel)
    letra = letra_orig
    obs: list[str] = []
    seta: str | None = None

    ac = aceitacao(letra, nqa)
    if isinstance(ac, str):
        seta = ac
        base = _indice(_LETRA_INICIAL_POR_NQA[nqa])
        # "baixo": o primeiro plano definido abaixo é exatamente a letra em que a
        # sequência começa. "cima": o último plano definido da coluna do NQA.
        alvo = base if ac == "baixo" else base + len(_SEQUENCIA_ACEITACAO) - 1
        letra = LETRAS[min(alvo, len(LETRAS) - 1)]
        ac = aceitacao(letra, nqa)
        direcao = "abaixo" if seta == "baixo" else "acima"
        obs.append(
            f"A célula (letra {letra_orig} × NQA {nqa}%) não tem plano na Tabela 2; "
            f"conforme a norma usou-se o primeiro plano {direcao} (letra {letra}), "
            f"o que também altera o tamanho da amostra."
        )

    if regime == "severo":
        # A norma tem tabela própria de Ac/Re para inspeção severa. O efeito prático é
        # exigir mais: aqui aplica-se o número de aceitação do plano imediatamente mais
        # rigoroso da mesma coluna, mantendo o tamanho de amostra do regime normal.
        ac_severo = aceitacao(LETRAS[max(_indice(letra) - 1, 0)], nqa)
        ac = ac_severo if isinstance(ac_severo, int) else 0
        obs.append(
            "Regime severo: o tamanho da amostra é o do regime normal e o critério de "
            "aceitação foi endurecido para o do plano imediatamente mais rigoroso."
        )

    tabela_tamanho = TAMANHO_ATENUADO if regime == "atenuado" else TAMANHO_NORMAL
    n = tabela_tamanho[letra]

    if regime == "atenuado":
        # O tamanho cai (Tabela 3 da norma), então o critério de aceitação não pode
        # continuar sendo o do regime normal — sobre uma amostra menor ele seria mais
        # permissivo que a norma. Usa-se o Ac do plano normal de mesmo tamanho de amostra.
        letra_equivalente = next(
            (l for l in LETRAS if TAMANHO_NORMAL[l] == n), letra
        )
        ac_atenuado = aceitacao(letra_equivalente, nqa)
        if isinstance(ac_atenuado, int):
            ac = ac_atenuado
        obs.append(
            "Regime atenuado: tamanho de amostra reduzido conforme a norma, com o "
            f"critério de aceitação do plano normal de mesmo tamanho (letra {letra_equivalente})."
        )

    inspecao_total = False
    if n >= tamanho_lote:
        n = tamanho_lote
        inspecao_total = True
        obs.append(
            "O tamanho de amostra da norma alcança ou supera o lote — a NBR 5426 manda "
            "inspecionar 100% das unidades."
        )

    return PlanoAmostragem(
        tamanho_lote=int(tamanho_lote),
        nivel=nivel,
        nqa=float(nqa),
        regime=regime,
        letra_original=letra_orig,
        letra_codigo=letra,
        tamanho_amostra=int(n),
        numero_aceitacao=int(ac),
        numero_rejeicao=int(ac) + 1,
        seta_aplicada=seta,
        inspecao_total=inspecao_total,
        observacoes=obs,
    )


__all__ = [
    "LETRAS", "NIVEIS", "NIVEIS_GERAIS", "NIVEIS_ESPECIAIS", "NIVEL_PADRAO",
    "NQAS", "NQA_PADRAO", "REGIMES", "TAMANHO_NORMAL", "TAMANHO_ATENUADO",
    "PlanoAmostragem", "NBR5426Error", "letra_codigo", "aceitacao", "plano",
]
