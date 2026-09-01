"""
Leitura da classificação viária e identificação das vias principais do município.

Duas responsabilidades, ambas defensivas quanto ao vocabulário do cadastro recebido:

1. **Classe de iluminação** — o mercado usa três vocabulários misturados: o da
   NBR 5101:2012/2018 (V1–V5 para tráfego motorizado, P1–P4 para pedestres), o da
   EN 13201/CIE adotado por boa parte das consultorias e já usado em
   `cadastro_ip/classe_via.py` (M1–M6, C0–C5, P1–P6) e o texto puro da hierarquia
   viária municipal ("arterial", "coletora", "local", "trânsito rápido"). O módulo
   normaliza os três para um código canônico **sem descartar** o que não reconhece:
   rótulo desconhecido vira estrato próprio, porque para efeito de amostragem o que
   importa é cobrir todos os rótulos presentes na base, não julgar a nomenclatura.

2. **Vias principais** — as que precisam obrigatoriamente ter ponto inspecionado.
   Uma via entra por dois caminhos independentes: pelo **tipo** (avenida, rodovia,
   estrada, anel, contorno, marginal — inclusive designações do tipo BR-040, MG-424)
   ou pela **classe** mais exigente presente na base (V1/V2, M1/M2, C0/C1, arterial).
   O ranking usa a contagem de pontos de IP do logradouro como proxy de extensão —
   é o único dado de porte disponível em cadastro de IP sem malha viária externa.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd


# ── Normalização de classe de iluminação ──────────────────────────────────────
# Rank de exigência: 0 = mais exigente. Usado para decidir quais classes puxam a via
# para o grupo das principais e para ordenar a tabela de cobertura no relatório.
RANK_EXIGENCIA: dict[str, int] = {
    # NBR 5101 — tráfego motorizado
    "V1": 0, "V2": 1, "V3": 2, "V4": 3, "V5": 4,
    # EN 13201 / CIE — vias motorizadas (M) e áreas de conflito (C)
    "M1": 0, "M2": 1, "M3": 2, "M4": 3, "M5": 4, "M6": 5,
    "C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5,
    # Pedestres — nunca são "via principal", mas entram como estrato próprio
    "P1": 10, "P2": 11, "P3": 12, "P4": 13, "P5": 14, "P6": 15,
}

# Hierarquia viária em texto → código canônico da NBR 5101
_TEXTO_PARA_CLASSE: list[tuple[str, str]] = [
    ("transito rapido", "V1"),
    ("transito r", "V1"),
    ("expressa", "V1"),
    ("arterial", "V2"),
    ("coletora", "V3"),
    ("local", "V4"),
    ("pedestre", "P3"),
    ("passeio", "P3"),
    ("ciclovia", "P3"),
]

# Classes que, por si só, qualificam a via como principal (as duas mais exigentes
# de cada vocabulário motorizado).
CLASSES_PRINCIPAIS = {"V1", "V2", "M1", "M2", "C0", "C1"}

ROTULO_SEM_CLASSE = "(sem classe)"


def _sem_acento(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn"
    )


def _slug(texto) -> str:
    s = _sem_acento(texto).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def normalizar_classe(valor) -> str:
    """
    Converte o valor da coluna de classificação viária em um código canônico.

    Aceita "V3", "v 3", "Classe V3", "M2", "C1", "P2", "Arterial", "Via Coletora".
    Valor vazio vira `ROTULO_SEM_CLASSE`; valor não reconhecido volta em caixa alta,
    preservado como estrato próprio.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ROTULO_SEM_CLASSE
    bruto = str(valor).strip()
    if not bruto or bruto.lower() in {"nan", "none", "-", "--"}:
        return ROTULO_SEM_CLASSE

    alfanumerico = re.search(r"\b([VMCP])\s*[-_]?\s*([0-9])\b", bruto.upper())
    if alfanumerico:
        return f"{alfanumerico.group(1)}{alfanumerico.group(2)}"

    slug = _slug(bruto)
    for chave, codigo in _TEXTO_PARA_CLASSE:
        if chave in slug:
            return codigo
    return bruto.upper()


def rank_exigencia(classe: str) -> int:
    """Rank de exigência da classe (0 = mais exigente). Desconhecida vai para o fim."""
    return RANK_EXIGENCIA.get(str(classe).upper(), 50)


def classe_e_principal(classe: str) -> bool:
    return str(classe).upper() in CLASSES_PRINCIPAIS


# ── Tipo de via a partir do nome do logradouro ────────────────────────────────
# Ordem importa: o primeiro padrão que casar define o tipo. "rodovia" antes de "rua"
# porque "Rodovia ..." não pode cair no genérico.
_PADROES_TIPO: list[tuple[str, str]] = [
    (r"\b(rodovia|rod)\b",                 "rodovia"),
    (r"\b(br|mg|sp|rj|ba|pr|rs|sc|go|mt|ms|pe|ce|pa|ma|pb|rn|al|se|pi|to|ro|ac|am|rr|ap|es|df)\s*-?\s*\d{2,3}\b",
                                            "rodovia"),
    (r"\b(anel|contorno)\b",               "anel viário"),
    (r"\bmarginal\b",                      "marginal"),
    (r"\b(avenida|av)\b",                  "avenida"),
    (r"\b(estrada|estr|rodoanel)\b",       "estrada"),
    (r"\b(alameda|al)\b",                  "alameda"),
    (r"\b(praca|pca|pc)\b",                "praça"),
    (r"\b(travessa|tv)\b",                 "travessa"),
    (r"\b(largo|lgo)\b",                   "largo"),
    (r"\b(viela|beco)\b",                  "viela"),
    (r"\b(rua|r)\b",                       "rua"),
]

# Tipos que caracterizam via estruturante do município.
TIPOS_PRINCIPAIS = {"rodovia", "anel viário", "marginal", "avenida", "estrada"}


def tipo_via(logradouro) -> str:
    """
    Deduz o tipo do logradouro pelo nome ("avenida", "rodovia", "rua", ...).

    Retorna "indefinido" quando o nome não traz o tipo — comum em cadastro que grava
    só "Sete de Setembro". Nesse caso a via ainda pode entrar como principal pela classe.
    """
    slug = _slug(logradouro)
    if not slug:
        return "indefinido"
    for padrao, tipo in _PADROES_TIPO:
        if re.search(padrao, slug):
            return tipo
    return "indefinido"


def tipo_e_principal(tipo: str) -> bool:
    return str(tipo).lower() in TIPOS_PRINCIPAIS


# ── Identificação das vias principais ─────────────────────────────────────────
@dataclass
class ViaPrincipal:
    """Uma via candidata a cobertura obrigatória na amostra."""

    chave: str            # chave normalizada do logradouro (agrupa grafias)
    nome: str             # nome como aparece no cadastro (primeira ocorrência)
    tipo: str
    classes: tuple[str, ...]
    pontos: int
    motivos: tuple[str, ...]

    @property
    def motivo_texto(self) -> str:
        return " + ".join(self.motivos)


def identificar_vias_principais(
    df: pd.DataFrame,
    col_chave: str = "_chave_logradouro",
    col_nome: str = "_logradouro",
    col_tipo: str = "_tipo_via",
    col_classe: str = "_classe",
    teto: int | None = 20,
    minimo_pontos: int = 1,
) -> list[ViaPrincipal]:
    """
    Ranqueia as vias estruturantes do município.

    Uma via entra na lista se o tipo for principal (avenida, rodovia, estrada, anel,
    marginal) **ou** se alguma de suas classes for das mais exigentes (V1/V2, M1/M2,
    C0/C1). O ranking é por número de pontos de IP — proxy de extensão da via — e o
    `teto` limita quantas viram cobertura obrigatória, para que um município com 300
    avenidas não consuma a amostra inteira só com cotas.

    Args:
        teto: máximo de vias devolvidas (None = todas).
        minimo_pontos: descarta vias com menos pontos que isso (ruído de cadastro).

    Returns:
        Lista de `ViaPrincipal` ordenada por relevância (classe mais exigente primeiro,
        depois número de pontos).
    """
    if col_chave not in df.columns or df.empty:
        return []

    vias: list[ViaPrincipal] = []
    for chave, grupo in df.groupby(col_chave, dropna=False):
        if not str(chave).strip():
            continue
        if len(grupo) < minimo_pontos:
            continue

        tipos = [t for t in grupo[col_tipo].dropna().unique().tolist()] if col_tipo in grupo else []
        tipo = next((t for t in tipos if tipo_e_principal(t)), tipos[0] if tipos else "indefinido")
        classes = tuple(sorted(
            {c for c in grupo[col_classe].dropna().unique().tolist() if c != ROTULO_SEM_CLASSE},
            key=rank_exigencia,
        )) if col_classe in grupo else ()

        motivos = []
        if tipo_e_principal(tipo):
            motivos.append(f"tipo de via: {tipo}")
        classes_exigentes = [c for c in classes if classe_e_principal(c)]
        if classes_exigentes:
            motivos.append(f"classe exigente: {', '.join(classes_exigentes)}")
        if not motivos:
            continue

        nome = str(grupo[col_nome].iloc[0]) if col_nome in grupo else str(chave)
        vias.append(
            ViaPrincipal(
                chave=str(chave),
                nome=nome,
                tipo=tipo,
                classes=classes,
                pontos=int(len(grupo)),
                motivos=tuple(motivos),
            )
        )

    vias.sort(
        key=lambda v: (
            min((rank_exigencia(c) for c in v.classes), default=99),
            -v.pontos,
            v.nome,
        )
    )
    return vias[:teto] if teto else vias


def vias_para_dataframe(vias: list[ViaPrincipal]) -> pd.DataFrame:
    """Tabela das vias principais para exibição na UI e para a aba do relatório."""
    if not vias:
        return pd.DataFrame(
            columns=["Via", "Tipo", "Classes", "Pontos no cadastro", "Motivo", "_chave"]
        )
    return pd.DataFrame(
        [
            {
                "Via": v.nome,
                "Tipo": v.tipo,
                "Classes": ", ".join(v.classes) if v.classes else "—",
                "Pontos no cadastro": v.pontos,
                "Motivo": v.motivo_texto,
                "_chave": v.chave,
            }
            for v in vias
        ]
    )


__all__ = [
    "RANK_EXIGENCIA", "CLASSES_PRINCIPAIS", "TIPOS_PRINCIPAIS", "ROTULO_SEM_CLASSE",
    "ViaPrincipal", "normalizar_classe", "rank_exigencia", "classe_e_principal",
    "tipo_via", "tipo_e_principal", "identificar_vias_principais", "vias_para_dataframe",
]
