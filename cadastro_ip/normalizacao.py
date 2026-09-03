"""
Normalização de colunas dos inputs (seção 4) e de nomes de logradouros (seção 10.4).

Reconhece cada conceito pela função semântica (não pelo nome literal), aceitando
variações de caixa e acentos. Quando uma coluna esperada não é encontrada nem por
sinônimo, o módulo retorna None para o conceito — o caller (UI) deve perguntar ao
usuário qual coluna usar.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd


# ── Sinônimos aceitos por conceito (case-insensitive, ignora acentos) ────────
# Lista não exaustiva — novos sinônimos podem ser adicionados sem mudar a lógica.
SINONIMOS_COLUNAS: dict[str, list[str]] = {
    "id_ponto":     ["etiqueta", "id pip", "id_ponto", "idg", "identificador", "ip", "codigo", "id",
                     "cod id", "cod_id"],
    "tecnologia":   ["tecnologia", "tipo lampada", "tipo_lampada", "tipolampada", "tipo de lampada",
                     "tipo de lampada houer", "tecnologia houer",
                     "tecnologia considerada", "tecnologia existente"],
    "potencia":     ["potencia", "potenciaLampada", "potencia (w)", "potencia w", "pot",
                     "potencia lampada", "potencia considerada", "potencia existente"],
    "quantidade":   ["qtd original", "qtd lampadas", "quantidade", "quantidadelampadas", "qtd", "qtde",
                     "quantidade de lampadas", "quantidade de lampadas houer",
                     "quantidade considerada", "quantidade existente"],
    # Eixos separados. Inclui os nomes de GIS (coord_y / northing / utm n) porque
    # cadastro exportado de QGIS/ArcGIS vem projetado — a conversão para graus é
    # tratada em `coordenadas.py`, mas a coluna precisa ser encontrada antes.
    # Eixos separados. Inclui os nomes de GIS (coord_y / northing / utm_n) porque
    # cadastro exportado de QGIS/ArcGIS vem projetado — converter para graus é papel de
    # `coordenadas.py`, mas a coluna precisa ser encontrada antes. Nomes de uma letra
    # ("X", "Y") ficam de fora de propósito: o casamento por prefixo do scoring não
    # filtra sinônimo curto, e "X" pegaria qualquer coluna começada por x.
    "latitude":     ["latitude", "latitude coletada", "latitude original", "latitude cad", "lat",
                     "latitude3", "latitude y", "coord y", "coordenada y", "utm n",
                     "northing", "norte utm"],
    "longitude":    ["longitude", "longitude coletada", "longitude original", "longitude cad",
                     "long", "lon", "lng", "longitude3", "longitude x", "coord x",
                     "coordenada x", "utm e", "easting", "leste utm"],
    # Par em uma célula só — o formato que sai de "copiar coordenada" no Google Maps.
    # Sem "gps" solto na lista: "GPS_LAT" é eixo, não par, e o prefixo confundiria os
    # dois conceitos. Quem decide no fim é `coordenadas.parece_par`, sobre os dados.
    "coordenadas":  ["coordenadas", "coordenada", "coordenada geografica", "lat long",
                     "latlong", "lat lon", "latlon", "geolocalizacao",
                     "georreferenciamento", "coordenada gps"],
    "logradouro":   ["endereco", "logradouro", "rua", "endereço"],
    "bairro":       ["bairro"],
    "local":        ["local", "localizacao", "tipo de local", "localizacão"],
    "classe_via":   ["classe", "classe via", "classeiluminacao", "classificacao viaria",
                     "classe de iluminacao", "classe da via de conflito 1", "tipo de via"],
    "classe_pedestre": ["classe pedestre", "classe passeio",
                        "classe da via de pedestres 1", "classe da via de pedestres 2"],
    "material":     ["material", "luminaria", "tipo_luminaria", "modelo da luminaria"],
    "municipio":    ["municipio", "nome_cidade", "cidade", "nome municipio"],
    "uf":           ["uf", "estado", "sigla uf"],
}

# Conceitos obrigatórios para o pipeline (faltar dispara interação com o usuário)
CONCEITOS_OBRIGATORIOS_CADASTRO = ["id_ponto", "tecnologia", "potencia", "quantidade"]
CONCEITOS_OBRIGATORIOS_INSPECAO = ["id_ponto", "tecnologia", "potencia"]
CONCEITOS_RECOMENDADOS_CADASTRO = ["latitude", "longitude", "logradouro", "bairro", "local"]


# ── Tipos atuais vs novos (seção 4) ───────────────────────────────────────────
# Cadastros tipo Timóteo têm versões "atual" e "novo" das mesmas colunas.
# Sempre prevalece a "atual"; a "nova" é proposta de projeto e é descartada.
TOKENS_ATUAL = ["atual", "existente", "original"]
TOKENS_NOVO  = ["novo", "nova", "proposta", "projeto", "futuro"]


# ── Abreviações de tipo de via (seção 10.4) ───────────────────────────────────
ABREV_VIA = {
    r"\br\.?\b":    "rua",
    r"\bav\.?\b":   "avenida",
    r"\btv\.?\b":   "travessa",
    r"\bpc\.?\b":   "praca",
    r"\bpca\.?\b":  "praca",
    r"\bal\.?\b":   "alameda",
    r"\brod\.?\b":  "rodovia",
    r"\bestr\.?\b": "estrada",
    r"\blgo\.?\b":  "largo",
    r"\bvl\.?\b":   "vila",
    r"\bjd\.?\b":   "jardim",
}


@dataclass
class MapeamentoColunas:
    """Resultado da normalização: o que foi mapeado e o que falta."""

    mapeados: dict[str, str] = field(default_factory=dict)   # conceito → nome real da coluna
    faltando: list[str] = field(default_factory=list)        # conceitos obrigatórios sem match
    ambiguos: dict[str, list[str]] = field(default_factory=dict)  # conceito → várias colunas candidatas

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna uma cópia do df com as colunas mapeadas renomeadas para o nome canônico."""
        rename = {real: conceito for conceito, real in self.mapeados.items()}
        return df.rename(columns=rename)


def limpar_id_serie(s: pd.Series) -> pd.Series:
    """
    Normaliza uma coluna de IDs para string canônica antes de qualquer merge.

    Trata o caso comum em que IDs numéricos vêm do Excel como float
    (`220693372.0`) e ao virar string têm um `.0` final que faz o merge
    silenciosamente não casar com a outra base (`'220693372'`).

    Retorna uma Series de strings, trimadas, sem `.0` final.
    """
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _slug(s: str) -> str:
    """Forma canônica para comparação: minúsculo, sem acentos, espaços e pontuação colapsados."""
    s = _strip_accents(str(s)).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _coluna_e_versao_nova(nome: str) -> bool:
    """Detecta se um nome de coluna se refere à versão 'nova/proposta', não à 'atual'."""
    sl = _slug(nome)
    return any(tok in sl.split() for tok in TOKENS_NOVO)


# Tokens que indicam que a coluna NÃO é a principal — penalizam o ranking
# Penalidade DURA (-60): variantes claramente não-principais (duplicadas, propostas, etc.)
TOKENS_PENALIDADE = [
    "duplicado", "segundo nivel", "segundo nível", "objectid",
    "se houver", "proposta", "futuro", "anterior", "novo", "nova",
]
# Penalidade LEVE (-15): tokens que sugerem variante secundária mas podem aparecer
# em colunas válidas (ex: "Endereço Cadastro" na inspeção é o endereço original).
# Não deve derrubar a coluna se for o único candidato.
TOKENS_PENALIDADE_LEVE = ["cadastro", "original"]
# Tokens que indicam que a coluna é a principal — bonificam.
#
# ATENÇÃO: "houer" aqui NÃO é branding do portal, é o texto que aparece de fato no
# cabeçalho das planilhas de cadastro recebidas ("Tipo de Lâmpada_Houer",
# "Quantidade de Lâmpadas_HOUER"), marcando a coluna já tratada em campo em oposição à
# coluna crua da prefeitura. É dado de entrada de terceiro, não identidade nossa:
# removê-lo cegaria o detector e a coluna tratada perderia para a crua.
# Coberto por tests/test_normalizacao.py::TestDetectorPreferenciaHouer.
TOKENS_BONIFICACAO = ["houer"]

# Bonificacoes por conceito - quando a coluna contem esses tokens, ganha pontos extras.
BONIFICACOES_POR_CONCEITO: dict[str, list[str]] = {
    "classe_via": ["via", "conflito", "iluminacao", "iluminacao", "viaria"],
    "latitude":   ["coletada", "houer"],
    "longitude":  ["coletada", "houer"],
    # Para IAE/ID: 'Considerada' > 'Existente' > 'Cadastro'
    "tecnologia": ["considerada", "houer", "existente"],
    "potencia":   ["considerada", "existente"],
    "quantidade": ["considerada", "existente"],
}


def _score_candidato(coluna: str, sinonimos_slug: set[str], conceito: str | None = None) -> int:
    """
    Calcula um score para a coluna como candidata de um conceito.
    Maior score = melhor candidato.
    Sinonimos muito curtos (len<=2) exigem match exato para evitar falsos
    positivos (ex: 'id' aparece como substring em 'quantidade').
    """
    col_slug = _slug(coluna)
    score = 0

    # Match exato com algum sinônimo (slug a slug)
    if col_slug in sinonimos_slug:
        score += 100
    else:
        # Algum sinônimo contém o slug da coluna (ou vice-versa) — match parcial
        contido = any(col_slug == s for s in sinonimos_slug)
        starts = any(col_slug.startswith(s) or s.startswith(col_slug) for s in sinonimos_slug)
        # Sinonimos curtos (<=2 chars) nunca fazem match por substring
        # para evitar 'id' matchear 'quantidade', 'consolidado', etc.
        sinonimos_longos = {s for s in sinonimos_slug if len(s) > 2}
        if contido:
            score += 80
        elif starts:
            score += 40
        elif any(s in col_slug for s in sinonimos_longos):
            score += 25

    # Bonus/bonificação só aplicam se já houve algum match com sinônimo
    # (evita que tokens secundários inflem score de coluna sem relação)
    if score > 0:
        # Bonifica tokens preferidos (ver TOKENS_BONIFICACAO: alias de planilha, nao marca)
        if any(t in col_slug for t in TOKENS_BONIFICACAO):
            score += 15

        # Bonificações específicas por conceito
        if conceito and conceito in BONIFICACOES_POR_CONCEITO:
            bonus_tokens = [_slug(t) for t in BONIFICACOES_POR_CONCEITO[conceito]]
            score += 25 * sum(1 for t in bonus_tokens if t in col_slug)

    # Penaliza tokens de variantes secundárias
    if any(t in col_slug for t in [_slug(p) for p in TOKENS_PENALIDADE]):
        score -= 60

    # Penalidade leve para tokens que NÃO desclassificam a coluna sozinhos.
    # Ex: "Endereço Cadastro" na inspeção: tem "cadastro" mas é o endereço útil
    # quando a inspeção não traz logradouro próprio.
    if any(t in col_slug for t in [_slug(p) for p in TOKENS_PENALIDADE_LEVE]):
        score -= 15

    # Penaliza nomes muito longos (geralmente são variantes)
    if len(coluna) > 35:
        score -= 5

    return score


def detectar_colunas(
    df: pd.DataFrame,
    obrigatorios: list[str],
    recomendados: list[str] | None = None,
) -> MapeamentoColunas:
    """
    Detecta colunas semanticamente para cada conceito, usando a tabela de sinônimos.

    Quando há colunas "atual" e "nova" duplicando o mesmo conceito, prevalece a "atual".
    Quando há múltiplos candidatos, usa um sistema de scoring que penaliza variantes
    secundárias (ex: "Tecnologia do ponto duplicado") e prioriza nomes diretos.
    """
    recomendados = recomendados or []
    colunas_reais = list(df.columns)

    resultado = MapeamentoColunas()

    for conceito, sinonimos in SINONIMOS_COLUNAS.items():
        if conceito not in obrigatorios and conceito not in recomendados:
            continue
        sinonimos_slug = {_slug(s) for s in sinonimos}

        # Score cada coluna real
        scored: list[tuple[int, str]] = []
        for col in colunas_reais:
            if col is None or (isinstance(col, float) and pd.isna(col)):
                continue
            score = _score_candidato(str(col), sinonimos_slug, conceito)
            if score > 0:
                scored.append((score, str(col)))

        if not scored:
            if conceito in obrigatorios:
                resultado.faltando.append(conceito)
            continue

        # Filtra versões "nova/proposta" se houver alternativa "atual"
        atuais = [(s, c) for s, c in scored if not _coluna_e_versao_nova(c)]
        if atuais:
            scored = atuais

        # Ordena por score desc; em empate, preserva ordem de aparecimento
        scored.sort(key=lambda x: -x[0])

        # Se top-1 está empatado com top-2, registra como ambíguo
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            resultado.ambiguos[conceito] = [c for _, c in scored if _ == scored[0][0]]
        elif len(scored) > 1:
            # Não empatado, mas mais de uma opção válida — registra como info
            outras = [c for _, c in scored[1:] if scored[1][0] > 20]
            if outras:
                resultado.ambiguos[conceito] = [scored[0][1]] + outras[:3]

        resultado.mapeados[conceito] = scored[0][1]

    return resultado


def _remover_linhas_total(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas de totais (onde alguma string seja exatamente 'TOTAL')."""
    if df is None or df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        mask = mask | (df[col].astype(str).str.strip().str.upper() == "TOTAL")
    return df[~mask].copy()


def normalizar_cadastro(df: pd.DataFrame) -> tuple[pd.DataFrame, MapeamentoColunas]:
    """Aplica detecção e renomeação no cadastro principal."""
    mapa = detectar_colunas(df, CONCEITOS_OBRIGATORIOS_CADASTRO, CONCEITOS_RECOMENDADOS_CADASTRO)
    df_mapeado = mapa.aplicar(df)
    return _remover_linhas_total(df_mapeado), mapa


def normalizar_inspecao(df: pd.DataFrame) -> tuple[pd.DataFrame, MapeamentoColunas]:
    """Aplica detecção e renomeação na base de inspeção."""
    recomendados = ["quantidade", "logradouro", "classe_via", "classe_pedestre", "latitude", "longitude"]
    mapa = detectar_colunas(df, CONCEITOS_OBRIGATORIOS_INSPECAO, recomendados)
    df_mapeado = mapa.aplicar(df)
    return _remover_linhas_total(df_mapeado), mapa


# Conceitos obrigatórios para IAE/ID: id_ponto é OPCIONAL
# (se ausente, IDs sintéticos serão gerados pelo roteamento)
CONCEITOS_OBRIGATORIOS_IAE_ID = ["tecnologia", "potencia", "quantidade"]
CONCEITOS_RECOMENDADOS_IAE_ID = ["id_ponto", "latitude", "longitude", "logradouro", "local"]


def normalizar_iae_id(df: pd.DataFrame) -> tuple[pd.DataFrame, MapeamentoColunas]:
    """Aplica detecção e renomeação nas bases IAE e ID.
    
    id_ponto é RECOMENDADO (não obrigatório) — se a base não tiver
    identificador real, o roteamento gera IDs sintéticos automaticamente.
    Isso evita que colunas não-identificadoras (ex: 'Quantidade Cadastro')
    sejam incorretamente mapeadas por conter 'id' como substring.
    """
    mapa = detectar_colunas(df, CONCEITOS_OBRIGATORIOS_IAE_ID, CONCEITOS_RECOMENDADOS_IAE_ID)
    df_mapeado = mapa.aplicar(df)
    return _remover_linhas_total(df_mapeado), mapa


# ── Normalização de logradouros (seção 10.4) ──────────────────────────────────
def normalizar_logradouro(nome: str) -> str:
    """
    Forma canônica para comparação de ruas:
    - minúsculo, sem acentos
    - abreviações expandidas (R. → Rua, Av. → Avenida, etc.)
    - espaços colapsados
    """
    if nome is None or (isinstance(nome, float) and pd.isna(nome)):
        return ""
    s = _slug(nome)
    for padrao, expansao in ABREV_VIA.items():
        s = re.sub(padrao, expansao, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chave_logradouro(logradouro: str, bairro: str | None = None) -> str:
    """
    Chave de agrupamento para propagação de Classe Via.
    Quando houver Bairro, usa Logradouro + Bairro para evitar homônimos de
    ruas em bairros diferentes (seção 10.4).
    """
    log_norm = normalizar_logradouro(logradouro)
    if bairro is None or (isinstance(bairro, float) and pd.isna(bairro)) or str(bairro).strip() == "":
        return log_norm
    bairro_norm = _slug(bairro)
    return f"{log_norm} || {bairro_norm}"


__all__ = [
    "MapeamentoColunas",
    "normalizar_cadastro",
    "normalizar_inspecao",
    "normalizar_iae_id",
    "detectar_colunas",
    "normalizar_logradouro",
    "chave_logradouro",
    "limpar_id_serie",
    "SINONIMOS_COLUNAS",
    "CONCEITOS_OBRIGATORIOS_CADASTRO",
    "CONCEITOS_OBRIGATORIOS_INSPECAO",
]
