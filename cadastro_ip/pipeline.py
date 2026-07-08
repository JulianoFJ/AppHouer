"""
Orquestração do pipeline de tratamento de cadastros de IP.

Combina todos os módulos:
  normalizacao → tecnologia → reator → aneel_2590 → extrapolacao →
  roteamento → considerada → classe_via → saidas/* → relatorio

Entry point: `executar(...)` — recebe os 4 DataFrames + município/UF e
retorna um `ResultadoPipeline` com todos os artefatos prontos para download.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import (
    aneel_2590,
    classe_via,
    extrapolacao,
    normalizacao,
    reator,
    roteamento,
    tecnologia,
)
from .normalizacao import limpar_id_serie


@dataclass
class ResultadoPipeline:
    # Identificação
    municipio: str
    uf: str
    tempo_operacao: aneel_2590.TempoOperacao | None

    # Métricas
    total_cadastro: int
    total_inspecao: int
    total_iae: int
    total_id: int
    fator_extrapolacao: extrapolacao.Extrapolacao

    # DataFrames intermediários (úteis para preview e debug)
    cadastro_normalizado: pd.DataFrame
    inspecao_normalizada: pd.DataFrame
    iae_normalizada: pd.DataFrame
    id_normalizada: pd.DataFrame

    # Roteamento dos pontos
    roteamento: roteamento.ResultadoRoteamento

    # Cruzamento ponto-a-ponto cadastro × inspeção
    comparacao: pd.DataFrame

    # Propagação de Classe Via
    propagacao_classe: classe_via.ResultadoPropagacao

    # Tratamentos (4 DataFrames, prontos para alimentar as abas)
    tratamento_convencional: pd.DataFrame
    tratamento_led_iv: pd.DataFrame
    tratamento_iae: pd.DataFrame
    tratamento_id: pd.DataFrame

    # Avisos para o relatório
    avisos: list[str] = field(default_factory=list)
    normalizacoes_tecnologia: dict[str, int] = field(default_factory=dict)
    # Códigos de tecnologia presentes nos inputs mas não reconhecidos pelo
    # classificador. Estrutura: { fonte: { codigo_bruto: quantidade } }.
    # `fonte` é uma de: 'cadastro', 'inspecao', 'iae', 'id'.
    codigos_desconhecidos: dict[str, dict[str, int]] = field(default_factory=dict)
    # Diferença entre qtd_corrigida e (qtd_recebida + IAE_novos + ID_novos).
    # Idealmente 0 — qualquer valor diferente indica que algum ponto foi perdido
    # ou duplicado em algum tratamento.
    desbalanceamento: float = 0.0

    # Mapeamentos de coluna (úteis para debug e relatório)
    mapeamento_cadastro: normalizacao.MapeamentoColunas | None = None
    mapeamento_inspecao: normalizacao.MapeamentoColunas | None = None

    # Bytes dos arquivos .xlsx gerados (preenchidos por `saidas/`)
    xlsx_classificacao: bytes | None = None
    xlsx_analise_cadastro: bytes | None = None
    xlsx_quantitativo: bytes | None = None

    # Tabela consolidada Tec × Pot pronta para a aba Resultado (precomputada,
    # evita SUMIFS frágeis no .xlsx). Colunas: tec, pot, qtd_recebida,
    # qtd_corrigida, trat_conv, trat_led, trat_iae, trat_id.
    resultado_por_tec_pot: pd.DataFrame | None = None

    relatorio_texto: str = ""


def _construir_comparacao(
    cadastro: pd.DataFrame,
    inspecao: pd.DataFrame,
    col_id: str = "id_ponto",
) -> pd.DataFrame:
    """
    Cruzamento ponto-a-ponto entre cadastro e inspeção (aba `Comparação` do
    arquivo Análise Cadastro). Adiciona flags de divergência.
    """
    cad = cadastro.copy()
    insp = inspecao.copy()
    # IDs vindos do Excel como float (220693372.0) precisam virar string canônica
    # antes do merge — senão "220693372.0" != "220693372" e o join falha silencioso.
    cad[col_id] = limpar_id_serie(cad[col_id])
    insp[col_id] = limpar_id_serie(insp[col_id])

    merged = cad.merge(
        insp,
        on=col_id,
        how="left",
        suffixes=("_cad", "_insp"),
    )

    def _flag_tec(row):
        cad_tec = row.get("codigo_tecnologia_cad")
        insp_tec = row.get("codigo_tecnologia_insp")
        if pd.isna(insp_tec) or insp_tec is None:
            return "Sem inspeção"
        return "Igual" if (cad_tec or "") == (insp_tec or "") else "Diferente"

    def _flag_pot(row):
        cad_pot = row.get("potencia_cad")
        insp_pot = row.get("potencia_insp")
        if pd.isna(insp_pot):
            return "Sem inspeção"
        try:
            return "Igual" if abs(float(cad_pot) - float(insp_pot)) <= 1e-6 else "Diferente"
        except (TypeError, ValueError):
            return "Diferente"

    merged["flag_tecnologia"] = merged.apply(_flag_tec, axis=1)
    merged["flag_potencia"] = merged.apply(_flag_pot, axis=1)

    def _flag_geral(row):
        ft, fp = row["flag_tecnologia"], row["flag_potencia"]
        if ft == "Sem inspeção" or fp == "Sem inspeção":
            return "Sem inspeção"
        if ft == "Diferente" and fp == "Diferente":
            return "Tecnologia e Potência Diferentes"
        if ft == "Diferente":
            return "Tecnologia Diferente"
        if fp == "Diferente":
            return "Potência Diferente"
        return "Tecnologia e Potência Iguais"

    merged["divergencia"] = merged.apply(_flag_geral, axis=1)
    
    # Flags binárias conforme regra 1.5
    merged["tecnologia_diferente"] = merged["flag_tecnologia"].apply(lambda x: 1 if x == "Diferente" else 0)
    merged["potencia_diferente"] = merged["flag_potencia"].apply(lambda x: 1 if x == "Diferente" else 0)
    
    return merged


def _construir_tratamento_convencional(
    rot: roteamento.ResultadoRoteamento,
    inspecao: pd.DataFrame,
    fator: int,
    col_id: str = "id_ponto",
) -> pd.DataFrame:
    """
    Tratamento Convencional: pontos com cadastro=Conv + inspeção=Conv divergentes.
    Quantidade de pontos = quantidade de divergências × fator (seção 9.2).
    """
    base = rot.pontos_convencional.copy()
    if base.empty:
        return base.assign(quantidade_extrapolada=pd.Series(dtype="int64"))

    insp_indexed = inspecao.set_index(limpar_id_serie(inspecao[col_id]))
    cad_id = limpar_id_serie(base[col_id])

    insp_tec, insp_pot, insp_qtd = [], [], []
    for pid in cad_id:
        if pid in insp_indexed.index:
            row = insp_indexed.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            insp_tec.append(row.get("codigo_tecnologia"))
            insp_pot.append(row.get("potencia"))
            insp_qtd.append(row.get("quantidade"))
        else:
            insp_tec.append(None)
            insp_pot.append(None)
            insp_qtd.append(None)

    base["tecnologia_inspecao"] = insp_tec
    base["potencia_inspecao"] = insp_pot
    base["quantidade_inspecao"] = insp_qtd
    base["fator_extrapolacao"] = fator
    # 1 divergência por linha (cada ponto é uma divergência) — extrapolada por fator
    base["quantidade_extrapolada"] = fator
    return base


def _construir_tratamento_led_iv(
    rot: roteamento.ResultadoRoteamento,
    inspecao: pd.DataFrame,
    fator: int,
    col_id: str = "id_ponto",
) -> pd.DataFrame:
    """
    Tratamento LED IV: cadastro=Conv + inspeção=LED (pontos efetivamente trocados).
    Quantidade Considerada = quantidade da inspeção × fator (seção 9.2).
    Coluna 'Executado' default = Sim (seção 11).
    """
    base = rot.pontos_led_iv.copy()
    if base.empty:
        return base.assign(quantidade_considerada=pd.Series(dtype="float64"), executado=pd.Series(dtype="object"))

    insp_indexed = inspecao.set_index(limpar_id_serie(inspecao[col_id]))
    cad_id = limpar_id_serie(base[col_id])

    insp_tec, insp_pot, insp_qtd = [], [], []
    for pid in cad_id:
        if pid in insp_indexed.index:
            row = insp_indexed.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            insp_tec.append(row.get("codigo_tecnologia"))
            insp_pot.append(row.get("potencia"))
            insp_qtd.append(row.get("quantidade", 1) or 1)
        else:
            insp_tec.append(None)
            insp_pot.append(None)
            insp_qtd.append(1)

    base["tecnologia_inspecao"] = insp_tec
    base["potencia_inspecao"] = insp_pot
    base["quantidade_inspecao"] = insp_qtd
    base["fator_extrapolacao"] = fator
    base["quantidade_considerada"] = [
        (q if q is not None and pd.notna(q) else 1) * fator for q in insp_qtd
    ]
    base["executado"] = "Sim"
    return base


def _construir_tratamento_iae_ou_id(
    pontos_existentes: pd.DataFrame,
    pontos_novos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tratamento IAE / ID: bases já completas (não amostrais), aplicação direta
    sem fator de extrapolação (seção 9.2 — observação).
    """
    out_existentes = pontos_existentes.copy()
    out_existentes["origem"] = "EXISTENTE_NO_CADASTRO"
    out_novos = pontos_novos.copy()
    out_novos["origem"] = "NOVO"
    combinado = pd.concat([out_existentes, out_novos], ignore_index=True, sort=False)
    if not combinado.empty:
        qtd_col = None
        for nome in ("quantidade_considerada", "Quantidade", "quantidade", "quantidadeLampadas"):
            if nome in combinado.columns:
                qtd_col = nome
                break
        if qtd_col:
            combinado["quantidade_considerada"] = combinado[qtd_col]
        else:
            combinado["quantidade_considerada"] = 1
    return combinado


def _agregar_qtd_por_tec_pot(
    df: pd.DataFrame,
    col_tec: str,
    col_pot: str,
    col_qtd: str,
) -> dict[tuple, float]:
    """
    Agrega quantidade por chave (tecnologia, potencia). Retorna dict {(tec, pot): qtd}.
    Trata Tec/Pot None/NaN — ignora linhas onde tec for vazio.
    """
    if df is None or df.empty or col_tec not in df.columns:
        return {}
    out: dict[tuple, float] = {}
    for _, row in df.iterrows():
        tec = row.get(col_tec)
        pot = row.get(col_pot)
        qtd = row.get(col_qtd)
        if tec is None or (isinstance(tec, float) and pd.isna(tec)):
            tec = "SEM CLASS."
        if qtd is None or (isinstance(qtd, float) and pd.isna(qtd)):
            qtd = 1
        try:
            qtd_f = float(qtd)
            if qtd_f < 0:
                qtd_f = 0.0
        except (TypeError, ValueError):
            qtd_f = 1.0
        try:
            pot_f = float(pot) if pot is not None and not (isinstance(pot, float) and pd.isna(pot)) else 0.0
        except (TypeError, ValueError):
            pot_f = 0.0
        key = (str(tec).strip().upper(), pot_f)
        out[key] = out.get(key, 0) + qtd_f
    return out


def _construir_resultado_por_tec_pot(
    cadastro_norm: pd.DataFrame,
    trat_convencional: pd.DataFrame,
    trat_led_iv: pd.DataFrame,
    pontos_iae_novos: pd.DataFrame,
    pontos_id_novos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida a tabela Tec × Pot da aba Resultado em Python — substitui
    SUMIFS frágeis no .xlsx.

    Para cada combinação (Tec, Pot) que aparece em qualquer fonte, calcula:
      - qtd_recebida: vinda do cadastro original
      - trat_conv:    saldo do Tratamento Convencional para essa (Tec, Pot)
                      = +qty se essa (Tec, Pot) é o estado *inspeção* (novo)
                        -qty se essa (Tec, Pot) é o estado *cadastro* (antigo)
      - trat_led:     idem para Tratamento LED IV
      - trat_iae:     +qty se (Tec, Pot) corresponde a um ponto IAE NOVO
      - trat_id:      idem para ID NOVO
      - qtd_corrigida = qtd_recebida + trat_conv + trat_led + trat_iae + trat_id

    O total geral da coluna `qtd_corrigida` deve ser:
      total_cadastro_recebido + len(IAE_novos) + len(ID_novos)
    (LED IV e Convencional são net-zero — só transferem entre Tec × Pot).
    """
    # 1) Cadastro Recebido por Tec x Pot
    _qtd_candidatas = ("quantidade", "Quantidade", "quantidade_considerada", "quantidadeLampadas")
    qtd_col_cad = next((c for c in _qtd_candidatas if c in cadastro_norm.columns), None)
    if qtd_col_cad is None:
        cadastro_norm = cadastro_norm.assign(quantidade=1)
        qtd_col_cad = "quantidade"
    recebido = _agregar_qtd_por_tec_pot(
        cadastro_norm, "codigo_tecnologia", "potencia", qtd_col_cad
    )

    # 2) Convencional — adições (estado inspeção) e remoções (estado cadastro)
    conv_add = _agregar_qtd_por_tec_pot(
        trat_convencional, "tecnologia_inspecao", "potencia_inspecao", "quantidade_extrapolada"
    ) if not trat_convencional.empty else {}
    conv_rem = _agregar_qtd_por_tec_pot(
        trat_convencional, "codigo_tecnologia", "potencia", "quantidade_extrapolada"
    ) if not trat_convencional.empty else {}

    # 3) LED IV — adições no estado inspeção, remoções no estado cadastro
    led_add = _agregar_qtd_por_tec_pot(
        trat_led_iv, "tecnologia_inspecao", "potencia_inspecao", "quantidade_considerada"
    ) if not trat_led_iv.empty else {}
    led_rem = _agregar_qtd_por_tec_pot(
        trat_led_iv, "codigo_tecnologia", "potencia", "quantidade_considerada"
    ) if not trat_led_iv.empty else {}

    # 4) IAE novos — soma Quantidade Considerada (ou Existente, fallback)
    # Ordem de prioridade: quantidade_considerada > quantidade (mapeado pelo normalizador,
    # que ja preferiu 'Considerada' sobre 'Existente' via bonificacao de score)
    iae_add = {}
    if not pontos_iae_novos.empty:
        qtd_col_iae = _detectar_coluna_qtd_priorizando_considerada(pontos_iae_novos)
        if qtd_col_iae and "codigo_tecnologia" in pontos_iae_novos.columns:
            iae_add = _agregar_qtd_por_tec_pot(
                pontos_iae_novos, "codigo_tecnologia", "potencia", qtd_col_iae
            )

    # 5) ID novos — soma Quantidade Considerada (ou Existente, fallback)
    id_add = {}
    if not pontos_id_novos.empty:
        qtd_col_id = _detectar_coluna_qtd_priorizando_considerada(pontos_id_novos)
        if qtd_col_id and "codigo_tecnologia" in pontos_id_novos.columns:
            id_add = _agregar_qtd_por_tec_pot(
                pontos_id_novos, "codigo_tecnologia", "potencia", qtd_col_id
            )

    # 6) Junta todas as chaves (Tec, Pot) que aparecem em qualquer fonte
    todas_chaves = set(recebido) | set(conv_add) | set(conv_rem) | set(led_add) | set(led_rem) | set(iae_add) | set(id_add)

    linhas = []
    for (tec, pot) in sorted(todas_chaves, key=lambda x: (_ordem_tec(x[0]), x[1])):
        qrec  = recebido.get((tec, pot), 0)
        tconv = conv_add.get((tec, pot), 0) - conv_rem.get((tec, pot), 0)
        tled  = led_add.get((tec, pot), 0) - led_rem.get((tec, pot), 0)
        tiae  = iae_add.get((tec, pot), 0)
        tid   = id_add.get((tec, pot), 0)
        qcor  = qrec + tconv + tled + tiae + tid
        linhas.append({
            "tec": tec,
            "pot": pot,
            "qtd_recebida": qrec,
            "qtd_corrigida": qcor,
            "trat_conv": tconv,
            "trat_led":  tled,
            "trat_iae":  tiae,
            "trat_id":   tid,
        })

    df_out = pd.DataFrame(linhas, columns=[
        "tec", "pot", "qtd_recebida", "qtd_corrigida",
        "trat_conv", "trat_led", "trat_iae", "trat_id",
    ])
    # Remove linhas onde TODAS as colunas numéricas são zero (combinações que
    # apareceram em algum dict mas não geraram nenhum saldo final — ruído).
    cols_numericas = ["qtd_recebida", "qtd_corrigida", "trat_conv", "trat_led", "trat_iae", "trat_id"]
    df_out = df_out[df_out[cols_numericas].abs().sum(axis=1) > 0].reset_index(drop=True)
    return df_out


_ORDEM_TEC = ["LED", "VS", "VM", "VMT", "MT", "FL", "IN", "GLOBO", "ORNAMENTAL", "REBATEDOR", "PROJETOR"]


def _ordem_tec(tec: str) -> int:
    try:
        return _ORDEM_TEC.index(str(tec).upper())
    except ValueError:
        return 99


def _detectar_coluna_qtd(df: pd.DataFrame) -> str | None:
    """Encontra a coluna de quantidade no DataFrame (várias capitalizações possíveis)."""
    if df is None or df.empty:
        return None
    for nome in ("quantidade_considerada", "Quantidade", "quantidade", "quantidadeLampadas"):
        if nome in df.columns:
            return nome
    return None


def _soma_quantidade_iae_id(df: pd.DataFrame) -> float:
    """
    Soma a quantidade de luminárias de uma base IAE/ID novos.
    Prioriza 'Quantidade Considerada' > 'Quantidade Existente' > 'quantidade'.
    Cada linha conta como 1 ponto se não houver coluna de quantidade.
    """
    if df is None or df.empty:
        return 0.0
    col = _detectar_coluna_qtd_priorizando_considerada(df)
    if col is None:
        return float(len(df))
    valores = pd.to_numeric(df[col], errors="coerce").fillna(1)
    return float(valores.sum())


def _detectar_coluna_qtd_priorizando_considerada(df: pd.DataFrame) -> str | None:
    """
    Para bases IAE/ID: prioriza 'Quantidade Considerada' (decisão final) sobre
    'Quantidade Existente' sobre 'quantidade' (campo genérico normalizado).
    Cada coluna representa o total de luminarias naquele ponto/linha.
    """
    if df is None or df.empty:
        return None
    # Tenta nomes exatos primeiro (pós-normalização pelo mapeamento de score)
    for nome in (
        "quantidade_considerada",   # nome normalizado ideal
        "Quantidade Considerada",   # nome original se normalizacao nao renomeou
        "quantidade existente",
        "Quantidade Existente",
        "quantidade",               # fallback generico pos-normalizacao
        "Quantidade",
    ):
        if nome in df.columns:
            return nome
    return None


def executar(
    cadastro: pd.DataFrame,
    inspecao: pd.DataFrame,
    iae: pd.DataFrame,
    id_: pd.DataFrame,
    municipio: str,
    uf: str,
    codigo_ibge: int | None = None,
    horas_operacao_manual: int | None = None,
    minutos_operacao_manual: int | None = None,
) -> ResultadoPipeline:
    """
    Executa o pipeline completo. Não gera os .xlsx ainda — isso fica para `saidas/*`,
    que recebem este resultado e produzem os bytes.

    Args:
        cadastro/inspecao/iae/id_: DataFrames brutos lidos das planilhas do usuário.
        municipio, uf: identificação confirmada pelo usuário.
        codigo_ibge: opcional, desempata homônimos no lookup ANEEL.
        horas/minutos_operacao_manual: fallback se o município não estiver na base ANEEL.
    """
    avisos: list[str] = []

    # ── 1) Normalização de colunas ──────────────────────────────────────────
    cad_norm, mapa_cad = normalizacao.normalizar_cadastro(cadastro)
    insp_norm, mapa_insp = normalizacao.normalizar_inspecao(inspecao)
    iae_norm, _ = normalizacao.normalizar_iae_id(iae)
    id_norm, _ = normalizacao.normalizar_iae_id(id_)

    if mapa_cad.faltando:
        avisos.append(f"Colunas obrigatórias não identificadas no cadastro: {mapa_cad.faltando}")
    if mapa_insp.faltando:
        avisos.append(f"Colunas obrigatórias não identificadas na inspeção: {mapa_insp.faltando}")

    # ── 2) Classificação de tecnologia ──────────────────────────────────────
    if "tecnologia" in cad_norm.columns:
        cad_norm = tecnologia.aplicar_classificacao(cad_norm, "tecnologia")
    if "tecnologia" in insp_norm.columns:
        insp_norm = tecnologia.aplicar_classificacao(insp_norm, "tecnologia")
    if "tecnologia" in iae_norm.columns:
        iae_norm = tecnologia.aplicar_classificacao(iae_norm, "tecnologia")
    if "tecnologia" in id_norm.columns:
        id_norm = tecnologia.aplicar_classificacao(id_norm, "tecnologia")

    normalizacoes_tec = tecnologia.normalizacoes_aplicadas(cad_norm, "tecnologia") if "tecnologia" in cad_norm.columns else {}

    # Captura códigos de tecnologia não reconhecidos em cada fonte.
    # Em vez de mascarar em "SEM CLASS." e quebrar o balanceamento, registramos
    # exatamente quais códigos brutos apareceram e quantos pontos cada um afeta.
    desconhecidos: dict[str, dict[str, int]] = {}
    for fonte, df in (("cadastro", cad_norm), ("inspecao", insp_norm),
                       ("iae", iae_norm), ("id", id_norm)):
        if "tecnologia" in df.columns:
            d = tecnologia.codigos_desconhecidos(df, "tecnologia")
            if d:
                desconhecidos[fonte] = d
    if desconhecidos:
        partes = []
        for fonte, d in desconhecidos.items():
            itens = ", ".join(f"{k!r}={v}" for k, v in sorted(d.items(), key=lambda x: -x[1]))
            partes.append(f"{fonte}: {itens}")
        avisos.append(
            "Códigos de tecnologia não reconhecidos pelo classificador "
            "(adicione-os em `tecnologia.VARIANTES_PARA_CODIGO` ou corrija na fonte) — "
            + " | ".join(partes)
        )

    # ── 3) Perda de reator (cadastro e inspeção) ────────────────────────────
    if "potencia" in cad_norm.columns:
        cad_norm = reator.aplicar_reator(cad_norm)
    if "potencia" in insp_norm.columns:
        insp_norm = reator.aplicar_reator(insp_norm)

    if reator.teve_1000w(cad_norm) or reator.teve_1000w(insp_norm):
        avisos.append(reator.ALERTA_1000W)

    # ── 4) Tempo de operação (ANEEL) ────────────────────────────────────────
    tempo = aneel_2590.buscar(municipio, uf, codigo_ibge=codigo_ibge)
    if tempo is None:
        if horas_operacao_manual is not None and minutos_operacao_manual is not None:
            tempo = aneel_2590.TempoOperacao(
                municipio=municipio,
                uf=uf,
                horas=int(horas_operacao_manual),
                minutos=int(minutos_operacao_manual),
                codigo_ibge=codigo_ibge,
            )
        else:
            avisos.append(
                f"Tempo de operação não encontrado na base ANEEL 2590/2019 para {municipio}/{uf}. "
                "Forneça manualmente no formato HHhMMmin."
            )

    # ── 5) Roteamento dos pontos ────────────────────────────────────────────
    rot = roteamento.rotear(cad_norm, insp_norm, iae_norm, id_norm)

    # ── 6) Fator de extrapolação ────────────────────────────────────────────
    fator = extrapolacao.calcular_fator(len(cad_norm), len(insp_norm))
    if fator.fator == 0:
        avisos.append(
            f"Fator de extrapolação = 0 (cadastro: {fator.total_cadastro}, "
            f"amostra: {fator.total_amostra}). Revise se o tamanho da amostra é adequado."
        )

    # ── 7) Tratamentos ──────────────────────────────────────────────────────
    trat_conv = _construir_tratamento_convencional(rot, insp_norm, fator.fator)
    trat_led_iv = _construir_tratamento_led_iv(rot, insp_norm, fator.fator)
    trat_iae = _construir_tratamento_iae_ou_id(rot.pontos_iae_existentes, rot.pontos_iae_novos)
    trat_id = _construir_tratamento_iae_ou_id(rot.pontos_id_existentes, rot.pontos_id_novos)

    avisos.append(
        "A coluna 'Executado' do Tratamento LED IV foi preenchida com 'Sim' por padrão. "
        "Revise individualmente os pontos onde a execução ainda não foi feita e altere para 'Não' antes de finalizar."
    )

    # ── 8) Comparação ponto-a-ponto ─────────────────────────────────────────
    comp = _construir_comparacao(cad_norm, insp_norm)

    # ── 9) Propagação de Classe Via ─────────────────────────────────────────
    prop = classe_via.propagar(cad_norm, insp_norm)
    if prop.aviso_sem_bairro:
        avisos.append(
            "Cadastro não tem coluna `Bairro` — homônimos de logradouros podem ter sido "
            "agrupados incorretamente. Revise."
        )

    # ── 10) Resultado consolidado Tec × Pot (para a aba Resultado) ───────────
    resultado_tp = _construir_resultado_por_tec_pot(
        cad_norm, trat_conv, trat_led_iv, rot.pontos_iae_novos, rot.pontos_id_novos
    )

    # ── 11) Invariante de balanceamento ─────────────────────────────────────
    # qtd_corrigida total deve ser exatamente:
    #   qtd_recebida total + IAE novos + ID novos
    # (Convencional e LED IV são net-zero — só transferem entre Tec×Pot).
    # Qualquer divergência indica que algum ponto foi perdido ou duplicado.
    qtd_recebida_total = float(resultado_tp["qtd_recebida"].sum())
    qtd_corrigida_total = float(resultado_tp["qtd_corrigida"].sum())
    iae_novos_qtd = _soma_quantidade_iae_id(rot.pontos_iae_novos)
    id_novos_qtd = _soma_quantidade_iae_id(rot.pontos_id_novos)
    esperado = qtd_recebida_total + iae_novos_qtd + id_novos_qtd
    desbalanceamento = qtd_corrigida_total - esperado
    if abs(desbalanceamento) > 0.5:  # tolerância para floats
        avisos.append(
            f"Invariante de balanceamento NÃO bate: "
            f"qtd_corrigida={qtd_corrigida_total:.0f}, "
            f"esperado={esperado:.0f} "
            f"(recebida={qtd_recebida_total:.0f} + IAE_novos={iae_novos_qtd:.0f} + "
            f"ID_novos={id_novos_qtd:.0f}). "
            f"Diferença={desbalanceamento:+.0f}. "
            "Isso geralmente significa códigos de tecnologia não reconhecidos "
            "ou pontos com tecnologia/potência ausentes. Revise os alertas acima."
        )

    return ResultadoPipeline(
        municipio=municipio,
        uf=uf,
        tempo_operacao=tempo,
        total_cadastro=len(cad_norm),
        total_inspecao=len(insp_norm),
        total_iae=len(iae_norm),
        total_id=len(id_norm),
        fator_extrapolacao=fator,
        cadastro_normalizado=cad_norm,
        inspecao_normalizada=insp_norm,
        iae_normalizada=iae_norm,
        id_normalizada=id_norm,
        roteamento=rot,
        comparacao=comp,
        propagacao_classe=prop,
        tratamento_convencional=trat_conv,
        tratamento_led_iv=trat_led_iv,
        tratamento_iae=trat_iae,
        tratamento_id=trat_id,
        avisos=avisos,
        normalizacoes_tecnologia=normalizacoes_tec,
        mapeamento_cadastro=mapa_cad,
        mapeamento_inspecao=mapa_insp,
        resultado_por_tec_pot=resultado_tp,
        codigos_desconhecidos=desconhecidos,
        desbalanceamento=desbalanceamento,
    )


__all__ = ["ResultadoPipeline", "executar"]
