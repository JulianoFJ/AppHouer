"""
Roteamento dos pontos nos 4 caminhos de tratamento (seção 7).

Chave única de cruzamento: identificador do ponto (id_ponto após normalização).
Cada ponto segue UM ÚNICO dos quatro caminhos. Ordem de avaliação:

  1. Ponto está na Base IAE? → Tratamento IAE
  2. Ponto está na Base ID? → Tratamento ID
  3. Caso contrário (é IV — Iluminação Viária), cruza com a inspeção:
       Cadastro=Conv E Inspeção=LED        → Tratamento LED IV
       Cadastro=Conv E Inspeção=Conv c/divergência → Tratamento Convencional
       Cadastro=LED  E Inspeção=LED s/divergência → fica no inventário (sem tratamento de troca)
       Cadastro=LED  E Inspeção=Conv → mantém LED (exceção da seção 8)

IAE/ID têm distinção adicional (seção 7.1):
  - JÁ EXISTE no cadastro → realocação (sai de IV, entra em IAE/ID — não soma)
  - NÃO EXISTE no cadastro → adição (linha nova no Cadastro Corrigido — soma)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .normalizacao import limpar_id_serie


class Caminho(str, Enum):
    IAE_EXISTENTE = "IAE_EXISTENTE"  # estava no cadastro como IV → realoca
    IAE_NOVO       = "IAE_NOVO"      # não estava no cadastro → adiciona
    ID_EXISTENTE  = "ID_EXISTENTE"
    ID_NOVO        = "ID_NOVO"
    LED_IV         = "LED_IV"        # convencional no cadastro, LED na inspeção
    CONVENCIONAL   = "CONVENCIONAL"  # divergência tec/pot dentro de convencionais
    LED_OK         = "LED_OK"        # LED em ambos, sem divergência (só inventário)
    LED_MANTIDO    = "LED_MANTIDO"   # cadastro LED, inspeção Conv → mantém LED
    NAO_INSPECIONADO = "NAO_INSPECIONADO"  # IV sem inspeção (representado pela amostra via fator)


@dataclass
class ResultadoRoteamento:
    df_roteado: pd.DataFrame              # cadastro completo + coluna 'caminho'
    pontos_iae_existentes: pd.DataFrame   # subset IAE que já estava no cadastro
    pontos_iae_novos: pd.DataFrame        # subset IAE ausente do cadastro
    pontos_id_existentes: pd.DataFrame
    pontos_id_novos: pd.DataFrame
    pontos_led_iv: pd.DataFrame
    pontos_convencional: pd.DataFrame     # divergências para tratamento conv
    pontos_led_ok: pd.DataFrame
    pontos_led_mantido: pd.DataFrame
    pontos_nao_inspecionados: pd.DataFrame
    resumo: dict[str, int] = field(default_factory=dict)


def _ha_divergencia(cad_tec, cad_pot, insp_tec, insp_pot) -> bool:
    """Divergência se tecnologia OU potência diferirem."""
    tec_diff = (cad_tec or "") != (insp_tec or "")
    pot_diff = False
    try:
        if pd.notna(cad_pot) and pd.notna(insp_pot):
            pot_diff = abs(float(cad_pot) - float(insp_pot)) > 1e-6
        elif pd.isna(cad_pot) != pd.isna(insp_pot):
            pot_diff = True
    except (TypeError, ValueError):
        pot_diff = str(cad_pot) != str(insp_pot)
    return tec_diff or pot_diff


def rotear(
    cadastro: pd.DataFrame,
    inspecao: pd.DataFrame,
    iae: pd.DataFrame,
    id_: pd.DataFrame,
    col_id: str = "id_ponto",
    col_codigo: str = "codigo_tecnologia",
    col_potencia: str = "potencia",
) -> ResultadoRoteamento:
    """
    Roteia cada ponto do universo (cadastro + IAE novos + ID novos) em um dos
    caminhos enumerados em `Caminho`.

    Espera que `tecnologia.aplicar_classificacao()` já tenha sido aplicado a
    cadastro e inspeção (coluna `codigo_tecnologia` presente).

    Args:
        cadastro / inspecao / iae / id_: DataFrames já normalizados
            (colunas semânticas: id_ponto, codigo_tecnologia, potencia, ...).
        col_id, col_codigo, col_potencia: nomes das colunas-chave.

    Returns:
        ResultadoRoteamento com o DataFrame consolidado e subsets por caminho.
    """
    cad = cadastro.copy()
    insp = inspecao.copy()

    cad_id_str = limpar_id_serie(cad[col_id])

    # Conjuntos de IDs (strings para garantir match consistente)
    ids_cadastro = set(cad_id_str)
    ids_inspecao = set(limpar_id_serie(insp[col_id])) if col_id in insp.columns else set()
    ids_iae      = set(limpar_id_serie(iae[col_id])) if col_id in iae.columns else set()
    ids_id       = set(limpar_id_serie(id_[col_id])) if col_id in id_.columns else set()

    # Index para lookup rápido de inspeção por id
    insp_indexed = insp.set_index(limpar_id_serie(insp[col_id])) if not insp.empty else insp

    # ── 1) Processa todo o cadastro ─────────────────────────────────────────
    caminhos: list[str] = []
    for idx, ponto_id in enumerate(cad_id_str):
        # Prioridade 1: IAE
        if ponto_id in ids_iae:
            caminhos.append(Caminho.IAE_EXISTENTE.value)
            continue
        # Prioridade 2: ID
        if ponto_id in ids_id:
            caminhos.append(Caminho.ID_EXISTENTE.value)
            continue
        # Prioridade 3: IV — cruza com inspeção
        if ponto_id not in ids_inspecao:
            caminhos.append(Caminho.NAO_INSPECIONADO.value)
            continue

        cad_tec = cad.iloc[idx].get(col_codigo)
        cad_pot = cad.iloc[idx].get(col_potencia)
        insp_row = insp_indexed.loc[ponto_id] if ponto_id in insp_indexed.index else None
        if isinstance(insp_row, pd.DataFrame):
            # Múltiplas inspeções para o mesmo ID — pega a primeira (não deveria acontecer)
            insp_row = insp_row.iloc[0]
        insp_tec = insp_row.get(col_codigo) if insp_row is not None else None
        insp_pot = insp_row.get(col_potencia) if insp_row is not None else None

        cad_is_led = str(cad_tec or "").upper() == "LED"
        insp_is_led = str(insp_tec or "").upper() == "LED"

        if not cad_is_led and insp_is_led:
            caminhos.append(Caminho.LED_IV.value)
        elif cad_is_led and not insp_is_led:
            caminhos.append(Caminho.LED_MANTIDO.value)
        elif cad_is_led and insp_is_led:
            if _ha_divergencia(cad_tec, cad_pot, insp_tec, insp_pot):
                # Mesmo entre LEDs pode haver divergência de potência — vai para LED_IV
                # (a inspeção prevalece pela regra Considerada).
                caminhos.append(Caminho.LED_IV.value)
            else:
                caminhos.append(Caminho.LED_OK.value)
        else:  # ambos convencionais
            if _ha_divergencia(cad_tec, cad_pot, insp_tec, insp_pot):
                caminhos.append(Caminho.CONVENCIONAL.value)
            else:
                caminhos.append(Caminho.NAO_INSPECIONADO.value)
                # Sem divergência entre convencionais → não há tratamento a fazer.
                # Permanece como representado pelo cadastro original.

    cad["caminho"] = caminhos

    # ── 2) Identifica pontos NOVOS de IAE/ID ──────────────────────────────
    def _garantir_id(df: pd.DataFrame, prefixo: str) -> pd.DataFrame:
        df_out = df.copy()
        if col_id not in df_out.columns:
            df_out[col_id] = [f"NOVO_{prefixo}_{i+1:02d}" for i in range(len(df_out))]
        else:
            df_out[col_id] = df_out[col_id].astype(object)
            # Trata nulos e strings vazias/nan
            str_col = df_out[col_id].astype(str).str.strip().str.lower()
            mask = df_out[col_id].isna() | (str_col == "") | (str_col == "nan") | (str_col == "none")
            if mask.any():
                nulos_idx = mask[mask].index
                synthetic_ids = [f"NOVO_{prefixo}_{i+1:02d}" for i in range(len(nulos_idx))]
                df_out.loc[nulos_idx, col_id] = synthetic_ids
        return df_out

    iae = _garantir_id(iae, "IAE")
    id_ = _garantir_id(id_, "ID")

    ids_iae = set(limpar_id_serie(iae[col_id]))
    ids_id  = set(limpar_id_serie(id_[col_id]))

    iae_novos_ids = ids_iae - ids_cadastro
    iae_novos = iae[limpar_id_serie(iae[col_id]).isin(iae_novos_ids)].copy()

    id_novos_ids  = ids_id  - ids_cadastro
    id_novos = id_[limpar_id_serie(id_[col_id]).isin(id_novos_ids)].copy()

    if not iae_novos.empty:
        iae_novos["caminho"] = Caminho.IAE_NOVO.value
    if not id_novos.empty:
        id_novos["caminho"] = Caminho.ID_NOVO.value

    # ── 3) Monta subsets por caminho ────────────────────────────────────────
    def _filtro(c: Caminho) -> pd.DataFrame:
        return cad[cad["caminho"] == c.value].copy()

    resumo = {
        Caminho.IAE_EXISTENTE.value:    int((cad["caminho"] == Caminho.IAE_EXISTENTE.value).sum()),
        Caminho.IAE_NOVO.value:         int(len(iae_novos)),
        Caminho.ID_EXISTENTE.value:     int((cad["caminho"] == Caminho.ID_EXISTENTE.value).sum()),
        Caminho.ID_NOVO.value:          int(len(id_novos)),
        Caminho.LED_IV.value:           int((cad["caminho"] == Caminho.LED_IV.value).sum()),
        Caminho.CONVENCIONAL.value:     int((cad["caminho"] == Caminho.CONVENCIONAL.value).sum()),
        Caminho.LED_OK.value:           int((cad["caminho"] == Caminho.LED_OK.value).sum()),
        Caminho.LED_MANTIDO.value:      int((cad["caminho"] == Caminho.LED_MANTIDO.value).sum()),
        Caminho.NAO_INSPECIONADO.value: int((cad["caminho"] == Caminho.NAO_INSPECIONADO.value).sum()),
    }

    return ResultadoRoteamento(
        df_roteado=cad,
        pontos_iae_existentes=_filtro(Caminho.IAE_EXISTENTE),
        pontos_iae_novos=iae_novos,
        pontos_id_existentes=_filtro(Caminho.ID_EXISTENTE),
        pontos_id_novos=id_novos,
        pontos_led_iv=_filtro(Caminho.LED_IV),
        pontos_convencional=_filtro(Caminho.CONVENCIONAL),
        pontos_led_ok=_filtro(Caminho.LED_OK),
        pontos_led_mantido=_filtro(Caminho.LED_MANTIDO),
        pontos_nao_inspecionados=_filtro(Caminho.NAO_INSPECIONADO),
        resumo=resumo,
    )


__all__ = ["Caminho", "ResultadoRoteamento", "rotear"]
