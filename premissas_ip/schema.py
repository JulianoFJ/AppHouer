"""
Catálogo neutro de premissas de IP — wrapper de runtime sobre `schema_inputs.json`.

A *estrutura* (seções, parâmetros, unidades, fontes) é fixa e reaproveitável por
qualquer município; os *valores* são coletados por município no wizard. Parâmetros
com `coleta=True` são específicos do município (default vazio, preenchidos no
formulário); os demais trazem um default editável vindo do modelo de referência
(premissa de PPP, mercado, SINAPI, CAGED, etc.).

O JSON é gerado por `_gerar_schema.py` a partir da `Planilha Modelo Inputs IP`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_JSON = Path(__file__).parent / "schema_inputs.json"

# Rótulos amigáveis dos tipos de input usados no wizard.
TIPOS_INPUT = {"numero", "moeda", "percentual", "data", "sim_nao", "texto"}


@dataclass(frozen=True)
class Parametro:
    id: str
    linha_modelo: int
    label: str
    tipo: str
    unidade: str = ""
    default: object = None
    fonte: str = ""
    coleta: bool = False

    @property
    def eh_grupo(self) -> bool:
        return self.tipo == "grupo"

    @property
    def eh_input(self) -> bool:
        return self.tipo in TIPOS_INPUT

    @property
    def label_unidade(self) -> str:
        """Rótulo para exibição, com unidade entre parênteses quando houver."""
        if self.unidade and self.unidade not in ("#",):
            return f"{self.label} ({self.unidade})"
        return self.label


@dataclass(frozen=True)
class ColunaTabela:
    id: str
    label: str
    tipo: str


@dataclass(frozen=True)
class Secao:
    id: str
    nome: str
    linha_modelo: int
    dinamica: bool
    parametros: list[Parametro]
    colunas_tabela: list[ColunaTabela] = field(default_factory=list)

    def inputs(self) -> list[Parametro]:
        """Apenas parâmetros que recebem valor (exclui cabeçalhos de grupo)."""
        return [p for p in self.parametros if p.eh_input]

    def coletados(self) -> list[Parametro]:
        """Parâmetros específicos do município (preenchidos no wizard)."""
        return [p for p in self.parametros if p.coleta]


@dataclass(frozen=True)
class Schema:
    secoes: list[Secao]
    meta: dict

    def secao(self, secao_id: str) -> Secao | None:
        return next((s for s in self.secoes if s.id == secao_id), None)

    def parametro(self, param_id: str) -> Parametro | None:
        for s in self.secoes:
            for p in s.parametros:
                if p.id == param_id:
                    return p
        return None

    def todos_inputs(self) -> list[Parametro]:
        return [p for s in self.secoes for p in s.inputs()]

    def defaults(self) -> dict[str, object]:
        """Mapa id->default para semear o estado do wizard."""
        return {p.id: p.default for p in self.todos_inputs()}


@lru_cache(maxsize=1)
def carregar() -> Schema:
    """Carrega (e cacheia) o catálogo a partir do JSON."""
    dados = json.loads(_JSON.read_text(encoding="utf-8"))
    secoes: list[Secao] = []
    for s in dados["secoes"]:
        params = [
            Parametro(
                id=p["id"],
                linha_modelo=p["linha_modelo"],
                label=p["label"],
                tipo=p["tipo"],
                unidade=p.get("unidade", ""),
                default=p.get("default"),
                fonte=p.get("fonte", ""),
                coleta=p.get("coleta", False),
            )
            for p in s["parametros"]
        ]
        colunas = [ColunaTabela(**c) for c in s.get("colunas_tabela", [])]
        secoes.append(
            Secao(
                id=s["id"],
                nome=s["nome"],
                linha_modelo=s["linha_modelo"],
                dinamica=s["dinamica"],
                parametros=params,
                colunas_tabela=colunas,
            )
        )
    return Schema(secoes=secoes, meta=dados.get("_meta", {}))


__all__ = ["Parametro", "ColunaTabela", "Secao", "Schema", "carregar", "TIPOS_INPUT"]
