"""
Gerador (one-shot, reexecutável) do catálogo neutro de premissas a partir do
modelo `Planilha Modelo Inputs IP`.

Não é usado em runtime. Produz `schema_inputs.json`, que é a fonte da verdade do
catálogo consumido por `schema.py`. Reexecute apenas se o modelo de referência
mudar de estrutura:

    py -3 app/premissas_ip/_gerar_schema.py "<caminho do modelo .xlsx>"

Regra de neutralidade (o portal precisa servir qualquer município): a coluna
`Fonte` do modelo diz a origem de cada dado. Dados específicos do município
(cadastro, campo, estudos de engenharia, contratos, distribuidora, legislação,
indicação da prefeitura) entram em branco e são COLETADOS no wizard. Premissas e
catálogos reusáveis (Premissa PPP, Mercado, SINAPI, CAGED, CADTERC, ANEEL, etc.)
mantêm o default do modelo como ponto de partida EDITÁVEL.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import openpyxl

ABA = "Inputs_IP"
COL_MARCA, COL_LABEL, COL_UNID, COL_VALOR = 0, 1, 2, 5  # 0-based (A, B, C, F)

# Fontes que indicam dado específico do município → coletar no wizard, default vazio.
FONTES_MUNICIPIO = (
    "cadastro de ip da prefeitura",
    "dados informados pela prefeitura",
    "trabalho de campo",
    "estudos de engenharia do projeto",
    "contrato caixa",
    "aditivo do contrato",
    "site da distribuidora",
    "legislacao municipal",
    "indicacao pela prefeitura",
)
# Seções inteiras tratadas como coleta de município (independe da fonte das linhas).
SECOES_COLETA = {
    "Quantitativo Parque",
    "Contribuição de Iluminação Pública",
    "Receita Corrente Líquida",
    "Distribuidora de Energia",
    "Despesas Contratos Atuais O&M",
}
# Seção com lista dinâmica (itens definidos pelo usuário, não fixos no catálogo).
SECAO_DINAMICA = "Iluminação Especial"


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _tipo(unidade: str, default) -> str:
    u = (unidade or "").strip().lower()
    sval = "" if default is None else str(default).strip()
    if u == "%":
        return "percentual"
    if u.startswith("r$"):
        return "moeda"
    if u == "data":
        return "data"
    if u == "#":
        return "sim_nao" if sval in ("Sim", "Não") else "texto"
    if u in ("meses", "mês", "mes", "anos", "ano", "horas", "dias", "litros",
             "unidade", "unidades", "pontos de ip", "kw", "w", "m²", "kg"):
        return "numero"
    if sval and not _eh_numero(sval):
        return "texto"
    return "numero"


def _eh_numero(s: str) -> bool:
    try:
        float(str(s).replace(",", "."))
        return True
    except (ValueError, TypeError):
        return False


def _fonte_municipio(fonte: str) -> bool:
    f = _slug(fonte)
    return any(_slug(k) in f for k in FONTES_MUNICIPIO)


def _ler_fonte(row) -> str:
    for c in row[11:16]:
        if c and "Fonte" in str(c):
            return str(c).replace("Fonte:", "").strip()
    return ""


def gerar(caminho_modelo: str) -> dict:
    wb = openpyxl.load_workbook(caminho_modelo, read_only=True, data_only=True)
    ws = wb[ABA]

    secoes: list[dict] = []
    secao_atual: dict | None = None

    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=1137, values_only=True)):
        r = i + 1
        marca = row[COL_MARCA]
        label = row[COL_LABEL]
        unidade = row[COL_UNID] if len(row) > COL_UNID else None
        valor = row[COL_VALOR] if len(row) > COL_VALOR else None
        fonte = _ler_fonte(row) if len(row) > 11 else ""

        # Marcador de seção (x na coluna A).
        if marca is not None and str(marca).strip().lower() == "x":
            nome = str(label).strip()
            secao_atual = {
                "id": _slug(nome),
                "nome": nome,
                "linha_modelo": r,
                "dinamica": nome == SECAO_DINAMICA,
                "parametros": [],
            }
            secoes.append(secao_atual)
            continue

        if secao_atual is None or label is None or str(label).strip() == "":
            continue

        nome_label = str(label).strip()
        unid = "" if unidade is None else str(unidade).strip()
        eh_grupo = (unid == "" and (valor is None or str(valor).strip() == ""))

        if eh_grupo:
            secao_atual["parametros"].append({
                "id": f"p{r}",
                "linha_modelo": r,
                "label": nome_label,
                "tipo": "grupo",
            })
            continue

        coleta = (secao_atual["nome"] in SECOES_COLETA) or _fonte_municipio(fonte)
        tipo = _tipo(unid, valor)

        # Default: específico do município entra em branco; premissa/catálogo mantém.
        if coleta:
            default = None
        else:
            default = valor
            if hasattr(default, "isoformat"):
                default = default.isoformat()

        secao_atual["parametros"].append({
            "id": f"p{r}",
            "linha_modelo": r,
            "label": nome_label,
            "unidade": unid,
            "tipo": tipo,
            "default": default,
            "fonte": fonte,
            "coleta": coleta,
        })

    wb.close()

    # Seção dinâmica: substitui linhas fixas por definição de tabela editável.
    for s in secoes:
        if s["dinamica"]:
            s["colunas_tabela"] = [
                {"id": "local", "label": "Local / Elemento", "tipo": "texto"},
                {"id": "pontos_atuais", "label": "Pontos de IP atuais", "tipo": "numero"},
                {"id": "pontos_futuros", "label": "Pontos de IP projeto futuro", "tipo": "numero"},
                {"id": "capex", "label": "CAPEX (R$)", "tipo": "moeda"},
            ]
            # Mantém apenas os parâmetros gerais (não-locais) da seção.
            s["parametros"] = [
                p for p in s["parametros"]
                if p["tipo"] == "grupo" or p["linha_modelo"] >= 531
            ]

    total_param = sum(len([p for p in s["parametros"] if p["tipo"] != "grupo"]) for s in secoes)
    total_coleta = sum(len([p for p in s["parametros"] if p.get("coleta")]) for s in secoes)
    return {
        "_meta": {
            "fonte_modelo": Path(caminho_modelo).name,
            "aba": ABA,
            "total_secoes": len(secoes),
            "total_parametros": total_param,
            "total_coleta_municipio": total_coleta,
        },
        "secoes": secoes,
    }


if __name__ == "__main__":
    modelo = sys.argv[1]
    dados = gerar(modelo)
    destino = Path(__file__).parent / "schema_inputs.json"
    destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    m = dados["_meta"]
    print(f"OK -> {destino}")
    print(f"  seções={m['total_secoes']}  parâmetros={m['total_parametros']}  "
          f"coleta_município={m['total_coleta_municipio']}")
