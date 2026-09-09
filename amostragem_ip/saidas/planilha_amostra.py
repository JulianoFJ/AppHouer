"""
Geração das duas planilhas de campo — estrutural e qualidade.

Cada arquivo sai com duas abas:

  1. **Amostra de Campo** — a lista dos pontos sorteados: colunas de identificação e
     localização seguidas de todas as colunas originais do cadastro municipal (a
     amostragem não altera nem resume, só filtra as linhas sorteadas). É cadastro
     puro dos pontos a inspecionar, não um formulário de coleta em campo — a equipe
     registra o levantamento em instrumento próprio, fora desta planilha.
  2. **Plano de Amostragem** — a memória de cálculo: parque, plano NBR 5426, semente
     do sorteio, cobertura por classe e vias principais contempladas. É a aba que
     sustenta o dado perante o poder concedente e a banca.

Reaproveita a estilização do portal (`cadastro_ip/saidas/_helpers.py`) para que as
planilhas de amostragem tenham a mesma cara das três saídas da Análise de Cadastro.
"""

from __future__ import annotations

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

from cadastro_ip.saidas._helpers import (
    aplicar_estilo_header,
    autoajustar_largura,
    escrever_dataframe,
    header_font,
    subheader_fill,
    workbook_para_bytes,
)

from ..amostrador import COLUNAS_AUXILIARES, GRUPO_ESTRUTURAL, GRUPO_QUALIDADE, ResultadoAmostragem

# Colunas geradas só para a amostragem (ordem/via principal) — nunca vieram do
# cadastro do cliente, então não entram na lista de "colunas originais" a preservar.
_COLUNAS_GERADAS = {"_via_principal", "_ordem"}

ROTULO_GRUPO = {
    GRUPO_ESTRUTURAL: "Medição Estrutural",
    GRUPO_QUALIDADE: "Medição de Qualidade",
}

# Colunas de identificação que abrem a planilha, antes das colunas originais do cadastro.
_COLUNAS_IDENTIFICACAO = [
    ("Nº", "_ordem"),
    ("ID do ponto", "_id"),
    ("Logradouro", "_logradouro"),
    ("Bairro", "_bairro"),
    ("Classe (cadastro)", "_classe"),
    ("Tipo de via", "_tipo_via"),
    ("Via principal", "_via_principal"),
    ("Latitude (cadastro)", "_lat"),
    ("Longitude (cadastro)", "_lon"),
]


def _montar_tabela(
    resultado: ResultadoAmostragem, grupo: str
) -> pd.DataFrame:
    """
    Monta o DataFrame da aba única: identificação do ponto sorteado seguida de
    todas as colunas originais do cadastro do município (tecnologia, potência,
    poste — o que quer que o cadastro traga). Nenhum campo em branco para
    preenchimento futuro: a inspeção em si é registrada em instrumento à parte.
    """
    amostra = (resultado.estrutural if grupo == GRUPO_ESTRUTURAL else resultado.qualidade).copy()
    chaves_principais = {v.chave for v in resultado.vias_principais}
    amostra["_via_principal"] = amostra["_chave_via"].map(
        lambda c: "Sim" if c in chaves_principais else "Não"
    )
    # Ordena por bairro e logradouro: é como a equipe percorre o município, e reduz
    # deslocamento entre pontos — a aleatoriedade já foi decidida no sorteio.
    amostra = amostra.sort_values(["_bairro", "_logradouro", "_id"], kind="stable").reset_index(drop=True)
    amostra["_ordem"] = amostra.index + 1

    dados = {rotulo: amostra[coluna] for rotulo, coluna in _COLUNAS_IDENTIFICACAO}
    df = pd.DataFrame(dados)
    originais = [c for c in amostra.columns if c not in COLUNAS_AUXILIARES and c not in _COLUNAS_GERADAS]
    for coluna in originais:
        df[coluna] = amostra[coluna].to_numpy()
    return df


def _aba_amostra(wb: Workbook, resultado: ResultadoAmostragem, grupo: str) -> None:
    ws = wb.active
    ws.title = "Amostra de Campo"
    df = _montar_tabela(resultado, grupo)

    titulo = (
        f"{ROTULO_GRUPO[grupo]} — {resultado.municipio or 'Município'}"
        f"{'/' + resultado.uf if resultado.uf else ''} · "
        f"{len(df)} pontos · sorteio com semente {resultado.config.semente}"
    )
    ws.cell(row=1, column=1, value=titulo)
    ws.cell(row=1, column=1).font = header_font()
    ws.cell(row=1, column=1).fill = subheader_fill()
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(df.columns), 2))

    escrever_dataframe(ws, df, linha_inicial=2)
    aplicar_estilo_header(ws, 2, len(df.columns))
    ws.freeze_panes = "C3"
    autoajustar_largura(ws, len(df.columns))
    # A linha 1 é o título mesclado; a largura tem que sair do cabeçalho real (linha 2).
    for indice, coluna in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(indice)].width = max(10, min(38, len(str(coluna)) + 4))


def _aba_plano(wb: Workbook, resultado: ResultadoAmostragem, grupo: str) -> None:
    """Memória de cálculo do sorteio."""
    ws = wb.create_sheet("Plano de Amostragem")
    linha = 1

    def _titulo(texto: str) -> None:
        nonlocal linha
        ws.cell(row=linha, column=1, value=texto)
        aplicar_estilo_header(ws, linha, 2)
        linha += 1

    def _item(rotulo: str, valor) -> None:
        nonlocal linha
        ws.cell(row=linha, column=1, value=rotulo)
        ws.cell(row=linha, column=2, value=valor)
        linha += 1

    _titulo("Identificação")
    _item("Município", f"{resultado.municipio}{'/' + resultado.uf if resultado.uf else ''}")
    _item("Planilha", ROTULO_GRUPO[grupo])
    _item("Pontos no cadastro (lote)", resultado.total_parque)
    _item("Pontos nesta planilha", len(resultado.estrutural if grupo == GRUPO_ESTRUTURAL else resultado.qualidade))
    _item("Amostra total (estrutural + qualidade)", resultado.total_amostra)
    _item("Semente do sorteio (reprodutibilidade)", resultado.config.semente)
    linha += 1

    plano = resultado.plano
    if plano is not None:
        _titulo("Dimensionamento — ABNT NBR 5426:1985")
        _item("Nível de inspeção", plano.nivel)
        _item("NQA (%)", plano.nqa)
        _item("Regime", plano.regime)
        _item("Letra-código (Tabela 1)", plano.letra_codigo)
        _item("Tamanho de amostra da norma", plano.tamanho_amostra)
        _item("Número de aceitação (Ac)", plano.numero_aceitacao)
        _item("Número de rejeição (Re)", plano.numero_rejeicao)
        _item("Amostra efetivamente sorteada", resultado.total_amostra)
        _item("Fração do parque inspecionada", f"{resultado.total_amostra / max(resultado.total_parque, 1):.2%}")
        for observacao in plano.observacoes:
            _item("Observação", observacao)
        linha += 1

    if not resultado.cobertura_classes.empty:
        _titulo("Cobertura por classe de iluminação")
        tabela = resultado.cobertura_classes.copy()
        tabela["% do parque"] = tabela["% do parque"].map(lambda v: f"{v:.1%}")
        linha = escrever_dataframe(ws, tabela, linha_inicial=linha) + 1

    if not resultado.cobertura_vias.empty:
        _titulo("Vias principais com cobertura obrigatória")
        linha = escrever_dataframe(ws, resultado.cobertura_vias, linha_inicial=linha) + 1

    abrangencia = resultado.abrangencia
    _titulo("Abrangência geográfica")
    _item("Bairros no cadastro / na amostra",
          f"{abrangencia.get('bairros_parque', 0)} / {abrangencia.get('bairros_amostra', 0)}")
    _item("Logradouros no cadastro / na amostra",
          f"{abrangencia.get('logradouros_parque', 0)} / {abrangencia.get('logradouros_amostra', 0)}")
    if abrangencia.get("cobertura_grid") is not None:
        _item("Células da malha 12×12 com parque atingidas pela amostra",
              f"{abrangencia['celulas_cobertas']} de {abrangencia['celulas_com_parque']} "
              f"({abrangencia['cobertura_grid']:.1%})")
        _item("Distância mediana de um ponto qualquer ao ponto inspecionado mais próximo",
              f"{abrangencia['distancia_mediana_km']:.2f} km")
        _item("Idem, percentil 90", f"{abrangencia['distancia_p90_km']:.2f} km")
    linha += 1

    if resultado.ressalvas:
        _titulo("Ressalvas")
        for ressalva in resultado.ressalvas:
            _item("•", ressalva)

    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 80


def gerar(resultado: ResultadoAmostragem, grupo: str) -> bytes:
    """
    Gera a planilha .xlsx de uma das duas frentes de campo.

    Args:
        resultado: saída de `amostrador.sortear`.
        grupo: `GRUPO_ESTRUTURAL` ou `GRUPO_QUALIDADE`.

    Returns:
        Bytes do arquivo .xlsx.
    """
    if grupo not in (GRUPO_ESTRUTURAL, GRUPO_QUALIDADE):
        raise ValueError(f"Grupo desconhecido: {grupo!r}")
    wb = Workbook()
    _aba_amostra(wb, resultado, grupo)
    _aba_plano(wb, resultado, grupo)
    return workbook_para_bytes(wb)


__all__ = ["gerar", "ROTULO_GRUPO"]
