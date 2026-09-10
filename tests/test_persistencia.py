import pandas as pd
import pytest
from sqlalchemy import text

import db
from amostragem_ip import amostrador, persistencia


@pytest.fixture
def registrar_execucao():
    """Devolve uma função que marca um id de execução para apagar no fim do teste.
    `amostragem_pontos` some junto — a FK é ON DELETE CASCADE (ver a migration)."""
    ids: list[int] = []
    yield ids.append
    with db.engine().begin() as conexao:
        for execucao_id in ids:
            conexao.execute(
                text("delete from amostragem_execucoes where id = :id"),
                {"id": execucao_id},
            )


def _base_teste(n: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({
        "id": [f"PT{i}" for i in range(n)],
        "lat": [-19.90 + i * 0.001 for i in range(n)],
        "lon": [-43.90 + i * 0.001 for i in range(n)],
    })
    colunas = {"id_ponto": "id", "latitude": "lat", "longitude": "lon"}
    base, _ = amostrador.preparar_base(df, colunas)
    return base


def test_salvar_execucao_grava_pontos_selecionados_e_excluidos(registrar_execucao):
    base = _base_teste()
    config = amostrador.ConfigAmostragem(tamanho_amostra=5, semente=1)
    resultado = amostrador.sortear(base, config, municipio="Pytest", uf="MG")

    pontos_excluidos = [{
        "_id": "PTX", "_logradouro": "Rua Fantasma", "_bairro": "Centro", "_classe": "LV",
        "categoria": "Parque/praça", "nome": "Parque Teste",
        "osm_url": "https://osm.example/1",
    }]

    execucao_id = persistencia.salvar_execucao(
        resultado, usuario_login=None, pontos_excluidos=pontos_excluidos)
    assert execucao_id is not None
    registrar_execucao(execucao_id)

    pontos = persistencia.carregar_pontos(execucao_id)
    selecionados = pontos[pontos["selecionado"]]
    excluidos = pontos[~pontos["selecionado"]]

    assert len(selecionados) == resultado.total_amostra
    assert len(excluidos) == 1
    assert excluidos.iloc[0]["revisao_manual"]["categoria"] == "Parque/praça"
    assert excluidos.iloc[0]["dados"]["_id"] == "PTX"


def test_listar_execucoes_inclui_a_execucao_recem_criada(registrar_execucao):
    base = _base_teste()
    config = amostrador.ConfigAmostragem(tamanho_amostra=3, semente=2)
    resultado = amostrador.sortear(base, config, municipio="Pytest Listagem", uf="MG")

    execucao_id = persistencia.salvar_execucao(
        resultado, usuario_login=None, pontos_excluidos=[])
    assert execucao_id is not None
    registrar_execucao(execucao_id)

    execucoes = persistencia.listar_execucoes(limite=200)
    assert execucao_id in execucoes["id"].values
