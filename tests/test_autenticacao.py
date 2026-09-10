import datetime
import uuid

import pytest
from sqlalchemy import text

import db
from acesso import autenticacao


@pytest.fixture
def login_temporario():
    login = f"pytest_{uuid.uuid4().hex[:8]}"
    yield login
    with db.engine().begin() as conexao:
        conexao.execute(text("delete from users where login = :login"), {"login": login})


@pytest.fixture
def intervalo_verificacao_zero():
    """Força `_ainda_valido` a reconsultar o banco a cada chamada, em vez de confiar
    na janela de 60s — necessário para testar a revogação sem esperar."""
    original = autenticacao.INTERVALO_VERIFICACAO_S
    autenticacao.INTERVALO_VERIFICACAO_S = 0
    yield
    autenticacao.INTERVALO_VERIFICACAO_S = original


def test_verificar_senha_aceita_senha_correta_e_rejeita_errada():
    registro = autenticacao.gerar_hash("uma-senha-bem-forte-123")
    assert autenticacao.verificar_senha("uma-senha-bem-forte-123", registro)
    assert not autenticacao.verificar_senha("senha-errada", registro)


def test_criar_usuario_aparece_em_usuarios_cadastrados(login_temporario):
    autenticacao.criar_ou_atualizar_usuario(
        login_temporario, "Nome Teste", "usuario", "senha-forte-123")

    cadastrados = autenticacao.usuarios_cadastrados()
    assert login_temporario in cadastrados
    assert cadastrados[login_temporario]["ativo"] is True
    assert cadastrados[login_temporario]["perfil"] == "usuario"


def test_atualizar_usuario_existente_troca_perfil_e_reativa(login_temporario):
    autenticacao.criar_ou_atualizar_usuario(
        login_temporario, "Nome Teste", "usuario", "senha-forte-123")
    autenticacao.definir_ativo(login_temporario, False)

    autenticacao.criar_ou_atualizar_usuario(
        login_temporario, "Nome Teste", "admin", "outra-senha-123")

    cadastrados = autenticacao.usuarios_cadastrados()
    assert cadastrados[login_temporario]["perfil"] == "admin"
    assert cadastrados[login_temporario]["ativo"] is True


def test_usuario_desativado_nao_aparece_entre_ativos(login_temporario):
    autenticacao.criar_ou_atualizar_usuario(
        login_temporario, "Nome Teste", "usuario", "senha-forte-123")
    autenticacao.definir_ativo(login_temporario, False)

    assert login_temporario not in autenticacao._usuarios()


def test_usuario_atual_invalida_sessao_apos_desativacao(
        login_temporario, intervalo_verificacao_zero):
    """Regressão do achado de revisão de segurança: revogar um acesso pela tela de
    administração só grava no Postgres — sem isto, uma sessão de navegador já aberta
    continuaria válida até expirar sozinha (`expiracao_horas`)."""
    autenticacao.criar_ou_atualizar_usuario(
        login_temporario, "Nome Teste", "usuario", "senha-forte-123")

    import streamlit as st
    usuario = autenticacao.Usuario(
        login=login_temporario, nome="Nome Teste", perfil="usuario",
        sessao_id="sessao-teste",
        autenticado_em=datetime.datetime.now(datetime.timezone.utc),
    )
    st.session_state[autenticacao._CHAVE_SESSAO] = usuario
    st.session_state[autenticacao._CHAVE_ULTIMA_VERIFICACAO] = 0.0
    assert autenticacao.usuario_atual() is not None

    autenticacao.definir_ativo(login_temporario, False)
    st.session_state[autenticacao._CHAVE_SESSAO] = usuario
    st.session_state[autenticacao._CHAVE_ULTIMA_VERIFICACAO] = 0.0
    assert autenticacao.usuario_atual() is None
