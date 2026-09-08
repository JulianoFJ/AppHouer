"""
Cadastra (ou reseta a senha de) um usuário na tabela `users` do Postgres.

    py -m acesso.gerar_hash                    # pergunta login e senha
    py -m acesso.gerar_hash --login jferreira --nome "Juliano Ferreira"

A senha é lida sem eco (`getpass`) e nunca é gravada em lugar nenhum — nem no histórico
do terminal, nem em arquivo. Requer `DATABASE_URL` no ambiente, apontando para o mesmo
Postgres que o app usa (local, ou o de produção via `railway run`).

Rodar de novo para um login que já existe faz UPSERT: atualiza nome/perfil/hash e
reativa a conta (`ativo = true`) — é assim que se troca a senha de alguém ou se
readmite um acesso revogado, sem precisar de um comando separado para cada caso.

Rode a partir de `app/`, que é onde o pacote `acesso` está no sys.path.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys

from sqlalchemy import text

import db
from .autenticacao import ITERACOES, gerar_hash

TAMANHO_SUGESTAO = 16


def main() -> int:
    # O console do Windows abre em cp1252, que não codifica nem `─` nem os acentos das
    # mensagens abaixo — sem isto o comando morre com UnicodeEncodeError DEPOIS de já
    # ter mostrado a senha sorteada, que é o pior momento possível para falhar.
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    p = argparse.ArgumentParser(description="Cadastra ou atualiza um usuário em `users`.")
    p.add_argument("--login", help="identificador de acesso (minúsculas, sem espaço)")
    p.add_argument("--nome", help="nome exibido no portal e na trilha de uso")
    p.add_argument("--perfil", default="usuario", help="rótulo livre (padrão: usuario)")
    p.add_argument("--sortear-senha", action="store_true",
                   help="sorteia uma senha forte em vez de pedir uma")
    args = p.parse_args()

    login = (args.login or input("Login: ")).strip().lower()
    if not login or " " in login:
        print("Login inválido: use minúsculas, sem espaços.", file=sys.stderr)
        return 1
    nome = args.nome or input("Nome exibido: ").strip() or login

    if args.sortear_senha:
        senha = secrets.token_urlsafe(TAMANHO_SUGESTAO)
        print(f"\nSenha sorteada (anote agora, não será exibida de novo): {senha}")
    else:
        senha = getpass.getpass("Senha: ")
        if senha != getpass.getpass("Repita a senha: "):
            print("As senhas não conferem.", file=sys.stderr)
            return 1
        if len(senha) < 10:
            print("Senha muito curta: use pelo menos 10 caracteres.", file=sys.stderr)
            return 1

    print(f"\nDerivando com PBKDF2-SHA256, {ITERACOES:,} iterações...".replace(",", "."))
    registro = gerar_hash(senha)

    try:
        with db.engine().begin() as conexao:
            conexao.execute(
                text(
                    "insert into users (login, nome, senha_hash, perfil, ativo) "
                    "values (:login, :nome, :senha_hash, :perfil, true) "
                    "on conflict (login) do update set "
                    "nome = excluded.nome, senha_hash = excluded.senha_hash, "
                    "perfil = excluded.perfil, ativo = true"
                ),
                {"login": login, "nome": nome, "senha_hash": registro, "perfil": args.perfil},
            )
    except Exception as erro:
        print(f"\nFalha ao gravar no Postgres: {erro!r}", file=sys.stderr)
        print("Confira se DATABASE_URL está definida e as migrations foram aplicadas "
              "(`py -m alembic upgrade head`).", file=sys.stderr)
        return 1

    print(f"\nUsuário '{login}' ({nome}, perfil={args.perfil}) gravado em `users`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
