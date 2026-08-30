"""
Gera o bloco de secrets de um usuário, com a senha já em hash.

    py -m acesso.gerar_hash                    # pergunta login e senha
    py -m acesso.gerar_hash --login jferreira --nome "Juliano Ferreira"

A senha é lida sem eco (`getpass`) e nunca é gravada em lugar nenhum — nem no histórico
do terminal, nem em arquivo. O que sai é só o hash, que é o que vai para o secrets.

Rode a partir de `app/`, que é onde o pacote `acesso` está no sys.path.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys

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

    p = argparse.ArgumentParser(description="Gera o bloco [auth.usuarios.X] do secrets.")
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

    print("\n" + "-" * 72)
    print("Cole o bloco abaixo em .streamlit/secrets.toml (local) ou no painel de")
    print("secrets do Streamlit Cloud (publicado). NÃO faça commit deste conteúdo.")
    print("-" * 72)
    print(f"""
[auth]
expiracao_horas = 12

[auth.usuarios.{login}]
nome = "{nome}"
perfil = "{args.perfil}"
senha_hash = "{registro}"
""")
    print("Se o bloco [auth] já existir no seu secrets, copie apenas a parte")
    print(f"[auth.usuarios.{login}].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
