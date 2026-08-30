"""
Converte a chave JSON de uma service account do Google no bloco TOML do `secrets`.

    py -m acesso.configurar_sheets --json chave.json --planilha <URL ou id da planilha>
    py -m acesso.configurar_sheets --json chave.json --planilha <URL> --testar

Por que este comando existe: a conversão à mão é o passo onde a configuração falha.
O JSON traz a chave privada com quebras de linha escritas como `\\n` dentro de uma
string de várias centenas de caracteres; copiada errada — e é fácil errar, porque
editores costumam quebrar a linha ou "consertar" as barras — o `secrets` fica
sintaticamente válido e a autenticação falha em silêncio, com o log caindo para CSV
sem avisar ninguém. Aqui a string é reemitida exatamente como o TOML espera.

`--testar` fecha o ciclo: autentica de verdade, escreve uma linha de teste na aba e a
apaga em seguida. É o único jeito de descobrir agora, e não daqui a um mês, que a
planilha não foi compartilhada com a service account — o erro mais comum e o mais
silencioso dos dois lados.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Campos que uma chave de service account sempre traz. Servem para recusar cedo o
# arquivo errado — é comum baixar por engano a credencial de "OAuth client ID", que
# tem outro formato e não serve aqui.
OBRIGATORIOS = ("type", "project_id", "private_key", "client_email", "token_uri")


def extrair_id(texto: str) -> str:
    """
    Aceita a URL inteira da planilha ou só o id.

    A URL tem a forma `.../spreadsheets/d/<id>/edit#gid=0`; pedir que o usuário recorte
    o id no meio dela é convite a erro, então o recorte é feito aqui.
    """
    texto = texto.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", texto)
    return m.group(1) if m else texto


def _escapar(valor: str) -> str:
    """Escapa a string para uma literal TOML de aspas duplas (barras e quebras)."""
    return valor.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def montar_bloco(cred: dict, planilha_id: str, aba: str) -> str:
    linhas = [
        "[log_uso]",
        f'planilha_id = "{planilha_id}"',
        f'aba = "{aba}"',
        "",
        "[gcp_service_account]",
    ]
    # Ordem estável e previsível, com os campos conhecidos primeiro: o bloco é lido por
    # gente, e um `secrets` cuja ordem muda a cada geração é impossível de comparar.
    conhecidos = ["type", "project_id", "private_key_id", "private_key", "client_email",
                  "client_id", "auth_uri", "token_uri",
                  "auth_provider_x509_cert_url", "client_x509_cert_url",
                  "universe_domain"]
    for chave in conhecidos + [k for k in cred if k not in conhecidos]:
        if chave in cred and cred[chave] not in (None, ""):
            linhas.append(f'{chave} = "{_escapar(str(cred[chave]))}"')
    return "\n".join(linhas)


def testar(cred: dict, planilha_id: str, aba: str) -> int:
    """Autentica, escreve uma linha de teste e a remove. Devolve 0 se tudo funcionou."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("\n[--testar] gspread/google-auth não instalados nesta máquina.")
        print("           Instale com:  py -m pip install gspread google-auth")
        return 2

    print("\n[--testar] Autenticando...")
    try:
        credenciais = Credentials.from_service_account_info(
            cred, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        planilha = gspread.authorize(credenciais).open_by_key(planilha_id)
    except Exception as erro:                     # noqa: BLE001
        print(f"[--testar] FALHOU ao abrir a planilha: {erro!r}")
        print("\n  A causa quase sempre é uma destas duas:")
        print(f"  1. A planilha não foi compartilhada com {cred.get('client_email')}")
        print("     (abra a planilha -> Compartilhar -> cole esse e-mail -> Editor).")
        print("  2. A Google Sheets API não está habilitada no projeto do Google Cloud.")
        return 1

    print(f"[--testar] Planilha aberta: {planilha.title!r}")
    try:
        try:
            ws = planilha.worksheet(aba)
        except Exception:
            ws = planilha.add_worksheet(aba, rows=1000, cols=9)
            print(f"[--testar] Aba {aba!r} não existia e foi criada.")
        ws.append_row(["__teste_de_configuracao__"], value_input_option="RAW")
        ws.delete_rows(len(ws.get_all_values()))
        print("[--testar] Escrita e remoção OK. A trilha de uso está pronta.")
        return 0
    except Exception as erro:                     # noqa: BLE001
        print(f"[--testar] Abriu para leitura mas FALHOU ao escrever: {erro!r}")
        print("  A service account provavelmente entrou como Leitor, não como Editor.")
        return 1


def main() -> int:
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    p = argparse.ArgumentParser(
        description="Monta o bloco [gcp_service_account] + [log_uso] do secrets.")
    p.add_argument("--json", required=True,
                   help="caminho do arquivo .json baixado do Google Cloud")
    p.add_argument("--planilha", required=True,
                   help="URL completa da planilha, ou apenas o id")
    p.add_argument("--aba", default="uso", help="nome da aba (padrão: uso)")
    p.add_argument("--testar", action="store_true",
                   help="autentica e escreve uma linha de teste, para validar de fato")
    args = p.parse_args()

    caminho = Path(args.json).expanduser()
    if not caminho.exists():
        print(f"Arquivo não encontrado: {caminho}", file=sys.stderr)
        return 1
    try:
        cred = json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError as erro:
        print(f"O arquivo não é um JSON válido: {erro}", file=sys.stderr)
        return 1

    faltando = [c for c in OBRIGATORIOS if c not in cred]
    if faltando:
        print("Este JSON não parece ser a chave de uma service account.", file=sys.stderr)
        print(f"Campos ausentes: {', '.join(faltando)}", file=sys.stderr)
        print("\nNo Google Cloud, o caminho certo é IAM & Admin -> Service Accounts ->"
              " sua conta -> Keys -> Add key -> Create new key -> JSON.", file=sys.stderr)
        return 1
    if cred.get("type") != "service_account":
        print(f"Tipo inesperado: {cred.get('type')!r} (esperado 'service_account').",
              file=sys.stderr)
        return 1

    planilha_id = extrair_id(args.planilha)

    print("-" * 72)
    print("PASSO QUE MAIS SE ESQUECE: compartilhe a planilha, como EDITOR, com")
    print(f"  {cred.get('client_email')}")
    print("Sem isso a API responde 403 e a trilha cai para CSV local sem avisar.")
    print("-" * 72)
    print("\nCole o bloco abaixo no painel de Secrets do Streamlit Cloud (Settings ->")
    print("Secrets), ABAIXO do que já estiver lá. Não faça commit deste conteúdo.\n")
    print(montar_bloco(cred, planilha_id, args.aba))
    print()

    if args.testar:
        return testar(cred, planilha_id, args.aba)

    print("-" * 72)
    print("Para validar a configuração de verdade, rode de novo com --testar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
