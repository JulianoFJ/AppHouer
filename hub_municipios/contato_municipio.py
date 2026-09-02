"""
Descoberta do canal institucional da prefeitura — site, e-mail e telefone públicos.

Serve à apresentação comercial: o time precisa saber por onde abordar o município. O
módulo busca **apenas dados institucionais publicados no site oficial** (endereço do
portal, e-mail de ouvidoria/contato, telefone da sede). Não coleta nome, e-mail nem
telefone de agentes públicos — o alvo é a instituição, não a pessoa.

## O método, e por que ele é um palpite verificável

Não existe API pública com o contato das 5.570 prefeituras. O que existe é uma
convenção: a esmagadora maioria usa domínio `<municipio>.<uf>.gov.br`. O módulo gera os
candidatos dessa convenção, testa qual responde, e lê a página de contato procurando
e-mail do próprio domínio e telefone.

Isso acerta boa parte, mas **não todos**: numa amostra de 6 municípios em 01/09/2026,
acertou 4 — Belo Horizonte usa `pbh.gov.br` e Uberlândia não respondeu ao padrão. Por
isso todo registro sai com `url_origem` e `coletado_em`, e o que não for encontrado
volta vazio em vez de chutado. Material comercial com telefone errado queima a
abordagem, que é pior do que campo em branco: o slide traz o campo vazio para o time
preencher.

Os dados podem estar desatualizados no próprio site da prefeitura. **Confira antes de
ligar** — é o que a legenda no slide diz, e é literal.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import pandas as pd
import requests

from . import config


TIMEOUT = 12
TAMANHO_MINIMO_PAGINA = 800

# Caminhos usuais da página de contato em portal de prefeitura.
CAMINHOS_CONTATO = ["/portal/contato", "/contato", "/fale-conosco", "/ouvidoria",
                    "/portal/telefones", "/telefones-uteis"]

# Prefixos de e-mail que caracterizam canal institucional (e não caixa pessoal).
PREFIXOS_INSTITUCIONAIS = ("ouvidoria", "contato", "faleconosco", "fale.conosco",
                           "gabinete", "atendimento", "protocolo", "imprensa",
                           "comunicacao", "administracao", "licitacao", "compras")

_RE_EMAIL = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-z]{2,}", re.I)

# DDDs efetivamente em uso no Brasil. A validação existe porque sem ela qualquer
# sequência de 10 ou 11 dígitos vira "telefone" — e página de prefeitura é cheia de
# timestamp e id em JavaScript. Na primeira versão isto devolveu "1786986951" como
# telefone de Divinópolis, que é um epoch, não um número de contato.
DDDS_VALIDOS = {
    11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 27, 28, 31, 32, 33, 34, 35, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 71, 73, 74, 75, 77, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95,
    96, 97, 98, 99,
}

# Só aceita número que venha FORMATADO como telefone: DDD entre parênteses, ou
# separadores explícitos entre os blocos. Dígitos corridos são recusados de propósito.
_RE_TELEFONE = re.compile(
    r"\(\s*(\d{2})\s*\)\s*(\d{4,5})[-.\s]?(\d{4})"      # (37) 3229-6500
    r"|\b(\d{2})[.\s](\d{4,5})[-.\s](\d{4})\b"            # 37 3229-6500
)

_RE_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_RE_TAG = re.compile(r"<[^>]+>")


def _texto_visivel(html: str) -> str:
    """
    Remove script, style e tags antes de procurar contato.

    Sem isso, a varredura lê o JavaScript da página — de onde saíram os "telefones"
    falsos da primeira versão.
    """
    limpo = _RE_SCRIPT.sub(" ", html or "")
    return _RE_TAG.sub(" ", limpo)

COLUNAS = ["codigo_municipio", "municipio", "uf", "site", "email", "telefone",
           "url_origem", "coletado_em", "observacao"]


@dataclass
class Contato:
    """Canal institucional encontrado, sempre com a procedência junto."""

    codigo_municipio: str = ""
    municipio: str = ""
    uf: str = ""
    site: Optional[str] = None
    email: Optional[str] = None
    telefone: Optional[str] = None
    url_origem: Optional[str] = None
    coletado_em: Optional[str] = None
    observacao: str = ""

    @property
    def encontrado(self) -> bool:
        return bool(self.site)


def _slug(texto) -> str:
    t = unicodedata.normalize("NFD", str(texto or ""))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]", "", t)


def candidatos_de_dominio(municipio: str, uf: str) -> list[str]:
    """Domínios a tentar, do mais provável ao menos."""
    nome, sigla = _slug(municipio), _slug(uf)
    if not nome or not sigla:
        return []
    base = f"{nome}.{sigla}.gov.br"
    return [f"https://www.{base}", f"https://{base}",
            f"https://prefeitura.{base}", f"https://site.{base}"]


def _sessao() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; Plataforma IP - Hub de Municipios)",
        "Accept": "text/html,application/xhtml+xml",
    })
    return s


SESSAO = _sessao()


def _melhor_email(textos: str, dominio: str) -> Optional[str]:
    """
    Escolhe o e-mail institucional mais provável.

    Prioriza endereço do próprio domínio da prefeitura e com prefixo de canal público
    (ouvidoria@, contato@). E-mail de outro domínio costuma ser do fornecedor que fez o
    site, e endereço com nome de pessoa é justamente o que não se quer coletar.
    """
    achados = {e.lower() for e in _RE_EMAIL.findall(_texto_visivel(textos))}
    do_dominio = [e for e in achados if dominio and dominio in e.split("@")[-1]]
    for prefixo in PREFIXOS_INSTITUCIONAIS:
        for email in sorted(do_dominio):
            if email.split("@")[0].startswith(prefixo):
                return email
    return sorted(do_dominio)[0] if do_dominio else None


def _melhor_telefone(html: str) -> Optional[str]:
    """Primeiro telefone formatado, com DDD válido, no texto visível da página."""
    for grupos in _RE_TELEFONE.findall(_texto_visivel(html)):
        ddd, prefixo, sufixo = (grupos[0], grupos[1], grupos[2]) if grupos[0] else (
            grupos[3], grupos[4], grupos[5])
        try:
            if int(ddd) in DDDS_VALIDOS:
                return f"({ddd}) {prefixo}-{sufixo}"
        except (TypeError, ValueError):
            continue
    return None


def buscar(municipio: str, uf: str, codigo_municipio: str = "") -> Contato:
    """
    Procura o canal institucional da prefeitura.

    Nunca levanta: portal fora do ar, domínio inexistente ou HTML ilegível devolvem um
    `Contato` vazio com a observação do que houve. A apresentação segue sem o dado.
    """
    contato = Contato(codigo_municipio=str(codigo_municipio), municipio=str(municipio),
                      uf=str(uf), coletado_em=date.today().isoformat())
    if not municipio or not uf:
        contato.observacao = "Município ou UF ausente."
        return contato

    dominio = f"{_slug(municipio)}.{_slug(uf)}.gov.br"
    home = None
    for candidato in candidatos_de_dominio(municipio, uf):
        try:
            resposta = SESSAO.get(candidato, timeout=TIMEOUT, allow_redirects=True)
        except requests.exceptions.RequestException:
            continue
        if resposta.status_code == 200 and len(resposta.text) > TAMANHO_MINIMO_PAGINA:
            home = resposta
            break

    if home is None:
        contato.observacao = (
            f"Nenhum portal respondeu no padrão {dominio}. Município que usa domínio "
            "próprio (como Belo Horizonte, em pbh.gov.br) precisa ser preenchido à mão.")
        return contato

    contato.site = str(home.url).rstrip("/")
    contato.url_origem = contato.site
    texto = home.text
    contato.email = _melhor_email(texto, dominio)
    contato.telefone = _melhor_telefone(texto)

    # A home costuma não trazer contato; a página dedicada quase sempre traz.
    if not (contato.email and contato.telefone):
        for caminho in CAMINHOS_CONTATO:
            try:
                pagina = SESSAO.get(contato.site + caminho, timeout=TIMEOUT,
                                    allow_redirects=True)
            except requests.exceptions.RequestException:
                continue
            if pagina.status_code != 200 or len(pagina.text) < TAMANHO_MINIMO_PAGINA:
                continue
            contato.email = contato.email or _melhor_email(pagina.text, dominio)
            telefone = _melhor_telefone(pagina.text)
            contato.telefone = contato.telefone or telefone
            if contato.email and contato.telefone:
                contato.url_origem = str(pagina.url).rstrip("/")
                break

    if not (contato.email or contato.telefone):
        contato.observacao = ("Portal localizado, mas sem e-mail ou telefone legível "
                              "na home nem nas páginas de contato usuais.")
    return contato


# ── Cache em disco ───────────────────────────────────────────────────────────
def _caminho_cache():
    return config.SICONFI_CACHE / "contatos_municipios.parquet"


def carregar_cache() -> pd.DataFrame:
    caminho = _caminho_cache()
    if caminho.exists():
        try:
            return pd.read_parquet(caminho)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUNAS)


def buscar_com_cache(municipio: str, uf: str, codigo_municipio: str = "",
                     revalidar: bool = False) -> Contato:
    """
    Igual a `buscar`, reaproveitando o cache local.

    O cache guarda inclusive a busca que não achou nada, para não bater no mesmo portal
    inexistente a cada geração de apresentação. `revalidar=True` força a nova coleta —
    contato de prefeitura muda com troca de gestão.
    """
    chave = f"{_slug(municipio)}|{_slug(uf)}"
    cache = carregar_cache()
    if not revalidar and not cache.empty:
        guardado = cache[
            (cache["municipio"].map(_slug) == _slug(municipio))
            & (cache["uf"].map(_slug) == _slug(uf))
        ]
        if not guardado.empty:
            registro = guardado.iloc[-1].to_dict()
            return Contato(**{c: (None if pd.isna(registro.get(c)) else registro.get(c))
                              for c in COLUNAS if c in registro})

    contato = buscar(municipio, uf, codigo_municipio)
    try:
        config.garantir_pastas()
        novo = pd.DataFrame([asdict(contato)], columns=COLUNAS)
        combinado = pd.concat([cache, novo], ignore_index=True)
        combinado = combinado.drop_duplicates(subset=["municipio", "uf"], keep="last")
        combinado.to_parquet(_caminho_cache(), index=False)
    except Exception:
        pass
    return contato


__all__ = ["Contato", "COLUNAS", "buscar", "buscar_com_cache",
           "candidatos_de_dominio", "carregar_cache"]
