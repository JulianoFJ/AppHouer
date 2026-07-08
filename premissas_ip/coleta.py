"""
Contrato de dados da coleta de premissas + utilidades de agrupamento.

`Respostas` é o objeto único que a página do wizard preenche e que os geradores de
saída (`saidas/planilha_inputs.py`, `saidas/blocos_relatorio.py`) consomem. Mantê-lo
explícito desacopla a UI da lógica de geração.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import schema


@dataclass
class Respostas:
    municipio: str = ""
    uf: str = ""
    data_base: str = ""
    # valores por parâmetro: param_id -> valor (float/str/bool). Frações para %.
    valores: dict[str, object] = field(default_factory=dict)
    # itens da seção dinâmica (Iluminação Especial): lista de dicts por coluna.
    iluminacao_especial: list[dict] = field(default_factory=list)

    def valor(self, param_id: str, padrao=None):
        v = self.valores.get(param_id, padrao)
        return padrao if v is None or v == "" else v


# ── Agrupamento das seções em macro-blocos navegáveis ─────────────────────────
# Apenas o que a engenharia produz. Financeiro/Receitas e Socioambiental NÃO são
# feitos pela engenharia (são preenchidos pela equipe econômico-financeira) — ficam
# fora do wizard. Continuam existindo no schema e nas saídas (com default) para
# preservar o layout do modelo eco-fin.
MACRO_BLOCOS: list[tuple[str, tuple[str, ...]]] = [
    ("Premissas & Prazos", ("Prazos",)),
    ("Parque & Cadastro", ("Quantitativo Parque",)),
    ("Engenharia & Proposições", (
        "Estudos de Engenharia",
        "Iluminação Viária",
        "Telegestão",
        "Iluminação de Áreas Especiais",
        "Expansão e Demanda Reprimida",
        "Iluminação Especial",
    )),
    ("Operação & Custos", (
        "Materiais de Manutenção",
        "Dimensionamento de Equipes",
        "Encargos e Benefícios",
        "Veículos",
        "Poda de Árvore",
        "Infraestrutura",
        "Despesas pré-operacionais",
        "Seguros e Garantias",
        "Verificador Independente",
        "Instituição Financeira Depositária",
    )),
]

# Seções fora do escopo da engenharia (não exibidas no wizard).
SECOES_EXCLUIDAS: tuple[str, ...] = (
    "Contribuição de Iluminação Pública", "Receita Corrente Líquida",
    "Distribuidora de Energia", "Despesas Contratos Atuais O&M",
    "Receitas Acessórias", "Venda de Sucatas", "Bônus de Energia",
    "Tributos Municipais", "Value for Money",
    "Estudos Ambientais", "Levantamento de Stakeholders",
    "Avaliação Preliminar de Áreas Contaminadas",
    "Sistema de Gestão Socioambiental",
    "Consultoria Especializada em Comunicação Social",
)


def secao_no_escopo(secao: schema.Secao) -> bool:
    """True se a seção é feita pela engenharia (exibida no wizard)."""
    return not any(secao.nome.startswith(p) for p in SECOES_EXCLUIDAS)


def macro_bloco(secao: schema.Secao) -> str:
    """Retorna o macro-bloco de uma seção (casamento por prefixo do nome)."""
    for nome_bloco, prefixos in MACRO_BLOCOS:
        if any(secao.nome.startswith(p) for p in prefixos):
            return nome_bloco
    return "Outros"


def secoes_por_bloco() -> list[tuple[str, list[schema.Secao]]]:
    """Lista (macro-bloco -> seções) na ordem de `MACRO_BLOCOS`, só do escopo de engenharia."""
    s = schema.carregar()
    ordem = [b[0] for b in MACRO_BLOCOS] + ["Outros"]
    agrupado: dict[str, list[schema.Secao]] = {nome: [] for nome in ordem}
    for sec in s.secoes:
        if secao_no_escopo(sec):
            agrupado[macro_bloco(sec)].append(sec)
    return [(nome, agrupado[nome]) for nome in ordem if agrupado[nome]]


def inputs_no_escopo() -> list[schema.Parametro]:
    """Parâmetros de input das seções de engenharia (para métricas de progresso)."""
    return [p for _, secoes in secoes_por_bloco() for sec in secoes for p in sec.inputs()]


__all__ = ["Respostas", "MACRO_BLOCOS", "SECOES_EXCLUIDAS", "secao_no_escopo",
           "macro_bloco", "secoes_por_bloco", "inputs_no_escopo"]
