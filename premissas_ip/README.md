# premissas_ip — Coleta neutra de premissas e geração de saídas

Substitui o processo manual de montar a **planilha de inputs de IP** (que alimenta o
modelo econômico-financeiro) e os **blocos do relatório de engenharia** a partir das
planilhas Excel legadas. O portal coleta as premissas de qualquer município e gera as
saídas parametrizadas.

## Fluxo

```
Wizard (paginas/premissas_ip.py)
   │  preenche →
   ▼
Respostas (coleta.py)  ── município, valores{param_id→valor}, iluminacao_especial[]
   │
   ├─► saidas/planilha_inputs.py ─► Inputs IP.xlsx
   │       • aba Inputs_IP: 32 seções no layout do modelo (cada parâmetro na sua linha)
   │       • aba Dimensionamento: bases CAPEX/OPEX e custo de equipe (fórmulas vivas)
   │       • aba Distribuição Temporal: expansão e reinvestimento ano a ano (fórmulas)
   │
   └─► saidas/blocos_relatorio.py ─► Blocos Relatorio.xlsx + Resumo.md
           • Resumo CAPEX (§6) e OPEX (§7) com coluna "Memória de cálculo"
```

## Componentes

| Arquivo | Papel |
|---|---|
| `schema_inputs.json` | Catálogo das 32 seções → parâmetros (fonte da verdade dos dados). |
| `_gerar_schema.py` | Gera o JSON a partir da `Planilha Modelo Inputs IP` (one-shot, reexecutável). |
| `schema.py` | Wrapper de runtime: dataclasses `Secao`/`Parametro`, `carregar()`. |
| `coleta.py` | Contrato `Respostas` + agrupamento das seções em macro-blocos. |
| `modelo.py` | Relações de dimensionamento como **fórmulas Excel** (`Inputs_IP!$F$linha`). |
| `saidas/planilha_inputs.py` | Monta a planilha de inputs parametrizada. |
| `saidas/blocos_relatorio.py` | Monta os blocos de dados do relatório. |

## Neutralidade (serve qualquer município)

A *estrutura* é fixa; os *valores* são por município. A separação vem da coluna
**Fonte** do modelo:

- **Coletado no wizard** (default em branco, marcado 📍): Cadastro/Campo da Prefeitura,
  Estudos de Engenharia do Projeto, Contratos, Distribuidora, Legislação Municipal,
  indicação da Prefeitura — além das seções Quantitativo Parque, CIP, RCL, Distribuidora.
- **Premissa/catálogo editável** (default do modelo como ponto de partida): Premissa PPP,
  Mercado, SINAPI, CAGED, CADTERC, ANEEL, etc.

A seção **Iluminação Especial** é dinâmica (lista de itens definida pelo usuário), pois
seus elementos são específicos de cada cidade.

## Parametrização viva (auditável)

Nada é congelado em Python na planilha de inputs: expansão, equipes, veículos e a
distribuição temporal por vida útil/reinvestimento são **fórmulas** que referenciam as
células de premissa em `Inputs_IP`. Alterar uma premissa no Excel recalcula tudo. Os
blocos do relatório documentam cada conta na coluna *Memória de cálculo*.

## Regenerar o catálogo

Apenas se a estrutura do modelo de referência mudar:

```
py -3 app/premissas_ip/_gerar_schema.py "docs/planilhas engenharia/Planilha Modelo Inputs IP - Tramandaí v1.xlsx"
```

## Auto-preenchimento por upload (independente das demais páginas)

Para reduzir a digitação, a página aceita uploads que preenchem os campos por
casamento de **palavra-chave** (neutro entre municípios). Nada é aplicado em silêncio —
tudo aparece em painéis de conferência e segue editável.

| Fonte (upload) | Módulo | Preenche |
|---|---|---|
| **DTO** (.docx) | `dto.py` | parque total, LED, expansão anual, demanda reprimida, vida útil LED |
| **Extrapolação** (.xlsx) | `planilhas.extrair_extrapolacao` | distribuição por classe viária (16 campos) + **custo de luminária por classe** (p167–p177) + CPE (custo unitário luminária/estrutura) + custo médio luminária + custo de braço (via aba "Inputs Técnicos - Eco-fin") + catálogo de luminárias |
| **Proposição de IAE** (.xlsx) | `planilhas.extrair_proposicao_iae` | custo da estrutura de IAE + contagens de pontos |
| **InvBens / ID** (.xlsx) | `planilhas.extrair_invbens` | lista de Iluminação Especial por bem (nome + pontos atuais + **CAPEX por bem**, somado do "Quadro Investimentos") |

Mapa de fluxo completo: [docs/fluxograma_premissas_ip.html](../../docs/fluxograma_premissas_ip.html).

## Testes

```
py -3 -m pytest tests/test_premissas_schema.py tests/test_premissas_saidas.py \
                tests/test_premissas_dto.py tests/test_premissas_planilhas.py
```
