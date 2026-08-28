# Hub de Municípios

Cruza a **receita de COSIP** declarada ao SICONFI com o **parque de iluminação pública** da
BDGD/ANEEL, para qualquer município do Brasil, e devolve os indicadores de triagem de uma
PPP de IP.

A pergunta que o módulo responde não é "quanto o município arrecada", e sim **se a
arrecadação sustenta o serviço** — o que só aparece dividindo a receita pelo parque que ela
tem de custear.

## Módulos

| Arquivo | Papel |
|---|---|
| `config.py` | caminhos das bases e parâmetros de negócio (tarifa, LED de referência) |
| `siconfi.py` | cliente da API do Tesouro; extrai a COSIP do DCA Anexo I-C, com cache local |
| `bdgd.py` | ingestão da entidade PIP do geodatabase e agregação municipal |
| `indicadores.py` | cruzamento das duas bases e os indicadores derivados |
| `etl_bdgd.py` | CLI do ETL offline |

A UI é `paginas/hub_municipios.py` — só apresentação, como nos demais módulos do portal.

## Uso programático

```python
from hub_municipios import siconfi, indicadores

cosip  = siconfi.consultar_com_cache(["3106200", "3170206"], [2023, 2024, 2025])
painel = indicadores.cruzar(cosip)                 # lê o parque agregado automaticamente

for _, linha in painel.iterrows():
    for aviso in indicadores.ressalvas(linha):     # ressalvas de due diligence
        print(aviso)
```

## Fluxo de dados

```
API SICONFI  ──────────────────────────────┐
  DCA Anexo I-C                            │
                                           ├──► indicadores.cruzar() ──► painel
BDGD .gdb ──ogr2ogr──► parquet PIP ──┐     │
  (dezenas de GB)      (dezenas MB)  │     │
                                     └─────┘
                     agregado municipal (centenas de KB, versionado)
```

O portal lê **apenas** o agregado. O bruto nunca entra no Streamlit.

## Duas armadilhas do domínio que o módulo trata

### 1. Quebra de série no plano de contas da COSIP

Verificado na API em 28/08/2026:

| Exercício | Conta |
|---|---|
| até 2017 | `1.2.3.0.00.00.00 — Contribuição para Custeio do Serviço de Iluminação Pública` |
| 2018 em diante | `1.2.4.0.00.0.0 — Contribuição para o Custeio do Serviço de Iluminação Pública` |

Muda o código **e** o texto — o artigo "o" só existe na redação nova. Filtrar pelo texto
literal devolve vazio para 2017 e anteriores, e esse vazio costuma ser lido como "o
município não cobra COSIP", apagando exercícios inteiros da série. O casamento aqui é por
padrão de nome **+** classe de conta orçamentária (`RO`, o que também exclui a
intraorçamentária `RI`, que duplicaria o valor).

### 2. Schema variável da BDGD

A PIP não tem o mesmo schema em todas as versões:

| Base | `TIPO_LAMP` / `POT_LAMP` / `PERDA_REAT` |
|---|---|
| Cemig-D V11 (2024) | presentes |
| Energisa_MT M10 (2017) | **ausentes** |

Pedir uma coluna inexistente faz o `ogr2ogr` abortar a base inteira. O ETL lê o schema real
antes de montar o SELECT e avisa quando o mix tecnológico não pôde ser apurado.

## Como a tecnologia da lâmpada é determinada

A BDGD não traz o domínio de `TIPO_LAMP` embutido, e o código varia entre distribuidoras.
Em vez de fixar uma tabela, o rótulo é **inferido da assinatura física** de cada código:

| Assinatura | Tecnologia |
|---|---|
| perda de reator ≈ 0 | LED (driver integrado) |
| potências 80/125 W | vapor de mercúrio (série normalizada exclusiva) |
| perda de reator < 10 W | vapor metálico |
| perda ~17 W nas potências 70/100/150 W | vapor de sódio |

Assinaturas medidas na Cemig-D V11: LED 0,0 W · mercúrio 12,2 W · metálico 4,8 W · sódio
17,5 W. 70/150 W servem tanto a sódio quanto a metálico — o que separa os dois é a perda do
reator, não a potência.

O ETL imprime a tabela inferida com as evidências. Para corrigir um rótulo sem tocar no
código, crie `data/tipo_lamp_override.csv` com as colunas `tipo_lamp,tecnologia`.

## Validação física do dado

O ETL e os testes checam **horas equivalentes de operação** (`consumo ÷ carga instalada`).
IP acionada por relé fotoelétrico opera 11–12 h/dia, ou seja 4.000–4.400 h/ano. A Cemig-D
V11 dá **4.160 h/ano** — o dado fecha. Fora da faixa de 3.000–5.000 h, consumo e carga
declarados à ANEEL estão inconsistentes e todo indicador por ponto fica suspeito.

## Filtro de declaração implausível

Abaixo de **R$ 12 por ponto por ano** (R$ 1/ponto/mês) o valor não pode ser arrecadação
real — é erro de preenchimento do DCA. Caso observado: Passos/MG declarou R$ 134,83 (2020)
e R$ 16,35 (2025) para um parque de 15.573 pontos. Sem esse filtro o município entra em
qualquer ranking como "o pior do estado" e derruba a mediana. Os registros são marcados em
`declaracao_implausivel`, excluídos dos gráficos e das medianas, e permanecem visíveis na
tabela.

## Ressalvas de uso

1. **Valores nominais.** Série plurianual exige deflacionamento (IPCA/IGP-M).
2. **O DCA é declaratório e não auditado.** Reclassificação de rubrica pelo ente aparece
   como oscilação de receita — cruze com o balancete municipal antes de projetar.
3. **Receita arrecadada ≠ faturada.** A inadimplência da COSIP está embutida no dado; a
   base faturada só vem da concessionária de energia.
4. **A BDGD tem data-base fixa.** Comparar COSIP de um ano com parque de outro embute
   defasagem — ela é calculada e sinalizada em cada ficha.
5. **COSIP arrecadada não é garantia de bancabilidade.** Ela limita o teto de
   contraprestação, mas o que sustenta o financiamento é o mecanismo de vinculação da
   arrecadação (conta vinculada, fundo garantidor), não o valor bruto.
6. **A tarifa e o LED de referência são defaults de triagem**, editáveis na interface. Em
   estudo, a tarifa vem da resolução homologatória da distribuidora e do regime tributário
   do município.

## Atualizar a BDGD

Ver `dados/LEIA-ME.md`. Em resumo, a partir de `app/`:

```bash
py -m hub_municipios.etl_bdgd --listar        # o que há e o que falta
py -m hub_municipios.etl_bdgd --paralelo 3    # processa o que falta
py -m hub_municipios.etl_bdgd --consolidar    # reconsolida sem reprocessar
```

O ETL varre a pasta local `dados/bdgd/brutos/` **e** os repositórios de
`config.BDGD_PASTAS_EXTRA` — por padrão, o acervo nacional no Drive compartilhado do time
(38 distribuidoras, ~129 GB). Quando a mesma distribuidora aparece nas duas origens, vence
a de data-base mais recente.

É **retomável**: cada base agregada é gravada em `processados/agregado_<slug>.parquet`
antes de a próxima começar. Interromper não custa o que já foi feito.

Requer GDAL (`ogr2ogr` + `ogrinfo`) — OSGeo4W ou QGIS. Ver `app/requirements-etl.txt`.

**O gargalo é rede, não CPU.** Base local de 15,9 GB: ~55 s. Base de 5,65 GB no Drive:
~7 min, porque o Drive File Stream baixa sob demanda. Por isso o ETL ordena da menor base
para a maior (valor entregue mais cedo) e `--paralelo` vale a pena — são conexões
simultâneas, não trabalho de processador.
