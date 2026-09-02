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
| `aneel_tarifas.py` | tarifas B4a da ANEEL — homologada (sem tributos) e faturada (com) |
| `indicadores.py` | cruzamento das bases e os indicadores derivados |
| `etl_bdgd.py` | CLI do ETL offline da BDGD |
| `etl_aneel.py` | CLI do ETL offline das tarifas |

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

## Energia: as duas tarifas e o anual × ciclo

Este bloco foi refeito em 28/08/2026 depois de uma divergência de fator 2 entre a
triagem de São José da Lapa e o modelo econômico-financeiro de Matozinhos. **Não havia
erro de cálculo físico** — havia duas grandezas diferentes sendo comparadas:

| | O que é | Onde erra |
|---|---|---|
| `tarifa_sem_tributos` | TUSD + TE da resolução homologatória | Cemig: R$ 0,3399/kWh |
| `tarifa_com_tributos` | R$/kWh **faturado**, receita ÷ mercado da classe IP | Cemig: R$ 0,4725/kWh |
| `economia_reais_ano` | ano 1, reais constantes | — |
| `economia_ciclo_reais` | acumulado **nominal** do prazo | 22 anos a 4% a.a. = ×34,25, não ×22 |

Usar a tarifa da REH onde cabe a faturada tira 28% do custo de energia; multiplicar o
anual pelo prazo tira outros 36% do acumulado. Juntos, fator 2 — que era exatamente a
distância entre as duas leituras. Com as duas corrigidas, Matozinhos fecha em
R$ 16,46 mi contra os R$ 17 mi do modelo da equipe.

Outras duas mudanças do mesmo lote:

- **Base de consumo**: o declarado à ANEEL passa na frente do derivado sempre que as
  horas equivalentes (recalculadas de consumo ÷ carga, não lidas da coluna) caem em
  3.000–5.000 h/ano. `origem_consumo` diz qual valeu — 3.537 municípios pelo declarado,
  526 pelo derivado.
- **Guarda de potência**: 1.354 municípios declaram potência média fora de 50–400 W por
  ponto (Neoenergia Elektro 208.625 W, Copel 37.736 W, CPFL e RGE ~1.250 W). Neles o
  bloco de energia inteiro sai nulo, com ressalva. Arrecadação e triagem seguem válidas.

## Atualizar as tarifas da ANEEL

```bash
py -m hub_municipios.etl_aneel --listar   # imprime o SCHEMA REAL; rode isto primeiro
py -m hub_municipios.etl_aneel            # grava data/tarifas_b4a.parquet
py -m hub_municipios.etl_aneel --ano 2025 # fixa o ano do SAMP
```

Uma vez por ano, depois do reajuste. Três armadilhas, todas tratadas e testadas:

1. **O B4a não está em `DscSubGrupo`.** O subgrupo é `"B4"`; o "a" só existe em
   `DscSubClasse` (`"Iluminação pública – B4a"`). Filtrar por `"B4a"` no subgrupo
   devolve zero linhas sem erro nenhum. `B4b` é bulbo de lâmpada e fica de fora.
2. **`Receita Energia (R$)` do SAMP JÁ INCLUI os tributos.** ICMS, PIS/PASEP e COFINS
   são o detalhamento do que está dentro dela, cobrados "por dentro". Somá-los por cima
   dá R$ 0,60/kWh na Cemig em vez de R$ 0,47 — 27% de inflação no custo de energia. A
   conferência que pega isso: descontando os tributos, o SAMP tem de reproduzir a
   tarifa da REH, que vem de outro dataset. Cemig: 0,3424 contra 0,3399, 0,7% de erro.
3. **O SAMP tem mês com erro de ordem de grandeza.** A Cemig declarou R$ 388 milhões de
   receita de IP em junho/2026 contra ~R$ 37 milhões nos demais meses. Por isso a
   agregação é **mediana das competências**, nunca soma anual — que devolvia
   R$ 1,25/kWh.

O de-para BDGD → sigla ANEEL está em `ALIAS_BDGD_ANEEL` e **não é derivável**: a ANEEL
identifica as Energisa por acrônimo regulatório (EAC, EMS, EMT, EMR, ENF, EPB, ERO,
ESE, ESS, ETO), a Enel SP pela razão social antiga (ELETROPAULO) e as duas do Norte
pelo controlador atual (ÂMBAR). Sem a tabela, o casamento por nome acerta 16 de 38.
Cobertura atual: 38/38 distribuidoras, 4.913 dos 5.417 municípios.

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


## Despesa declarada, contratos, ranking e apresentação (01/09/2026)

O Hub passou a olhar também o **outro lado da conta**. Até aqui ele cruzava a receita
de COSIP com o parque; sem a despesa, a "sobra" era uma conta de um lado só.

### `despesas.py` — SICONFI, DCA Anexo I-E

Mesma API do módulo de receita, outro anexo: despesa por função e subfunção.

- **25.752 Energia Elétrica** e **15.452 Serviços Urbanos** são **envelopes**, não gasto
  de IP. O primeiro soma iluminação com escolas, prédios e poços; o segundo soma IP com
  limpeza, capina, praças e cemitério. A classificação funcional nacional não tem
  subfunção de iluminação pública, ponto final.
- **45% dos municípios não declaram a função 25** — medido em 60 municípios de MG,
  exercício 2023 (31 com 25.752, 27 sem função 25, 2 só com a função 25 agregada).
  Belo Horizonte e Uberlândia estão no grupo do meio. Tratar essa ausência como "não
  gastou com energia" seria erro grosseiro: o módulo cai para o envelope disponível e
  grava a origem em `origem_despesa_energia`, no mesmo espírito de `origem_pontos`.
- **A coluna padrão é "Despesas Liquidadas"**, não empenhadas nem pagas: liquidação é o
  reconhecimento do fato gerador e é o estágio comparável com a receita arrecadada do
  Anexo I-C. Em Mateus Leme/2023 a diferença entre empenhado e pago foi de 24% só na
  energia — escolher errado distorce a conta.
- **O anexo não tem linha "Total Geral"**: o total é a soma de "Despesas Exceto
  Intraorçamentárias" com "Despesas Intraorçamentárias".

Dois indicadores nascem daí, em `indicadores.cruzar(despesas_declaradas=...)`:

| Indicador | O que responde |
|---|---|
| `cosip_cobre_energia` | COSIP ÷ despesa de energia. "A CIP paga a conta de luz?" Mateus Leme/2023: **71,5%**, não paga. |
| `ip_sobre_energia_declarada` | Custo de energia da IP estimado pela BDGD ÷ despesa declarada. Teste de consistência: a IP é parte da conta, nunca mais que ela. Mateus Leme deu 29,8%, faixa plausível. |

Ambos só são calculados onde há a função 25. Com o denominador vindo de Serviços
Urbanos a razão não teria significado, e o módulo prefere não responder a responder errado.

### `contratos_ip.py` — PNCP

Resolve pelo outro lado: o contrato, com objeto em texto livre e valor global. Três
limitações medidas na API, todas tratadas e nenhuma eliminável:

1. **Busca fuzzy** — casa por relevância, não por termo obrigatório. Entre os contratos
   de IP de MG veio um de cessão de "Salão de Festas". Daí o filtro `_e_relevante`.
2. **`municipio_id` do PNCP não é o código IBGE** (BH é 2310 lá e 3106200 no IBGE). O
   casamento é por nome normalizado + UF, com o risco de homônimo que isso carrega.
3. **Rate limiting agressivo** — chamadas seguidas caem com `ConnectionReset`, sem
   status HTTP. Exige sessão com retry, backoff e pausa entre chamadas.

Por isso o resultado é **evidência para conferência humana**, não número de cálculo. O
`valor_global` cobre todo o prazo do contrato, não um ano, e município grande costuma
ter vários contratos vigentes.

### `ranking.py`

Critérios de ordenação da carteira, cada um com o **sentido natural** da sua ordenação:
parque com pouco LED e CIP que cobre pouco da conta ordenam do **menor** para o maior,
porque ali está a oportunidade, não o pior colocado. Município sem o dado do critério
vai sempre para o fim — ausência de informação não é desempenho.

### `apresentacao.py`

Gera o `.pptx` para levar à prefeitura, no **padrão visual do deck de Itabira**
("Estruturação da PPP de Cidade Inteligente", 05/08/2026), adotado em 02/09/2026:

```
Capa → De onde vem esta análise → Possíveis escopos
  → [ município: panorama | pré-viabilidade | potenciais impactos | a conta fecha? |
      soluções digitais ] × N
  → Metodologia e fontes
```

- **A marca da peça é Houer, e isso é exceção deliberada** à diretriz de 30/08/2026 que
  tirou a marca do portal. Vale só para o `.pptx`, que é material comercial apresentado
  pela consultoria — planilhas, relatórios e a UI seguem como Plataforma IP.
- **A paleta foi amostrada pixel a pixel do deck original**, não estimada: degradê
  `#003264 → #00508E` na faixa-título, verde `#00FF29` para arrecadação, vermelho
  `#FF3131` para despesa, fundo `#F6F6F6` com arcos tênues. Trocar por "azul
  corporativo genérico" descaracteriza a peça ao lado das que a equipe já entregou.
- **Tudo em vetor, inclusive o mapa e a silhueta urbana**, desenhados com `build_freeform`
  e formas nativas (`malhas.carregar_contorno_uf` traz o estado como um polígono só —
  empilhar 800 polígonos municipais por slide inviabilizaria o arquivo). Nada de PNG: o
  cliente edita qualquer elemento, o texto continua selecionável e não é preciso
  instalar o `kaleido`. Os ícones são emblemas montados com formas nativas até a equipe
  passar a biblioteca licenciada — decisão do usuário em 02/09/2026.
- **O percentual do slide de pré-viabilidade replica o critério do deck**:
  `(arrecadação − despesa declarada) / arrecadação`, para manter comparabilidade com as
  peças já entregues. A ressalva sai no rodapé do próprio slide, porque é material: a
  despesa declarada é o custeio de hoje, e a contraprestação da PPP — com CAPEX
  amortizado, telegestão e reinvestimento — é estruturalmente maior. Quem desconta a
  contraprestação é `sobra_percentual`, no slide "A conta fecha?".
- **Os números do potencial ficam só no slide de impactos.** Quando o slide da p13
  entrou, os mesmos seis cartões passaram a aparecer duas vezes no mesmo deck; o slide
  de fechamento foi enxugado para o custo do serviço, o contrato atual do PNCP e a
  frase-síntese. Travado em teste.
- **A tese muda conforme o parque.** Onde a economia de energia fica abaixo de 5%, o
  slide de impactos troca de argumento: em vez de estampar "economia R$ 0" três vezes,
  mostra parque em LED, potência média e conta de energia atual. Em MG isso é a regra —
  75,6% do parque estadual já é LED, e Mateus Leme tem 95,8% de LED com 54,6 W médios,
  abaixo da potência de referência.
- **A frase de fechamento é condicionada ao número.** Sobra negativa não vira silêncio
  nem promessa: o slide diz que a CIP não banca o serviço nas premissas adotadas e que
  viabilizar exige revisar a lei de custeio, ajustar escopo ou prever contrapartida.
- Densidade (pontos por mil habitantes) foi deliberadamente deixada **fora** do slide:
  ela estoura a faixa plausível sempre que o cadastro da distribuidora está inflado, e
  peça de cliente não é lugar de exibir indicador que denuncia problema de base.
