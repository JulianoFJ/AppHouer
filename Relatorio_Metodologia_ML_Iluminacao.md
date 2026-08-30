# Relatório Técnico de Metodologia
## Sistema Preditivo de Iluminação Pública com Aprendizado de Máquina
### Concessão de Cidade Inteligente / Iluminação Pública Municipal

---

**Versão:** 1.0  
**Data:** Abril de 2026  
**Classificação:** Técnico-Interno  

---

## Sumário

1. [Contexto e Objetivo](#1-contexto-e-objetivo)
2. [Base de Dados](#2-base-de-dados)
3. [Arquitetura do Sistema Preditivo](#3-arquitetura-do-sistema-preditivo)
4. [Modelos de Aprendizado de Máquina](#4-modelos-de-aprendizado-de-máquina)
5. [Métricas de Desempenho Estatístico](#5-métricas-de-desempenho-estatístico)
6. [Lógica de Conformidade NBR 5101:2024](#6-lógica-de-conformidade-nbr-51012024)
7. [Módulos Auxiliares de Inteligência](#7-módulos-auxiliares-de-inteligência)
8. [Limitações Reconhecidas e Controles Adotados](#8-limitações-reconhecidas-e-controles-adotados)
9. [Justificativa do Uso Preditivo em Concessões](#9-justificativa-do-uso-preditivo-em-concessões)
10. [Conclusão](#10-conclusão)
11. [Referências Normativas e Técnicas](#11-referências-normativas-e-técnicas)

---

## 1. Contexto e Objetivo

### 1.1 Cenário de Concessão

No modelo de concessão de iluminação pública municipal — modalidade cada vez mais adotada em projetos de Cidade Inteligente — a empresa concessionária assume a responsabilidade integral pelo inventário, modernização, operação e manutenção do parque de luminárias por um período contratual tipicamente superior a vinte anos. O município transfere o risco de eficiência energética e a obrigação de conformidade normativa para o concessionário.

Nesse contexto, dois desafios técnicos tornam-se críticos:

1. **Escala de inventário:** municípios de médio e grande porte possuem dezenas de milhares de pontos de iluminação pública, cada um com geometria de via, tipo de estrutura e demanda fotométrica distintos. Dimensionar luminária a luminária por meio de simulação fotométrica completa (software Dialux, Relux ou equivalente) é inviável operacionalmente no prazo de proposta ou mobilização.

2. **Pressão de conformidade normativa:** a ABNT NBR 5101:2024 estabelece requisitos mínimos de luminância média (Lmed), iluminância média (Emed), uniformidade global (Uo) e uniformidade longitudinal (Ul) por subclasse de via. O não-atendimento implica penalidades contratuais e, em última instância, risco à segurança viária.

### 1.2 Objetivo do Sistema

O sistema preditivo tem por objetivo **estimar com rapidez e rastreabilidade** as métricas fotométricas esperadas e a potência necessária para cada ponto de iluminação, dado o conjunto de características geométricas e classificação da via, permitindo:

- Pré-dimensionamento de potência por fornecedor para toda a malha inventariada;
- Identificação automatizada de pontos de risco de não-conformidade NBR 5101;
- Detecção de necessidade de Correção de Ponto Escuro (CPE);
- Sugestão de tipo de braço adequado à geometria instalada;
- Suporte à análise de viabilidade econômica (CAPEX estimado) e eficientização energética.

O sistema **não substitui** o projeto luminotécnico definitivo exigido contratualmente, mas provê uma camada de inteligência de primeira triagem sobre inventários de larga escala, reduzindo de semanas para minutos o tempo de análise.

---

## 2. Base de Dados

### 2.1 Origem dos Dados

Os dados de treinamento são provenientes de simulações fotométricas históricas reais executadas para concessões municipais anteriores, consolidadas em planilhas de inventário e resultado de simulação no padrão interno da empresa. Cada registro corresponde a um ponto de iluminação simulado em software fotométrico, com as colunas de entrada (geometria) e resultado (métricas fotométricas e potência escolhida).

### 2.2 Composição do Dataset

| Item | Dataset Bruto | Dataset Limpo (sem outliers) |
|---|---|---|
| Registros totais | ~3.100 | **2.830** |
| Registros com Braço Novo preenchido | 288 | 288 |
| Variáveis preditoras numéricas | 10 | 10 |
| Variáveis preditoras categóricas | 5 | 5 |
| Subclasses de via representadas | C0–C5 | C0–C5 |

> **Nota:** O inventário histórico disponível concentra-se em vias da categoria C (Áreas de Conflito segundo NBR 5101), que representam a maior parcela das concessões processadas. Categorias M (vias motorizadas) e P (vias pedonais/ciclovias) aparecem com volume reduzido de registros rotulados, o que impacta diretamente a confiabilidade preditiva para essas classes — detalhe tratado na Seção 8.

### 2.3 Variáveis Preditoras (Features)

#### Numéricas (10 features)

| Feature | Descrição | Unidade |
|---|---|---|
| Faixas de Rodagem | Número de faixas da via principal | adimensional |
| Largura Via 1 | Largura da pista principal | m |
| Largura Via 2 | Largura da pista secundária (canteiro) | m |
| Largura Passeio 1 | Largura do passeio/calçada lado 1 | m |
| Largura Passeio 2 | Largura do passeio/calçada lado 2 | m |
| Largura Canteiro Central | Largura do canteiro central, se houver | m |
| Altura da luminária | Altura de montagem da luminária no poste | m |
| Projeção do braço | Distância horizontal do braço ao eixo do poste | m |
| Distância entre postes | Espaçamento entre postes consecutivos | m |
| Distância poste à via | Distância horizontal do poste à borda da via | m |

#### Categóricas (5 features)

| Feature | Categorias possíveis |
|---|---|
| Classificação viária | M1–M6, C0–C5, P1–P6 |
| Tipo de estrutura | Braço, Suporte, Chicote duplo |
| Posteação | Unilateral, Bilateral alternada, Bilateral frontal, Canteiro central |
| Braço Novo | Curto I, Curto II, Médio I, Médio II, Longo I, Longo II |
| Fornecedor | LEDSTAR, SX LIGHTING, TECNOWATT |

### 2.4 Variáveis-Alvo (Targets)

| Código | Nome completo | Unidade | Norma de referência |
|---|---|---|---|
| lmed | Luminância Média | cd/m² | NBR 5101 (classes M) |
| uo | Fator de Uniformidade Global | adimensional | NBR 5101 |
| ul | Uniformidade Longitudinal | adimensional | NBR 5101 (classes M) |
| emed | Iluminância Média horizontal | lux | NBR 5101 (classes C e P) |
| emin | Iluminância Mínima horizontal | lux | NBR 5101 (classes P) |
| w | Potência simulada — IP Principal | W | — |

### 2.5 Tratamento de Outliers

O dataset passou por filtragem de outliers físicos — valores impossíveis oriundos de erros de preenchimento nas planilhas originais (ex: luminâncias negativas, potências nulas, distâncias entre postes de 0 m). O dataset "limpo" é a versão utilizada como base primária de treinamento. Ambas as bases são mantidas e podem ser selecionadas pelo usuário na interface, permitindo comparação do impacto dos outliers sobre as predições.

---

## 3. Arquitetura do Sistema Preditivo

### 3.1 Pipeline de Predição

O sistema segue uma arquitetura de pipeline sequencial com dependência entre modelos:

```
Entradas geométricas + classificação + fornecedor
        │
        ▼
┌─────────────────────────────┐
│  Pré-processamento          │
│  • Imputação (mediana/moda) │
│  • Normalização (Z-score)   │
│  • OneHotEncoding           │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Modelo W         Modelo lmed / uo / ul
   (Potência)       (independentes de W)
       │
       │  W_previsto injetado como feature
       ▼
   Modelo emed    Modelo emin
   (dependem de W para maior acurácia)
```

**Justificativa da dependência W → emed/emin:** fisicamente, a iluminância produzida por uma luminária é função direta do seu fluxo luminoso, que por sua vez é altamente correlacionado com a potência consumida. Ao treinar emed e emin com W como feature adicional, o modelo aprende esta relação física implicitamente, elevando significativamente o R² (de ~0,60 para ~0,94 no caso do emed).

### 3.2 Ajuste pela Hierarquia NBR 5101

Após as predições brutas dos modelos, aplica-se um ajuste de escalonamento de potência baseado na subclasse NBR:

**Classes M (luminância):**

$$W_{ajustado} = W_{previsto} \times \frac{L_{med,req}^{classe}}{L_{med,req}^{M3}}$$

Onde M3 serve como classe de referência (baseline), garantindo a hierarquia de potência M1 > M2 > ... > M6 sem depender da predição direta de lmed (cujo R² é insuficiente para conformidade).

**Classes C e P (iluminância):**

$$W_{ajustado} = W_{previsto} \times \frac{E_{med,req}}{E_{med,previsto}} \quad \text{se } E_{med,previsto} < E_{med,req}$$

Este ajuste só é aplicado quando o modelo prevê iluminância abaixo do requisito normativo, preservando a potência original quando o atendimento já é satisfeito.

---

## 4. Modelos de Aprendizado de Máquina

### 4.1 Regressão — Random Forest (lmed, emed)

**Algoritmo:** `RandomForestRegressor` (scikit-learn)  
**Hiperparâmetros:** 300 estimadores, `min_samples_leaf=2`, `random_state=42`

O Random Forest é um método de ensemble que constrói múltiplas árvores de decisão sobre subamostras aleatórias do dataset (técnica de *bagging*) e sobre subconjuntos aleatórios de features em cada nó (*feature subsampling*). A predição final é a média das predições de todas as árvores.

**Propriedades relevantes para este contexto:**
- Robusto a outliers remanescentes no dataset;
- Capaz de capturar interações não-lineares entre geometria de via e resultado fotométrico;
- Naturalmente fornece estimativa de importância de variável, útil para auditoria do modelo;
- Não requer escalonamento das features numéricas (mantido no pipeline por uniformidade).

**Aplicação:** emed (R²=0,94) e lmed (R²=0,28). A divergência de desempenho entre os dois modelos reflete a qualidade distinta dos dados: registros de iluminância (classes C/P) são mais abundantes e consistentes no dataset histórico do que registros de luminância (classes M).

### 4.2 Regressão — Histogram Gradient Boosting (uo, ul, emin, w)

**Algoritmo:** `HistGradientBoostingRegressor` (scikit-learn)  
**Hiperparâmetros:** 300 iterações, `learning_rate=0.05`, `max_depth=6`, `random_state=42`

O Gradient Boosting constrói árvores sequencialmente, cada uma corrigindo os erros residuais da anterior (técnica de *boosting*). A variante Histogram utiliza histogramas discretizados das features para acelerar o processo, tornando-o escalável a datasets maiores.

**Propriedades relevantes:**
- Excelente capacidade preditiva em dados tabulares estruturados, particularmente quando há relações de ordem entre variáveis (ex: potência cresce com distância entre postes);
- Tratamento nativo de valores ausentes, dispensando imputação prévia (mantida no pipeline por consistência);
- Regularização implícita via `learning_rate` e `max_depth` para evitar sobreajuste.

**Aplicação:** w (R²=0,88), emin (R²=0,82), uo (R²=-27), ul (R²=-0,05). Os modelos de uniformidade (uo, ul) apresentam R² fortemente negativo, indicando que o conjunto de features disponível não captura os fatores que determinam a distribuição espacial do fluxo luminoso — um fenômeno que depende do diagrama polar da luminária (curve de distribuição fotométrica), não disponível no inventário.

### 4.3 Classificação — Random Forest para Braço Novo

**Algoritmo:** `RandomForestClassifier` (scikit-learn)  
**Hiperparâmetros:** 400 estimadores, `random_state=42`  
**Classes:** Curto I (1,2m), Curto II (1,4m), Médio I (1,8m), Médio II (2,4m), Longo I (2,8m), Longo II (3,5m)  
**Amostras de treino:** 288 registros com braço catalogado

Este classificador aprende a associar o perfil geométrico da via ao tipo de braço historicamente utilizado nas simulações. A lógica é: engenheiros experientes escolheram determinado braço para determinada geometria — o modelo generaliza esse padrão.

**Limiar de confiança aplicado:** sugestão de troca de braço é emitida apenas quando a probabilidade da classe predita supera **50%**, reduzindo falsos positivos em geometrias ambíguas. Com este limiar, o modelo gera sugestões em aproximadamente 6,7% das instalações analisadas no inventário de referência, concentradas em casos de subprojeção (braços curtos em vias mais largas).

### 4.4 Classificação — Random Forest para CPE

**Algoritmo:** `RandomForestClassifier` (scikit-learn)  
**Hiperparâmetros:** 400 estimadores, `class_weight='balanced'`, `random_state=42`  
**Classes:** sem CPE (0), com CPE (1)

O modelo de Correção de Ponto Escuro aprende a identificar configurações que historicamente demandaram a inserção de um ponto intermediário de iluminação para eliminar trecho sem cobertura adequada (ponto escuro). O parâmetro `class_weight='balanced'` compensa o desbalanceamento natural da classe positiva (CPE é a minoria dos registros).

**Integração com regras determinísticas:** o classificador ML funciona como sinal complementar a um conjunto de regras fixas:
- Distância entre postes ≥ 45 m **E**
- Potência prevista > 1,40 × média histórica da classe (desvio estatístico significativo)

O CPE é acionado se qualquer um dos dois caminhos (ML ou regras) indicar risco.

### 4.5 Pré-processamento (Pipeline Unificado)

Todos os modelos de regressão e classificação são encapsulados em `sklearn.pipeline.Pipeline` com o seguinte `ColumnTransformer`:

| Transformação | Features | Método |
|---|---|---|
| Imputação numérica | Features numéricas | Mediana do conjunto de treino |
| Normalização | Features numéricas | StandardScaler (Z-score) |
| Imputação categórica | Features categóricas | Moda do conjunto de treino |
| Codificação | Features categóricas | OneHotEncoder (`handle_unknown='ignore'`) |

O encapsulamento em pipeline garante que os parâmetros de pré-processamento (mediana, desvio padrão, categorias conhecidas) sejam aprendidos exclusivamente sobre o conjunto de treino e aplicados consistentemente em inferência, eliminando *data leakage*.

---

## 5. Métricas de Desempenho Estatístico

### 5.1 Resultados por Modelo (Dataset Limpo)

| Modelo | Algoritmo | R² (test) | MAE | Confiabilidade para NBR |
|---|---|---|---|---|
| **w** (Potência) | Hist. Gradient Boosting | **0,882** | 5,09 W | Alta |
| **emed** (Ilum. Média) | Random Forest | **0,938** | 0,75 lux | Alta |
| **emin** (Ilum. Mínima) | Hist. Gradient Boosting | **0,817** | 1,08 lux | Alta |
| **lmed** (Luminância Média) | Random Forest | 0,279 | 4,15 cd/m² | Baixa |
| **ul** (Uniformidade Long.) | Hist. Gradient Boosting | -0,051 | 1,48 | Nenhuma |
| **uo** (Uniformidade Global) | Hist. Gradient Boosting | -27,07 | 0,19 | Nenhuma |

### 5.2 Interpretação do R²

O coeficiente de determinação R² mede a proporção da variância do target explicada pelo modelo:

- **R² = 1,0:** predição perfeita
- **R² = 0,0:** modelo equivale à média do target (sem poder preditivo)
- **R² < 0:** modelo pior que simplesmente prever a média — indica que as features disponíveis não contêm a informação necessária

Os valores negativos de uo e ul indicam que a distribuição espacial da luz (uniformidade) depende fundamentalmente do diagrama polar da luminária — informação não presente no inventário de entradas. Este é um limite físico do problema, não uma falha do algoritmo.

### 5.3 Limiar de Confiabilidade para Verificação NBR

O sistema aplica um limiar conservador de **R² ≥ 0,50** para determinar quais métricas são utilizadas na verificação de conformidade NBR automática. Abaixo deste limiar, o valor é exibido como "Estimado" na interface, sem implicar badge de conformidade verde/vermelho. Esta decisão técnica evita que modelos imprecisos gerem falsos alertas de não-conformidade em larga escala.

| Métrica | Confiável (R²≥0,5)? | Verificação NBR automática |
|---|---|---|
| emed | Sim (0,938) | Sim |
| emin | Sim (0,817) | Sim |
| w | Sim (0,882) | N/A (usada no ajuste) |
| lmed | **Não** (0,279) | **Não** |
| ul | **Não** (-0,051) | **Não** |
| uo | **Não** (-27,07) | **Não** |

### 5.4 Divisão Treino/Teste

Todos os modelos são avaliados com divisão estratificada de 80%/20% (`train_test_split`, `random_state=42`). As métricas reportadas na Seção 5.1 referem-se ao conjunto de teste — dados que o modelo nunca viu durante o treinamento —, garantindo que os resultados reflitam capacidade de generalização e não sobreajuste.

---

## 6. Lógica de Conformidade NBR 5101:2024

### 6.1 Requisitos por Subclasse

A ABNT NBR 5101:2024 define requisitos fotométricos mínimos por subclasse de via:

**Vias Motorizadas (M) — Luminância**

| Subclasse | Lmed mín. (cd/m²) | Uo mín. | Ul mín. |
|---|---|---|---|
| M1 | 2,00 | 0,40 | 0,70 |
| M2 | 1,50 | 0,40 | 0,70 |
| M3 | 1,00 | 0,40 | 0,60 |
| M4 | 0,75 | 0,40 | 0,60 |
| M5 | 0,50 | 0,35 | 0,40 |
| M6 | 0,30 | 0,35 | 0,40 |

**Áreas de Conflito (C) — Iluminância**

| Subclasse | Emed mín. (lux) | Uo mín. |
|---|---|---|
| C0 | 50,0 | 0,40 |
| C1 | 30,0 | 0,40 |
| C2 | 20,0 | 0,40 |
| C3 | 15,0 | 0,35 |
| C4 | 10,0 | 0,35 |
| C5 | 5,0 | 0,35 |

**Vias Pedonais e Ciclovias (P) — Iluminância**

| Subclasse | Emed mín. (lux) | Emin mín. (lux) |
|---|---|---|
| P1 | 20,0 | 7,5 |
| P2 | 15,0 | 5,0 |
| P3 | 10,0 | 3,0 |
| P4 | 7,5 | 1,5 |
| P5 | 5,0 | 1,0 |
| P6 | 3,0 | 0,6 |

### 6.2 Lógica de Ajuste de Potência

O ajuste de potência garante que, mesmo quando o modelo de iluminância prevê valor abaixo do requisito, a potência indicada seja incrementada proporcionalmente — assumindo relação linear entre potência e iluminância produzida, que é fisicamente válida dentro da faixa de operação nominal das luminárias LED:

```
Para classes C e P:
  Se emed_previsto < emed_requisito:
    W_final = W_previsto × (emed_requisito / emed_previsto)
  Senão:
    W_final = W_previsto

Para classes M:
  W_final = W_previsto × (lmed_requisito_classe / lmed_requisito_M3)
  # M3 como baseline: hierarquia M1 > M2 > ... > M6 garantida
```

---

## 7. Módulos Auxiliares de Inteligência

### 7.1 Correção de Ponto Escuro (CPE)

**Definição:** Ponto escuro é um trecho de via sem cobertura fotométrica adequada, tipicamente resultante de espaçamento excessivo entre postes.

**Critério de acionamento:**
1. Distância entre postes ≥ 45 m **E** (potência prevista > 1,40 × média histórica da classe **OU** classe sem histórico disponível); **OU**
2. Classificador ML de CPE prediz risco com base no perfil geométrico.

**Ação proposta:** inserção de ponto intermediário, reduzindo o vão para metade. O sistema recalcula as métricas fotométricas para o novo vão e apresenta o comparativo antes/depois ao usuário.

### 7.2 Sugestão de Braço Novo

**Objetivo:** identificar instalações onde o tipo de braço instalado diverge do padrão geométrico historicamente associado àquela configuração de via.

**Funcionamento:** o classificador `clf_braco` recebe o perfil geométrico da instalação (sem o braço atual como input, para evitar viés) e prediz o braço mais provável segundo o padrão histórico das simulações. A sugestão é emitida apenas quando:
- A classe predita difere do braço atual identificado pela projeção instalada; **E**
- A probabilidade da classe predita supera **50%** (limiar de confiança mínima).

**Impacto prático:** nos 312 pontos do inventário analisado (todos classe C), a sugestão de troca foi emitida em **21 instalações (6,7%)**, todas com confiança ≥ 50%, predominantemente indicando transição de braços médios para Longo II em geometrias de via mais larga.

### 7.3 Assistente de Conformidade

Quando o modelo prevê que a configuração instalada não atende a norma (para as métricas confiáveis), o sistema testa automaticamente, em ordem crescente de impacto estrutural:

1. Aumento de altura da luminária (+1m, +2m);
2. Substituição por braço de maior projeção (ordem crescente de tamanho);
3. Redução da distância entre postes (−5m, −10m);
4. Instalação de poste intermediário (CPE).

A primeira configuração que atinge conformidade é apresentada como recomendação ao projetista.

---

## 8. Limitações Reconhecidas e Controles Adotados

### 8.1 Representatividade de Classes M e P

O dataset histórico contém predominantemente registros de vias classe C. Vias M e P têm poucos ou nenhum registro rotulado, resultando em modelos com menor poder preditivo para essas categorias. **Controle adotado:** para vias M, o sistema utiliza escalonamento proporcional à hierarquia NBR em vez de verificação direta de lmed (cujo R²=0,28 inviabiliza badges de conformidade confiáveis).

### 8.2 Ausência de Dados de Curva Fotométrica (IES/LDT)

A uniformidade (uo, ul) é função do diagrama polar da luminária — informação que não consta no inventário de entradas e é impossível de inferir apenas pela geometria. Os modelos uo e ul têm R² negativos e são exibidos como estimativas sem badge de conformidade. **Controle adotado:** esses valores são sinalizados como "Estimado" na interface, e o projetista é alertado de que a validação de uniformidade exige projeto fotométrico definitivo.

### 8.3 Variabilidade do Braço por Fornecedor/Produto

A projeção real do braço pode variar por produto e fixação. O sistema classifica o braço em 6 categorias discretas (Curto I a Longo II) baseando-se na projeção informada no inventário, não no código do produto. Divergências de até ±0,3 m são absorvidas pelo mapeamento para a categoria mais próxima.

### 8.4 Extrapolação Fora do Domínio de Treino

Modelos de aprendizado de máquina são confiáveis apenas no interior do espaço de features do conjunto de treino. Geometrias de via extremas (alturas > 14 m, distâncias > 60 m, vias > 20 m de largura) estão fora da densidade de dados históricos e podem produzir estimativas com erro elevado. **Controle adotado:** o sistema exibe os valores computados sem artifício de extrapolação — cabe ao engenheiro avaliar a plausibilidade física do resultado e acionar projeto fotométrico definitivo quando necessário.

### 8.5 Natureza Probabilística das Predições

Toda predição por aprendizado de máquina carrega incerteza intrínseca. O MAE do modelo de potência é de 5,09 W (dataset limpo), o que para luminárias de 30–150 W representa um erro percentual de 3% a 17%. Este erro é aceitável para pré-dimensionamento e triagem em larga escala, mas não para orçamentação unitária de alta precisão.

---

## 9. Justificativa do Uso Preditivo em Concessões

### 9.1 Comparativo com o Método Tradicional

| Critério | Simulação fotométrica completa | Sistema preditivo ML |
|---|---|---|
| Tempo por ponto | 15–60 min (engenheiro + software) | < 1 segundo |
| Capacidade de processamento | ~200 pontos/semana | 10.000+ pontos/hora |
| Dependência de IES/LDT | Obrigatória | Não necessária |
| Custo operacional | Alto (hora técnica especializada) | Negligível após implantação |
| Precisão fotométrica | Alta (resultado definitivo) | Moderada (pré-dimensionamento) |
| Conformidade regulatória | Definitiva | Triagem — requer validação |

### 9.2 Posição no Fluxo de Trabalho da Concessão

```
Inventário municipal
        │
        ▼
 [Sistema ML]           ← Este sistema
  Triagem em larga escala
  Pré-dimensionamento de W
  Identificação de CPE e braço
        │
        ▼
 Priorização de projetos
 (pontos críticos identificados)
        │
        ▼
 Projeto fotométrico definitivo  ← Software fotométrico (Dialux/Relux)
 (Dialux/Relux — amostra ≥ 10%)
        │
        ▼
 Execução e medição em campo
```

O sistema ML opera na camada de **triagem e planejamento**, jamais substituindo o projeto fotométrico definitivo exigido pela ABNT NBR 5101 e pelos contratos de concessão. Sua função é direcionar o esforço de engenharia para os pontos que realmente demandam atenção, em vez de distribuir uniformemente sobre o inventário inteiro.

### 9.3 Validade Estatística para Tomada de Decisão

O modelo de potência (R²=0,882, MAE=5,09 W) foi validado em conjunto de teste independente (20% dos dados, sem contato com o treino). Esta métrica indica que o modelo explica 88,2% da variância de potência no inventário histórico, o que é estatisticamente robusto para um problema com tanta diversidade de geometrias e fornecedores.

O modelo de iluminância média emed (R²=0,938, MAE=0,75 lux) é ainda mais preciso, refletindo a forte relação física entre a geometria do poste e a iluminância ao nível do pavimento.

Estes coeficientes são comparáveis ao desempenho de modelos ML em aplicações consolidadas de engenharia, como predição de consumo energético em edificações (R² típico: 0,80–0,95) e estimativa de cargas estruturais (R² típico: 0,75–0,92).

### 9.4 Rastreabilidade e Auditabilidade

Cada predição é rastreável ao conjunto de inputs que a gerou. O sistema registra, para cada ponto simulado:
- Geometria de entrada completa;
- Fornecedor considerado;
- Métricas previstas por modelo;
- Status de conformidade NBR por métrica;
- Flags de CPE e sugestão de braço (com confiança);
- Modelo sugerido do banco de dados de luminária e custo estimado.

Esta rastreabilidade é um requisito contratual típico em concessões de serviço público e é plenamente atendida pelo sistema.

---

## 10. Conclusão

O sistema preditivo de iluminação pública representa uma aplicação de engenharia de aprendizado de máquina diretamente alinhada às demandas operacionais de concessões de cidade inteligente. A combinação de modelos de regressão (Random Forest e Gradient Boosting) para estimativa de métricas fotométricas e potência, com classificadores para decisões auxiliares (CPE e tipo de braço), cria uma camada de inteligência de triagem que:

1. **Reduz drasticamente o tempo de análise** de inventários de larga escala, de semanas para minutos;
2. **Concentra o esforço de engenharia** nos pontos de maior risco de não-conformidade NBR 5101;
3. **Mantém rastreabilidade técnica completa** para fins de auditoria contratual;
4. **Opera com controles explícitos de confiabilidade**, rejeitando predições de métricas cujos modelos não atingem R² mínimo adequado;
5. **É extensível conforme o inventário histórico cresce**, pois o retreinamento com novos dados de simulação melhora continuamente o desempenho preditivo.

A metodologia adotada está aderente às boas práticas de ciência de dados aplicada à engenharia (validação em conjunto de teste independente, encapsulamento em pipeline scikit-learn, controle de limiar de confiabilidade) e ao arcabouço normativo vigente (ABNT NBR 5101:2024).

**O sistema não substitui o projeto fotométrico definitivo.** Ele é uma ferramenta de primeira triagem, projetada para multiplicar a capacidade analítica da equipe de engenharia sem comprometer a precisão das decisões finais de projeto.

---

## 11. Referências Normativas e Técnicas

- **ABNT NBR 5101:2024** — Iluminação Pública — Procedimento. Associação Brasileira de Normas Técnicas, Rio de Janeiro.
- **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.
- **Friedman, J. H. (2001).** Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.
- **Pedregosa, F. et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- **CIE 140:2019** — Road Lighting Calculations. Commission Internationale de l'Éclairage.
- **ABNT NBR ISO/CIE 8995-1:2013** — Iluminação de ambientes de trabalho.

---

*Este relatório foi produzido pela equipe técnica com base no sistema preditivo em operação. Os dados de desempenho dos modelos referem-se ao dataset histórico disponível até abril de 2026. Retreinamento periódico com novos dados de simulação é recomendado para manutenção da qualidade preditiva.*
