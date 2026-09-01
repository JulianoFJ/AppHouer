# Amostragem para Inspeção de Campo

Recebe o cadastro de iluminação pública de um município e devolve **duas planilhas de
campo disjuntas** — 60% medição estrutural, 40% medição de qualidade — dimensionadas
pela ABNT NBR 5426 e sorteadas de forma aleatória, porém com abrangência garantida.

Página: `paginas/amostragem_campo.py` (só UI). Lógica: este pacote.

```
leitura → preparar_base → nbr5426.plano → vias.identificar_vias_principais →
amostrador.sortear → saidas/planilha_amostra + relatorio
```

## Por que não sortear aleatoriamente e pronto

Parque de IP brasileiro é dominado por via local — tipicamente 70 a 85% dos pontos.
Sorteio aleatório simples devolve quase só rua residencial: a equipe volta de campo
sem nenhuma medição em avenida, rodovia ou área de conflito, que é exatamente onde a
NBR 5101 é mais exigente e onde a não conformidade tem consequência contratual. No
extremo oposto, amostra escolhida a dedo não sustenta inferência sobre o parque nem
resiste a questionamento de banca.

O sorteio é, então, **estratificado com cotas de cobertura e dispersão espacial**, em
três camadas:

1. **Cotas obrigatórias** — ≥1 ponto de cada classe de iluminação e ≥1 ponto em cada
   via estruturante, **em cada uma das duas planilhas**. Ao sortear a cota de uma
   classe, prefere-se um ponto que também cubra uma via principal ainda descoberta,
   para não gastar amostra à toa.
2. **Alocação proporcional** — o restante distribuído entre as classes na proporção do
   parque (método do maior resto).
3. **Dispersão espacial** — dentro de cada estrato, k-means sobre as coordenadas com
   k = número de pontos a sortear e um ponto aleatório por cluster. Espalha a amostra
   pela mancha urbana sem tirar a aleatoriedade: quem sai de cada cluster continua
   sendo decisão do gerador.

Efeito medido no cadastro de Matozinhos (5.012 pontos, amostra de 200): cobertura de
**95% da malha 12×12** contra **76%** de um sorteio aleatório simples do mesmo tamanho.
Em 20 sementes, a dispersão venceu em 20.

## Decisões travadas (01/09/2026)

- **Amostras disjuntas.** Nenhum ponto aparece nas duas planilhas — não se conta duas
  vezes a mesma equipe no mesmo poste. `n_estrutural + n_qualidade = n`.
- **Cobertura independente por planilha.** A frente de qualidade precisa de uma via de
  cada classe para confrontar com a NBR 5101; a estrutural precisa representar o parque.
- **Vias principais: ≥1 ponto em cada uma**, com teto configurável (padrão 20) para que
  um município com 300 avenidas não consuma a amostra só com cotas. Cada via obrigatória
  custa 2 pontos (um por planilha) — o custo aparece na UI antes do sorteio.
- **Tamanho pré-preenchido pela norma, mas editável.** É prática levar folga sobre o
  mínimo normativo para absorver perda de campo (ponto inexistente, inacessível,
  coordenada errada). Abaixo do mínimo a UI avisa que o critério Ac/Re perde lastro.

## Armadilhas tratadas, todas com teste

- **Chave de via ≠ chave de logradouro.** `cadastro_ip.normalizacao.chave_logradouro`
  inclui o bairro, e com razão: "Rua A" do Centro e "Rua A" do distrito são ruas
  diferentes. Mas uma rodovia atravessa vários bairros e continua sendo **uma** via —
  com a chave do cadastro, a Rodovia Idsel Costa Martins de Matozinhos virava 8 vias
  principais e consumia 16 cotas da amostra. Daí a coluna `_chave_via`, sem bairro,
  usada **só** para vias principais.
- **Três vocabulários de classe misturados no mercado**: NBR 5101 (V1–V5, P1–P4),
  EN 13201/CIE (M1–M6, C0–C5, P1–P6) e o texto da hierarquia municipal (arterial,
  coletora, local). `vias.normalizar_classe` converte os três e **preserva** o que não
  reconhece como estrato próprio — para amostragem o que importa é cobrir todo rótulo
  presente, não julgar a nomenclatura.
- **Coordenada implausível.** Cadastro municipal traz lat/long trocados, zerados ou em
  UTM. Fora da faixa do território brasileiro a coordenada é tratada como ausente, com
  ressalva — o ponto continua elegível ao sorteio, apenas não puxa o k-means para o
  oceano.
- **Setas da Tabela 2 da NBR 5426.** Célula sem plano manda usar o primeiro plano acima
  ou abaixo, **e isso muda o tamanho da amostra** porque muda a letra-código. Lote
  pequeno com NQA rigoroso frequentemente cai nesse caso e acaba em inspeção 100%.
- **Regime atenuado não pode manter o Ac do normal.** A amostra encolhe (315 → 125);
  manter o mesmo número de aceitação tornaria o critério mais frouxo que a norma. Usa-se
  o Ac do plano normal de mesmo tamanho de amostra.

## A amostra é deliberadamente não auto-ponderada

As cotas sobre-representam as classes exigentes: em Matozinhos, V2 é 7,2% do parque e
24% da amostra. É o objetivo — mas significa que **extrapolar a inspeção para o parque
pela média simples da amostra produz viés**. A extrapolação correta é a média ponderada
por estrato, com `w_h = N_h / n_h`, entregue na coluna *Peso p/ extrapolação* da tabela
de cobertura, na aba do plano de cada planilha e no relatório:

```
total_parque = Σ_h (média do estrato h × N_h)
```

Vale para taxa de divergência cadastral, potência média, tecnologia e qualquer
indicador que suba para o EVTE. O teste `test_peso_de_extrapolacao_reconstroi_o_parque`
trava a identidade `Σ (n_h × w_h) = N`.

## Reprodutibilidade

Todo o sorteio depende de uma **semente** exposta na UI e registrada na planilha e no
relatório. Mesmo cadastro + mesma semente = mesma amostra. É o que permite ao poder
concedente refazer o sorteio e verificar que a amostra não foi escolhida a dedo.

## Saídas

Cada planilha (`saidas/planilha_amostra.gerar(resultado, grupo)`) tem três abas:

| Aba | Conteúdo |
|---|---|
| **Amostra de Campo** | Formulário: identificação/localização do ponto + colunas em branco para preencher em campo, com validação por lista onde o domínio é fechado. Ordenada por bairro e logradouro, que é como a equipe percorre o município. |
| **Cadastro (referência)** | As mesmas linhas com todas as colunas originais do cadastro, para conferir o que estava declarado. |
| **Plano de Amostragem** | Memória de cálculo: parque, plano NBR 5426 (letra, n, Ac/Re), semente, cobertura por classe com peso de extrapolação, vias principais, abrangência e ressalvas. |

`relatorio.gerar(resultado)` produz o mesmo conteúdo em Markdown, para anexar ao
relatório de engenharia.

## Sobre a norma

A NBR 5426:1985 foi cancelada pela ABNT em 2018 sem substituta nacional — a referência
internacional equivalente é a ISO 2859-1. Segue sendo a norma citada nos termos de
referência de concessões e PPPs de IP, e é nesse uso que o módulo a aplica. A ressalva
está no rodapé do relatório gerado, para não passar como norma vigente.
