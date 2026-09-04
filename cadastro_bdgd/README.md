# `cadastro_bdgd` — cadastro ponto a ponto de IP a partir da BDGD

Gera o cadastro de iluminação pública de um município a partir da BDGD da ANEEL, para
que a **Amostragem para Inspeção** funcione em município que não tem cadastro próprio.

O Hub de Municípios já lê a BDGD, mas o **agregado**: totais por município, que
respondem "quantos pontos e quanta carga". Este pacote responde outra coisa — *quais*
são os pontos, um a um, com coordenada. São artefatos diferentes e não se substituem.

## Cadeia

```
.gdb da distribuidora  --extracao-->  dados/bdgd/cadastros/<ibge>.parquet
                       --vias_osm-->  + logradouro, hierarquia, classe NBR
                       --montagem-->  DataFrame para a amostragem + .xlsx para baixar
```

| Módulo | Papel |
|---|---|
| `extracao` | ETL offline (exige GDAL): PIP + PONNOT → parquet por município |
| `vias_osm` | logradouro e classe viária pelo OpenStreetMap, que a BDGD não tem |
| `classe_nbr` | classe de iluminação M pela Tabela 1 da ABNT NBR 5101:2024 |
| `montagem` | junta tudo e produz a planilha com aba de procedência |
| `caminhos` | onde os artefatos ficam em disco |

## Comandos

```bash
# a partir de app/ — exige ogr2ogr (OSGeo4W ou QGIS) e o .gdb da distribuidora
py -m cadastro_bdgd.extracao 3162955          # São José da Lapa/MG
py -m cadastro_bdgd.extracao 3162955 --forcar # refaz
py -m cadastro_bdgd.extracao --listar         # o que já está extraído
```

## As três decisões que sustentam o pacote

### A coordenada vem do PONNOT, não da PIP

A entidade PIP **não tem geometria** (`Geometry: None` no `ogrinfo`) nem campo de
endereço. O que ela tem é `PN_CON`, o ponto notável de conexão, que é chave para a
entidade PONNOT — essa sim uma camada de pontos em SIRGAS 2000.

O join `PIP.PN_CON → PONNOT.COD_ID` casou **100% dos 10.590 pontos** de Ponta Porã/MS
(04/09/2026), sem duplicidade de `COD_ID`. Cuidado com `PAC`: parece a chave natural e
casou **0%** — é o ponto de atendimento da UC, com codificação própria.

O que se ganha é a posição do **poste da rede**, não a do luminário. Para dispersão
espacial da amostra e para o mapa isso basta.

### Logradouro e classe vêm do OSM, e saem marcados como inferidos

A BDGD não tem nome de rua nem classe viária em lugar nenhum, e são as duas colunas de
que o sorteio depende para garantir cobertura de avenida e rodovia. Cada ponto é casado
com a via mais próxima do OpenStreetMap.

Medido em Ponta Porã: 4,8 s de download, 0,4 s de casamento, **79%** dos pontos com
nome de via, **100%** com hierarquia funcional, distância mediana ao eixo de **6,3 m**
(p90 = 13,8 m; 97,6% dentro de 20 m).

É atribuição por proximidade, não cadastro: serve para dizer "este ponto pertence à
Avenida Brasil", não para dizer o número. A coluna `dist_via_m` sai junto justamente
para permitir descartar o que ficou longe.

A distância é medida contra vértices **densificados a cada 10 m**, não contra os nós
originais da via. Sem isso a mediana era 24,7 m — o erro não era do casamento, era de
medir distância a vértice esparso em vez de ao segmento.

### A classe replica o método da norma, não um "de-para"

A NBR 5101:2024 determina a classe M por soma de ponderações (Tabela 1) e a fórmula
`classe = 6 − V_PS`, não por tipo de via. Um de-para (`primary` → M2) seria opinião; a
soma de parcelas é o procedimento da norma aplicado com os parâmetros disponíveis, e
cada parcela fica registrada com a origem.

Dos sete parâmetros, o OSM cobre três com dado real (velocidade, separação de faixas,
densidade de interseções), um por proxy (luminância ambiente, pelo `ARE_LOC` da BDGD) e
um raramente (veículos estacionados). **Sinalização** exige vistoria e entra no neutro.

**Volume de tráfego é inferido da hierarquia viária, e essa foi uma correção.** Na
primeira versão ele ficou no neutro, como manda a prudência — e o resultado foi inútil:
avenida, coletora e rua local urbanas caíam **todas em M4**, porque em via urbana a
velocidade quase nunca passa de 60 km/h e o que sobra não separa uma via da outra.
Classe que não distingue avenida de rua local não estratifica amostra nenhuma. A
hierarquia funcional existe para refletir volume, então é o proxy defensável — e é o
primeiro parâmetro a substituir quando houver contagem de tráfego real.

Com isso a estimativa produz M3 (avenida), M4 (coletora), M5 (local urbana) e M6 (local
rural), que é a discriminação de que o sorteio precisa.

**A classe daqui é estimativa para dimensionar inspeção. Não é enquadramento
normativo** — esse exige contagem de tráfego e vistoria.

## Integração com a Amostragem

A página ganhou um seletor de origem no Passo 1. Os dois caminhos — planilha enviada e
cadastro da BDGD — terminam escrevendo em `SS["cadastro"]`, e daí para a frente os
passos 2 a 6 **não sabem de onde o cadastro veio**. Nada em `amostragem_ip`,
`cadastro_ip` ou `hub_municipios` foi alterado.

O upload continua sendo a opção padrão, de propósito: quando o município tem cadastro
próprio, ele é melhor que o derivado — tem logradouro e classe de origem, não inferidos.

As colunas de `COLUNAS_SAIDA` usam os nomes canônicos que `cadastro_ip.normalizacao`
detecta sozinho, então a origem BDGD não pede mapeamento manual. Há teste travando esse
contrato.

## Limites que precisam ser ditos ao cliente

- **Cobertura é a da distribuidora, não a do município.** A BDGD tem o que é faturado
  como IP. Ponto não faturado, em litígio ou faturado noutro município não aparece.
- **Não roda no Streamlit Cloud.** A extração exige o `.gdb` (dezenas de GB) e GDAL. A
  página lista apenas os municípios já extraídos na máquina; o parquet de um município
  é pequeno o bastante para ser versionado por decisão explícita, se for preciso.
- **`CONTROLE` (telegestão) tem preenchimento irregular** entre distribuidoras — trate
  como indício, confirme em campo.
- **A tecnologia é inferida** da assinatura física do `TIPO_LAMP` (perda de reator e
  série de potências), porque a BDGD não publica o domínio do campo. A inferência é a do
  Hub, reaproveitada.

## Custo

| Etapa | Tempo |
|---|---|
| Extração, base local (Energisa MS, 4,4 GB) | ~45 s por município |
| Extração, base no Drive (Cemig-D, 15,9 GB) | dezenas de minutos — o Drive File Stream baixa sob demanda |
| Download da malha OSM | ~5 s por município, com cache em disco |
| Casamento de 10 mil pontos | < 1 s |
