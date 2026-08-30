# `acesso` — controle de entrada e trilha de uso

Dois serviços que só fazem sentido juntos: barrar quem não é da equipe, e registrar o
que a equipe fez. Sem o primeiro, o segundo registraria "alguém" — que é exatamente a
informação inútil que existia antes deste pacote.

## O que ele protege, e o que não protege

Protege o **portal**: as cinco ferramentas, os dados que elas carregam em memória e os
entregáveis que elas geram.

**Não protege os arquivos do repositório.** Modelos `.pkl`, `dataset.csv`, planilhas de
simulação e os `.parquet` agregados são legíveis por qualquer pessoa com acesso ao
repositório, com ou sem senha aqui. Enquanto o repositório for público, a senha do
portal é uma fechadura na porta da frente de uma casa sem parede nos fundos.

---

## 1. Cadastrar o primeiro usuário

Sem nenhum usuário cadastrado o portal **não abre** — falha fechado, de propósito: um
`secrets` ausente no deploy não pode virar "portal público" silenciosamente.

```bash
cd app
py -m acesso.gerar_hash --login jferreira --nome "Juliano Ferreira" --perfil admin
```

A senha é digitada sem eco e nunca é gravada em lugar nenhum. O comando imprime um
bloco TOML pronto:

```toml
[auth]
expiracao_horas = 12

[auth.usuarios.jferreira]
nome = "Juliano Ferreira"
perfil = "admin"
senha_hash = "pbkdf2_sha256$600000$…$…"
```

Onde colar:

| Ambiente | Destino |
|---|---|
| Local | `app/.streamlit/secrets.toml` (está no `.gitignore`) |
| Streamlit Cloud | painel do app → **Settings → Secrets** |

Para adicionar mais pessoas, rode de novo e copie **apenas** o bloco
`[auth.usuarios.<login>]` — o `[auth]` fica uma vez só.

**Revogar um acesso:** apague o bloco daquela pessoa. Ninguém mais é afetado, e é
justamente isso que uma senha compartilhada não permitiria.

### Parâmetros de `[auth]`

| Chave | Padrão | Efeito |
|---|---|---|
| `expiracao_horas` | 12 | sessão inativa expira; `0` desliga |
| `max_tentativas` | 5 | falhas seguidas antes do bloqueio |
| `bloqueio_minutos` | 15 | duração do bloqueio |

O bloqueio vive no `session_state`, portanto é **por aba do navegador**. Ele encarece a
tentativa manual, mas não impede força bruta distribuída — para isso seria preciso
estado compartilhado entre sessões, que o Community Cloud não oferece. A defesa real
contra isso é o log: cinco falhas viram cinco linhas com hora e login tentado.

---

## 2. Ligar o histórico no Google Sheets

Sem este passo o log funciona, mas em CSV local — e **no Streamlit Cloud o disco é
efêmero**: o container é recriado a cada redeploy e o app hiberna por inatividade, de
modo que o CSV dura dias, não meses. Para rodar só na sua máquina, pode pular.

1. **Criar a planilha** no seu Drive. Pode ficar vazia; o cabeçalho é escrito sozinho.
   O `planilha_id` é o trecho da URL entre `/d/` e `/edit`.
2. **Criar uma service account** no [Google Cloud Console](https://console.cloud.google.com):
   *IAM & Admin → Service Accounts → Create*. Não precisa de papel nenhum no projeto.
3. **Habilitar a Google Sheets API** no mesmo projeto (*APIs & Services → Library*).
4. **Gerar uma chave JSON** para a service account (*Keys → Add key → JSON*).
5. **Compartilhar a planilha** com o `client_email` do JSON, como **Editor**. Este é o
   passo que se esquece: sem ele a API responde 403 e o log cai para CSV em silêncio.
6. **Colar no secrets** o conteúdo do JSON sob `[gcp_service_account]` e o id sob
   `[log_uso]` — ver `app/.streamlit/secrets.toml.example`.

Confira em qual backend está com `acesso.backend_ativo()`.

---

## 3. O que fica registrado

Uma linha por evento, com estas colunas:

| Coluna | Conteúdo |
|---|---|
| `timestamp_utc` | ISO-8601, sempre UTC |
| `evento` | `login`, `login_falha`, `logout`, `pagina`, ou a ação |
| `usuario`, `nome`, `perfil` | quem; em `login_falha`, o login tentado |
| `sessao_id` | correlaciona os eventos de uma mesma sessão |
| `segundos_sessao` | tempo desde o login **naquele evento** |
| `alvo` | objeto da ação: município, nº de pontos, página |
| `detalhe` | parâmetro relevante (código IBGE, tamanho do arquivo) |

Ações instrumentadas hoje: `cadastro_processado`, `inputs_gerados`,
`blocos_relatorio_gerados`, `municipio_consultado`, `carteira_triada`.

**Duração de sessão.** Não existe evento de saída confiável — o usuário fecha a aba e o
servidor não fica sabendo. Em vez de um heartbeat que poluiria o log, a duração é
derivada: é o maior `segundos_sessao` entre os eventos de um mesmo `sessao_id`. Como
consequência, uma sessão em que a pessoa só fez login e saiu aparece com duração zero,
não com duração desconhecida.

**Páginas são registradas na mudança, não a cada rerun.** O Streamlit reexecuta o
script inteiro a cada clique em qualquer widget; sem essa guarda, uma tarde de trabalho
viraria milhares de linhas e estouraria a quota da API do Sheets (60 req/min).

**O que não é registrado, por decisão:** conteúdo de planilha de cliente. A trilha
responde "quem, quando, o quê" — ela não é cópia do dado.

---

## 4. Limites conhecidos

- **Escrita assíncrona.** Eventos vão para uma fila em memória e uma thread grava em
  lote a cada 4 s. Se o container morrer com a fila cheia, os eventos pendentes se
  perdem. É log de uso, não livro-razão — a alternativa (escrita síncrona) somaria
  ~400 ms a cada clique.
- **Auditoria nunca derruba o app.** Toda falha de escrita é engolida e reportada só no
  stdout, que no Cloud vira log de aplicação. Um portal que cai porque a planilha de log
  ficou indisponível seria pior do que um portal sem log.
- **Um usuário logado pode baixar tudo o que o portal gera.** O controle é de entrada,
  não de uso: não há perfil que restrinja páginas. O campo `perfil` existe e é
  registrado, mas hoje é informativo.

---

## 5. Migrar para login corporativo

Se o portal um dia for exposto a cliente externo, o caminho é `st.login` (OIDC,
disponível desde o Streamlit 1.42; aqui roda 1.56). A mudança fica contida neste
pacote: `exigir_login()` passa a consultar `st.user` e a validar o e-mail contra uma
allowlist, e `Usuario` é construído a partir do token. Nada nas páginas muda — elas só
conhecem `registrar_acao`.

Não foi feito agora porque exige projeto no Google Cloud, credencial OAuth, URI de
redirect e conta Google para cada pessoa — custo que não se paga para uma equipe
pequena e conhecida.
