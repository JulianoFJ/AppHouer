"""
Relatório de execução em texto markdown (seção 12.4).

Produz uma mensagem que deve ser exibida ao usuário ao final do processamento,
contendo todas as métricas e avisos exigidos pelas instruções v1.4.
"""

from __future__ import annotations

import pandas as pd

from . import pipeline as _pipeline


def _fmt_qtd(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _soma_luminarias(df: pd.DataFrame) -> int:
    """
    Soma a quantidade de luminárias em uma base IAE/ID.
    Cada linha pode representar 1 ponto com múltiplas lâmpadas — a coluna
    `quantidade_considerada` (ou fallback `quantidade`) traz esse total.
    """
    if df is None or df.empty:
        return 0
    for col in ("quantidade_considerada", "quantidade", "Quantidade"):
        if col in df.columns:
            return int(round(pd.to_numeric(df[col], errors="coerce").fillna(1).sum()))
    return len(df)


def _bloco_classe_via(r) -> str:
    prop = r.propagacao_classe
    linhas = [
        f"- Pontos do cadastro com classe propagada: **{_fmt_qtd(prop.pontos_com_classe)}**",
        f"- Pontos do cadastro **sem classe** (rua sem inspeção ou inspeção não propagável): **{_fmt_qtd(prop.pontos_sem_classe)}**",
    ]
    if prop.quantidades_por_classe:
        distrib = ", ".join(f"{c}: {_fmt_qtd(n)}" for c, n in sorted(prop.quantidades_por_classe.items()))
        linhas.append(f"- Distribuição por classe: {distrib}")
    if prop.logradouros_divergentes:
        linhas.append(f"- Logradouros com **divergência de classe** entre inspeções: **{len(prop.logradouros_divergentes)}**")
        for div in prop.logradouros_divergentes[:10]:  # mostra até 10 para não inflar
            linhas.append(
                f"    - `{div['rua']}` → observadas {div['classes_observadas']}, "
                f"escolhida **{div['escolhida']}** ({div['contagens']})"
            )
        if len(prop.logradouros_divergentes) > 10:
            linhas.append(f"    - … e mais {len(prop.logradouros_divergentes) - 10} logradouros.")
    if prop.aviso_sem_bairro:
        linhas.append("- ⚠️ Cadastro **sem coluna Bairro** — homônimos podem ter sido agrupados.")
    return "\n".join(linhas)


def _bloco_acerto(r) -> str:
    """% de acerto cadastro vs inspeção."""
    comp = r.comparacao
    insp_validas = comp[comp["divergencia"] != "Sem inspeção"]
    total = len(insp_validas)
    if total == 0:
        return "- Sem pontos inspecionados — % de acerto não calculável."

    tec_ok = (insp_validas["flag_tecnologia"] == "Igual").sum()
    pot_ok = (insp_validas["flag_potencia"] == "Igual").sum()
    ambos_ok = ((insp_validas["flag_tecnologia"] == "Igual") & (insp_validas["flag_potencia"] == "Igual")).sum()
    pct = lambda n: f"{(n / total * 100):.1f}%"
    return (
        f"- Tecnologia correta: **{pct(tec_ok)}** ({_fmt_qtd(int(tec_ok))} de {_fmt_qtd(total)})\n"
        f"- Potência correta: **{pct(pot_ok)}** ({_fmt_qtd(int(pot_ok))} de {_fmt_qtd(total)})\n"
        f"- Tecnologia e Potência corretas: **{pct(ambos_ok)}** ({_fmt_qtd(int(ambos_ok))} de {_fmt_qtd(total)})"
    )


def gerar(r: _pipeline.ResultadoPipeline) -> str:
    """Monta o relatório completo em markdown."""
    rot = r.roteamento.resumo
    tempo_str = r.tempo_operacao.formato_hhmm if r.tempo_operacao else "**não localizado**"
    fator = r.fator_extrapolacao

    # Distinção crítica: "linhas" da planilha vs "luminárias" reais. Uma base
    # IAE de 20 linhas pode representar 215 luminárias quando cada ponto tem
    # uma coluna `Quantidade Considerada` com valores > 1.
    iae_lamp_novos       = _soma_luminarias(r.roteamento.pontos_iae_novos)
    id_lamp_novos        = _soma_luminarias(r.roteamento.pontos_id_novos)
    iae_lamp_existentes  = _soma_luminarias(r.roteamento.pontos_iae_existentes)
    id_lamp_existentes   = _soma_luminarias(r.roteamento.pontos_id_existentes)
    iae_lamp_total       = _soma_luminarias(r.iae_normalizada)
    id_lamp_total        = _soma_luminarias(r.id_normalizada)

    total_corrigido_luminarias = r.total_cadastro + iae_lamp_novos + id_lamp_novos

    texto = f"""
# Relatório de Execução — Análise de Cadastro de IP

## Identificação
- **Município / UF:** {r.municipio} / {r.uf}
- **Tempo de operação aplicado (ANEEL 2590/2019):** {tempo_str}

## Volumes
- Total de pontos no **cadastro recebido**: **{_fmt_qtd(r.total_cadastro)}**
- Total de pontos **inspecionados** (amostra): **{_fmt_qtd(r.total_inspecao)}**
- Base IAE recebida: **{_fmt_qtd(r.total_iae)} linhas** ({_fmt_qtd(iae_lamp_total)} luminárias)
- Base ID recebida: **{_fmt_qtd(r.total_id)} linhas** ({_fmt_qtd(id_lamp_total)} luminárias)
- **Fator de extrapolação** = FLOOR({_fmt_qtd(fator.total_cadastro)} / {_fmt_qtd(fator.total_amostra)}) = **{fator.fator}**

## Roteamento dos pontos
- Tratamento **Convencional**: **{_fmt_qtd(rot.get('CONVENCIONAL', 0))}** pontos
- Tratamento **LED IV** (troca conv → LED): **{_fmt_qtd(rot.get('LED_IV', 0))}** pontos
- Tratamento **IAE**: **{_fmt_qtd(rot.get('IAE_EXISTENTE', 0) + rot.get('IAE_NOVO', 0))}** linhas ({_fmt_qtd(iae_lamp_existentes + iae_lamp_novos)} luminárias)
    - Existentes no cadastro (realocados): {_fmt_qtd(rot.get('IAE_EXISTENTE', 0))} linhas ({_fmt_qtd(iae_lamp_existentes)} luminárias)
    - Novos (adicionados ao Cadastro Corrigido): {_fmt_qtd(rot.get('IAE_NOVO', 0))} linhas ({_fmt_qtd(iae_lamp_novos)} luminárias)
- Tratamento **ID**: **{_fmt_qtd(rot.get('ID_EXISTENTE', 0) + rot.get('ID_NOVO', 0))}** linhas ({_fmt_qtd(id_lamp_existentes + id_lamp_novos)} luminárias)
    - Existentes no cadastro (realocados): {_fmt_qtd(rot.get('ID_EXISTENTE', 0))} linhas ({_fmt_qtd(id_lamp_existentes)} luminárias)
    - Novos (adicionados ao Cadastro Corrigido): {_fmt_qtd(rot.get('ID_NOVO', 0))} linhas ({_fmt_qtd(id_lamp_novos)} luminárias)
- LED já corretamente cadastrado (sem tratamento de troca): {_fmt_qtd(rot.get('LED_OK', 0))}
- LED mantido (inspeção convencional, regra da seção 8): {_fmt_qtd(rot.get('LED_MANTIDO', 0))}
- IV não inspecionados (representados pela amostra via fator): {_fmt_qtd(rot.get('NAO_INSPECIONADO', 0))}

## Cadastro Corrigido
- **Total geral do Cadastro Corrigido** (Recebido + IAE novos + ID novos, em luminárias): **{_fmt_qtd(total_corrigido_luminarias)}**
    - Cadastro recebido: {_fmt_qtd(r.total_cadastro)} luminárias
    - IAE novos: +{_fmt_qtd(iae_lamp_novos)} luminárias ({_fmt_qtd(rot.get('IAE_NOVO', 0))} linhas)
    - ID novos: +{_fmt_qtd(id_lamp_novos)} luminárias ({_fmt_qtd(rot.get('ID_NOVO', 0))} linhas)

## Propagação de Classe Via
{_bloco_classe_via(r)}

## % de acerto cadastro vs inspeção
{_bloco_acerto(r)}

## Normalizações de tecnologia aplicadas
"""
    if r.normalizacoes_tecnologia:
        for codif, n in sorted(r.normalizacoes_tecnologia.items(), key=lambda x: -x[1]):
            texto += f"- `{codif}` — {_fmt_qtd(n)} ocorrências\n"
    else:
        texto += "- Nenhuma normalização necessária (códigos já no padrão).\n"

    texto += "\n## Códigos de tecnologia não reconhecidos\n"
    if r.codigos_desconhecidos:
        texto += (
            "Os códigos abaixo apareceram nos inputs mas não foram classificados. "
            "Em vez de mascarar essas linhas, o pipeline as registrou por fonte. "
            "Adicione-os a `cadastro_ip/tecnologia.py:VARIANTES_PARA_CODIGO` "
            "ou corrija na planilha original.\n\n"
        )
        for fonte, mapa in r.codigos_desconhecidos.items():
            total = sum(mapa.values())
            texto += f"- **{fonte}** ({_fmt_qtd(total)} pontos):\n"
            for codigo, qtd in sorted(mapa.items(), key=lambda x: -x[1]):
                texto += f"    - `{codigo}` — {_fmt_qtd(qtd)} pontos\n"
    else:
        texto += "- Todos os códigos de tecnologia foram reconhecidos.\n"

    texto += "\n## Validação de balanceamento\n"
    if abs(r.desbalanceamento) <= 0.5:
        texto += "- ✅ Cadastro Corrigido bate com Recebido + IAE novos + ID novos.\n"
    else:
        texto += (
            f"- ❌ **Desbalanceamento de {r.desbalanceamento:+.0f} pontos** entre "
            "Cadastro Corrigido e o esperado (Recebido + IAE novos + ID novos). "
            "Provavelmente causado por códigos de tecnologia não reconhecidos.\n"
        )

    texto += "\n## Avisos para revisão\n"
    if r.avisos:
        for av in r.avisos:
            texto += f"- ⚠️ {av}\n"
    else:
        texto += "- Nenhum aviso.\n"

    if r.mapeamento_cadastro and r.mapeamento_cadastro.ambiguos:
        texto += "\n## Colunas com múltiplas candidatas no cadastro (revise)\n"
        for conceito, opcoes in r.mapeamento_cadastro.ambiguos.items():
            texto += f"- `{conceito}`: candidatas {opcoes} — escolhida `{r.mapeamento_cadastro.mapeados.get(conceito)}`\n"

    return texto.strip()


__all__ = ["gerar"]
