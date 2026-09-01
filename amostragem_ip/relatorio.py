"""
Relatório de execução do sorteio, em Markdown.

Serve a dois leitores diferentes com o mesmo texto: a equipe de campo, que precisa
saber quantos pontos vai percorrer e onde, e o poder concedente / a banca, que
precisa poder refazer o sorteio e verificar que a amostra não foi escolhida a dedo.
Daí registrar sempre a semente, o plano da norma e as ressalvas — inclusive as
desfavoráveis.
"""

from __future__ import annotations

from datetime import datetime

from .amostrador import ResultadoAmostragem


def _numero(valor) -> str:
    return f"{valor:,}".replace(",", ".")


def gerar(resultado: ResultadoAmostragem) -> str:
    """Monta o relatório de execução do sorteio em Markdown."""
    plano = resultado.plano
    config = resultado.config
    abrangencia = resultado.abrangencia
    municipio = resultado.municipio or "Município não informado"
    uf = f"/{resultado.uf}" if resultado.uf else ""

    linhas: list[str] = []
    add = linhas.append

    add(f"# Plano de Amostragem para Inspeção de Campo — {municipio}{uf}")
    add("")
    add(f"*Gerado em {datetime.now():%d/%m/%Y %H:%M}*")
    add("")
    add("## 1. Dimensionamento da amostra")
    add("")
    add(f"- **Parque cadastrado (lote):** {_numero(resultado.total_parque)} pontos")
    if plano is not None:
        add(f"- **Norma aplicada:** ABNT NBR 5426:1985 — amostragem simples, "
            f"nível de inspeção **{plano.nivel}**, NQA **{plano.nqa}%**, regime **{plano.regime}**")
        add(f"- **Letra-código (Tabela 1):** {plano.letra_codigo}"
            + (f" (originalmente {plano.letra_original}, ajustada pela seta da Tabela 2)"
               if plano.seta_aplicada else ""))
        add(f"- **Amostra exigida pela norma:** {_numero(plano.tamanho_amostra)} pontos")
        add(f"- **Critério de aceitação:** Ac = {plano.numero_aceitacao} / "
            f"Re = {plano.numero_rejeicao} — até {plano.numero_aceitacao} pontos não conformes "
            "na amostra o lote é aceito; a partir de "
            f"{plano.numero_rejeicao} o cadastro é rejeitado e demanda retrabalho.")
    add(f"- **Amostra efetivamente sorteada:** {_numero(resultado.total_amostra)} pontos "
        f"({resultado.total_amostra / max(resultado.total_parque, 1):.2%} do parque)")
    if plano is not None and resultado.total_amostra > plano.tamanho_amostra:
        folga = resultado.total_amostra - plano.tamanho_amostra
        add(f"- **Folga sobre a norma:** +{_numero(folga)} pontos "
            f"({folga / plano.tamanho_amostra:.1%}), margem de segurança para perdas de campo "
            "(ponto inexistente, inacessível ou com coordenada errada).")
    add(f"- **Medição estrutural:** {_numero(len(resultado.estrutural))} pontos "
        f"({config.proporcao_estrutural:.0%})")
    add(f"- **Medição de qualidade:** {_numero(len(resultado.qualidade))} pontos "
        f"({config.proporcao_qualidade:.0%})")
    add("- As duas amostras são **disjuntas**: nenhum ponto aparece nas duas planilhas.")
    add("")

    add("## 2. Método do sorteio")
    add("")
    add("Amostragem aleatória **estratificada com cotas de cobertura e dispersão espacial**, "
        "em três camadas:")
    add("")
    add("1. **Cotas obrigatórias** — pelo menos um ponto de cada classe de iluminação e um "
        "ponto em cada via estruturante (avenida, rodovia, estrada, anel viário, marginal ou "
        "via de classe mais exigente), em **cada uma** das duas planilhas.")
    add("2. **Alocação proporcional** — o restante distribuído entre as classes na proporção "
        "do parque, preservando a estrutura viária do município.")
    add("3. **Dispersão espacial** — dentro de cada estrato o sorteio é balanceado por "
        "agrupamento geográfico (k-means sobre as coordenadas, um ponto aleatório por "
        "agrupamento), o que impede a amostra de se concentrar na área de maior densidade.")
    add("")
    add(f"**Semente do gerador aleatório: `{config.semente}`.** O mesmo cadastro com a mesma "
        "semente reproduz exatamente esta amostra — é o que permite auditar o sorteio.")
    add("")

    if not resultado.cobertura_classes.empty:
        add("## 3. Cobertura por classe de iluminação")
        add("")
        add("| Classe | Parque | % do parque | % da amostra | Estrutural | Qualidade | "
            "Nas duas | Peso w_h |")
        add("|---|---:|---:|---:|---:|---:|:--:|---:|")
        total_amostra = max(resultado.total_amostra, 1)
        for _, linha in resultado.cobertura_classes.iterrows():
            peso = linha["Peso p/ extrapolação"]
            peso_txt = f"{peso:.1f}" if peso else "—"
            add(
                f"| {linha['Classe']} | {_numero(linha['Pontos no parque'])} "
                f"| {linha['% do parque']:.1%} "
                f"| {linha['Amostra total'] / total_amostra:.1%} "
                f"| {linha['Estrutural']} | {linha['Qualidade']} "
                f"| {linha['Coberta nas duas']} | {peso_txt} |"
            )
        add("")
        add("> **A amostra é deliberadamente não auto-ponderada.** As cotas de cobertura "
            "sobre-representam as classes exigentes e as vias estruturantes — que é o "
            "objetivo, porque é onde a NBR 5101 aperta e onde o risco contratual mora. "
            "A consequência é que **extrapolar o resultado da inspeção para o parque pela "
            "média simples da amostra produz viés**. A extrapolação correta é a média "
            "ponderada por estrato, com o peso `w_h = N_h / n_h` da última coluna: "
            "`total_parque = Σ_h (média do estrato h × N_h)`. Vale para taxa de divergência "
            "cadastral, potência média, tecnologia e qualquer indicador levado ao EVTE.")
        add("")

    if not resultado.cobertura_vias.empty:
        add("## 4. Vias principais contempladas")
        add("")
        add("| Via | Tipo | Classes | Parque | Estrutural | Qualidade |")
        add("|---|---|---|---:|---:|---:|")
        for _, linha in resultado.cobertura_vias.iterrows():
            add(f"| {linha['Via']} | {linha['Tipo']} | {linha['Classes']} | "
                f"{_numero(linha['Pontos no parque'])} | {linha['Estrutural']} | "
                f"{linha['Qualidade']} |")
        add("")
        add(f"*Critério de seleção: tipo de via estruturante ou classe de iluminação mais "
            f"exigente, limitado às {config.teto_vias_principais} vias mais relevantes por "
            "número de pontos.*")
        add("")

    add("## 5. Abrangência geográfica")
    add("")
    add(f"- **Bairros:** {abrangencia.get('bairros_amostra', 0)} de "
        f"{abrangencia.get('bairros_parque', 0)} do cadastro")
    add(f"- **Logradouros:** {_numero(abrangencia.get('logradouros_amostra', 0))} de "
        f"{_numero(abrangencia.get('logradouros_parque', 0))} do cadastro")
    if abrangencia.get("cobertura_grid") is not None:
        add(f"- **Malha 12×12 sobre a mancha do cadastro:** "
            f"{abrangencia['celulas_cobertas']} de {abrangencia['celulas_com_parque']} células "
            f"com pontos de IP receberam amostra (**{abrangencia['cobertura_grid']:.1%}**)")
        add(f"- **Distância de um ponto qualquer do município ao ponto inspecionado mais "
            f"próximo:** mediana de {abrangencia['distancia_mediana_km']:.2f} km, "
            f"percentil 90 de {abrangencia['distancia_p90_km']:.2f} km")
    else:
        add("- Sem coordenadas válidas no cadastro — não foi possível medir a abrangência "
            "geográfica nem gerar o mapa de conferência.")
    add("")

    if resultado.ressalvas:
        add("## 6. Ressalvas")
        add("")
        for ressalva in resultado.ressalvas:
            add(f"- {ressalva}")
        add("")

    add("---")
    add("")
    add("*A NBR 5426:1985 foi cancelada pela ABNT em 2018 sem substituta nacional; a "
        "referência internacional equivalente é a ISO 2859-1. Ela segue sendo a norma citada "
        "nos termos de referência de concessões e PPPs de iluminação pública, e é nesse uso "
        "que este plano a aplica: dimensionar a verificação amostral do cadastro.*")
    return "\n".join(linhas)


__all__ = ["gerar"]
