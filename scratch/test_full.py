"""
test_full.py â€” Suite de testes do modelo Houer ML
==================================================
Testa todos os comportamentos crÃ­ticos:
  1. Carregamento de modelos
  2. Ajuste de PotÃªncia pela Hierarquia NBR (individual e lote)
  3. CPE â€” detecÃ§Ã£o por distÃ¢ncia e por desvio de potÃªncia
  4. CPE â€” prediÃ§Ãµes com distÃ¢ncia/2
  5. CoerÃªncia entre classes (M1 > M6, C0 > C5, P1 > P6 em potÃªncia)
  6. EquivalÃªncia individual vs lote para mesma configuraÃ§Ã£o
  7. Classes C e P retornam mÃ©tricas corretas
"""

import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
import pandas as pd
import numpy as np
import json

PASTA = os.path.join(os.path.dirname(__file__), '..')

# â”€â”€ Carrega artefatos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with open(os.path.join(PASTA, 'features_limpo.json'), encoding='utf-8') as f:
    meta = json.load(f)

num_ok = meta['features_numericas']
cat_ok = meta['features_categoricas']
feature_w_col = meta.get('feature_w_col', 'Potencia simulada - IP Principal (W)')
modelos_dependem_de_w = set(meta.get('modelos_dependem_de_w', []))

modelos = {}
for key in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']:
    path = os.path.join(PASTA, f'modelo_{key}_limpo.pkl')
    if os.path.exists(path):
        modelos[key] = joblib.load(path)
        try:
            # Evita falha de permissão do ambiente ao abrir pool paralelo no predict.
            if hasattr(modelos[key], 'named_steps') and 'model' in modelos[key].named_steps:
                est = modelos[key].named_steps['model']
                if hasattr(est, 'n_jobs'):
                    est.n_jobs = 1
        except Exception:
            pass

NBR5101 = {
    'M1': {'metricas': ['lmed','uo','ul','w'], 'lmed': 2.0,  'uo': 0.40, 'ul': 0.70},
    'M2': {'metricas': ['lmed','uo','ul','w'], 'lmed': 1.5,  'uo': 0.40, 'ul': 0.70},
    'M3': {'metricas': ['lmed','uo','ul','w'], 'lmed': 1.0,  'uo': 0.40, 'ul': 0.60},
    'M4': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.75, 'uo': 0.40, 'ul': 0.60},
    'M5': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.50, 'uo': 0.35, 'ul': 0.40},
    'M6': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.30, 'uo': 0.35, 'ul': 0.40},
    'C0': {'metricas': ['emed','uo','w'], 'emed': 50.0, 'uo': 0.40},
    'C1': {'metricas': ['emed','uo','w'], 'emed': 30.0, 'uo': 0.40},
    'C2': {'metricas': ['emed','uo','w'], 'emed': 20.0, 'uo': 0.40},
    'C3': {'metricas': ['emed','uo','w'], 'emed': 15.0, 'uo': 0.35},
    'C4': {'metricas': ['emed','uo','w'], 'emed': 10.0, 'uo': 0.35},
    'C5': {'metricas': ['emed','uo','w'], 'emed':  5.0, 'uo': 0.35},
    'P1': {'metricas': ['emed','emin','w'], 'emed': 20.0, 'emin': 7.5},
    'P2': {'metricas': ['emed','emin','w'], 'emed': 15.0, 'emin': 5.0},
    'P3': {'metricas': ['emed','emin','w'], 'emed': 10.0, 'emin': 3.0},
    'P4': {'metricas': ['emed','emin','w'], 'emed':  7.5, 'emin': 1.5},
    'P5': {'metricas': ['emed','emin','w'], 'emed':  5.0, 'emin': 1.0},
    'P6': {'metricas': ['emed','emin','w'], 'emed':  3.0, 'emin': 0.6},
}

# Config base padrÃ£o para todos os testes
BASE_CFG = {
    'Faixas de Rodagem':        2,
    'Largura Via 1':            7.0,
    'Largura Via 2':            0.0,
    'Largura Passeio 1':        2.0,
    'largura Passeio 2':        2.0,
    'largura Canteiro Central': 0.0,
    'altura da luminaria':      10.0,
    'projecao do braÃ§o':        1.5,
    'distancia entre postes':   35.0,
    'distancia Poste a via':    0.5,
    'Tipo de estrutura':        'BraÃ§o',
    'posteacao':                'Unilateral',
    'BraÃ§o Novo':               'Longo II',
    'Fornecedor':               'LEDSTAR',
}

PASSOU = []
FALHOU = []
sep = "â”€" * 70

def ok(nome):
    PASSOU.append(nome)
    print(f"  âœ… {nome}")

def fail(nome, detalhe=""):
    FALHOU.append(nome)
    print(f"  âŒ {nome}" + (f"\n     â†’ {detalhe}" if detalhe else ""))

def prever(cfg_dict, dist_override=None):
    """Roda predição e retorna dict {metrica: valor}."""
    dados = {**BASE_CFG, **cfg_dict}
    if dist_override is not None:
        dados['distancia entre postes'] = dist_override
    colunas = list(dict.fromkeys(num_ok + cat_ok + [feature_w_col]))
    X = pd.DataFrame([{k: dados.get(k, np.nan) for k in colunas}])

    resultados = {}
    if 'w' in modelos:
        resultados['w'] = max(modelos['w'].predict(X)[0], 0)

    for m in ['lmed', 'uo', 'ul', 'emed', 'emin']:
        if m not in modelos:
            continue
        if m in modelos_dependem_de_w:
            X_m = X.copy()
            X_m[feature_w_col] = resultados.get('w', np.nan)
            resultados[m] = max(modelos[m].predict(X_m)[0], 0)
        else:
            resultados[m] = max(modelos[m].predict(X)[0], 0)

    return resultados

REQ_M3_BASE = NBR5101['M3']['lmed']  # 1.0 cd/mÂ² â€” baseline para classes M

def ajustar_nbr(resultados, subclasse):
    """Replica a lÃ³gica de ajuste NBR do app.
    M: fator proporcional direto (lmed RÂ²=0.28 nÃ£o Ã© confiÃ¡vel).
    C/P: escalonamento condicional por emed (RÂ²>0.8).
    """
    info = NBR5101.get(subclasse, {})
    resultados = dict(resultados)
    if 'w' not in resultados:
        return resultados

    if subclasse.startswith('M'):
        req_classe = info.get('lmed', REQ_M3_BASE)
        fator = req_classe / REQ_M3_BASE
        resultados['w'] = resultados['w'] * fator
    else:
        req = info.get('emed')
        pred_ilum = resultados.get('emed', 0)
        if req is not None and pred_ilum > 0 and pred_ilum < req:
            resultados['w'] = resultados['w'] * (req / pred_ilum)
    return resultados


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 1 â€” Carregamento de Modelos e Features")
print(sep)
# ==============================================================================

modelos_esperados = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
for key in modelos_esperados:
    if key in modelos:
        ok(f"modelo_{key}_limpo.pkl carregado")
    else:
        fail(f"modelo_{key}_limpo.pkl", "arquivo nÃ£o encontrado")

for feat in num_ok:
    ok(f"feature numÃ©rica: {feat}")
for feat in cat_ok:
    ok(f"feature categÃ³rica: {feat}")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 2 â€” Ajuste NBR: PotÃªncia deve variar com a Classe")
print(sep)
# ==============================================================================

# M1 mais exigente que M6 â€” mesma geometria, M1 deve ter mais potÃªncia apÃ³s ajuste
pots_m = {}
for classe in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
    r = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe})
    r = ajustar_nbr(r, classe)
    pots_m[classe] = r['w']
    print(f"  {classe}: W = {r['w']:.1f}")

m1_maior_m6 = pots_m['M1'] > pots_m['M6']
if m1_maior_m6:
    ok("M1 tem maior potÃªncia que M6 apÃ³s ajuste NBR")
else:
    fail("Hierarquia M1 > M6 nÃ£o respeitada", f"M1={pots_m['M1']:.1f} vs M6={pots_m['M6']:.1f}")

# Verifica ordem monotÃ´nica (tolerÃ¢ncia: cada classe deve ser >= prÃ³xima)
hierarquia_ok = True
for i, (a, b) in enumerate(zip(['M1','M2','M3','M4','M5'], ['M2','M3','M4','M5','M6'])):
    if pots_m[a] < pots_m[b] * 0.95:  # 5% de tolerÃ¢ncia
        hierarquia_ok = False
        print(f"     [{a}={pots_m[a]:.1f}] < [{b}={pots_m[b]:.1f}] â€” inversÃ£o detectada")
if hierarquia_ok:
    ok("Hierarquia M1â‰¥M2â‰¥...â‰¥M6 respeitada (potÃªncia)")
else:
    fail("Hierarquia M1â†’M6 nÃ£o Ã© monotÃ´nica")

# Idem para C
pots_c = {}
for classe in ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']:
    r = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe})
    r = ajustar_nbr(r, classe)
    pots_c[classe] = r['w']
    print(f"  {classe}: W = {r['w']:.1f}")

c0_maior_c5 = pots_c['C0'] > pots_c['C5']
if c0_maior_c5:
    ok("C0 tem maior potÃªncia que C5 apÃ³s ajuste NBR")
else:
    fail("Hierarquia C0 > C5 nÃ£o respeitada", f"C0={pots_c['C0']:.1f} vs C5={pots_c['C5']:.1f}")

# Idem para P
pots_p = {}
for classe in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    r = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe})
    r = ajustar_nbr(r, classe)
    pots_p[classe] = r['w']
    print(f"  {classe}: W = {r['w']:.1f}")

p1_maior_p6 = pots_p['P1'] > pots_p['P6']
if p1_maior_p6:
    ok("P1 tem maior potÃªncia que P6 apÃ³s ajuste NBR")
else:
    fail("Hierarquia P1 > P6 nÃ£o respeitada", f"P1={pots_p['P1']:.1f} vs P6={pots_p['P6']:.1f}")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 3 â€” CPE: DistÃ¢ncia (â‰¥ 45m) + desvio histÃ³rico")
print(sep)
# ==============================================================================

# Regra atual do app:
# aciona se dist >= 45m e houver desvio de potÃªncia histÃ³rica relevante
# (ou se nÃ£o houver histÃ³rico disponÃ­vel para a classe)
DIST_CPE_MIN = 45.0
POT_DESVIO_FATOR = 1.40

def aciona_cpe(dist, pot_prev, media_hist):
    cpe_por_dist = dist >= DIST_CPE_MIN
    cpe_por_desvio = (
        media_hist is not None and pot_prev is not None
        and pot_prev > media_hist * POT_DESVIO_FATOR
    )
    return cpe_por_dist and (cpe_por_desvio or media_hist is None)

# Deve disparar
casos_on = [
    (45.0, 150.0, 100.0),  # dist limiar + desvio
    (55.0, 180.0, 100.0),  # dist alta + desvio
    (50.0, 120.0, None),   # sem histÃ³rico, fallback por distÃ¢ncia
]
for dist, pot, hist in casos_on:
    if aciona_cpe(dist, pot, hist):
        ok(f"CPE acionado para dist={dist}m, pot={pot}, hist={hist}")
    else:
        fail(f"CPE nÃ£o acionado para dist={dist}m, pot={pot}, hist={hist}")

# NÃ£o deve disparar
casos_off = [
    (44.9, 200.0, 100.0),  # abaixo de 45m
    (50.0, 130.0, 100.0),  # sem desvio suficiente (1.3x)
    (35.0, 180.0, None),   # sem histÃ³rico, mas dist baixa
]
for dist, pot, hist in casos_off:
    if not aciona_cpe(dist, pot, hist):
        ok(f"CPE NÃƒO acionado para dist={dist}m, pot={pot}, hist={hist} (correto)")
    else:
        fail(f"CPE disparou incorretamente para dist={dist}m, pot={pot}, hist={hist}")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 4 â€” CPE: PrediÃ§Ãµes com distÃ¢ncia/2 diferem das originais")
print(sep)
# ==============================================================================

DIST_ORIG = 50.0
DIST_CPE  = DIST_ORIG / 2

for forn in ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']:
    for classe in ['M3', 'C3', 'P3']:
        r_orig = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe, 'Fornecedor': forn}, dist_override=DIST_ORIG)
        r_cpe  = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe, 'Fornecedor': forn}, dist_override=DIST_CPE)
        r_orig = ajustar_nbr(r_orig, classe)
        r_cpe  = ajustar_nbr(r_cpe,  classe)

        w_orig, w_cpe = r_orig['w'], r_cpe['w']
        # CPE com dist menor tende a ter W menor ou igual (melhor distribuiÃ§Ã£o)
        diferem = abs(w_orig - w_cpe) > 0.1
        if diferem:
            sentido = "menor" if w_cpe < w_orig else "maior"
            ok(f"{classe}/{forn}: CPE({DIST_CPE:.0f}m)={w_cpe:.1f}W {sentido} que Orig({DIST_ORIG:.0f}m)={w_orig:.1f}W")
        else:
            fail(f"{classe}/{forn}: prediÃ§Ã£o CPE idÃªntica Ã  original ({w_orig:.1f}W) â€” distÃ¢ncia nÃ£o foi aplicada")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 5 â€” MÃ©tricas corretas por tipo de via")
print(sep)
# ==============================================================================

# Classe M: deve retornar lmed (nÃ£o emed como mÃ©trica principal)
r_m = prever({'ClassificaÃ§Ã£o viÃ¡ria': 'M3'})
r_m = ajustar_nbr(r_m, 'M3')
if 'lmed' in r_m and r_m['lmed'] > 0:
    ok(f"Classe M3: lmed = {r_m['lmed']:.3f} cd/mÂ² (mÃ©trica de luminÃ¢ncia correta)")
else:
    fail("Classe M3: lmed nÃ£o disponÃ­vel ou zero")

# Classe C: deve retornar emed como mÃ©trica principal
r_c = prever({'ClassificaÃ§Ã£o viÃ¡ria': 'C3'})
r_c = ajustar_nbr(r_c, 'C3')
if 'emed' in r_c and r_c['emed'] > 0:
    ok(f"Classe C3: emed = {r_c['emed']:.2f} lux (mÃ©trica de iluminÃ¢ncia correta)")
else:
    fail("Classe C3: emed nÃ£o disponÃ­vel ou zero")

# Classe P: deve retornar emed e emin
r_p = prever({'ClassificaÃ§Ã£o viÃ¡ria': 'P3'})
r_p = ajustar_nbr(r_p, 'P3')
if 'emed' in r_p and r_p['emed'] > 0:
    ok(f"Classe P3: emed = {r_p['emed']:.2f} lux")
else:
    fail("Classe P3: emed nÃ£o disponÃ­vel ou zero")
if 'emin' in r_p and r_p['emin'] > 0:
    ok(f"Classe P3: emin = {r_p['emin']:.2f} lux")
else:
    fail("Classe P3: emin nÃ£o disponÃ­vel ou zero")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 6 â€” EquivalÃªncia Individual Ã— Lote (mesma config = mesmo resultado)")
print(sep)
# ==============================================================================

cfg_teste = {'ClassificaÃ§Ã£o viÃ¡ria': 'M3', 'Fornecedor': 'LEDSTAR'}

# Simula Individual
r_ind = prever(cfg_teste)
r_ind = ajustar_nbr(r_ind, 'M3')

# Simula Lote (vetorizado â€” mesmo dado, um DataFrame de 1 linha)
dados_lote = {**BASE_CFG, **cfg_teste}
df_lote = pd.DataFrame([{k: dados_lote.get(k, np.nan) for k in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col]))}])
r_lote = {} 
if 'w' in modelos:
    r_lote['w'] = max(modelos['w'].predict(df_lote)[0], 0)
for m in ['lmed', 'uo', 'ul', 'emed', 'emin']:
    if m not in modelos:
        continue
    if m in modelos_dependem_de_w:
        df_m = df_lote.copy()
        df_m[feature_w_col] = r_lote.get('w', np.nan)
        r_lote[m] = max(modelos[m].predict(df_m)[0], 0)
    else:
        r_lote[m] = max(modelos[m].predict(df_lote)[0], 0)

# Ajuste NBR no lote (replica lógica do app)
r_lote = ajustar_nbr(r_lote, 'M3')

for m in ['lmed', 'emed', 'w']:
    if m in r_ind and m in r_lote:
        diff = abs(r_ind[m] - r_lote[m])
        if diff < 0.01:
            ok(f"Individual == Lote para {m}: {r_ind[m]:.4f}")
        else:
            fail(f"DivergÃªncia em {m}", f"individual={r_ind[m]:.4f}, lote={r_lote[m]:.4f}, diff={diff:.4f}")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 7 â€” PotÃªncia pÃ³s-ajuste atende ao mÃ­nimo NBR (via simulaÃ§Ã£o)")
print(sep)
# ==============================================================================

# Para cada classe, apÃ³s ajuste, a iluminÃ¢ncia ajustada deve ser >= req
for classe in ['M1', 'M3', 'M6', 'C0', 'C3', 'C5', 'P1', 'P3', 'P6']:
    info = NBR5101[classe]
    r = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe})
    r_ajust = ajustar_nbr(dict(r), classe)

    metric_ref = 'lmed' if classe.startswith('M') else 'emed'
    req = info.get(metric_ref)
    pred_ilum_orig = r.get(metric_ref, 0)
    pot_orig = r.get('w', 0)
    pot_ajust = r_ajust.get('w', 0)

    # ApÃ³s ajuste: se pred_ilum < req â†’ potÃªncia foi escalonada
    if pred_ilum_orig < req:
        escalonado = pot_ajust > pot_orig * 1.001
        if escalonado:
            ok(f"{classe}: ajuste aplicado {pot_orig:.1f}W â†’ {pot_ajust:.1f}W (ilum={pred_ilum_orig:.3f} < req={req})")
        else:
            fail(f"{classe}: ilum={pred_ilum_orig:.3f} < req={req} mas potÃªncia NÃƒO foi escalonada")
    else:
        ok(f"{classe}: ilum={pred_ilum_orig:.3f} >= req={req} â€” sem ajuste necessÃ¡rio ({pot_orig:.1f}W)")


# ==============================================================================
print(f"\n{sep}")
print("  TESTE 8 â€” Todos os Fornecedores retornam valores distintos")
print(sep)
# ==============================================================================

for classe in ['M3', 'C3', 'P3']:
    pots_forn = {}
    for forn in ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']:
        r = prever({'ClassificaÃ§Ã£o viÃ¡ria': classe, 'Fornecedor': forn})
        r = ajustar_nbr(r, classe)
        pots_forn[forn] = r['w']

    todos_iguais = len(set(round(v, 1) for v in pots_forn.values())) == 1
    if not todos_iguais:
        ok(f"Classe {classe}: fornecedores retornam potÃªncias distintas â€” " +
           ", ".join(f"{k}={v:.1f}W" for k, v in pots_forn.items()))
    else:
        fail(f"Classe {classe}: todos os fornecedores retornam a mesma potÃªncia ({list(pots_forn.values())[0]:.1f}W)")


# ==============================================================================
print(f"\n{sep}")
print("  RESUMO FINAL")
print(sep)
# ==============================================================================

total = len(PASSOU) + len(FALHOU)
print(f"\n  Testes executados : {total}")
print(f"  âœ… Passaram        : {len(PASSOU)}")
print(f"  âŒ Falharam        : {len(FALHOU)}")

if FALHOU:
    print("\n  Falhas:")
    for f in FALHOU:
        print(f"    - {f}")
    sys.exit(1)
else:
    print("\n  Todos os testes passaram.")
    sys.exit(0)




