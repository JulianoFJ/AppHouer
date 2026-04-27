"""
Diagnóstico: varre variações de configurações para C-class e detecta
quais parâmetros combinados fazem o emed ou uo cair abaixo da NBR.
"""
import joblib, pandas as pd, numpy as np, json, itertools

with open('features.json', encoding='utf-8') as f:
    meta = json.load(f)

modelo_uo   = joblib.load('modelo_uo.pkl')
modelo_emed = joblib.load('modelo_emed.pkl')
num_ok = meta['features_numericas']
cat_ok = meta['features_categoricas']

NBR_C = {
    'C0': {'emed': 50.0, 'uo': 0.40},
    'C1': {'emed': 30.0, 'uo': 0.40},
    'C2': {'emed': 20.0, 'uo': 0.40},
    'C3': {'emed': 15.0, 'uo': 0.35},
    'C4': {'emed': 10.0, 'uo': 0.35},
    'C5': {'emed':  5.0, 'uo': 0.35},
}

# Varre combinações de parametros de geometria
classes     = ['C3', 'C2', 'C4']
alturas     = [8.0, 9.0, 10.0, 12.0]
distancias  = [25.0, 30.0, 35.0, 40.0, 45.0]
larguras    = [5.0, 7.0, 9.0, 12.0]
fornecedores = ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']

rows = []
for classe, alt, dist, larg, forn in itertools.product(classes, alturas, distancias, larguras, fornecedores):
    cfg = {
        'Classificação viária': classe, 'Faixas de Rodagem': 2,
        'Largura Via 1': larg, 'Largura Via 2': 0.0,
        'Largura Passeio 1': 2.0, 'largura Passeio 2': 2.0,
        'largura Canteiro Central': 0.0, 'altura da luminaria': alt,
        'projecao do braço': 1.5, 'distancia entre postes': dist,
        'distancia Poste a via': 0.5, 'Tipo de estrutura': 'Braço',
        'posteacao': 'Unilateral', 'Braço Novo': 'Longo II', 'Fornecedor': forn
    }
    rows.append(cfg)

df_test = pd.DataFrame([{k: r.get(k, np.nan) for k in num_ok + cat_ok} for r in rows])
preds_uo   = modelo_uo.predict(df_test)
preds_emed = modelo_emed.predict(df_test)

# Analisa quais falham
results = []
for i, cfg in enumerate(rows):
    classe = cfg['Classificação viária']
    req = NBR_C.get(classe, {})
    emed_val = preds_emed[i]
    uo_val   = preds_uo[i]
    emed_ok  = emed_val >= req.get('emed', 0)
    uo_ok    = uo_val   >= req.get('uo', 0)
    atende   = emed_ok and uo_ok
    results.append({
        'Classe': classe, 'Altura': cfg['altura da luminaria'],
        'Distância': cfg['distancia entre postes'], 'Largura': cfg['Largura Via 1'],
        'Fornecedor': cfg['Fornecedor'],
        'emed_pred': round(emed_val, 2), 'uo_pred': round(uo_val, 4),
        'emed_req': req.get('emed'), 'uo_req': req.get('uo'),
        'emed_ok': emed_ok, 'uo_ok': uo_ok, 'Atende': atende
    })

df_res = pd.DataFrame(results)
total = len(df_res)
falhas = df_res[~df_res['Atende']]
print(f"Total testado: {total}")
print(f"Falhas: {len(falhas)} ({len(falhas)/total*100:.1f}%)")
print(f"Falhas só por uo: {len(df_res[~df_res['uo_ok'] & df_res['emed_ok']])}")
print(f"Falhas só por emed: {len(df_res[df_res['uo_ok'] & ~df_res['emed_ok']])}")
print(f"Falhas em ambos: {len(df_res[~df_res['uo_ok'] & ~df_res['emed_ok']])}")

print("\n=== Distribuição de uo predito ===")
print(df_res.groupby('Classe')['uo_pred'].describe().round(4))

print("\n=== Distribuição de emed predito por classe ===")
print(df_res.groupby('Classe')['emed_pred'].describe().round(2))

print("\n=== Top 10 falhas (ordenado por emed) ===")
print(falhas.sort_values('emed_pred').head(10).to_string(index=False))

print("\n=== Configurações que sempre falham (todas forns) ===")
grp = df_res.groupby(['Classe','Altura','Distância','Largura'])['Atende'].all()
sempre_falha = grp[~grp].index
print(f"Combinações que falham para todos fornecedores: {len(sempre_falha)}")
for idx in list(sempre_falha)[:5]:
    print(f"  {idx}")
