import joblib, pandas as pd, numpy as np, json

with open('features.json', encoding='utf-8') as f:
    meta = json.load(f)

modelo_uo   = joblib.load('modelo_uo.pkl')
modelo_emed = joblib.load('modelo_emed.pkl')
num_ok = meta['features_numericas']
cat_ok = meta['features_categoricas']

configs = [
    {
        'Classificação viária': 'C3', 'Faixas de Rodagem': 2, 'Largura Via 1': 7.0,
        'Largura Via 2': 0.0, 'Largura Passeio 1': 2.0, 'largura Passeio 2': 2.0,
        'largura Canteiro Central': 0.0, 'altura da luminaria': 10.0,
        'projecao do braço': 1.5, 'distancia entre postes': 35.0,
        'distancia Poste a via': 0.5, 'Tipo de estrutura': 'Braço',
        'posteacao': 'Unilateral', 'Braço Novo': 'Longo II', 'Fornecedor': 'LEDSTAR'
    },
    {
        'Classificação viária': 'C2', 'Faixas de Rodagem': 4, 'Largura Via 1': 10.0,
        'Largura Via 2': 0.0, 'Largura Passeio 1': 3.0, 'largura Passeio 2': 3.0,
        'largura Canteiro Central': 0.0, 'altura da luminaria': 12.0,
        'projecao do braço': 2.0, 'distancia entre postes': 30.0,
        'distancia Poste a via': 0.5, 'Tipo de estrutura': 'Braço',
        'posteacao': 'Bilateral alternada', 'Braço Novo': 'Longo II', 'Fornecedor': 'SX LIGHTING'
    },
]

df_test = pd.DataFrame([{k: c.get(k, np.nan) for k in num_ok + cat_ok} for c in configs])
preds_uo   = modelo_uo.predict(df_test)
preds_emed = modelo_emed.predict(df_test)

print('=== Resultados do teste ===')
for i, cfg in enumerate(configs):
    classe = cfg['Classificação viária']
    print(f"\nConfig {i+1} - Via {classe} ({cfg['Fornecedor']}):")
    print(f"  uo   = {preds_uo[i]:.4f}")
    print(f"  emed = {preds_emed[i]:.2f} lux")

print("\n=== Requisitos NBR ===")
print("C3: emed >= 15 lux, uo >= 0.35")
print("C2: emed >= 20 lux, uo >= 0.40")

# Verifica pass/fail
norms = {'C3': {'emed': 15.0, 'uo': 0.35}, 'C2': {'emed': 20.0, 'uo': 0.40}}
print("\n=== Resultado Status NBR ===")
for i, cfg in enumerate(configs):
    classe = cfg['Classificação viária']
    req = norms.get(classe, {})
    uo_ok   = preds_uo[i] >= req.get('uo', 0)
    emed_ok = preds_emed[i] >= req.get('emed', 0)
    status = 'ATENDE' if (uo_ok and emed_ok) else 'NAO ATENDE'
    print(f"Config {i+1} ({classe}): uo_ok={uo_ok} ({preds_uo[i]:.4f}>={req.get('uo')}), emed_ok={emed_ok} ({preds_emed[i]:.2f}>={req.get('emed')}) => {status}")
