# %% [markdown]
# # Cadeia gate -> x -> y -> raio com validação cruzada aninhada
#
# - Gate (presença) treinado em todas as amostras; x, y e raio treinados só nas realmente presentes.
# - Cada estágio recebe as previsões dos anteriores. No treino elas são *out-of-fold* (CV interna),
#   no teste vêm do modelo treinado no fold; assim a métrica externa não vaza.
# - Métricas dos regressores: nas amostras realmente presentes (cadeia isolada) e nas que também
#   passaram pelo gate (ponta a ponta).
# - Um modelo por alvo, trocável por string: 'hgb' | 'lgbm' | 'xgb'.
# - Saídas: oof_predicoes.csv e posicoes_previstas.csv (entrada do fe_raio.py).

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

SEED = 42
CSV = 'databasev4_2.csv'          # features do fe3.py + pos_x, pos_y, radius e a coluna do gate
COL_GATE = 'presence'             # 0/1 (também serve 'apice_visivel' do presenca_adaptada.py)
MODELO = {'pos_x': 'hgb', 'pos_y': 'hgb', 'radius': 'hgb'}     # o LightGBM travou no seu ambiente: use 'hgb'
N_EXT, N_INT = 5, 5               # folds externos e internos
LIMIAR_GATE = 0.5
ORACULO = False                   # True: x e y REAIS como entrada dos estágios seguintes (limite superior)
FEATURES_RAIO_CSV = None          # ex.: 'features_raio.csv' (fe_raio.py) para somar features ao estágio do raio

# lista explícita: colunas derivadas do rótulo (classe, erro_x, top_y...) nunca entram como feature
FEATURES_FE3 = [
    'x_pico_env', 't_pico_env', 'x0_fit', 't0_fit', 'v_fit', 'd_apex', 'score_fit', 'score_contraste',
    'flag_borda_x', 'flag_borda_t', 'a_max', 'larg_x_50', 'larg_x_25', 'a_rel_10mm', 'a_rel_20mm',
    'a_rel_30mm', 'a_rel_40mm', 'frac_energia_banda', 'rms_banda', 'clutter_rms', 'clutter_env_med',
    'snr_env', 'snr_rms', 'larg_t_50', 'larg_t_25', 'eco2_atraso', 'eco2_razao', 'freq_centroid_apex',
    'freq_peak_apex', 'freq_bw_apex', 'freq_centroid_global', 'freq_centroid_razao', 'freq_bw_razao',
    'rms_global', 'kurtosis_global', 'q99_env', 'apex_sobre_rms', 'bg_rms', 'bg_t_pico']
ALVOS = ['pos_x', 'pos_y', 'radius']              # ordem da cadeia
COL_PREV = {'pos_x': 'x_hat', 'pos_y': 'y_hat'}   # nome da previsão que alimenta o estágio seguinte


def fabrica(nome):
    """Devolve uma função que cria um regressor novo (importa a biblioteca só se for usada)."""
    if nome == 'hgb':
        return lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=2000, max_leaf_nodes=31, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=50, random_state=SEED)
    if nome == 'lgbm':
        from lightgbm import LGBMRegressor
        return lambda: LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=20, random_state=SEED, n_jobs=1, verbose=-1)
    if nome == 'xgb':
        from xgboost import XGBRegressor
        return lambda: XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, tree_method='hist', random_state=SEED, n_jobs=1)
    raise ValueError(f'modelo desconhecido: {nome}')


# %% [markdown]
# ## Dados

# %%
df = pd.read_csv(CSV)
if 'id' in df.columns:
    ids = df['id'].to_numpy()
else:
    ids = np.arange(1, len(df) + 1)
    print('AVISO: sem coluna id; assumindo que a linha i corresponde ao arquivo Bscan{i+1}')

feat = [c for c in FEATURES_FE3 if c in df.columns]
print(f'{len(feat)} features usadas; ausentes no CSV: {[c for c in FEATURES_FE3 if c not in df.columns]}')
X = df[feat].reset_index(drop=True)
Y = df[ALVOS].reset_index(drop=True)
gate = df[COL_GATE].to_numpy().astype(int)

EXTRA = None
if FEATURES_RAIO_CSV:
    fr = pd.read_csv(FEATURES_RAIO_CSV).set_index('id').drop(columns=['x_prev', 'y_prev'], errors='ignore')
    EXTRA = fr.reindex(ids).reset_index(drop=True).add_prefix('fr_')   # NaN onde não há features
    print(f'{EXTRA.shape[1]} features extras do raio; linhas sem extras: {EXTRA.isna().all(axis=1).sum()}')


# %% [markdown]
# ## Cadeia dentro de um fold externo

# %%
def cadeia_fold(i_tr, i_te):
    """Treina x -> y -> raio nos realmente presentes do treino e prevê TODAS as amostras do teste."""
    tr = i_tr[gate[i_tr] == 1]
    cv_int = KFold(N_INT, shuffle=True, random_state=SEED)
    Xtr = X.iloc[tr].reset_index(drop=True)
    Xte = X.iloc[i_te].reset_index(drop=True)
    saida = {}
    for alvo in ALVOS:
        if alvo == 'radius' and EXTRA is not None:
            Xtr = pd.concat([Xtr, EXTRA.iloc[tr].reset_index(drop=True)], axis=1)
            Xte = pd.concat([Xte, EXTRA.iloc[i_te].reset_index(drop=True)], axis=1)
        ytr = Y[alvo].to_numpy()[tr]
        make = fabrica(MODELO[alvo])
        oof = cross_val_predict(make(), Xtr, ytr, cv=cv_int)    # previsões "limpas" para o próximo estágio
        pte = make().fit(Xtr, ytr).predict(Xte)
        saida[alvo] = pte
        if alvo in COL_PREV:
            Xtr[COL_PREV[alvo]] = ytr if ORACULO else oof
            Xte[COL_PREV[alvo]] = Y[alvo].to_numpy()[i_te] if ORACULO else pte
    return saida


# %% [markdown]
# ## Validação cruzada externa (leva alguns minutos)

# %%
n = len(df)
proba = np.zeros(n)
pred = {a: np.full(n, np.nan) for a in ALVOS}
ext = StratifiedKFold(N_EXT, shuffle=True, random_state=SEED)

for k, (i_tr, i_te) in enumerate(ext.split(X, gate), 1):
    g = HistGradientBoostingClassifier(random_state=SEED).fit(X.iloc[i_tr], gate[i_tr])
    proba[i_te] = g.predict_proba(X.iloc[i_te])[:, 1]
    for a, p in cadeia_fold(i_tr, i_te).items():
        pred[a][i_te] = p
    print(f'fold {k}/{N_EXT} concluído')

gate_pred = (proba >= LIMIAR_GATE).astype(int)

# %% [markdown]
# ## Resultados

# %%
print('=== Gate ===')
print(confusion_matrix(gate, gate_pred))
print(classification_report(gate, gate_pred, digits=3))


def tabela(mask, titulo):
    linhas = []
    alvos = {a: (Y[a].to_numpy()[mask], pred[a][mask]) for a in ALVOS}
    alvos['topo (y+raio)'] = ((Y['pos_y'] + Y['radius']).to_numpy()[mask],
                              (pred['pos_y'] + pred['radius'])[mask])
    for nome, (yt, yp) in alvos.items():
        rmse = np.sqrt(mean_squared_error(yt, yp))
        linhas.append({'alvo': nome, 'R2': r2_score(yt, yp), 'RMSE': rmse,
                       'MAE': mean_absolute_error(yt, yp), 'desvio do alvo': yt.std(),
                       'RMSE/desvio': rmse / yt.std()})
    print(f'\n=== {titulo} (n = {mask.sum()}) ===')
    print(pd.DataFrame(linhas).round(4).to_string(index=False))


tabela(gate == 1, 'Cadeia nas realmente presentes')
tabela((gate == 1) & (gate_pred == 1), 'Ponta a ponta (presentes que passaram no gate)')

# %% [markdown]
# ## Saídas

# %%
out = pd.DataFrame({'id': ids, 'gate': gate, 'gate_proba': proba, 'gate_pred': gate_pred,
                    'x_hat': pred['pos_x'], 'y_hat': pred['pos_y'], 'r_hat': pred['radius'],
                    'pos_x': Y['pos_x'], 'pos_y': Y['pos_y'], 'radius': Y['radius']})
out.to_csv('oof_predicoes.csv', index=False)

m = gate_pred == 1
pd.DataFrame({'id': ids[m], 'x_prev': pred['pos_x'][m], 'y_prev': pred['pos_y'][m]}) \
    .to_csv('posicoes_previstas.csv', index=False)
print(f'salvos: oof_predicoes.csv e posicoes_previstas.csv ({m.sum()} linhas, entrada do fe_raio.py)')
