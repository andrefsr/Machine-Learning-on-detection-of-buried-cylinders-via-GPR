# %% [markdown]
# # Rótulo de presença do cilindro: regra geométrica vs. visibilidade medida
# Convenção do gprMax: y cresce PARA CIMA, então o topo do cilindro (ponto mais próximo da antena)
# está em y = pos_y + radius. Quanto menor o y do topo, mais tarde chega o eco.
# Requer o CSV com as features do fe3.py + pos_x, pos_y, radius, soil_type.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

T_JAN = 3.0e-9                    # s, fim da janela de tempo
T_INI = 361 * 3.0e-9 / 637        # s, início do ROI do fe3.py (amostra 361, ~1,7 ns)
X_INI = 0.08                      # posição do 1º traço no solo [m]
X_FIM = 0.08 + 59 * 0.002         # posição do último traço (60 traços de 2 mm) = 0.198 m
METRICA = 'snr_env'               # medida de visibilidade (SNR do ápice)

df = pd.read_csv('databasev4_alt.csv')
px, py, r = df['pos_x'].to_numpy(), df['pos_y'].to_numpy(), df['radius'].to_numpy()
df['top_y'] = py + r

# %% [markdown]
# ## A. A sua regra, critério por critério
# Com y para cima, `pos_y + radius > 0.446` testa se o topo está ACIMA de 0,446 m, o que não é um
# limite de profundidade.

# %%
lat = (px + r < 0.08) | (px - r > 0.2)
prof = py + r > 0.446
df['presence_v1'] = (~(lat | prof)).astype(int)

print(df['presence_v1'].value_counts().to_string())
print(f'ausentes por lateral: {lat.sum()} | por "profundidade": {prof.sum()} | ambos: {(lat & prof).sum()}\n')
print(df[['pos_x', 'pos_y', 'radius', 'top_y']].describe().round(4).to_string())

# %% [markdown]
# ## B. Visibilidade medida
# Esquerda: `gap_lat` > 0 = cilindro totalmente fora da varredura; < 0 = sobreposto (vermelho = sua regra).
# Direita: mediana da métrica no plano (pos_x, topo do cilindro), sem nenhuma hipótese de modelo.

# %%
df['gap_lat'] = np.maximum(X_INI - (px + r), (px - r) - X_FIM)
df['apice_dentro'] = (px >= X_INI) & (px <= X_FIM)

bins = np.linspace(df['gap_lat'].min(), df['gap_lat'].max(), 21)
g = df.groupby(pd.cut(df['gap_lat'], bins), observed=True)[METRICA].quantile([.25, .5, .75]).unstack()
xm = np.array([iv.mid for iv in g.index])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].fill_between(xm, g[.25], g[.75], alpha=.3)
ax[0].plot(xm, g[.5], 'o-')
ax[0].axvline(0, color='r', ls='--')
ax[0].set(xlabel='gap lateral [m]', ylabel=METRICA, title='Visibilidade vs. distância lateral à varredura')

hb = ax[1].hexbin(px, df['top_y'], C=df[METRICA], reduce_C_function=np.median, gridsize=22, mincnt=3)
plt.colorbar(hb, ax=ax[1], label=f'mediana de {METRICA}')
ax[1].axvline(X_INI, color='r', ls='--')
ax[1].axvline(X_FIM, color='r', ls='--')
ax[1].set(xlabel='pos_x [m]', ylabel='topo do cilindro: pos_y + radius [m]', title='Mapa de visibilidade')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## C. Calibração por solo: t0 = a + b·top_y (b < 0, com v_eff = -2/b)
# Usa só cilindros com ápice dentro da varredura, sem ápice na borda do tempo e com bom ajuste
# (score_contraste acima da mediana do solo), com um corte de outliers. Daí saem os limites em y:
# - `y_min`: abaixo disso o eco do topo chega depois de 3 ns (fora da janela)
# - `y_max`: acima disso o ápice chega antes do início do ROI (flancos ainda podem aparecer)

# %%
cal = df[df['apice_dentro'] & (df['flag_borda_t'] == 0)]
cal = cal[cal['score_contraste'] >= cal.groupby('soil_type')['score_contraste'].transform('median')]

LIM = {}
plt.figure(figsize=(6, 4))
for s, gs in cal.groupby('soil_type'):
    b, a = np.polyfit(gs['top_y'], gs['t0_fit'], 1)
    res = gs['t0_fit'] - (a + b * gs['top_y'])
    ok = res.abs() < 3 * 1.4826 * res.abs().median()
    b, a = np.polyfit(gs.loc[ok, 'top_y'], gs.loc[ok, 't0_fit'], 1)
    if b >= 0:
        print(f'solo {s}: inclinação positiva, confira a orientação de y')
        continue
    v_eff = -2 / b
    LIM[s] = dict(y_min=(T_JAN - a) / b, y_max=(T_INI - a) / b)
    print(f'solo {s}: v_eff = {v_eff / 1e8:.3f}e8 m/s | εr_eff = {(2.998e8 / v_eff) ** 2:.1f} | '
          f'y_min = {LIM[s]["y_min"] * 100:.1f} cm | y_max = {LIM[s]["y_max"] * 100:.1f} cm')
    p = plt.scatter(gs['top_y'], gs['t0_fit'] * 1e9, s=4, alpha=.4, label=f'solo {s}')
    xx = np.array([gs['top_y'].min(), gs['top_y'].max()])
    plt.plot(xx, (a + b * xx) * 1e9, color=p.get_facecolor()[0][:3])
plt.axhline(T_JAN * 1e9, color='r', ls='--')
plt.axhline(T_INI * 1e9, color='r', ls='--')
plt.xlabel('topo do cilindro: pos_y + radius [m]')
plt.ylabel('t0_fit [ns]')
plt.legend()
plt.show()

# %% [markdown]
# ## D. Rótulos refinados e comparação
# - `presence_v2`: dentro da varredura (com tolerância `TOL_LAT`) e eco do topo dentro da janela
# - `presence_v3`: v2 e ainda com o ápice dentro do ROI (mais restrito)
# `TOL_LAT`: quanto o cilindro pode estar fora da abertura e ainda contar como presente
# (leia no gráfico B; 0 reproduz a sua regra lateral).

# %%
TOL_LAT = 0.0

df['y_min'] = df['soil_type'].map({s: d['y_min'] for s, d in LIM.items()})
df['y_max'] = df['soil_type'].map({s: d['y_max'] for s, d in LIM.items()})
df['tarde'] = df['top_y'] < df['y_min']
df['apice_cortado'] = df['top_y'] > df['y_max']

df['presence_v2'] = ((df['gap_lat'] < TOL_LAT) & ~df['tarde']).astype(int)
df['presence_v3'] = (df['presence_v2'].astype(bool) & ~df['apice_cortado']).astype(int)

print(pd.crosstab(df['soil_type'], [df['tarde'], df['apice_cortado']],
                  rownames=['solo'], colnames=['tarde', 'ápice cortado']), '\n')


def auc(rotulo):
    return roc_auc_score(df[rotulo], df[METRICA]) if df[rotulo].nunique() == 2 else float('nan')


for k in ['presence_v1', 'presence_v2', 'presence_v3']:
    print(f'{k}: presentes = {df[k].sum()} / {len(df)} | AUC da {METRICA} = {auc(k):.3f}')

df.to_csv('databasev5_presenca.csv', index=False)
