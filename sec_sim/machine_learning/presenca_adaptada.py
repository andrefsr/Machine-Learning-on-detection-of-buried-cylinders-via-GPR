# %% [markdown]
# # Rótulos de presença adaptados à calibração
# Usa o modelo do tempo do ápice por solo (calibrado no presenca_cilindro_v2.py):
#     t0 = A0 - 2*(pos_y + radius)/V_EFF        (y do gprMax cresce para cima)
# `pos_antenna` deixa de ser necessário: a calibração já embute a geometria.

# %%
import numpy as np
import pandas as pd

DT = 3e-9 / 637
I0_ROI = 361                          # amostra inicial do ROI no fe3.py (mude se reextrair)
T_INI = I0_ROI * DT                   # s, início do ROI (~1,7 ns para 361)
T_JAN = 3e-9                          # s, fim da janela
X_INI = 0.08                          # posição do 1º traço no solo [m]
X_FIM = 0.08 + 59 * 0.002             # último traço (60 traços de 2 mm) = 0.198 m

# resultado da calibração por solo (v_eff em m/s; A0 reconstruído de v_eff e y_min)
V_EFF = {1: 0.862e8, 2: 0.849e8, 3: 0.887e8}
A0 = {1: 4.763e-9, 2: 4.932e-9, 3: 4.826e-9}

df = pd.read_csv('databasev4_alt.csv')
pos_x = df['pos_x'].to_numpy()
pos_y = df['pos_y'].to_numpy()
radius = df['radius'].to_numpy()

v = df['soil_type'].map(V_EFF).to_numpy()
a0 = df['soil_type'].map(A0).to_numpy()
t0_pred = a0 - 2 * (pos_y + radius) / v          # tempo previsto do ápice (topo do cilindro) [s]

# %% [markdown]
# ## Classes
# - 0 ausente: corpo todo fora da varredura, ou eco do topo depois do fim da janela
# - 1 parcial: o corpo toca a varredura, mas o ápice está fora da abertura ou fora do ROI no tempo
# - 2 ápice visível: ápice dentro da abertura e dentro do ROI (o caso em que x0_fit e t0_fit fazem sentido)

# %%
corpo_fora = (pos_x + radius < X_INI) | (pos_x - radius > X_FIM)
tarde = t0_pred > T_JAN
apice_x = (pos_x >= X_INI) & (pos_x <= X_FIM)
apice_t = (t0_pred >= T_INI) & (t0_pred <= T_JAN)

classe = np.where(corpo_fora | tarde, 0, np.where(apice_x & apice_t, 2, 1))
presence = (classe >= 1).astype(int)             # "há algum vestígio do cilindro"
apice_visivel = (classe == 2).astype(int)        # "ápice utilizável"

df['classe'] = classe
df['presence'] = presence
df['apice_visivel'] = apice_visivel

print(df['classe'].value_counts().sort_index().rename({0: '0 ausente', 1: '1 parcial', 2: '2 ápice visível'}))
print(f"\ncorpo fora: {corpo_fora.sum()} | tarde: {tarde.sum()} | "
      f"ápice fora da abertura: {(~apice_x).sum()} | ápice antes do ROI: {(t0_pred < T_INI).sum()}")

# %% [markdown]
# ## Validação: o ajuste da hipérbole acerta o ápice onde o rótulo diz que ele é visível?
# Árbitro independente do SNR: erro de `x0_fit` (e de `t0_fit`) em relação ao valor real/previsto.
# Se `x0_fit` ainda está em coordenadas do B-scan, mantenha `X_OFF = X_INI`; se já somou 0,08, use 0.

# %%
X_OFF = X_INI
df['erro_x'] = (df['x0_fit'] + X_OFF - df['pos_x']).abs()
df['erro_t'] = (df['t0_fit'] - t0_pred).abs()

resumo = df.groupby('classe').agg(
    n=('erro_x', 'size'),
    erro_x_mediano_mm=('erro_x', lambda s: 1e3 * s.median()),
    x_dentro_5mm=('erro_x', lambda s: (s < 0.005).mean()),
    erro_t_mediano_ps=('erro_t', lambda s: 1e12 * s.median()),
)
print(resumo.round(3).to_string())

df.to_csv('databasev5_presenca.csv', index=False)
