# %% [markdown]
# # CNN multitarefa: B-scan (ROI) -> presença, x, y e raio
#
# Entrada : bscans_cnn.npz (cnn_dataset.py): canais = campo sem fundo e envelope, só o ROI.
# Saídas  : 5 neurônios = [presença (logit), x, y, raio, topo = y + raio]. O topo é um alvo auxiliar:
#           ele é bem determinado pelo tempo do ápice e ajuda a rede a separar y de r.
# Perdas  : BCE na presença (todas as amostras) + Huber nos alvos padronizados, só nas amostras
#           com cilindro presente (máscara).
# Aumento : espelhamento horizontal com remapeamento de x (x' = C_FLIP - x) + ruído leve; TTA opcional.
# Rode primeiro com RAPIDO = True (poucos segundos) para checar o ambiente.

# %%
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

SEED = 42
CSV_ROTULOS = 'databasev4_2.csv'          # pos_x, pos_y, radius, COL_GATE (+ x0_fit e score_contraste, se houver)
COL_GATE = 'presence'                     # 0/1; None = todas as amostras têm cilindro (sem cabeça de presença)
FEATURES_CSV = 'features5k.csv'           # só para o baseline HGB no mesmo split (opcional)
EPOCAS, LOTE, LR, PACIENCIA = 60, 64, 1e-3, 15
PESO_GATE = 0.5
PESOS_REG = [1.0, 1.0, 1.0, 0.5]          # pesos da perda: x, y, raio, topo (auxiliar)
FLIP = True                               # espelhamento horizontal (ver C_FLIP abaixo)
TTA = True                                # média das previsões do B-scan e do espelhado, no teste
RAPIDO = False                            # True: teste de fumaça (300 amostras, 2 épocas)

X_INI = 0.08                              # posição do 1º traço no solo [m]
X_FIM = 0.08 + 59 * 0.002                 # posição do último traço (60 traços de 2 mm)

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% [markdown]
# ## Dados

# %%
d = np.load('bscans_cnn.npz')
X, ids = d['X'], d['ids']

df = pd.read_csv(CSV_ROTULOS)
if 'id' in df.columns:
    df = df.set_index('id').loc[ids]
else:
    assert len(df) == len(ids), 'CSV e dataset com tamanhos diferentes'
    print('AVISO: o CSV não tem coluna id; assumindo a mesma ordem dos arquivos Bscan1..N')
    df = df.reset_index(drop=True)

if COL_GATE:
    gate = df[COL_GATE].to_numpy().astype(np.float32)
else:
    gate = np.ones(len(df), dtype=np.float32)
    PESO_GATE = 0.0
T = np.stack([df['pos_x'], df['pos_y'], df['radius'], df['pos_y'] + df['radius']], axis=1).astype(np.float64)


def estimar_delta():
    """Deslocamento entre o x do ápice no B-scan e pos_x (ex.: metade do afastamento Tx-Rx).
    O espelhamento mapeia x -> C_FLIP - x com C_FLIP = X_INI + X_FIM + 2*delta."""
    if not {'x0_fit', 'score_contraste'} <= set(df.columns):
        print('AVISO: sem x0_fit/score_contraste no CSV; assumindo delta = 0')
        return 0.0
    ok = (gate == 1) & (T[:, 0] > X_INI + 0.01) & (T[:, 0] < X_FIM - 0.01)
    ok &= df['score_contraste'].to_numpy() >= np.median(df['score_contraste'].to_numpy()[gate == 1])
    x0 = df['x0_fit'].to_numpy()[ok]
    d1 = np.median(T[ok, 0] - (x0 + X_INI))      # x0_fit em coordenadas do B-scan
    d0 = np.median(T[ok, 0] - x0)                # x0_fit já em coordenadas do solo
    return d1 if abs(d1) <= abs(d0) else d0


DELTA = estimar_delta()
C_FLIP = X_INI + X_FIM + 2 * DELTA
print(f'delta = {DELTA * 1e3:.2f} mm | C_FLIP = {C_FLIP:.4f} m')
if abs(DELTA) > 0.005:
    print('ATENÇÃO: |delta| > 5 mm; confira a convenção de x (ou use FLIP = False e TTA = False)')

if RAPIDO:
    sub = rng.choice(len(T), 300, replace=False)
    X, ids, T, gate = X[sub], ids[sub], T[sub], gate[sub]
    EPOCAS = 2

idx = np.arange(len(T))
strat = gate.astype(int) if COL_GATE else None
i_tmp, i_te = train_test_split(idx, test_size=0.15, stratify=strat, random_state=SEED)
i_tr, i_va = train_test_split(i_tmp, test_size=0.15 / 0.85, random_state=SEED,
                              stratify=None if strat is None else strat[i_tmp])

pres_tr = i_tr[gate[i_tr] == 1]
mu, sd = T[pres_tr].mean(axis=0), T[pres_tr].std(axis=0)     # padronização dos alvos (só presentes, só treino)
Tz = ((T - mu) / sd).astype(np.float32)
print(f'X {X.shape} | presentes {int(gate.sum())}/{len(gate)} | treino/val/teste = '
      f'{len(i_tr)}/{len(i_va)}/{len(i_te)} | {dev}')


def aumentar(xb, tb):
    """Espelhamento horizontal aleatório do B-scan, com remapeamento de x. Devolve cópias."""
    xb, tb = xb.copy(), tb.copy()
    if FLIP:
        fl = rng.random(len(xb)) < 0.5
        xb[fl] = xb[fl][..., ::-1]
        tb[fl, 0] = C_FLIP - tb[fl, 0]
    return xb, tb


# %% [markdown]
# ## Modelo

# %%
def coords(x):
    """Dois canais com tempo e posição normalizados (CoordConv): a rede precisa da posição absoluta."""
    B, _, H, W = x.shape
    t = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    p = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([t, p], dim=1)


class CNN(nn.Module):
    """4 blocos conv + pooling. As coordenadas são reinjetadas na entrada de cada bloco e a cabeça
    usa Flatten (não pooling global), para preservar a informação espacial que x e y exigem."""

    def __init__(self, n_out=5, in_ch=2):
        super().__init__()
        ch = [in_ch, 16, 32, 64, 64]

        def bloco(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.blocos = nn.ModuleList([bloco(ch[k] + 2, ch[k + 1]) for k in range(4)])
        self.pool = nn.AdaptiveAvgPool2d((4, 3))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 3, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, n_out))

    def forward(self, x):
        for b in self.blocos:
            x = b(torch.cat([x, coords(x)], dim=1))
        return self.head(self.pool(x))


def perda_total(saida, tz, gb):
    """saida: (B,5) = [logit, x, y, r, topo] padronizados; tz: (B,4); gb: (B,) 0/1."""
    l_gate = nnF.binary_cross_entropy_with_logits(saida[:, 0], gb)
    pesos = torch.tensor(PESOS_REG, dtype=torch.float32, device=saida.device)
    l_reg = (nnF.smooth_l1_loss(saida[:, 1:], tz, reduction='none') * pesos).sum(dim=1)
    l_reg = (l_reg * gb).sum() / gb.sum().clamp(min=1.0)
    return PESO_GATE * l_gate + l_reg


def perda_val(modelo, idx_v):
    modelo.eval()
    tot = 0.0
    with torch.no_grad():
        for k in range(0, len(idx_v), 256):
            ib = idx_v[k:k + 256]
            xb = torch.from_numpy(X[ib]).to(dev)
            tz = torch.from_numpy(Tz[ib]).to(dev)
            gb = torch.from_numpy(gate[ib]).to(dev)
            tot += perda_total(modelo(xb), tz, gb).item() * len(ib)
    return tot / len(idx_v)


def _prever(modelo, Xs, bs=256):
    """Saídas em unidades originais: [prob. de presença, x, y, raio, topo]."""
    modelo.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(Xs), bs):
            xb = torch.from_numpy(np.ascontiguousarray(Xs[k:k + bs])).to(dev)
            out.append(modelo(xb).cpu().numpy())
    s = np.concatenate(out).astype(np.float64)
    s[:, 0] = 1.0 / (1.0 + np.exp(-s[:, 0]))
    s[:, 1:] = s[:, 1:] * sd + mu
    return s


def prever(modelo, Xs, tta=False):
    p = _prever(modelo, Xs)
    if tta:
        q = _prever(modelo, Xs[..., ::-1])
        q[:, 1] = C_FLIP - q[:, 1]                  # volta o x do B-scan espelhado
        p = (p + q) / 2
    return p


# %% [markdown]
# ## Treino e avaliação

# %%
def treinar():
    modelo = CNN().to(dev)
    print('parâmetros:', sum(p.numel() for p in modelo.parameters()))
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)
    melhor, estado, espera = np.inf, None, 0

    for ep in range(1, EPOCAS + 1):
        modelo.train()
        perm = rng.permutation(i_tr)
        soma = 0.0
        for k in range(0, len(perm), LOTE):
            ib = perm[k:k + LOTE]
            xb, tb = aumentar(X[ib], T[ib])
            xt = torch.from_numpy(xb).to(dev)
            xt = xt + 0.05 * torch.randn_like(xt)
            tz = torch.from_numpy(((tb - mu) / sd).astype(np.float32)).to(dev)
            gb = torch.from_numpy(gate[ib]).to(dev)
            opt.zero_grad()
            loss = perda_total(modelo(xt), tz, gb)
            loss.backward()
            opt.step()
            soma += loss.item() * len(ib)
        sched.step()

        lv = perda_val(modelo, i_va)
        if lv < melhor - 1e-4:
            melhor, estado, espera = lv, copy.deepcopy(modelo.state_dict()), 0
        else:
            espera += 1
        if ep == 1 or ep % 5 == 0:
            print(f'época {ep:3d} | perda treino {soma / len(perm):.4f} | perda val {lv:.4f} | melhor {melhor:.4f}')
        if espera >= PACIENCIA:
            print(f'parada antecipada na época {ep}')
            break

    modelo.load_state_dict(estado)
    return modelo


def baseline_hgb():
    """HGB direto nas features do fe3.py (sem cadeia), mesmo split: referência para a CNN."""
    try:
        feats = pd.read_csv(FEATURES_CSV).set_index('id').loc[ids]
    except Exception as e:
        print('baseline HGB ignorado:', repr(e))
        return None
    i_fit = np.r_[i_tr, i_va]
    i_fit = i_fit[gate[i_fit] == 1]
    B = np.zeros((len(i_te), 3))
    for c in range(3):
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=2000, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=50, random_state=SEED)
        B[:, c] = m.fit(feats.iloc[i_fit], T[i_fit, c]).predict(feats.iloc[i_te])
    return B


def tabela(titulo, mask, P, B):
    Tt, g = T[i_te], mask
    yt = {'pos_x': Tt[g, 0], 'pos_y': Tt[g, 1], 'radius': Tt[g, 2], 'topo (y+raio)': Tt[g, 3],
          'y = topo - raio': Tt[g, 1]}
    yp = {'pos_x': P[g, 1], 'pos_y': P[g, 2], 'radius': P[g, 3], 'topo (y+raio)': P[g, 4],
          'y = topo - raio': P[g, 4] - P[g, 3]}
    yb = {} if B is None else {'pos_x': B[g, 0], 'pos_y': B[g, 1], 'radius': B[g, 2],
                               'topo (y+raio)': B[g, 1] + B[g, 2]}
    linhas = []
    for k in yt:
        rmse = np.sqrt(mean_squared_error(yt[k], yp[k]))
        linhas.append({'alvo': k, 'R2 CNN': r2_score(yt[k], yp[k]), 'RMSE': rmse,
                       'MAE': mean_absolute_error(yt[k], yp[k]), 'RMSE/desvio': rmse / yt[k].std(),
                       'R2 HGB': r2_score(yt[k], yb[k]) if k in yb else np.nan})
    print(f'\n=== {titulo} (n = {int(g.sum())}) ===')
    print(pd.DataFrame(linhas).round(4).to_string(index=False))


def avaliar_teste(P, B):
    gte = gate[i_te].astype(int)
    gp = (P[:, 0] >= 0.5).astype(int)
    if COL_GATE:
        print('=== Presença (teste) ===')
        print(confusion_matrix(gte, gp))
        print(classification_report(gte, gp, digits=3))
    tabela('Alvos nas realmente presentes', gte == 1, P, B)
    tabela('Ponta a ponta (presentes que passaram no gate)', (gte == 1) & (gp == 1), P, B)
    pd.DataFrame({'id': ids[i_te], 'gate': gte, 'gate_proba': P[:, 0],
                  'x_hat': P[:, 1], 'y_hat': P[:, 2], 'r_hat': P[:, 3], 'topo_hat': P[:, 4],
                  'pos_x': T[i_te, 0], 'pos_y': T[i_te, 1], 'radius': T[i_te, 2]}) \
        .to_csv('cnn_pred_teste.csv', index=False)


# %%
modelo = treinar()
torch.save({'estado': modelo.state_dict(), 'mu': mu.tolist(), 'sd': sd.tolist(), 'c_flip': float(C_FLIP)},
           'cnn_regressao.pt')
P = prever(modelo, X[i_te], tta=TTA)
avaliar_teste(P, baseline_hgb())
