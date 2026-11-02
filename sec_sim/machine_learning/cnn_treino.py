"""
cnn_treino.py - CNN 2D para detectar o cilindro no B-scan (classe 0/1/2 ou presença 0/1).

Entradas: bscans_cnn.npz (cnn_dataset.py), rótulos em databasev5_presenca.csv (presenca_adaptada.py)
e features5k.csv (fe3.py, só para o baseline HGB no MESMO split).
Rode primeiro com RAPIDO = True (poucos segundos) para checar o ambiente.
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
COL_ROTULO = 'classe'                    # 'classe' (0/1/2) ou 'presence' (0/1)
CSV_ROTULOS = 'databasev5_presenca.csv'
EPOCAS, LOTE, LR = 40, 64, 2e-3
RAPIDO = False                           # True: teste de fumaça (300 amostras, 2 épocas)

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------ dados
d = np.load('bscans_cnn.npz')
X, ids = d['X'], d['ids']

df = pd.read_csv(CSV_ROTULOS)
if 'id' in df.columns:
    df = df.set_index('id').loc[ids]
else:
    assert len(df) == len(ids), 'CSV e dataset com tamanhos diferentes'
    print('AVISO: o CSV não tem coluna id; assumindo a mesma ordem dos arquivos Bscan1..N')
y = df[COL_ROTULO].to_numpy().astype(np.int64)
n_cls = int(y.max()) + 1

if RAPIDO:
    sub = rng.choice(len(y), 300, replace=False)
    X, y, ids = X[sub], y[sub], ids[sub]
    EPOCAS = 2

idx = np.arange(len(y))
i_tmp, i_te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
i_tr, i_va = train_test_split(i_tmp, test_size=0.15 / 0.85, stratify=y[i_tmp], random_state=SEED)
print(f'X {X.shape} | classes {np.bincount(y)} | treino/val/teste = {len(i_tr)}/{len(i_va)}/{len(i_te)} | {dev}')


# ------------------------------------------------------------------ modelo
class CNN(nn.Module):
    """4 blocos conv + pooling global. Dois canais de coordenadas (tempo e posição) são
    concatenados na entrada, porque a classe depende da posição ABSOLUTA do ápice
    (dentro da abertura e da janela de tempo) e a convolução sozinha é invariante a translação."""

    def __init__(self, n_cls, in_ch=2):
        super().__init__()

        def bloco(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(bloco(in_ch + 2, 16), bloco(16, 32), bloco(32, 64), bloco(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        B, _, H, W = x.shape
        t = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        p = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return self.head(self.features(torch.cat([x, t, p], dim=1)))


def prever(modelo, Xs, bs=256):
    modelo.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(Xs), bs):
            xb = torch.from_numpy(Xs[k:k + bs]).to(dev)
            out.append(torch.softmax(modelo(xb), 1).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ treino
modelo = CNN(n_cls).to(dev)
print('parâmetros:', sum(p.numel() for p in modelo.parameters()))
opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)
perda = nn.CrossEntropyLoss()
melhor, estado = -1.0, None

for ep in range(1, EPOCAS + 1):
    modelo.train()
    perm = rng.permutation(i_tr)
    for k in range(0, len(perm), LOTE):
        ib = perm[k:k + LOTE]
        xb = torch.from_numpy(X[ib]).to(dev)
        yb = torch.from_numpy(y[ib]).to(dev)

        # aumento: espelhamento horizontal (preserva a classe, pois a abertura é simétrica) + ruído leve
        flip = torch.rand(len(ib), device=dev) < 0.5
        xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
        xb = xb + 0.05 * torch.randn_like(xb)

        opt.zero_grad()
        loss = perda(modelo(xb), yb)
        loss.backward()
        opt.step()
    sched.step()

    f1 = f1_score(y[i_va], prever(modelo, X[i_va]).argmax(1), average='macro')
    if f1 > melhor:
        melhor, estado = f1, copy.deepcopy(modelo.state_dict())
    if ep == 1 or ep % 5 == 0:
        print(f'época {ep:3d} | perda (último lote) {loss.item():.3f} | F1 macro (val) {f1:.3f}')

modelo.load_state_dict(estado)
torch.save(estado, 'cnn_presenca.pt')

# ------------------------------------------------------------------ avaliação no teste (uma vez)
pred = prever(modelo, X[i_te]).argmax(1)
print('\n=== CNN (teste) ===')
print(confusion_matrix(y[i_te], pred))
print(classification_report(y[i_te], pred, digits=3))

# baseline: HGB nas features do fe3.py, mesmo split (treino + validação juntos)
feats = pd.read_csv('features5k.csv').set_index('id').loc[ids]
i_fit = np.r_[i_tr, i_va]
hgb = HistGradientBoostingClassifier(random_state=SEED).fit(feats.iloc[i_fit], y[i_fit])
pred_h = hgb.predict(feats.iloc[i_te])
print('=== HGB nas features (teste) ===')
print(confusion_matrix(y[i_te], pred_h))
print(classification_report(y[i_te], pred_h, digits=3))

if n_cls == 3:
    print('Decisões binárias derivadas (acurácia no teste):')
    for nome, f in [('presença (classe >= 1)', lambda a: a >= 1), ('ápice visível (classe == 2)', lambda a: a == 2)]:
        print(f'  {nome:28s} CNN = {(f(pred) == f(y[i_te])).mean():.3f} | HGB = {(f(pred_h) == f(y[i_te])).mean():.3f}')
