import h5py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt

N = 2000
x = []
dt = 3e-9/637
tf = 1.6e-9
Nf = int(tf/dt)
T = 3e-9

for i in range(1,N+1):
    with h5py.File(f'Bscan{i}__merged.out', 'r') as f:
        rx = f['rxs/rx1']
        ez = rx['Ez'][:]

    ez_corr = np.zeros((637,60)) ### ROI - Region of Interest  
    ez_corr = ez[Nf+1:637,0:60]

    # plt.imshow(ez,cmap='viridis')
    # plt.colorbar()
    # plt.show()

    # plt.imshow(ez_corr,cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    sinal = ez_corr.flatten()

    features = [
        np.mean(sinal),
        np.var(sinal),
        np.max(np.abs(sinal)),
        np.sum(sinal**2),
        skew(sinal),
        kurtosis(sinal)
        #primeiro harmonico de fourier,
        #segundo, etc...
        #BUSCAR MAIS FEATURES
        #HÁ CORRELAÇÃO ENTRE AS FEATURES!!! VARIANCIA PRECISA DA MÉDIA PARA SER CALCULADA (NÃO NECESSARIAMENTE É UM PROBLEMA) - VERIFICAR
    ]

    x.append(features)

df_features = pd.DataFrame(x)
df_features = df_features.rename(columns={0:'Média',1:'Variância',2:'Máximo',3:'Energia',4:'Skewness',5:'Kurtosis'})

df_features.to_csv('Features.csv',index=False)