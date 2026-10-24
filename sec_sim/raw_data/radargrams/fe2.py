
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import h5py


def extrair_features(Ez, dt, dx):
    """
    Ez: matriz numérica com formato (N_t, N_x)
    dt: intervalo temporal entre amostras [s]
    dx: intervalo espacial entre traços [m]

    Retorna um dicionário de features por B-scan.
    """

    E = np.asarray(Ez, dtype=float)

    if E.ndim != 2 or min(E.shape) < 2:
        raise ValueError("Ez deve ser uma matriz 2D válida.")

    if not np.all(np.isfinite(E)):
        raise ValueError("Ez contém NaN ou valores infinitos.")

    Nt, Nx = E.shape
    A = np.abs(E)
    E2 = E**2

    # Estatísticas globais
    features = {
        "mean": np.mean(E),
        "std": np.std(E),
        "variance": np.var(E),
        "amp_max": np.max(A),
        "amp_mean": np.mean(A),
        "rms": np.sqrt(np.mean(E2)),
        "energy": np.sum(E2),
        "skewness": skew(E.ravel()),
        "kurtosis": kurtosis(E.ravel()),
        "median_abs": np.median(A),
        "q90_abs": np.quantile(A, 0.90),
        "q99_abs": np.quantile(A, 0.99),
    }

    # Posição e tempo do maior pico absoluto
    it, ix = np.unravel_index(np.argmax(A), A.shape)

    features["x_peak"] = ix * dx
    features["t_peak"] = it * dt

    # Estatísticas por A-scan (eixo temporal)
    energia_x = np.sum(E2, axis=0)
    rms_x = np.sqrt(np.mean(E2, axis=0))
    amp_x = np.max(A, axis=0)

    features["energy_x_mean"] = np.mean(energia_x)
    features["energy_x_std"] = np.std(energia_x)
    features["energy_x_max"] = np.max(energia_x)
    features["rms_x_mean"] = np.mean(rms_x)
    features["rms_x_std"] = np.std(rms_x)
    features["amp_x_std"] = np.std(amp_x)

    # Centroide e largura espacial da energia
    x = np.arange(Nx) * dx
    soma = np.sum(energia_x)

    if soma > 0:
        x_centro = np.sum(x * energia_x) / soma
        largura_x = np.sqrt(
            np.sum(energia_x * (x - x_centro)**2) / soma
        )
    else:
        x_centro = 0.0
        largura_x = 0.0

    features["x_energy_centroid"] = x_centro
    features["x_energy_width"] = largura_x

    # Correlação entre A-scans vizinhos
    correlacoes = []

    for i in range(Nx - 1):
        a = E[:, i]
        b = E[:, i + 1]

        if np.std(a) > 0 and np.std(b) > 0:
            correlacoes.append(np.corrcoef(a, b)[0, 1])

    features["corr_adj_mean"] = (
        np.mean(correlacoes) if correlacoes else 0.0
    )
    features["corr_adj_std"] = (
        np.std(correlacoes) if correlacoes else 0.0
    )

    # Features espectrais médias dos A-scans
    espectro = np.abs(np.fft.rfft(E, axis=0))**2
    frequencias = np.fft.rfftfreq(Nt, d=dt)

    espectro_medio = np.mean(espectro, axis=1)

    # Ignora o componente DC quando procura o pico espectral
    if len(frequencias) > 1:
        k = 1 + np.argmax(espectro_medio[1:])
        features["freq_peak"] = frequencias[k]

        soma_espectral = np.sum(espectro_medio)

        if soma_espectral > 0:
            features["freq_centroid"] = (
                np.sum(frequencias * espectro_medio)
                / soma_espectral
            )
        else:
            features["freq_centroid"] = 0.0
    else:
        features["freq_peak"] = 0.0
        features["freq_centroid"] = 0.0

    return features

N = 3000
x = []
dt = 3e-9/637
dx = 0.002
tf = 1.6e-9
# Nf = int(tf/dt)
Nf = 360
T = 3e-9

# x = []

# for i in range(1,N+1):
#     with h5py.File(f'Bscan{i}__merged.out', 'r') as f:
#         rx = f['rxs/rx1']
#         ez = rx['Ez'][:]

#     ez_corr = np.zeros((637,60)) ### ROI - Region of Interest  
#     ez_corr = ez[Nf+1:637,0:60]

#     lista_features = [extrair_features(ez_corr, dt=dt, dx=dx)]

#     x.append(lista_features)

# X = pd.DataFrame(x)
# X.to_csv('Features.csv',index=False)

# print(X.shape)
# print(X.head())

lista_features = []

for i in range(1, N + 1):

    arquivo = f'Bscan{i}__merged.out'

    with h5py.File(arquivo, 'r') as f:
        rx = f['rxs/rx1']
        ez = rx['Ez'][:]

    # Recorte da região de interesse - ROI
    ez_corr = ez[Nf + 1:637, 0:60]

    # Extrai um dicionário de features por B-scan
    features = extrair_features(
        ez_corr,
        dt=dt,
        dx=dx
    )

    # Identificação da simulação
    features['id'] = i

    # Adiciona o dicionário à lista
    lista_features.append(features)

    if i % 100 == 0:
        print(f'{i}/{N} B-scans processados')

# Converte os dicionários em uma tabela
X = pd.DataFrame(lista_features)

# Coloca o identificador na primeira coluna
colunas = ['id'] + [c for c in X.columns if c != 'id']
X = X[colunas]

# Exporta para CSV
X.to_csv('features_teste2.csv', index=False)

print('\nDimensões da tabela:', X.shape)
print(X.head())
print('\nCSV salvo em Features.csv')