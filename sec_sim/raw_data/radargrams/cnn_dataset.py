"""
cnn_dataset.py - Monta o dataset de B-scans completos (ROI) para a CNN.

Para cada arquivo: remove o traço mediano, calcula o envelope (Hilbert), recorta o ROI do fe3.py,
decima o tempo (média de blocos) e normaliza cada B-scan por estatísticas robustas.
Canais: 0 = campo Ez sem fundo, 1 = envelope.

Saída: bscans_cnn.npz com X (N, 2, Nt/DS, NX) float32 e ids (N,).
"""

import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import hilbert

N = 5000
ARQUIVO = 'Bscan{}__merged.out'
I0, I1 = 361, 637          # igual ao fe3.py (mude junto com ele se reextrair, ex.: I0 = 240)
NX = 60
DS = 4                     # decimação no tempo (média de blocos de DS amostras)


def processar(i):
    with h5py.File(ARQUIVO.format(i), 'r') as h:
        ez = h['rxs/rx1/Ez'][:, :NX].astype(float)
    nfull = ez.shape[0]

    ez = ez - np.median(ez, axis=1, keepdims=True)                   # remove o fundo
    env = np.abs(hilbert(ez, N=2 * nfull, axis=0))[:nfull]           # envelope (zero-padding)

    E, A = ez[I0:I1], env[I0:I1]
    nt = (E.shape[0] // DS) * DS
    E = E[:nt].reshape(-1, DS, E.shape[1]).mean(axis=1)
    A = A[:nt].reshape(-1, DS, A.shape[1]).mean(axis=1)

    # contraste relativo ao clutter do próprio B-scan + compressão da dinâmica
    mad = 1.4826 * np.median(np.abs(E - np.median(E)))
    E = np.arcsinh(E / (mad + 1e-30))
    A = np.log1p(A / (np.median(A) + 1e-30))
    return np.stack([E, A]).astype(np.float32)


def seguro(i):
    try:
        return processar(i), None
    except Exception as e:
        return None, repr(e)


def main(n=N):
    X, falhas = [], []
    with ProcessPoolExecutor() as ex:
        for k, (x, erro) in enumerate(ex.map(seguro, range(1, n + 1), chunksize=16), 1):
            if erro:
                falhas.append((k, erro))
            else:
                X.append(x)
            if k % 500 == 0:
                print(f'{k}/{n}')
    if falhas:
        raise RuntimeError(f'{len(falhas)} arquivos falharam (o dataset ficaria desalinhado), '
                           f'por exemplo: {falhas[:3]}')

    X = np.stack(X)
    np.savez_compressed('bscans_cnn.npz', X=X, ids=np.arange(1, n + 1))
    print('X:', X.shape, X.dtype, f'| {X.nbytes / 1e6:.0f} MB')
    print('média/desvio por canal:', X.mean(axis=(0, 2, 3)).round(3), X.std(axis=(0, 2, 3)).round(3))


if __name__ == '__main__':
    main()
