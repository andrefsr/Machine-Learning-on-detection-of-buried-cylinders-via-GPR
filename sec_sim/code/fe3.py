"""
fe3.py - Extração de features de B-scans do gprMax para prever raio, x e y
do cilindro enterrado.

Para cada B-scan (Ez, formato [tempo, traço]):
  1. remove o fundo (traço mediano) e calcula o envelope (Hilbert);
  2. localiza a hipérbole por empilhamento (busca grossa + busca fina);
  3. extrai features ao longo da hipérbole, no traço do ápice e na banda ao redor;
  4. salva um recorte 2D centrado no ápice (opcional, para uma CNN).

Saídas: features5k.csv e patches5k.npy (mesma ordem das linhas do CSV).
"""

import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis

# ============================== CONFIGURAÇÃO ==============================
N = 5000
ARQUIVO = 'Bscan{}__merged.out'
DT = 3e-9 / 637            # s
DX = 0.002                 # m entre traços
I0, I1 = 361, 637          # janela temporal em amostras (mesmo ROI do fe2.py)
NX = 60                    # nº de traços usados

C0 = 299792458.0
# Velocidades candidatas para a hipérbole (εr de 3 a 25). AJUSTE ao seu solo.
V_GRID = C0 / np.sqrt(np.linspace(3, 25, 12))

SIG_T, SIG_X = 8, 1        # suavização do envelope na busca da hipérbole
W_AMP = 15                 # meia-janela (amostras) do pico de amplitude em cada traço
W_BANDA = 32               # meia-largura (amostras) da banda ao redor da hipérbole
JT = 64                    # meia-janela temporal (amostras) do espectro no ápice
OFFSETS = (0.01, 0.02, 0.03, 0.04)   # afastamentos laterais [m] do perfil de amplitude

SALVAR_PATCHES = True
PATCH_ANTES, PATCH_DEPOIS, PATCH_LADO = 32, 96, 20   # amostras antes/depois; traços de cada lado


# ============================== FUNÇÕES AUXILIARES ==============================
def empilhar(env, x, t, x0s, t0s, vs):
    """Média do envelope ao longo de hipérboles t(x) = sqrt(t0² + (2(x-x0)/v)²).

    Retorna score com shape (len(x0s), len(t0s), len(vs)). Hipérboles que saem
    da janela temporal em mais de 50% dos traços recebem score 0.
    """
    Nt, Nx = env.shape
    dt = t[1] - t[0]
    dxm = x[None, :] - x0s[:, None]                                   # (n_x0, Nx)
    th = np.sqrt(t0s[None, :, None, None] ** 2
                 + (2 * dxm[:, None, None, :] / vs[None, None, :, None]) ** 2)
    idx = np.rint((th - t[0]) / dt).astype(np.intp)                   # (n_x0, n_t0, n_v, Nx)
    valido = (idx >= 0) & (idx < Nt)
    vals = env[np.clip(idx, 0, Nt - 1), np.arange(Nx)]
    nval = valido.sum(-1)
    score = (vals * valido).sum(-1) / np.maximum(nval, 1)
    score[nval < 0.5 * Nx] = 0.0
    return score


def ajustar_hiperbole(env, x, t):
    """Busca grossa em todo o B-scan e busca fina ao redor do melhor ponto."""
    dt = t[1] - t[0]

    x0s, t0s = x[::2], t[::4]
    s = empilhar(env, x, t, x0s, t0s, V_GRID)
    a, b, c = np.unravel_index(np.argmax(s), s.shape)
    x0, t0, v = x0s[a], t0s[b], V_GRID[c]

    x0s = x0 + DX * np.arange(-3, 3.5, 0.5)       # ±3 traços, passo de meio traço
    t0s = t0 + dt * np.arange(-8, 9)
    vs = v * np.linspace(0.85, 1.15, 9)
    s = empilhar(env, x, t, x0s, t0s, vs)
    a, b, c = np.unravel_index(np.argmax(s), s.shape)
    return x0s[a], t0s[b], vs[c], s[a, b, c]


def largura_contigua(y, i0, frac):
    """Nº de amostras contíguas em torno de i0 com y >= frac * y[i0] (NaN interrompe)."""
    lim = frac * y[i0]
    esq = i0
    while esq - 1 >= 0 and y[esq - 1] >= lim:
        esq -= 1
    dire = i0
    while dire + 1 < len(y) and y[dire + 1] >= lim:
        dire += 1
    return dire - esq + 1


def espectro(M, nfft=512):
    """M: (n_t, n_traços). Retorna (centroide, frequência de pico, largura de banda) em Hz."""
    w = np.hanning(M.shape[0])[:, None]
    P = np.mean(np.abs(np.fft.rfft(M * w, n=nfft, axis=0)) ** 2, axis=1)
    f = np.fft.rfftfreq(nfft, d=DT)
    P[0] = 0.0                                    # ignora DC
    s = P.sum()
    if s <= 0:
        return 0.0, 0.0, 0.0
    fc = (f * P).sum() / s
    bw = np.sqrt(((f - fc) ** 2 * P).sum() / s)
    return fc, f[np.argmax(P)], bw


def recortar(M, i0, j0):
    """Recorte de tamanho fixo centrado em (i0, j0); completa com zeros fora da matriz."""
    out = np.zeros((PATCH_ANTES + PATCH_DEPOIS, 2 * PATCH_LADO + 1), dtype=np.float32)
    ta, tb = i0 - PATCH_ANTES, i0 + PATCH_DEPOIS
    xa, xb = j0 - PATCH_LADO, j0 + PATCH_LADO + 1
    tac, tbc = max(ta, 0), min(tb, M.shape[0])
    xac, xbc = max(xa, 0), min(xb, M.shape[1])
    out[tac - ta:tbc - ta, xac - xa:xbc - xa] = M[tac:tbc, xac:xbc]
    return out


# ============================== EXTRAÇÃO ==============================
def extrair_features(ez):
    """ez: (N_t_total, N_x). Retorna (dict de features, patch (2, nt, nx) ou None)."""
    Nfull = ez.shape[0]
    if Nfull < I1:
        raise ValueError(f'Ez tem {Nfull} amostras de tempo; esperado >= {I1}.')
    if not np.all(np.isfinite(ez)):
        raise ValueError('Ez contém NaN ou infinito.')

    # ---- pré-processamento: remove o fundo e calcula o envelope no registro completo
    # (zero-padding no Hilbert evita artefatos de borda circular) e só depois recorta o ROI
    bg = np.median(ez, axis=1, keepdims=True)
    ez_bg = ez - bg
    env_full = np.abs(hilbert(ez_bg, N=2 * Nfull, axis=0))[:Nfull]

    E = ez_bg[I0:I1]
    env = env_full[I0:I1]
    bg_c = bg[I0:I1, 0]
    Nt, Nx = E.shape
    t = np.arange(I0, I1) * DT                    # tempo absoluto [s]
    x = np.arange(Nx) * DX                        # posição ao longo do B-scan [m]
    E2 = E ** 2

    # ---- localização: pico simples do envelope e ajuste da hipérbole
    env_s = gaussian_filter(env, (SIG_T, SIG_X))
    it_pico, ix_pico = np.unravel_index(np.argmax(env_s), env_s.shape)
    x0, t0, v, score = ajustar_hiperbole(env_s, x, t)

    j0 = int(np.clip(np.rint(x0 / DX), 0, Nx - 1))
    i0 = int(np.clip(np.rint((t0 - t[0]) / DT), 0, Nt - 1))

    f = {
        # localização
        'x_pico_env': ix_pico * DX,
        't_pico_env': t[it_pico],
        'x0_fit': x0,
        't0_fit': t0,
        'v_fit': v,
        'd_apex': v * t0 / 2,                     # profundidade aparente do topo
        'score_fit': score,
        'score_contraste': score / env.mean(),
        'flag_borda_x': float(j0 < 3 or j0 > Nx - 4),
        'flag_borda_t': float(i0 < 10 or i0 > Nt - 10),
    }

    # ---- amplitude ao longo da hipérbole ajustada
    t_hyp = np.sqrt(t0 ** 2 + (2 * (x - x0) / v) ** 2)
    i_hyp = np.rint((t_hyp - t[0]) / DT).astype(int)
    a = np.full(Nx, np.nan)
    for j in np.where((i_hyp >= 0) & (i_hyp < Nt))[0]:
        lo, hi = max(i_hyp[j] - W_AMP, 0), min(i_hyp[j] + W_AMP + 1, Nt)
        a[j] = env[lo:hi, j].max()
    a_max = np.nanmax(a)
    jm = int(np.nanargmax(a))

    f['a_max'] = a_max
    f['larg_x_50'] = largura_contigua(a, jm, 0.50) * DX
    f['larg_x_25'] = largura_contigua(a, jm, 0.25) * DX
    for d in OFFSETS:
        vals = []
        for sgn in (-1, 1):
            j = int(np.rint((x0 + sgn * d) / DX))
            if 0 <= j < Nx and np.isfinite(a[j]):
                vals.append(a[j])
        f[f'a_rel_{int(round(d * 1000))}mm'] = np.mean(vals) / a_max if vals else np.nan

    # ---- banda ao redor da hipérbole vs. clutter
    linhas = np.arange(Nt)[:, None]
    banda = np.abs(linhas - i_hyp[None, :]) <= W_BANDA
    f['frac_energia_banda'] = E2[banda].sum() / E2.sum()
    f['rms_banda'] = np.sqrt(E2[banda].mean())
    f['clutter_rms'] = np.sqrt(E2[~banda].mean())
    f['clutter_env_med'] = np.median(env[~banda])
    f['snr_env'] = a_max / f['clutter_env_med']
    f['snr_rms'] = a_max / f['clutter_rms']

    # ---- traço do ápice: largura temporal do pulso e segundo eco
    env_tr = env[:, j0]
    lo, hi = max(i0 - W_AMP, 0), min(i0 + W_AMP + 1, Nt)
    ip = lo + int(np.argmax(env_tr[lo:hi]))
    f['larg_t_50'] = largura_contigua(env_tr, ip, 0.50) * DT
    f['larg_t_25'] = largura_contigua(env_tr, ip, 0.25) * DT

    ini = max(i0 - 60, 0)
    seg = env_tr[ini:]
    picos, props = find_peaks(seg, prominence=0.1 * seg.max())
    if len(picos) >= 2:
        top2 = np.sort(picos[np.argsort(props['prominences'])[-2:]])
        f['eco2_atraso'] = (top2[1] - top2[0]) * DT
        f['eco2_razao'] = seg[top2[1]] / seg[top2[0]]
    else:
        f['eco2_atraso'] = np.nan
        f['eco2_razao'] = np.nan

    # ---- espectro no ápice vs. espectro global do ROI
    ta, tb = max(i0 - JT, 0), min(i0 + JT, Nt)
    fc_ap, fp_ap, bw_ap = espectro(E[ta:tb, max(j0 - 1, 0):j0 + 2])
    fc_gl, fp_gl, bw_gl = espectro(E)
    f['freq_centroid_apex'] = fc_ap
    f['freq_peak_apex'] = fp_ap
    f['freq_bw_apex'] = bw_ap
    f['freq_centroid_global'] = fc_gl
    f['freq_centroid_razao'] = fc_ap / fc_gl if fc_gl > 0 else np.nan
    f['freq_bw_razao'] = bw_ap / bw_gl if bw_gl > 0 else np.nan

    # ---- estatísticas globais (após remover o fundo) e covariáveis do fundo
    f['rms_global'] = np.sqrt(E2.mean())
    f['kurtosis_global'] = kurtosis(E.ravel())
    f['q99_env'] = np.quantile(env, 0.99)
    f['apex_sobre_rms'] = a_max / f['rms_global']
    f['bg_rms'] = np.sqrt(np.mean(bg_c ** 2))
    f['bg_t_pico'] = t[np.argmax(np.abs(bg_c))]

    patch = None
    if SALVAR_PATCHES:
        patch = np.stack([recortar(E, i0, j0), recortar(env, i0, j0)])

    return {k: float(val) for k, val in f.items()}, patch


# ============================== EXECUÇÃO ==============================
def processar(i):
    try:
        with h5py.File(ARQUIVO.format(i), 'r') as h:
            ez = h['rxs/rx1/Ez'][:, :NX].astype(float)
        feats, patch = extrair_features(ez)
        feats['id'] = i
        return i, feats, patch, None
    except Exception as e:
        return i, None, None, repr(e)


def main(n=N):
    with h5py.File(ARQUIVO.format(1), 'r') as h:
        print('Ez shape:', h['rxs/rx1/Ez'].shape,
              '| dt no arquivo:', h.attrs.get('dt'), '| dt usado:', DT)

    linhas, patches, falhas = [], [], []
    with ProcessPoolExecutor() as ex:
        for k, (i, feats, patch, erro) in enumerate(
                ex.map(processar, range(1, n + 1), chunksize=16), 1):
            if erro:
                falhas.append((i, erro))
                continue
            linhas.append(feats)
            patches.append(patch)
            if k % 100 == 0:
                print(f'{k}/{n} B-scans processados')

    X = pd.DataFrame(linhas)
    X = X[['id'] + [c for c in X.columns if c != 'id']]
    X.to_csv('features5k.csv', index=False)
    if SALVAR_PATCHES:
        np.save('patches5k.npy', np.stack(patches))

    print('\nDimensões da tabela:', X.shape)
    print('NaN por coluna (só as que têm):')
    print(X.isna().sum()[X.isna().sum() > 0])
    if falhas:
        print(f'\n{len(falhas)} arquivos falharam, por exemplo:', falhas[:3])
    print('\nCSV salvo em features5k.csv')


if __name__ == '__main__':
    main()
