"""
fe_raio.py - Features para prever o RAIO do cilindro, dado que a posição (x, y) já foi prevista.

A posição prevista é usada para (1) restringir a busca da hipérbole (x perto de x_prev, tempo do ápice
dentro da janela compatível com y_prev e com raios de 4 a 24 mm) e (2) montar features que dependem da
profundidade. Depois, o eco é alinhado ao longo da hipérbole e empilhado coerentemente, o que dá uma
forma de onda limpa do alvo, da qual saem amplitude, forma, espectro e padrão lateral.

Entrada : POS_CSV com colunas  id, x_prev, y_prev  (y_prev = CENTRO do cilindro, y do gprMax;
          x_prev em coordenadas do solo) e, opcionalmente, soil_type. Para gerar o CSV no notebook:

              mask = y1['presence_pred'].to_numpy() == 1
              pd.DataFrame({'id': df.index[mask] + 1,              # arquivo Bscan{id}
                            'x_prev': y2['pos_x_pred'].to_numpy(),
                            'y_prev': y3['pos_y_pred'].to_numpy()}).to_csv('posicoes_previstas.csv', index=False)

          Diagnóstico: rode de novo com as posições REAIS (x_prev = pos_x, y_prev = pos_y) e compare o R²
          do raio. Se subir muito, o erro de posição é o gargalo; se não, o limite está no próprio sinal.
Saída   : SAIDA_CSV com uma linha por B-scan (id + features).
"""

import warnings
import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter

# ============================== CONFIGURAÇÃO ==============================
ARQUIVO = 'Bscan{}__merged.out'
POS_CSV = 'posicoes_previstas.csv'
SAIDA_CSV = 'features_raio.csv'
DT = 3e-9 / 637
DX = 0.002
NX = 60
X_INI = 0.08                 # posição do 1º traço no solo [m]
I0, I1 = 240, 637            # ROI de localização (o fe3.py usava 361, que corta os ápices rasos)

# calibração do tempo do ápice por solo (presenca_cilindro_v2.py):  t0 = A0 - 2*(y_centro + raio)/V
V_EFF = {1: 0.862e8, 2: 0.849e8, 3: 0.887e8}
A0 = {1: 4.763e-9, 2: 4.932e-9, 3: 4.826e-9}
Y0 = {s: V_EFF[s] * A0[s] / 2 for s in V_EFF}        # plano "virtual" de tempo zero: y_topo = Y0 - V*t0/2
V_NOM = float(np.mean(list(V_EFF.values())))
Y0_NOM = float(np.mean(list(Y0.values())))
USAR_SOLO = False            # True: usa a calibração do solo de cada linha (precisa da coluna soil_type)

R_MIN, R_MAX = 0.004, 0.024  # faixa de raios considerada ao restringir o tempo do ápice [m]
MARG_T = 0.25e-9             # folga na janela do tempo do ápice [s]
SIG_T, SIG_X = 8, 1          # suavização do envelope na busca
W = 96                       # meia-janela (amostras) do eco alinhado
W_AMP = 15                   # meia-janela do pico de amplitude em cada traço
W_BANDA = 32                 # meia-largura da banda em torno da hipérbole
RHO_STACK = 0.02             # só traços a até 2 cm do ápice entram no empilhamento [m]
JT, JT_REF = 64, 96          # meia-janelas do espectro (eco e onda direta)
OFFS = (0.01, 0.02, 0.03, 0.04, 0.05)


# ============================== AUXILIARES ==============================
def empilhar(env, x, t, x0s, t0s, vs):
    """Média do envelope ao longo de hipérboles t(x) = sqrt(t0² + (2(x-x0)/v)²) -> (n_x0, n_t0, n_v)."""
    Nt, Nx = env.shape
    dt = t[1] - t[0]
    dxm = x[None, :] - x0s[:, None]
    th = np.sqrt(t0s[None, :, None, None] ** 2
                 + (2 * dxm[:, None, None, :] / vs[None, None, :, None]) ** 2)
    idx = np.rint((th - t[0]) / dt).astype(np.intp)
    valido = (idx >= 0) & (idx < Nt)
    vals = env[np.clip(idx, 0, Nt - 1), np.arange(Nx)]
    nval = valido.sum(-1)
    score = (vals * valido).sum(-1) / np.maximum(nval, 1)
    score[nval < 0.5 * Nx] = 0.0
    return score


def localizar(env_s, x, t, xp, y_prev, V0, Y00):
    """Busca restrita pela posição prevista: grossa, fina e, por fim, só em v (curvatura)."""
    dt = t[1] - t[0]
    tlo = max(2 * (Y00 - y_prev - R_MAX) / V0 - MARG_T, t[0])
    thi = min(2 * (Y00 - y_prev - R_MIN) / V0 + MARG_T, t[-1])
    if thi <= tlo:                                   # janela inválida: busca no ROI todo
        tlo, thi = t[0], t[-1]

    x0s = xp + DX * np.arange(-4, 4.5, 0.5)
    t0s = np.arange(tlo, thi, 2 * dt)
    vs = V0 * np.linspace(0.8, 1.3, 11)
    s = empilhar(env_s, x, t, x0s, t0s, vs)
    a, b, c = np.unravel_index(np.argmax(s), s.shape)
    x0, t0, v = x0s[a], t0s[b], vs[c]

    x0s = x0 + DX * np.arange(-1, 1.01, 0.25)
    t0s = t0 + dt * np.arange(-6, 7)
    vs = v * np.linspace(0.9, 1.1, 9)
    s = empilhar(env_s, x, t, x0s, t0s, vs)
    a, b, c = np.unravel_index(np.argmax(s), s.shape)
    x0, t0, v = x0s[a], t0s[b], vs[c]

    vs = v * np.linspace(0.85, 1.15, 31)
    s = empilhar(env_s, x, t, np.array([x0]), np.array([t0]), vs)[0, 0]
    c = int(np.argmax(s))
    return x0, t0, vs[c], s[c]


def achatar(M, i_c, w):
    """Recorta w amostras antes/depois de i_c[j] (índice no registro completo) em cada traço j.
    Retorna (Nx, 2w+1), com NaN onde a janela sai do registro."""
    Nt, Nx = M.shape
    F = np.full((Nx, 2 * w + 1), np.nan)
    for j in range(Nx):
        a, b = i_c[j] - w, i_c[j] + w + 1
        if b <= 0 or a >= Nt:
            continue
        lo, hi = max(a, 0), min(b, Nt)
        F[j, lo - a:hi - a] = M[lo:hi, j]
    return F


def largura_contigua(y, i0, frac):
    """Nº de amostras contíguas em torno de i0 com y >= frac*y[i0] (NaN interrompe)."""
    lim = frac * y[i0]
    esq = i0
    while esq - 1 >= 0 and y[esq - 1] >= lim:
        esq -= 1
    dire = i0
    while dire + 1 < len(y) and y[dire + 1] >= lim:
        dire += 1
    return dire - esq + 1


def espectro_amp(seg, nfft=1024):
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=nfft))
    return np.fft.rfftfreq(nfft, d=DT), S


def momentos(f, S):
    P = S ** 2
    P[0] = 0.0
    s = P.sum()
    if s <= 0:
        return np.nan, np.nan, np.nan
    fc = (f * P).sum() / s
    return fc, f[np.argmax(P)], np.sqrt(((f - fc) ** 2 * P).sum() / s)


# ============================== EXTRAÇÃO ==============================
def extrair(ez, x_prev, y_prev, V0, Y00):
    nfull = ez.shape[0]
    if nfull < I1:
        raise ValueError(f'Ez tem {nfull} amostras; esperado >= {I1}')

    # pré-processamento: remove o fundo e calcula o envelope no registro completo
    bg = np.median(ez, axis=1, keepdims=True)
    ezb = ez - bg
    envf = np.abs(hilbert(ezb, N=2 * nfull, axis=0))[:nfull]
    A, Eb = envf[I0:I1], ezb[I0:I1]
    Nt, Nx = A.shape
    t = np.arange(I0, I1) * DT
    x = np.arange(Nx) * DX
    A_s = gaussian_filter(A, (SIG_T, SIG_X))

    # --- localização guiada pela posição prevista
    x0, t0, v, score = localizar(A_s, x, t, x_prev - X_INI, y_prev, V0, Y00)
    i0 = int(np.rint((t0 - t[0]) / DT))
    d_ap = V0 * t0 / 2                                   # distância aparente do topo ao plano de tempo zero
    f = {
        'x_prev': x_prev, 'y_prev': y_prev,
        'x0_m': x0 + X_INI, 't0_m': t0, 'dx_m': x0 + X_INI - x_prev, 'score_m': score,
        'd_apice': d_ap,
        'y_topo_t': Y00 - d_ap,                          # topo do cilindro estimado pelo tempo
        'r_geo': Y00 - d_ap - y_prev,                    # topo (tempo) - centro (previsto)
        'v_m': v, 'v_ratio': v / V0,                     # curvatura da hipérbole vs. velocidade nominal
        'flag_borda_t': float(i0 < 10 or i0 > Nt - 10),
        'flag_borda_x': float(x0 < x[0] + 3 * DX or x0 > x[-1] - 3 * DX),
    }

    # --- eco alinhado ao longo da hipérbole ajustada
    rho = x - x0
    i_hyp = np.rint((np.sqrt(t0 ** 2 + (2 * rho / v) ** 2) - t[0]) / DT).astype(int)   # relativo ao ROI
    i_full = i_hyp + I0
    F = achatar(ezb, i_full, W)
    Fe = achatar(envf, i_full, W)
    f['cobertura'] = float(np.mean((i_hyp >= 0) & (i_hyp < Nt)))

    # --- amplitude e padrão lateral ao longo da hipérbole
    a = np.nanmax(Fe[:, W - W_AMP:W + W_AMP + 1], axis=1)
    if not np.isfinite(a).any():
        raise ValueError('hipérbole fora do registro')
    a_pico, jm = np.nanmax(a), int(np.nanargmax(a))
    ap = np.abs(rho) <= 2 * DX
    f['a_pico'] = a_pico
    f['a_apice'] = np.nanmean(a[ap]) if np.isfinite(a[ap]).any() else np.nan
    f['larg_x_50'] = largura_contigua(a, jm, 0.50) * DX
    f['larg_x_25'] = largura_contigua(a, jm, 0.25) * DX
    for d in OFFS:
        vals = [a[j] for j in (int(np.rint((x0 + s * d) / DX)) for s in (-1, 1))
                if 0 <= j < Nx and np.isfinite(a[j])]
        f[f'a_rel_{int(round(d * 1000))}mm'] = np.mean(vals) / a_pico if vals else np.nan
    m = np.isfinite(a) & (a > 0) & (np.abs(rho) <= 0.05)
    f['a_gauss'] = np.polyfit(rho[m] ** 2, np.log(a[m]), 1)[0] if m.sum() >= 6 else np.nan

    # --- forma de onda do eco empilhado coerentemente
    sel = (np.abs(rho) <= RHO_STACK) & np.isfinite(F[:, W])
    if sel.sum() < 3:
        raise ValueError('poucos traços válidos para o empilhamento')
    wv = np.nanmean(F[sel], axis=0)
    f['flag_fim_registro'] = float(np.isnan(wv).any())
    wv = np.nan_to_num(wv)

    c0 = W - 30
    k = c0 + int(np.argmax(np.abs(wv[c0:W + 31])))
    lo, hi = max(k - 48, 0), min(k + 49, len(wv))
    seg = wv[lo:hi]
    wmax, wmin = seg.max(), seg.min()
    ew = np.abs(hilbert(wv, N=512))[:len(wv)]
    ke = lo + int(np.argmax(ew[lo:hi]))
    en = (seg ** 2).sum()

    f['w_env_pico'] = ew[ke]
    f['w_pp'] = wmax - wmin
    f['w_assim'] = (wmax + wmin) / (wmax - wmin) if wmax > wmin else np.nan
    f['w_dt_ext'] = (int(np.argmax(seg)) - int(np.argmin(seg))) * DT
    f['w_sinal'] = float(np.sign(wv[k]))
    f['w_larg_50'] = largura_contigua(ew, ke, 0.50) * DT
    f['w_larg_25'] = largura_contigua(ew, ke, 0.25) * DT
    f['w_cauda'] = (wv[hi:hi + 96] ** 2).sum() / en if en > 0 else np.nan
    Fs = np.nan_to_num(F[sel][:, lo:hi])
    den = sel.sum() * (Fs ** 2).sum()
    f['coer'] = (Fs.sum(axis=0) ** 2).sum() / den if den > 0 else np.nan

    f['eco2_atraso'], f['eco2_razao'] = np.nan, np.nan
    if len(ew) - (ke + 5) > 3:
        pk, _ = find_peaks(ew[ke + 5:], prominence=0.15 * ew[ke])
        if len(pk):
            i2 = ke + 5 + pk[0]
            f['eco2_atraso'], f['eco2_razao'] = (i2 - ke) * DT, ew[i2] / ew[ke]

    # --- amplitude corrigida pela espalhamento (2D: ~1/sqrt(distância)) e contexto de clutter
    banda = np.abs(np.arange(Nt)[:, None] - i_hyp[None, :]) <= W_BANDA
    f['clutter_env_med'] = np.median(A[~banda])
    f['clutter_rms'] = np.sqrt(np.mean(Eb[~banda] ** 2))
    f['frac_energia_banda'] = (Eb[banda] ** 2).sum() / (Eb ** 2).sum()
    f['snr_w'] = ew[ke] / f['clutter_env_med']
    f['a_corr'] = ew[ke] * np.sqrt(d_ap)
    f['log_a'] = np.log(ew[ke] + 1e-30)

    # --- espectro do eco vs. onda direta (proxy do pulso incidente, vem da trilha mediana)
    ka, kb = max(k - JT, 0), min(k + JT + 1, len(wv))
    fr, S_e = espectro_amp(wv[ka:kb])
    ref = bg[:I0, 0]
    kr = int(np.argmax(np.abs(ref)))
    ra, rb = max(kr - JT_REF, 0), min(kr + JT_REF + 1, len(ref))
    _, S_r = espectro_amp(ref[ra:rb])
    fc, fp, bw = momentos(fr, S_e)
    fc_r, _, bw_r = momentos(fr, S_r)
    f['w_fc'], f['w_fpico'], f['w_bw'] = fc, fp, bw
    f['w_fc_razao'] = fc / fc_r if fc_r > 0 else np.nan
    f['w_bw_razao'] = bw / bw_r if bw_r > 0 else np.nan
    mb = (S_r >= 0.2 * S_r.max()) & (fr > 0) & (S_e > 0)
    f['h_inclinacao'] = (np.polyfit(np.log(fr[mb]), np.log(S_e[mb] / S_r[mb]), 1)[0]
                         if mb.sum() >= 5 else np.nan)
    f['e_rel_ref'] = np.log10((wv[ka:kb] ** 2).sum() / ((ref[ra:rb] ** 2).sum() + 1e-300) + 1e-300)

    return {k_: float(val) for k_, val in f.items()}


# ============================== EXECUÇÃO ==============================
def processar(linha):
    i, x_prev, y_prev, solo = linha
    try:
        with h5py.File(ARQUIVO.format(i), 'r') as h:
            ez = h['rxs/rx1/Ez'][:, :NX].astype(float)
        if USAR_SOLO:
            V0, Y00 = V_EFF[solo], Y0[solo]
        else:
            V0, Y00 = V_NOM, Y0_NOM
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            feats = extrair(ez, x_prev, y_prev, V0, Y00)
        feats['id'] = i
        return i, feats, None
    except Exception as e:
        return i, None, repr(e)


def main(pos_csv=POS_CSV, saida=SAIDA_CSV):
    pos = pd.read_csv(pos_csv)
    solo = pos['soil_type'] if 'soil_type' in pos.columns else pd.Series(0, index=pos.index)
    linhas = list(zip(pos['id'].astype(int), pos['x_prev'], pos['y_prev'], solo.astype(int)))

    ok, falhas = [], []
    with ProcessPoolExecutor() as ex:
        for k, (i, feats, erro) in enumerate(ex.map(processar, linhas, chunksize=16), 1):
            if erro:
                falhas.append((i, erro))
            else:
                ok.append(feats)
            if k % 500 == 0:
                print(f'{k}/{len(linhas)}')

    X = pd.DataFrame(ok)
    X = X[['id'] + [c for c in X.columns if c != 'id']]
    X.to_csv(saida, index=False)
    print('\nDimensões:', X.shape)
    nan = X.isna().sum()
    print('NaN por coluna (só as que têm):\n', nan[nan > 0].to_string() if (nan > 0).any() else 'nenhum')
    print(f'flags: borda_t = {X["flag_borda_t"].mean():.3f} | borda_x = {X["flag_borda_x"].mean():.3f} '
          f'| fim_registro = {X["flag_fim_registro"].mean():.3f}')
    if falhas:
        print(f'\n{len(falhas)} arquivos falharam, por exemplo:', falhas[:3])
    print('\nSalvo em', saida)


if __name__ == '__main__':
    main()
