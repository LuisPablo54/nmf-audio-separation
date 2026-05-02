'''
INTENTO DE OPTIMZIAR, NO EJECUTAR
ESTADO EN PRUEBAS

'''

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from scipy.linalg import svd
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==================== CONFIGURACIÓN GLOBAL ====================
DATA_ROOT = r'data'
OUTPUT_DIR = r'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PER_CAT = 30
N_FFT = 1024
HOP_LENGTH = 512
SR_TARGET = 22050
CATEGORIES = ['music', 'noise', 'speech']

PALETTE = {
    'music': '#4C72B0', 'noise': '#DD8452', 'speech': '#55A868',
    'bg': '#F8F9FA', 'grid': '#DEE2E6', 'text': '#212529'
}
CAT_LABELS = {'music': 'Music', 'noise': 'Noise', 'speech': 'Speech'}

# ==================== FUNCIONES DE CARGA Y PROCESAMIENTO ====================
def scan_files(data_root, categories, n_per_cat=None, exts=('.wav', '.flac', '.mp3')):
    """Escanea carpetas y devuelve DataFrame con metadatos de archivos."""
    rows = []
    for cat in categories:
        path = os.path.join(data_root, cat)
        if not os.path.exists(path):
            continue
        files = []
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(root, f))
        if n_per_cat and len(files) > n_per_cat:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(files), n_per_cat, replace=False)
            files = [files[i] for i in sorted(idx)]
        for f in files:
            rows.append({
                'category': cat,
                'path': f,
                'filename': os.path.basename(f),
                'size_bytes': os.path.getsize(f),
                'subdir': os.path.relpath(os.path.dirname(f), path)
            })
    return pd.DataFrame(rows)

def analyze_file(path, sr_target, n_fft, hop_length):
    """Extrae características y la matriz STFT de un archivo."""
    try:
        info = sf.info(path)
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=30.0)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        return {
            'ok': True,
            'sr_original': info.samplerate,
            'channels': info.channels,
            'duration_s': info.duration,
            'format': info.format,
            'rms_energy': float(np.sqrt(np.mean(y**2))),
            'zero_cross_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            'spectral_centroid_hz': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            'spectral_rolloff_hz': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            'stft_F': S.shape[0],
            'stft_T': S.shape[1],
            'stft_mean': float(S.mean()),
            'stft_max': float(S.max()),
            'stft_matrix': S
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}

def load_and_process_dataset(data_root, categories, n_per_cat, sr_target, n_fft, hop_length, force_recompute=False):
    """Carga en paralelo los archivos y devuelve DataFrame con matriz STFT."""
    csv_path = os.path.join(OUTPUT_DIR, 'metadata_MUSAN_with_stft.csv')
    stft_file = os.path.join(OUTPUT_DIR, 'stft_matrices.npz')
    if not force_recompute and os.path.exists(csv_path) and os.path.exists(stft_file):
        print("Cargando metadatos y matrices STFT previamente guardados...")
        df = pd.read_csv(csv_path)
        stft_data = np.load(stft_file, allow_pickle=True)
        df['stft_matrix'] = [stft_data[f] for f in df['filename']]
        return df
    print("Escaneando archivos...")
    df_files = scan_files(data_root, categories, n_per_cat)
    print(f"Procesando {len(df_files)} archivos en paralelo...")
    partial_analyze = partial(analyze_file, sr_target=sr_target, n_fft=n_fft, hop_length=hop_length)
    with Pool() as pool:
        results = list(tqdm(pool.imap(partial_analyze, df_files['path']), total=len(df_files)))
    records = []
    stft_dict = {}
    for meta, row in zip(results, df_files.to_dict('records')):
        if meta['ok']:
            stft = meta.pop('stft_matrix')
            stft_dict[row['filename']] = stft
            meta.update(row)
            records.append(meta)
    df = pd.DataFrame(records)
    df_ok = df[df['ok'] == True].copy()
    print(f"{len(df_ok)}/{len(df_files)} archivos OK")
    df_ok_no_stft = df_ok.drop(columns=['stft_matrix'])
    df_ok_no_stft.to_csv(csv_path, index=False)
    np.savez_compressed(stft_file, **stft_dict)
    return df_ok

# ==================== ALGORITMO UNIFIED BCGD (MEJORADO) ====================
def update_block(Block, grad_fn, v_Block, alpha, beta, method, is_nmf=True):
    """Actualiza un bloque (W o H) con la variante de descenso elegida."""
    if method == 'gd':
        Grad = grad_fn(Block)
        Block = Block - alpha * Grad
    elif method == 'momentum':
        Grad = grad_fn(Block)
        v_Block = beta * v_Block + Grad
        Block = Block - alpha * v_Block
    elif method == 'nesterov':
        Block_look = Block - alpha * beta * v_Block
        Grad_look = grad_fn(Block_look)
        v_Block = beta * v_Block + Grad_look
        Block = Block - alpha * v_Block
    if is_nmf:
        Block = np.maximum(Block, 0)
    return Block, v_Block

def nndsvd_initialization(X, k):
    """Inicialización NNDSVD (simplificada) para NMF."""
    m, n = X.shape
    U, s, Vt = svd(X, full_matrices=False)
    W = np.zeros((m, k))
    H = np.zeros((k, n))
    w = U[:, 0] * np.sqrt(s[0])
    h = Vt[0, :] * np.sqrt(s[0])
    W[:, 0] = np.maximum(w, 0)
    H[0, :] = np.maximum(h, 0)
    for i in range(1, k):
        w = U[:, i] * np.sqrt(s[i])
        h = Vt[i, :] * np.sqrt(s[i])
        w_plus = np.maximum(w, 0)
        h_plus = np.maximum(h, 0)
        w_minus = np.maximum(-w, 0)
        h_minus = np.maximum(-h, 0)
        W[:, i] = w_plus
        H[i, :] = h_plus
    return W, H

def unified_bcgd(X, k, steps, innerW=1, innerH=1, alpha_w=1e-3, alpha_h=1e-3,
                 method='gd', beta=0.9, is_nmf=True, mask=None, lambda_reg=0.0,
                 init='random', early_stopping=True, tol=1e-6):
    """Algoritmo BCGD unificado con early stopping e inicialización mejorada."""
    m, n = X.shape
    if mask is None:
        mask = np.ones_like(X)

    if init == 'nndsvd':
        W, H = nndsvd_initialization(X, k)
    else:
        W = np.random.uniform(0, 1/np.sqrt(k), size=(m, k))
        H = np.random.uniform(0, 1/np.sqrt(k), size=(k, n))

    v_W = np.zeros_like(W)
    v_H = np.zeros_like(H)
    loss_history = []
    best_loss = np.inf
    no_improve = 0

    for s in range(steps):
        for _ in range(innerW):
            grad_W_fn = lambda Wt: (mask * (Wt @ H - X)) @ H.T + lambda_reg * Wt
            W, v_W = update_block(W, grad_W_fn, v_W, alpha_w, beta, method, is_nmf)
        for _ in range(innerH):
            grad_H_fn = lambda Ht: W.T @ (mask * (W @ Ht - X)) + lambda_reg * Ht
            H, v_H = update_block(H, grad_H_fn, v_H, alpha_h, beta, method, is_nmf)

        residual = mask * (W @ H - X)
        loss = 0.5 * np.sum(residual ** 2) + (lambda_reg / 2.0) * (np.sum(W ** 2) + np.sum(H ** 2))
        loss_history.append(loss)

        if early_stopping:
            if loss < best_loss - tol:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 10:
                print(f"Early stopping en iteración {s}")
                break
    return W, H, loss_history

def select_k_by_elbow(X, k_range, steps=100, **bcgd_kwargs):
    """Selecciona k por el método del codo sobre la pérdida final."""
    losses = []
    for k in k_range:
        print(f"Probando k={k}...")
        _, _, loss_hist = unified_bcgd(X, k, steps=steps, **bcgd_kwargs)
        losses.append(loss_hist[-1])
    x = np.array(k_range)
    y = np.array(losses)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    dist = np.abs((x_norm + y_norm - 1) / np.sqrt(2))
    best_k = k_range[np.argmax(dist)]
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.axvline(best_k, color='r', linestyle='--', label=f'k elegido = {best_k}')
    plt.xlabel('rango k')
    plt.ylabel('pérdida final')
    plt.title('Selección de k por el método del codo')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_selection.png'), dpi=150)
    plt.show()
    return best_k, losses

# ==================== VISUALIZACIONES ====================
def plot_distributions(df_ok, cats, colors, labels):
    """Gráficas de distribución de duración y RMS."""
    # Duración
    dur_tot = [df_ok[df_ok['category']==c]['duration_s'].sum()/60 for c in cats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=PALETTE['bg'])
    wedges, _, autotexts = ax1.pie(dur_tot, labels=None, colors=colors, autopct='%1.1f%%',
                                   startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2.5))
    for autotext in autotexts:
        autotext.set_color('white')
    centre_circle = plt.Circle((0, 0), 0.60, fc='white', edgecolor=PALETTE['grid'], linewidth=1.5)
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, f'Total\n{sum(dur_tot):.1f}\nmin', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title('Duración Total', fontweight='bold')
    for cat, color, label in zip(cats, colors, labels):
        d = df_ok[df_ok['category']==cat]['duration_s'].dropna()
        ax2.hist(d, bins=35, color=color, alpha=0.6, label=label, edgecolor='white', density=True)
        kde = gaussian_kde(d)
        x_range = np.linspace(d.min(), d.max(), 200)
        ax2.plot(x_range, kde(x_range), color=color, lw=2)
    ax2.set_xlabel('Segundos')
    ax2.set_title('Distribución de Duración')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_distribuciones.png'), dpi=150)
    plt.show()

def plot_spectral_analysis(df_ok, sr_target, n_fft, hop_length):
    """Waveform, espectrograma, ZCR y distribución STFT para un ejemplo por categoría."""
    cats = df_ok['category'].unique()
    fig, axes = plt.subplots(len(cats), 4, figsize=(22, 12), facecolor=PALETTE['bg'])
    for row_idx, cat in enumerate(cats):
        path = df_ok[df_ok['category']==cat]['path'].iloc[0]
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=5.0)
        t = np.linspace(0, len(y)/sr, len(y))
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        ax = axes[row_idx, 0]
        ax.plot(t, y, color=PALETTE[cat], lw=0.8)
        ax.fill_between(t, y, 0, color=PALETTE[cat], alpha=0.2)
        ax.set_title(f'{CAT_LABELS[cat]} - Waveform')
        ax = axes[row_idx, 1]
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
        ax.set_title('Espectrograma')
        ax = axes[row_idx, 2]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        ax.plot(zcr, color=PALETTE[cat])
        ax.set_title('Zero Crossing Rate')
        ax = axes[row_idx, 3]
        ax.hist(S.flatten(), bins=50, color=PALETTE[cat], alpha=0.7, log=True)
        ax.set_title('Distribución STFT (Matriz X)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_spectral_analysis.png'), dpi=150)
    plt.show()

def plot_bases_espectrales(W, sr, hop_length, n_fft, k):
    """Visualiza las columnas de W como espectros de frecuencia."""
    fig, axes = plt.subplots(1, min(k, 6), figsize=(18, 3))
    for i in range(min(k, 6)):
        ax = axes[i]
        mag = W[:, i].reshape(-1, 1)
        freq_axis = np.linspace(0, sr/2, len(mag))
        ax.plot(freq_axis, librosa.amplitude_to_db(mag, ref=np.max), color='b')
        ax.set_title(f'Base {i+1}')
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('dB')
    plt.suptitle('Bases espectrales aprendidas (W)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bases_espectrales.png'), dpi=150)
    plt.show()

# ==================== MAIN ====================
def main():
    # 1. Cargar datos (con persistencia)
    df = load_and_process_dataset(DATA_ROOT, CATEGORIES, N_PER_CAT,
                                  SR_TARGET, N_FFT, HOP_LENGTH, force_recompute=False)

    # 2. Visualizaciones del EDA
    cats = df['category'].unique()
    colors = [PALETTE[c] for c in cats]
    labels = [CAT_LABELS[c] for c in cats]
    plot_distributions(df, cats, colors, labels)
    plot_spectral_analysis(df, SR_TARGET, N_FFT, HOP_LENGTH)

    # 3. Seleccionar matriz STFT de ejemplo (primera de music) y normalizar
    music_stft = df[df['category']=='music']['stft_matrix'].iloc[0]
    X = music_stft / np.max(music_stft)

    # 4. Selección automática de k
    k_range = [5, 10, 15, 20, 25, 30]
    best_k, _ = select_k_by_elbow(X, k_range, steps=100,
                                  method='nesterov', is_nmf=True,
                                  alpha_w=0.0005, alpha_h=0.0005, init='nndsvd')
    print(f"Mejor k elegido: {best_k}")

    # 5. Entrenamiento final
    print("Entrenando modelo final...")
    start_time = time.time()
    W, H, loss_hist = unified_bcgd(X, best_k, steps=500,
                                   method='nesterov', is_nmf=True,
                                   alpha_w=0.0005, alpha_h=0.0005,
                                   init='nndsvd', early_stopping=True)
    elapsed = time.time() - start_time
    print(f"Tiempo de entrenamiento: {elapsed:.2f} segundos")

    # 6. Curva de convergencia
    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.title('Curva de convergencia del NMF')
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergencia_nmf.png'), dpi=150)
    plt.show()

    # 7. Visualizar bases espectrales
    plot_bases_espectrales(W, SR_TARGET, HOP_LENGTH, N_FFT, best_k)

    # 8. Guardar resultados
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'nmf_result.npz'), W=W, H=H, loss=loss_hist)

if __name__ == '__main__':
    main()