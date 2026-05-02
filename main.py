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
# ==================================================
# VERIFICACIÓN DE DATOS Y DISTRIBUCIÓN
# ==================================================

def verify_data_distribution():
    """
    Verifica que las carpetas de datos existan, cuenta los archivos de audio
    por categoría, comprueba tasas de muestreo y muestra la distribución.
    """
    print("\n=== VERIFICACIÓN DE DATOS ===\n")
    
    # 1. Verificar estructura de directorios
    missing_dirs = []
    for cat in CATEGORIES:
        cat_path = os.path.join(DATA_ROOT, cat)
        if not os.path.isdir(cat_path):
            missing_dirs.append(cat_path)
    
    mixed_path = os.path.join(DATA_ROOT, 'mixed_audio')
    if not os.path.isdir(mixed_path):
        missing_dirs.append(mixed_path)
    
    if missing_dirs:
        print("⚠️  Faltan las siguientes carpetas:")
        for d in missing_dirs:
            print(f"   - {d}")
        print("Por favor, revisa la estructura de 'data/'.")
        return False
    
    # 2. Contar archivos de audio por categoría
    distribution = {}
    file_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
    
    for cat in CATEGORIES:
        cat_path = os.path.join(DATA_ROOT, cat)
        files = [f for f in os.listdir(cat_path) 
                 if f.lower().endswith(file_extensions)]
        distribution[cat] = len(files)
        print(f"📁 {cat.upper()}: {len(files)} archivos")
    
    # 3. Contar mezclas en mixed_audio
    mixed_files = [f for f in os.listdir(mixed_path) 
                   if f.lower().endswith(file_extensions)]
    distribution['mixed_audio'] = len(mixed_files)
    print(f"🎧 MIXED_AUDIO: {len(mixed_files)} archivos")
    
    # 4. Verificar que hay al menos un archivo por categoría (si se espera)
    for cat in CATEGORIES:
        if distribution[cat] == 0:
            print(f"   ⚠️  No hay archivos en {cat} — revisa su contenido.")
    
    # 5. Opcional: probar carga de un archivo por categoría (validar SR)
    print("\n--- Comprobación de tasa de muestreo (primer archivo por categoría) ---")
    sr_ok = True
    for cat in CATEGORIES:
        cat_path = os.path.join(DATA_ROOT, cat)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(file_extensions)]
        if files:
            sample_file = os.path.join(cat_path, files[0])
            try:
                y, sr = librosa.load(sample_file, sr=None, duration=2.0)  # sin resamplear
                if sr != SR_TARGET:
                    print(f"   {cat}: {os.path.basename(sample_file)} -> SR={sr} Hz (esperado {SR_TARGET})")
                    sr_ok = False
                else:
                    print(f"   ✅ {cat}: SR={sr} Hz correcto")
            except Exception as e:
                print(f"   ❌ Error al leer {cat}/{files[0]}: {e}")
                sr_ok = False
        else:
            print(f"   ⚠️  No se pudo probar {cat} (0 archivos)")
    
    # 6. Resumen visual de distribución
    print("\n--- DISTRIBUCIÓN ACTUAL ---")
    total = sum(distribution.get(c, 0) for c in CATEGORIES)
    print(f"Total de archivos fuente: {total}")
    for cat in CATEGORIES:
        count = distribution.get(cat, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"{cat:10} : {count:3} archivos ({pct:5.1f}%) {bar}")
    
    if total == 0:
        print("❌ No se encontraron archivos de audio en ninguna categoría.")
        return False
    
    if not sr_ok:
        print("\n⚠️  Advertencia: Algunas tasas de muestreo no coinciden con SR_TARGET.")
        print("   Se resamplearán automáticamente durante la carga (puede afectar rendimiento).")
    
    print("\n✅ Verificación completada.\n")
    return True

# ================= LLAMADA A LA VERIFICACIÓN =================
if __name__ == "__main__":
    # Aquí iría tu código principal, pero antes verificamos datos
    data_ok = verify_data_distribution()
    if not data_ok:
        print("Abortando ejecución por problemas con los datos.")
        exit(1)
    
    # El resto de tu pipeline (ej. cargar y procesar) continúa aquí...
    # Por ejemplo:
    # print("Iniciando procesamiento...")