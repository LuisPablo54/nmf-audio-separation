"""
Fase 2 — Mezcla sintética MUSAN
Proyecto C: Audio NMF (separación de fuentes)
 
Restricciones:
  - 400 archivos de speech disponibles
  - 5 niveles de SNR: 0, 5, 10, 15, 20 dB
  - Split por archivo de speech: 70/15/15 (train/val/test)
  - Mismo speech aparece en los 5 niveles → comparación justa entre niveles
  - Contaminante: 60% noise, 40% music
  - Sin repetición consecutiva del mismo contaminante dentro de un nivel
"""
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import random
import json
from collections import defaultdict
from tqdm import tqdm

# Configuración
semilla = 54 # Para reproducibilidad
random.seed(semilla)
np.random.seed(semilla)

SR          = 16000
SNR_LEVELS  = [0, 5, 10, 15, 20]
NOISE_RATIO = 0.60          # 60% noise ambiental, 40% music
SPLIT       = (0.70, 0.15, 0.15)   # train - val - test
 
def load_audio(path, sr=SR):
    """Carga y resamplea a mono 16kHz."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)
 
def rms(x):
    """RMS con epsilon para evitar división por cero."""
    return float(np.sqrt(np.mean(x ** 2) + 1e-9))
 
def mix_at_snr(speech, contaminant, snr_db):
    """
    Mezcla speech + contaminant al SNR objetivo.
    
    Si el contaminante es más corto que el speech, se repite (tile).
    Se toma un offset aleatorio para no usar siempre el inicio.
    La mezcla se normaliza al final para evitar clipping.
    
    Retorna: (mixed, alpha, snr_verificado)
    """
    # Ajustar longitud del contaminante
    if len(contaminant) < len(speech):
        reps = int(np.ceil(len(speech) / len(contaminant)))
        contaminant = np.tile(contaminant, reps)
 
    max_start = len(contaminant) - len(speech)
    start = random.randint(0, max_start) if max_start > 0 else 0
    contaminant = contaminant[start : start + len(speech)]
 
    # Escalar α para alcanzar el SNR objetivo
    alpha = rms(speech) / (rms(contaminant) * 10 ** (snr_db / 20.0))
    mixed = speech + alpha * contaminant
 
    # Normalización peak para evitar clipping
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak
 
    # Verificación del SNR real de la mezcla (sanity check)
    snr_real = 20 * np.log10(rms(speech) / (rms(alpha * contaminant) + 1e-9))
 
    return mixed.astype(np.float32), float(alpha), round(float(snr_real), 2)
 
# ── Muestreo sin repetición excesiva ──────────────────────────
 
def cycle_sample(file_list, n):
    """
    Devuelve n paths sin repetición inmediata excesiva.
    Si n > len(file_list): ciclos completos + muestra del resto.
    Siempre hace shuffle para variar el orden entre llamadas.
    """
    if len(file_list) == 0:
        raise ValueError("La lista de archivos está vacía.")
    full_cycles = n // len(file_list)
    remainder   = n %  len(file_list)
    pool = file_list * full_cycles
    if remainder > 0:
        pool += random.sample(file_list, remainder)
    random.shuffle(pool)
    return pool
 
# ── Split de speech (fijo para todos los niveles de SNR) ──────
 
def split_speech(speech_files, split=(0.70, 0.15, 0.15)):
    """
    Divide los 400 archivos de speech en train/val/test.
    El split es FIJO para todos los niveles de SNR — mismo hablante
    no aparece en train y test simultáneamente.
    """
    files = speech_files.copy()
    random.shuffle(files)
    n = len(files)
    n_train = int(n * split[0])              # 280
    n_val   = int(n * split[1])              # 60
    # test toma el resto para evitar pérdida por redondeo
    return (
        files[:n_train],
        files[n_train : n_train + n_val],
        files[n_train + n_val :]
    )

# ── Función principal ──────────────────────────────────────────
 
def build_dataset(speech_dir, noise_dir, music_dir, out_dir):
    """
    Genera el dataset de mezclas sintéticas.
    
    Estructura de salida:
        out_dir/
          train/snr_Xdb/mix_XXXX.wav
          val/snr_Xdb/mix_XXXX.wav
          test/snr_Xdb/mix_XXXX.wav
          metadata.json
          split_info.json
    """
    out_dir = Path(out_dir)
 
    


    speech_files = sorted(Path(speech_dir).rglob("*.wav"))
    noise_files  = sorted(Path(noise_dir).rglob("*.wav"))
    music_files  = sorted(Path(music_dir).rglob("*.wav"))
 
    speech_files = [str(p) for p in speech_files]
    noise_files  = [str(p) for p in noise_files]
    music_files  = [str(p) for p in music_files]

    speech_files = speech_files[:100] #para trabajar solo con 100
 
    print(f"Speech disponible : {len(speech_files):>4} archivos")
    print(f"Noise disponible  : {len(noise_files):>4} archivos")
    print(f"Music disponible  : {len(music_files):>4} archivos")
    print()
 
    if len(speech_files) == 0:
        raise FileNotFoundError(f"No se encontraron .wav en {speech_dir}")
    if len(noise_files) == 0:
        raise FileNotFoundError(f"No se encontraron .wav en {noise_dir}")
    if len(music_files) == 0:
        raise FileNotFoundError(f"No se encontraron .wav en {music_dir}")
 
    # Split fijo de speech
    train_sp, val_sp, test_sp = split_speech(speech_files, SPLIT)
    split_info = {
        "seed"       : semilla,
        "total_speech": len(speech_files),
        "train"      : len(train_sp),   # 280
        "val"        : len(val_sp),     # 60
        "test"       : len(test_sp),    # 60
        "snr_levels" : SNR_LEVELS,
        "noise_ratio": NOISE_RATIO,
    }
    print(f"Split de speech → train: {len(train_sp)} | val: {len(val_sp)} | test: {len(test_sp)}")
    print()
 
    metadata   = []
    snr_errors = defaultdict(list)   # para registrar discrepancias de SNR
 
    subsets = [("train", train_sp), ("val", val_sp), ("test", test_sp)]
 
    for subset_name, speech_subset in subsets:
        n = len(speech_subset)
        # Proporciones de contaminante para este subset
        n_noise = int(n * NOISE_RATIO)     # 60%
        n_music = n - n_noise              # 40%
 
        for snr in SNR_LEVELS:
            out_snr_dir = out_dir / subset_name / f"snr_{snr}db"
            out_snr_dir.mkdir(parents=True, exist_ok=True)
 
            # Pool de contaminantes — sin repetición excesiva
            noise_pool = cycle_sample(noise_files, n_noise)
            music_pool = cycle_sample(music_files, n_music)
 
            # Mezclar tipos y alinear con lista de speech
            cont_list = (
                [(p, "noise") for p in noise_pool] +
                [(p, "music") for p in music_pool]
            )
            random.shuffle(cont_list)
            # Asegurar que cont_list tiene exactamente n elementos
            cont_list = cont_list[:n]
 
            for i, (sp_path, (cont_path, cont_type)) in enumerate(
                zip(speech_subset, cont_list)
            ):
                try:
                    speech     = load_audio(sp_path)
                    contaminant = load_audio(cont_path)
                    mixed, alpha, snr_real = mix_at_snr(speech, contaminant, snr)
 
                    out_path = out_snr_dir / f"mix_{i:04d}.wav"
                    sf.write(str(out_path), mixed, SR)
 
                    metadata.append({
                        "file"            : str(out_path),
                        "subset"          : subset_name,
                        "speech_src"      : sp_path,
                        "contaminant_src" : cont_path,
                        "contaminant_type": cont_type,
                        "snr_target_db"   : snr,
                        "snr_real_db"     : snr_real,
                        "alpha"           : alpha,
                    })
 
                    # Registrar si el SNR real se desvía > 1 dB del objetivo
                    error = abs(snr_real - snr)
                    if error > 1.0:
                        snr_errors[snr].append({
                            "file": str(out_path),
                            "snr_real": snr_real,
                            "error_db": round(error, 2)
                        })
 
                except Exception as e:
                    print(f"  [ERROR] {sp_path} | {cont_path} → {e}")
 
            n_done = len(speech_subset)
            print(f"  [{subset_name:5s}] SNR {snr:>2} dB — {n_done} mezclas")
 
        print()
 
    # ── Guardar metadata y split info ─────────────────────────
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
 
    split_path = out_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
 
    # ── Reporte final ──────────────────────────────────────────
    total = len(metadata)
    noise_count = sum(1 for m in metadata if m["contaminant_type"] == "noise")
    music_count = sum(1 for m in metadata if m["contaminant_type"] == "music")
    snr_warn    = sum(len(v) for v in snr_errors.values())
 
    print("=" * 52)
    print(f"  Total de mezclas generadas : {total}")
    print(f"  Contaminante noise         : {noise_count} ({100*noise_count/total:.0f}%)")
    print(f"  Contaminante music         : {music_count} ({100*music_count/total:.0f}%)")
    print(f"  Mezclas con error SNR >1dB : {snr_warn}")
    print(f"  Metadata guardado en       : {meta_path}")
    print("=" * 52)
 
    return metadata, split_info

import os
os.chdir(r"C:\Users\xdieg\OptimizacionAudios\data\musan")

build_dataset(
    speech_dir = "speech",
    noise_dir  = "noise",
    music_dir  = "music",
    out_dir    = "mixed_audio"
)

def build_dataset_final(speech_dir, noise_dir, music_dir, out_dir, dry_run=False, max_files=None):
    """
    Versión final (Fase 2) con las tareas de Mares integradas.
    No reemplaza a la original, convive con ella.
    """
    out_dir = Path(out_dir)
    
    # Búsqueda de archivos (Mantiene el fix de Paula)
    speech_files = sorted(Path(speech_dir).rglob("*.wav"))
    noise_files  = sorted(Path(noise_dir).rglob("*.wav"))
    music_files  = sorted(Path(music_dir).rglob("*.wav"))
    
    speech_files = [str(p) for p in speech_files]
    noise_files  = [str(p) for p in noise_files]
    music_files  = [str(p) for p in music_files]

    # TAREA MARES 1: Limitar archivos para dry-run / pruebas rápidas
    if max_files:
        speech_files = speech_files[:max_files]
        print(f"Modo prueba: Trabajando solo con {max_files} audios de speech.")

    train_sp, val_sp, test_sp = split_speech(speech_files, SPLIT)
    
    # TAREA MARES 2: Incluir rutas reales en el split_info
    split_info = {
        "seed"        : semilla,
        "total_speech": len(speech_files),
        "train"       : len(train_sp),
        "val"         : len(val_sp),
        "test"        : len(test_sp),
        "train_files" : train_sp,   # <-- Rutas reales añadidas
        "val_files"   : val_sp,     # <-- Rutas reales añadidas
        "test_files"  : test_sp,    # <-- Rutas reales añadidas
        "snr_levels"  : SNR_LEVELS,
        "noise_ratio" : NOISE_RATIO,
    }
    
    metadata   = []
    snr_errors = defaultdict(list)
    subsets = [("train", train_sp), ("val", val_sp), ("test", test_sp)]

    for subset_name, speech_subset in subsets:
        n = len(speech_subset)
        n_noise = int(n * NOISE_RATIO)
        n_music = n - n_noise

        for snr in SNR_LEVELS:
            out_snr_dir = out_dir / subset_name / f"snr_{snr}db"
            if not dry_run: 
                out_snr_dir.mkdir(parents=True, exist_ok=True)

            noise_pool = cycle_sample(noise_files, n_noise)
            music_pool = cycle_sample(music_files, n_music)
            cont_list = ([(p, "noise") for p in noise_pool] + [(p, "music") for p in music_pool])
            random.shuffle(cont_list)
            cont_list = cont_list[:n]

            # Mantiene el tqdm que agregó Paula
            for i, (sp_path, (cont_path, cont_type)) in enumerate(tqdm(zip(speech_subset, cont_list), total=n, desc=f"[{subset_name}] SNR {snr}dB")):
                try:
                    speech = load_audio(sp_path)
                    contaminant = load_audio(cont_path)
                    mixed, alpha, snr_real = mix_at_snr(speech, contaminant, snr)

                    out_path = out_snr_dir / f"mix_{i:04d}.wav"
                    
                    # TAREA MARES 1 (Cont.): Solo escribe en disco si NO es dry-run
                    if not dry_run:
                        sf.write(str(out_path), mixed, SR)

                    metadata.append({
                        "file": str(out_path),
                        "subset": subset_name,
                        "speech_src": sp_path,
                        "contaminant_src": cont_path,
                        "contaminant_type": cont_type,
                        "snr_target_db": snr,
                        "snr_real_db": snr_real,
                        "alpha": alpha
                    })

                    error = abs(snr_real - snr)
                    if error > 1.0:
                        snr_errors[snr].append({
                            "file": str(out_path), 
                            "snr_real": snr_real,
                            "error_db": round(error, 2)
                        })

                except Exception as e:
                    print(f"  [ERROR] {sp_path} → {e}")

    # TAREA MARES 3: Guardar los JSONs en la carpeta que se le pase (outputs)
    if not dry_run:
        with open(out_dir / "metadata_MIXING.json", "w") as f:
            json.dump(metadata, f, indent=2)
        with open(out_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)
        
        if snr_errors:
            err_path = out_dir / "snr_errors.json"
            with open(err_path, "w") as f:
                json.dump(snr_errors, f, indent=2)
            print(f"\n[!] Detalle de errores SNR guardado en {err_path}")
            
    print("\n=== Proceso Finalizado ===")
    if dry_run:
        print("NOTA: MODO DRY-RUN ACTIVO. No se crearon archivos en el disco, fue solo simulación.")
    
    return metadata, split_info

import os

# Asegúrate de usar la ruta relativa o absoluta correcta hacia la carpeta "outputs"
# Basado en tu estructura, si estás en data/musan, outputs debería estar dos niveles arriba
carpeta_salida = "../../outputs" 

# Probamos con dry_run=True y max_files=5 para verificar que todo funciona en segundos
metadata, split_info = build_dataset_final(
    speech_dir = "speech",
    noise_dir  = "noise",
    music_dir  = "music",
    out_dir    = carpeta_salida, 
    dry_run    = True,     # Cámbialo a False cuando quieras generar los .wav reales
    max_files  = 5         # Cámbialo a None para procesar los 400 audios completos
)