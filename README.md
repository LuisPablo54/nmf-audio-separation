<div align="center">

# 🎵 Audio Source Separation via NMF

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USUARIO/audio-nmf-separation/blob/main/notebooks/01_BCGD_implementation.ipynb)

## 📖 Descripción
Este proyecto implementa la separación de fuentes de audio monoaural utilizando Factorización de Matrices No Negativas (NMF) optimizada mediante Block-Coordinate Gradient Descent (BCGD). El objetivo es desmezclar señales de voz y ruido combinadas digitalmente, comparando la convergencia y calidad de reconstrucción entre el Descenso de Gradiente (GD) estándar, Momentum y Nesterov Accelerated Gradient (NAG).
</div> 

## ⚙️ Pipeline del Proyecto

1. **Carga de Datos:** Lectura de señales independientes de voz y ruido desde el corpus MUSAN.
2. **Mezcla Digital:** Creación de la señal observada $y_{\text{mix}} = y_{\text{speech}} + y_{\text{noise}}$.
3. **Transformación (STFT):** Cálculo del espectrograma de magnitud $X = |\text{STFT}(y_{\text{mix}})|$, donde $X \in \mathbb{R}^{F \times T}$ con $F=513$ bins.
4. **Factorización NMF:** Aproximación $X \approx WH$ sujeto a $W \geq 0, H \geq 0$. $W \in \mathbb{R}^{F \times k}$ captura las plantillas espectrales y $H \in \mathbb{R}^{k \times T}$ las activaciones temporales. Proyección al ortante no-negativo usando $\max(\cdot,0)$.
5. **Reconstrucción (iSTFT):** Agrupación de las columnas de $W$ correspondientes a cada fuente y aplicación de la Transformada Inversa de Fourier a Corto Plazo utilizando la fase original de la mezcla.
6. **Evaluación:** Cálculo del RMSE sobre particiones temporales contiguas (70% Train, 15% Val, 15% Test).

## 📂 Estructura del Repositorio

```text
audio-nmf-separation/
│
├── README.md                        ← Documentación principal del proyecto
├── CONTRIBUTING.md                  ← Instrucciones para el equipo de desarrollo
├── LICENSE                          ← Licencia MIT
├── .gitignore                       ← Reglas de exclusión para Git
├── requirements.txt                 ← Dependencias del proyecto
│
├── notebooks/                       ← Jupyter Notebooks para Colab
│   ├── 00_EDA.ipynb                 ← Análisis exploratorio de MUSAN
│   ├── 01_BCGD_implementation.ipynb ← Algoritmo 1 (BCGD) y optimizadores
│   ├── 02_experiments.ipynb         ← Comparativa GD vs Momentum vs Nesterov
│   └── 03_source_separation.ipynb   ← Reconstrucción de audio y evaluación
│
├── src/                             ← Código fuente empaquetado
│   ├── __init__.py
│   ├── bcgd.py                      ← Implementación core de NMF y optimizadores
│   ├── audio_utils.py               ← Procesamiento de señales (STFT, iSTFT, mezclas)
│   ├── nmf_eval.py                  ← Algoritmo 2 (held-out evaluation) y métricas
│   └── visualization.py             ← Generación de espectrogramas y curvas de pérdida
│
├── data/                            ← Datos locales (No versionados en GitHub)
│   └── .gitkeep                     
│
├── outputs/                         ← Artefactos generados
│   ├── figures/                     ← Gráficas en alta calidad (PNG/PDF)
│   └── audio/                       ← Archivos .wav de señales reconstruidas
│
├── report/                          
│   └── report_draft.md              ← Borrador del reporte académico (max. 6 pág)
│
└── poster/
    └── poster_draft.md              ← Borrador del póster final

```
