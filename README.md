<div align="center">

# 🎵 Audio Source Separation via NMF

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 📖 Descripción
Este proyecto implementa la separación de fuentes de audio monoaural utilizando Factorización de Matrices No Negativas (NMF) optimizada mediante Block-Coordinate Gradient Descent (BCGD). El objetivo es desmezclar señales de voz y ruido combinadas digitalmente, comparando la convergencia y calidad de reconstrucción entre el Descenso de Gradiente (GD) estándar, Momentum y Nesterov Accelerated Gradient (NAG).
</div> 

## Cómo ejecutar
```bash
git clone <repo>
cd nmf-audio-separation
pip install -r requirements.txt
jupyter notebook notebooks/main_final.ipynb
```

## ⚙️ Pipeline del Proyecto

1. **Carga de Datos:** Lectura de señales independientes de voz y ruido desde el corpus MUSAN.
2. **Mezcla Digital:** Creación de la señal observada $y_{\text{mix}} = y_{\text{speech}} + y_{\text{noise}}$.
3. **Transformación (STFT):** Cálculo del espectrograma de magnitud $X = |\text{STFT}(y_{\text{mix}})|$, donde $X \in \mathbb{R}^{F \times T}$ con $F=513$ bins.
4. **Factorización NMF:** Aproximación $X \approx WH$ sujeto a $W \geq 0, H \geq 0$. $W \in \mathbb{R}^{F \times k}$ captura las plantillas espectrales y $H \in \mathbb{R}^{k \times T}$ las activaciones temporales. Proyección al ortante no-negativo usando $\max(\cdot,0)$.
5. **Reconstrucción (iSTFT):** Agrupación de las columnas de $W$ correspondientes a cada fuente y aplicación de la Transformada Inversa de Fourier a Corto Plazo utilizando la fase original de la mezcla.
6. **Evaluación:** Cálculo del RMSE sobre particiones temporales contiguas (70% Train, 15% Val, 15% Test).
