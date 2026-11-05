"""Preprocesamiento compartido por el entrenamiento (train.py) y la interfaz.

Una sola fuente de verdad: si el entrenamiento y la interfaz no preparan el audio exactamente igual,
las probabilidades que muestra la interfaz no significan lo mismo que las medidas en el entrenamiento.

Dos partes:
  1. audio -> mel-espectrograma en dB (solo lo usa la interfaz; los .npy del entrenamiento ya lo traen).
     LAS CONSTANTES DE ABAJO SON UNA SUPOSICIÓN: 431 frames equivalen a 10 s con sr=22050 y hop=512
     (valores por defecto de librosa). Compruébalas con training/probar_preprocesamiento.py.
  2. espectrograma -> entrada de la red: recortar relleno, repetir o tomar ventanas, normalizar.

Por qué se acondiciona el clip (hallazgo del diagnóstico): en el conjunto multilingüe, más de la mitad de
los frames de un clip típico son relleno digital (ceros) sin voz, mientras que en español casi no hay
relleno. Con recortes al azar, muchas ventanas de entrenamiento caían en puro relleno (ejemplos sin
información) y la duración del clip se volvía una pista que no tiene nada que ver con la voz.
El modo 'repetir' recorta ese relleno y repite el contenido hasta llenar la ventana, de modo que toda
muestra tiene voz de principio a fin, igual que los clips en español.
"""
from pathlib import Path

import numpy as np

# ---- parámetros del espectrograma (CONFIRMAR) ----
SR = 22050
DURACION = 10.0
N_FFT = 2048
HOP = 512
N_MELS = 128
TOP_DB = 80
N_FRAMES = 1 + int(SR * DURACION) // HOP          # 431 con los valores anteriores


# ----------------------------------------------------------------------------- 1. audio -> espectrograma
def audio_a_mel(origen):
    """Mel-espectrograma en dB, forma (n_mels, frames), de un archivo de audio COMPLETO (no se recorta)."""
    import librosa
    y, _ = librosa.load(str(origen), sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB).astype(np.float32)


# ----------------------------------------------------------------------------- 2. espectrograma -> red
def recortar_relleno(s):
    """Quita al inicio y al final los frames 'planos' (todas las bandas iguales: silencio digital o relleno).
    No toca las pausas con ruido de fondo, que sí son parte de la grabación."""
    plano = (s.max(0) - s.min(0)) < 1e-3
    util = np.flatnonzero(~plano)
    if len(util) == 0:
        return s
    return s[:, util[0]: util[-1] + 1]


def acondicionar(s, n_frames=N_FRAMES, modo="repetir", aleatorio=False):
    """Deja el espectrograma en (n_mels, n_frames).

    modo='repetir'  : recorta el relleno; si queda corto repite el contenido hasta llenar; si queda largo toma
                      una ventana (al azar si aleatorio=True, desde el inicio si no).
    modo='mantener' : comportamiento anterior: no toca el relleno; recorta o rellena con el mínimo.
    """
    if modo == "repetir":
        s = recortar_relleno(s)
        if s.shape[1] < n_frames:
            s = np.tile(s, (1, int(np.ceil(n_frames / s.shape[1]))))
    T = s.shape[1]
    if T >= n_frames:
        inicio = np.random.randint(0, T - n_frames + 1) if aleatorio else 0
        return s[:, inicio:inicio + n_frames]
    return np.pad(s, ((0, 0), (0, n_frames - T)), constant_values=s.min())


def normalizar(s):
    """z-score por muestra: quita diferencias de volumen/ganancia entre grabaciones."""
    return (s - s.mean()) / (s.std() + 1e-8)


def preparar(s, n_frames=N_FRAMES, modo="repetir", max_ventanas=8):
    """Para la interfaz: devuelve (V, n_mels, n_frames) ya normalizado.
    Un audio corto da una ventana; uno largo da varias (con solape del 50 %), hasta max_ventanas, y la
    interfaz promedia las probabilidades de las ventanas."""
    if modo == "repetir":
        s = recortar_relleno(s)
    T = s.shape[1]
    if T <= n_frames:
        ventanas = [acondicionar(s, n_frames, modo)]
    else:
        inicios = list(range(0, T - n_frames + 1, n_frames // 2))
        if inicios[-1] != T - n_frames:
            inicios.append(T - n_frames)
        if len(inicios) > max_ventanas:
            inicios = [inicios[i] for i in np.linspace(0, len(inicios) - 1, max_ventanas).round().astype(int)]
        ventanas = [s[:, i:i + n_frames] for i in inicios]
    return np.stack([normalizar(v) for v in ventanas]).astype(np.float32)