"""Preprocesamiento compartido por el entrenamiento (train.py) y la interfaz.

Una sola fuente de verdad: si el entrenamiento y la interfaz no preparan el audio exactamente igual,
las probabilidades que muestra la interfaz no significan lo mismo que las medidas en el entrenamiento.

Dos partes:
  1. audio -> mel-espectrograma en dB (solo lo usa la interfaz; los .npy del entrenamiento ya lo traen).
     Los parámetros son los del script con el que se generaron los .npy en español (confirmado por las
     alumnas): sr 22050, n_fft 2048, hop 512, 128 mels, fmin 20, fmax 11025, potencia 2, dB ABSOLUTOS
     (ref=1.0, sin top_db) y clips de 10 s rellenados con ceros. Los clips multilingües pueden haberse
     generado con otro script: ahí solo se sabe que miden 431 frames.
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

# ---- parámetros del espectrograma (los del script de generación de los .npy en español) ----
SR = 22050
DURACION = 10.0
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 20
FMAX = SR // 2
N_FRAMES = 1 + int(SR * DURACION) // HOP          # 431 con los valores anteriores


# ----------------------------------------------------------------------------- 1. audio -> espectrograma
def audio_a_mel(origen):
    """Mel-espectrograma en dB absolutos, forma (n_mels, frames), de un archivo de audio COMPLETO.
    Igual que el script de generación, salvo que no recorta a 10 s (preparar() toma ventanas) y que un audio
    más corto de 10 s se rellena con ceros para dar el mismo espectrograma que un .npy de entrenamiento."""
    import librosa
    y, _ = librosa.load(str(origen), sr=SR, mono=True)
    largo = int(SR * DURACION)
    if len(y) < largo:
        y = np.pad(y, (0, largo - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                         fmin=FMIN, fmax=FMAX, power=2.0)
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


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


# ----------------------------------------------------------------------------- aumento "de canal" (solo entrenamiento)
def perturbar_canal(s, rng=None, p=0.8):
    """Simula otra grabación del MISMO audio: otro ruido de fondo, otra respuesta en frecuencia y, a veces,
    otro ancho de banda. Se aplica sobre el espectrograma en dB, antes de normalizar.

    Por qué: el diagnóstico mostró que real y fake en español se distinguen por el suelo de ruido, el rango
    dinámico y la forma espectral (condiciones de grabación), no por la voz. Una red pequeña aprende ese atajo
    y saca ~100 % en el dataset, pero falla con grabaciones nuevas. Al variar esas propiedades al azar, la red
    ya no puede apoyarse en ellas y tiene que usar lo que sí cambia entre voz real y sintética (textura fina,
    armónicos, transiciones). Los artefactos de la síntesis no viven en el suelo de ruido ni en la inclinación.
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p:
        return s
    n_mels, T = s.shape
    s = s.astype(np.float32, copy=False)
    lo, hi = np.percentile(s[::4, ::4], (5, 95))            # submuestra: mismo valor práctico, 16 veces menos cálculo
    rango = max(float(hi - lo), 1.0)
    # 1) ruido de fondo de nivel y color aleatorios, sumado en potencia
    nivel = rng.uniform(lo - 5.0, lo + 0.4 * rango)
    color = np.interp(np.arange(n_mels), np.linspace(0, n_mels - 1, 4), rng.normal(0, 5.0, 4))
    ruido = (nivel + color[:, None]).astype(np.float32) + rng.standard_normal((n_mels, T), dtype=np.float32)
    k = np.float32(np.log(10) / 10)
    s = np.logaddexp(s * k, ruido * k) / k
    # 2) respuesta en frecuencia suave (micrófono / canal)
    s = s + np.interp(np.arange(n_mels), np.linspace(0, n_mels - 1, 6), rng.normal(0, 4.0, 6))[:, None]
    # 3) a veces, ancho de banda limitado (paso bajo con caída progresiva)
    if rng.random() < 0.3:
        corte = int(rng.uniform(0.5, 0.95) * n_mels)
        caida = np.zeros(n_mels); caida[corte:] = np.linspace(0, rng.uniform(10, 30), n_mels - corte)
        s = s - caida[:, None]
    return s.astype(np.float32)