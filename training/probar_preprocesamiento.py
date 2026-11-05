"""Averigua con qué parámetros se generaron los .npy, comparando un audio original con su espectrograma.

La interfaz recibe AUDIO y debe convertirlo a espectrograma igual que se hizo con los .npy de entrenamiento;
si los parámetros no coinciden, la red recibe datos distintos a los que conoce y sus porcentajes no valen.
Este script prueba combinaciones de parámetros y dice cuál reproduce mejor el .npy.

Uso (un audio original y SU .npy):
    python training/probar_preprocesamiento.py --audio ejemplo.wav --npy ejemplo.npy
Salida: ranking de combinaciones por error medio (en dB). Un error ~0 (< 0.01) indica coincidencia exacta.
Si ninguna se acerca, el .npy se generó con otro método (cuéntame cómo) o tiene recorte/relleno distinto.
"""
import argparse, itertools

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--npy", required=True)
    a = ap.parse_args()
    import librosa

    ref = np.load(a.npy).astype(np.float32)
    n_mels, T = ref.shape
    print(f".npy: {n_mels} mels x {T} frames | rango {ref.min():.1f} a {ref.max():.1f} dB (media {ref.mean():.1f})")

    resultados = []
    for sr, n_fft, hop, refm, top_db, dur in itertools.product(
            (22050, 16000, 44100), (2048, 1024), (512, 256, 160), ("max", "1.0"), (80, None), (None, 10.0)):
        try:
            y, _ = librosa.load(a.audio, sr=sr, mono=True, duration=dur)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
            s = librosa.power_to_db(mel, ref=np.max if refm == "max" else 1.0, top_db=top_db)
        except Exception:
            continue
        n = min(T, s.shape[1])
        err = float(np.abs(s[:, :n] - ref[:, :n]).mean())
        resultados.append((err, dict(sr=sr, n_fft=n_fft, hop=hop, ref=refm, top_db=top_db, duracion=dur),
                           s.shape[1]))
    resultados.sort(key=lambda r: r[0])
    print("\nMejores combinaciones (error medio en dB; frames que produce):")
    for err, p, fr in resultados[:5]:
        print(f"  error={err:7.3f}  frames={fr:4d}  {p}")
    err, p, fr = resultados[0]
    if err < 0.05:
        print(f"\nCoincide. Pon estos valores en training/preprocesamiento.py: {p}")
        if fr != T:
            print(f"  Ojo: genera {fr} frames y el .npy tiene {T}: hubo recorte o relleno al guardar.")
    else:
        print("\nNinguna reproduce el .npy. Dime cómo lo generaste (código o notebook) antes de usar la interfaz.")


if __name__ == "__main__":
    main()