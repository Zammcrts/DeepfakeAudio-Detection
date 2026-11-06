"""Inferencia: carga las 3 etapas de CNN y LSTM y analiza un audio con el MISMO preprocesamiento del entrenamiento.

Modelos que carga (los genera training/train.py en la carpeta de resultados):
    A  <modelo>_pre.pt       preentrenado con audios multilingües (nunca vio español)
    B  <modelo>.pt           preentrenado + capa final nueva + afinado con español
    C  <modelo>_solo_es.pt   entrenado solo con español (línea base)

Porcentajes: se calibran con la temperatura y el umbral que cada modelo obtuvo en la validación en español
(metricas.json / modelos.json), y se promedian entre los modelos del conjunto.

Uso desde Django:  from detector.inference import get_detector;  get_detector().analizar(ruta_audio)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "training"))
from training.models import build                                   # noqa: E402
from training.preprocesamiento import audio_a_mel, preparar, N_FRAMES, N_MELS   # noqa: E402

ETAPAS = [("A", "{m}_pre.pt", "A_preentrenado_sin_ver_es", "preentrenado"),
          ("B", "{m}.pt", "B_preentrenado_y_afinado_es", "preentrenado + afinado"),
          ("C", "{m}_solo_es.pt", "C_solo_espanol", "solo español")]


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


class Detector:
    """Carga los modelos de una carpeta de resultados.

    - Si existe modelos.json (lo escribe training/entrenar_cv.py): cada etapa es un CONJUNTO de modelos
      (semillas × folds) cuyas salidas se promedian, ya calibradas. Es lo más estable.
    - Si no, usa los .pt sueltos de un solo train.py (un modelo por etapa).
    max_miembros limita cuántos modelos del conjunto se cargan por etapa (menos memoria y tiempo por audio).
    """

    def __init__(self, resultados=None, device=None, etapas=("A", "B", "C"), max_miembros=None):
        self.dir = Path(resultados or RAIZ / "resultados_cv")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modelos = []                       # (etiqueta, [(red, umbral, temperatura), ...])
        conjunto = self.dir / "modelos.json"
        if conjunto.exists():
            lista = json.load(open(conjunto, encoding="utf-8"))
            for m in ("cnn", "lstm"):
                for letra, _, _, desc in ETAPAS:
                    miembros = lista.get(m, {}).get(letra, [])[:max_miembros]
                    if letra in etapas and miembros:
                        self.modelos.append((f"{m.upper()} · {letra} {desc}",
                                             [(self._cargar(m, self.dir / d["ckpt"]), d["thr"], d["T"]) for d in miembros]))
        else:
            try:
                met = json.load(open(self.dir / "metricas.json", encoding="utf-8"))["resultados"]
            except FileNotFoundError:
                met = {}
            for m in ("cnn", "lstm"):
                for letra, patron, clave, desc in ETAPAS:
                    f = self.dir / patron.format(m=m)
                    if letra not in etapas or not f.exists():
                        continue
                    thr = met.get(m, {}).get(clave, {}).get("test@umbral_val", {}).get("umbral", 0.5)
                    T = met.get(m, {}).get("temperaturas", {}).get(letra, 1.0)
                    self.modelos.append((f"{m.upper()} · {letra} {desc}", [(self._cargar(m, f), thr, T)]))
        if not self.modelos:
            raise FileNotFoundError(f"No encontré modelos (.pt) en {self.dir}")

    def _cargar(self, m, ruta):
        red = build(m, N_MELS)
        red.load_state_dict(torch.load(ruta, map_location=self.device))
        return red.to(self.device).eval()

    @torch.no_grad()
    def probabilidades(self, mel):
        """mel: (n_mels, frames) de un audio completo -> {etiqueta: prob_fake (0-1)}.
        Por modelo: v = (logit_fake - logit(umbral)) / T; se promedia entre ventanas y entre miembros del conjunto.
        v = 0 es el punto de decisión del modelo, y 1/(1+e^-v) es el porcentaje que ve la persona."""
        x = torch.from_numpy(preparar(mel, N_FRAMES)).to(self.device)       # (ventanas, n_mels, frames)
        out = {}
        for etiqueta, miembros in self.modelos:
            vs = []
            for red, thr, T in miembros:
                z = red(x).float()
                d = (z[:, 1] - z[:, 0]).cpu().numpy()
                vs.append(((d - _logit(min(max(thr, 0.02), 0.98))) / max(T, 1.0)).mean())
            out[etiqueta] = float(1 / (1 + np.exp(-np.mean(vs))))
        return out

    def analizar(self, ruta_audio):
        """-> [(nombre, {'fake': %, 'real': %}), ...] listo para la plantilla."""
        mel = audio_a_mel(ruta_audio)
        if mel.shape[1] < 10 or np.ptp(mel) < 1e-3:
            raise ValueError("El audio está vacío o es demasiado corto.")
        return [(n, {"fake": 100 * p, "real": 100 * (1 - p)}) for n, p in self.probabilidades(mel).items()]


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector