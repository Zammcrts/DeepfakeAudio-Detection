"""Revisa la integridad de los datos antes de entrenar.

Busca: conteos por clase, formas de los espectrogramas, archivos DUPLICADOS (mismo contenido),
solapamiento entre el conjunto multilingüe y el de español, y archivos con la misma huella en
clases distintas (real y fake a la vez).

Uso:
    python training/verificar_datos.py --multi_dir datos/multilingue --es_dir datos/espanol
"""
import argparse, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


def scan(root):
    info = {}
    for cls in ["real", "fake"]:
        for p in sorted(Path(root, cls).rglob("*.npy")):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            shp = np.load(p, mmap_mode="r").shape
            info[str(p)] = (cls, h, shp)
    return info


def resumen(nombre, info):
    print(f"\n=== {nombre} ===")
    for cls in ["real", "fake"]:
        s = [v[2] for v in info.values() if v[0] == cls]
        if s:
            frames = np.array([x[1] for x in s]); mels = {x[0] for x in s}
            print(f"{cls:5s} n={len(s):6d}  n_mels={sorted(mels)}  frames min/mediana/max = "
                  f"{frames.min()}/{int(np.median(frames))}/{frames.max()}")


def grupos(info):
    g = defaultdict(list)
    for path, (cls, h, _) in info.items():
        g[h].append((path, cls))
    return g


def mostrar(titulo, lst, n=5):
    print(f"\n{titulo}: {len(lst)}")
    for item in lst[:n]:
        print("   ", item)
    if len(lst) > n:
        print(f"    ... y {len(lst) - n} más")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_dir", required=True)
    ap.add_argument("--es_dir", required=True)
    a = ap.parse_args()

    multi, es = scan(a.multi_dir), scan(a.es_dir)
    resumen("MULTILINGÜE", multi); resumen("ESPAÑOL", es)

    gm, ge = grupos(multi), grupos(es)
    dup_multi = [v for v in gm.values() if len(v) > 1]
    dup_es = [v for v in ge.values() if len(v) > 1]
    solapan = [(gm[h], ge[h]) for h in gm.keys() & ge.keys()]
    conflicto = [v for g in (gm, ge) for v in g.values() if len({c for _, c in v}) > 1]

    print("\n" + "=" * 60 + "\nRESULTADOS DE LA REVISIÓN\n" + "=" * 60)
    mostrar("Grupos de archivos duplicados dentro de MULTILINGÜE", dup_multi)
    mostrar("Grupos de archivos duplicados dentro de ESPAÑOL", dup_es)
    mostrar("Contenidos que aparecen en MULTILINGÜE y también en ESPAÑOL (fuga)", solapan)
    mostrar("Contenidos idénticos etiquetados como real Y fake (conflicto de etiquetas)", conflicto)

    ok = not (dup_multi or dup_es or solapan or conflicto)
    print("\n" + ("Sin duplicados ni fugas exactas detectadas." if ok else
                  "HAY PROBLEMAS: corrígelos antes de entrenar o los resultados serán optimistas."))
    print("Nota: solo detecta copias exactas del archivo. Un mismo audio exportado de otra forma "
          "o un mismo hablante no se detectan aquí.")


if __name__ == "__main__":
    main()