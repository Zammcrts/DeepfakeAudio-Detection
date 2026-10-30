"""Aparta los espectrogramas duplicados (mismo contenido) sin borrarlos.

Por defecto SOLO MUESTRA lo que haría (simulacro). Con --aplicar mueve los duplicados a
<carpeta>_duplicados/ (fuera del conjunto de datos), conservando uno por grupo: el que no lleve
sufijo tipo " (1)" y tenga el nombre más corto.

Los grupos con el mismo contenido en clases distintas (real y fake) NO se tocan: se reportan.

Uso:
    python training/deduplicar_datos.py --dirs datos/multilingue datos/espanol            # simulacro
    python training/deduplicar_datos.py --dirs datos/multilingue datos/espanol --aplicar  # mueve
"""
import argparse, hashlib, re, shutil
from collections import defaultdict
from pathlib import Path


def clave(p):
    n = p.name
    return (bool(re.search(r"\(\d+\)", n)), len(n), n)


def procesar(root, aplicar):
    root = Path(root)
    grupos = defaultdict(list)
    for cls in ["real", "fake"]:
        for p in sorted(root.joinpath(cls).rglob("*.npy")):
            grupos[hashlib.md5(p.read_bytes()).hexdigest()].append((p, cls))

    destino = root.parent / (root.name + "_duplicados")
    mover, conflictos = [], []
    for lst in grupos.values():
        if len(lst) < 2:
            continue
        if len({c for _, c in lst}) > 1:
            conflictos.append(lst); continue
        lst = sorted(lst, key=lambda x: clave(x[0]))
        mover += lst[1:]                      # se conserva el primero

    print(f"\n=== {root} ===")
    for cls in ["real", "fake"]:
        print(f"  {cls}: {sum(1 for _, c in mover if c == cls)} duplicados a apartar")
    print(f"  conflictos real/fake (no se tocan): {len(conflictos)}")
    for p, cls in mover[:5]:
        print(f"    apartar: {p}")
    if len(mover) > 5:
        print(f"    ... y {len(mover) - 5} más")

    if aplicar:
        for p, cls in mover:
            d = destino / cls / p.name
            d.parent.mkdir(parents=True, exist_ok=True)
            i = 1
            while d.exists():
                d = d.with_name(f"{p.stem}_{i}{p.suffix}"); i += 1
            shutil.move(str(p), str(d))
        print(f"  -> movidos {len(mover)} archivos a {destino}")
    return len(mover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    total = sum(procesar(d, a.aplicar) for d in a.dirs)
    if not a.aplicar:
        print(f"\nSIMULACRO: no se movió nada. Si estás de acuerdo con los {total} archivos, "
              "repite el comando con --aplicar.")
    else:
        print("\nListo. Vuelve a correr verificar_datos.py para confirmar que quedó limpio.")


if __name__ == "__main__":
    main()
     