"""Lista los 'grupos' (hablante o voz) que se deducen de los nombres de archivo en español.

Sirve para decidir si se puede separar entrenamiento y prueba por hablante/voz, de modo que una misma
persona o voz sintética no aparezca en los dos lados. Regla aplicada al nombre (sin extensión):
  1. se quita el número final:            'Hansel-nazir_027'  -> 'Hansel-nazir'
  2. si hay guion, se toma lo que sigue:   'Hansel-nazir'      -> 'nazir'      (cuento-hablante)
  3. se quitan dígitos finales y mayúsculas: 'CarlangasIA2'    -> 'carlangasia'
  4. la voz sintética es clon de la persona real: se quita el 'ia' final ('karinaia' -> 'karina')
     y 'anap' se une a 'anapau'. Así la persona real y su clon quedan en el MISMO grupo.

Uso:
    python training/grupos_es.py --es_dir datos/espanol
Revisa la lista: si algún grupo está mal deducido, dime cómo se nombran esos archivos.
"""
import argparse, re
from collections import Counter
from pathlib import Path


def grupo(nombre):
    # Audios de HISPASpoof (hs_<generador>_speaker_<n>_sentence_<m>): el grupo es el hablante, y el real y
    # todos sus falsos quedan juntos.
    h = re.match(r"^hs_.+?_speaker_(\d+)_sentence_\d+$", Path(nombre).stem)
    if h:
        return f"hs{h.group(1)}"
    e = re.match(r"^ext_([a-z0-9]+)_([A-Za-z0-9]+)_\d+$", Path(nombre).stem)
    if e:                              # audios reales de otras fuentes (audios_a_npy.py): una persona/charla = un grupo
        return f"ext{e.group(1)}{e.group(2).lower()}"
    n = re.sub(r"_\d+$", "", Path(nombre).stem)
    if "-" in n:
        n = n.split("-")[-1]
    n = re.sub(r"\d+$", "", n).lower()
    if n.endswith("ia"):
        n = n[:-2]
    return {"anap": "anapau"}.get(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es_dir", required=True)
    a = ap.parse_args()
    for cls in ("real", "fake"):
        archivos = sorted(Path(a.es_dir, cls).rglob("*.npy"))
        c = Counter(grupo(p.name) for p in archivos)
        print(f"\n{cls.upper()}: {len(archivos)} audios en {len(c)} grupos")
        for g, n in c.most_common():
            ej = next(p.name for p in archivos if grupo(p.name) == g)
            print(f"   {g:28s}{n:5d}   p. ej. {ej}")
    print("\nSi hay pocos grupos de una clase (por ejemplo, 4 voces sintéticas), separar por grupo deja\n"
          "una voz entera para la prueba: es una evaluación más exigente y honesta, pero con mucha varianza.")


if __name__ == "__main__":
    main()