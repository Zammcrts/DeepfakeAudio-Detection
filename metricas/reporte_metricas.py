"""Reporte de las métricas del protocolo (objetivo 3 y matriz de registro 4.2) a partir de resultados_cv/.

Calcula, para CNN y LSTM en cada condición (A, B, C), sobre los audios en español probados fuera de muestra
(cada audio lo predijo un modelo que no vio a su hablante):

    exactitud, precisión, sensibilidad (recall), especificidad, F1, AUC, matriz de confusión, curva ROC
    y tiempo de procesamiento por audio,

con intervalo de confianza 95 % (bootstrap) y con DOS reglas de decisión:
    · umbral 0.5
    · umbral elegido en la validación de cada fold (nunca en la prueba)

Genera en la carpeta --destino (por defecto, metricas/):
    metricas_completas.csv   todas las métricas con sus intervalos (para Excel)
    metricas_completas.md    la misma tabla en Markdown (para pegar en documentos)
    matrices_confusion.png   matrices de confusión (umbral 0.5 y umbral de validación)
    roc.png                  curvas ROC de A, B y C por modelo
    registro_pruebas_total.csv   matriz de registro 4.2: una fila por audio, modelo y condición, de todos los folds

Uso (desde la raíz del proyecto):   python metricas/reporte_metricas.py --out resultados_cv
Los resultados se escriben en metricas/ (cambiable con --destino).
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELOS = ("cnn", "lstm")
ETAPAS = {"A": ("A_preentrenado_sin_ver_es", "A_preentrenado", "A · preentrenado"),
          "B": ("B_preentrenado_y_afinado_es", "B_afinado_es", "B · preentrenado + afinado"),
          "C": ("C_solo_espanol", "C_solo_es", "C · solo español")}


def descubrir(out):
    carpetas = sorted(Path(out).glob("seed*/fold*"), key=lambda p: (p.parent.name, int(p.name[4:])))
    return [d for d in carpetas if (d / "metricas.json").exists() and (d / "es_test_idx.npy").exists()]


def cargar(carpetas):
    """(modelo, etapa) -> (y, p, umbral por audio, ids), promediando p entre semillas si hay varias."""
    acum = {}
    for d in carpetas:
        met = json.load(open(d / "metricas.json", encoding="utf-8"))["resultados"]
        idx = np.load(d / "es_test_idx.npy")
        for m in MODELOS:
            for e, (clave, _, _) in ETAPAS.items():
                y, p = np.load(d / f"pred_{m}_{e}.npy")
                thr = met[m][clave]["test@umbral_val"]["umbral"]
                for i, yy, pp in zip(idx, y, p):
                    acum.setdefault((m, e), {}).setdefault(int(i), []).append((int(yy), float(pp), float(thr)))
    out = {}
    for k, dd in acum.items():
        ids = sorted(dd)
        out[k] = (np.array([dd[i][0][0] for i in ids]),
                  np.array([np.mean([t[1] for t in dd[i]]) for i in ids]),
                  np.array([np.mean([t[2] for t in dd[i]]) for i in ids]), ids)
    return out


def calcular(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"exactitud": div(tp + tn, len(y)), "precision": prec, "recall": rec,
            "especificidad": div(tn, tn + fp), "f1": div(2 * prec * rec, prec + rec),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def con_intervalos(y, p, umbral, n_boot, seed=0):
    base = calcular(y, (p >= umbral).astype(int)); base["auc"] = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    claves = ["exactitud", "precision", "recall", "especificidad", "f1", "auc"]
    muestras = {k: [] for k in claves}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        c = calcular(y[i], (p[i] >= (umbral[i] if np.ndim(umbral) else umbral)).astype(int))
        for k in claves[:-1]:
            muestras[k].append(c[k])
        muestras["auc"].append(roc_auc_score(y[i], p[i]))
    base["ic95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in muestras.items() if v}
    return base


def tiempos(carpetas):
    acum = {}
    for d in carpetas:
        f = d / "registro_pruebas.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            acum.setdefault((r["modelo"].lower(), r["condicion"]), []).append(float(r["tiempo_ms"]))
    return {k: (float(np.mean(v)), float(np.median(v))) for k, v in acum.items()}


def registro_total(carpetas, destino):
    filas = []
    for d in carpetas:
        f = d / "registro_pruebas.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            r = {"semilla": d.parent.name.replace("seed", ""), "fold": d.name.replace("fold", ""), **r}
            filas.append(r)
    if filas:
        with open(destino, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    return len(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resultados_cv", help="carpeta con los resultados del entrenamiento")
    ap.add_argument("--destino", default="metricas", help="carpeta donde se guardan las métricas")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    carpetas = descubrir(a.out)
    if not carpetas:
        raise SystemExit(f"No encontré folds terminados en {a.out}/seed*/fold*")
    destino = Path(a.destino); destino.mkdir(parents=True, exist_ok=True)
    datos = cargar(carpetas); tmp = tiempos(carpetas)
    n = len(next(iter(datos.values()))[0])
    print(f"{len(carpetas)} fold(s) terminados · {n} audios probados fuera de muestra\n")

    filas, res = [], {}
    for m in MODELOS:
        for e, (_, cond, nombre) in ETAPAS.items():
            y, p, thr_v, _ = datos[(m, e)]
            ms = tmp.get((m, cond), (float("nan"), float("nan")))
            for regla, umbral in (("0.5", 0.5), ("validación", thr_v)):
                r = con_intervalos(y, p, umbral, a.n_boot)
                res[(m, e, regla)] = r
                fila = {"modelo": m.upper(), "condicion": nombre, "umbral": regla,
                        **{k: round(r[k], 4) for k in ("exactitud", "precision", "recall", "especificidad", "f1", "auc")},
                        "TN": r["TN"], "FP": r["FP"], "FN": r["FN"], "TP": r["TP"], "ms_por_audio": round(ms[0], 2)}
                for k, (lo, hi) in r["ic95"].items():
                    fila[f"{k}_ic95"] = f"[{lo:.3f}, {hi:.3f}]"
                filas.append(fila)

    with open(destino / "metricas_completas.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)

    cab = "| Modelo | Condición | Umbral | Exactitud | Precisión | Recall | F1 | AUC | ms/audio |\n|---|---|---|---|---|---|---|---|---|\n"
    md = cab + "\n".join(f"| {f['modelo']} | {f['condicion']} | {f['umbral']} | {f['exactitud']:.3f} | {f['precision']:.3f} | "
                         f"{f['recall']:.3f} | {f['f1']:.3f} | {f['auc']:.3f} | {f['ms_por_audio']:.1f} |" for f in filas)
    (destino / "metricas_completas.md").write_text(md + "\n", encoding="utf-8")

    for regla in ("0.5", "validación"):
        print(f"Umbral {regla}" + (" (el elegido en la validación de cada fold)" if regla != "0.5" else ""))
        print(f"{'modelo':7s}{'condición':30s}{'exact.':>8s}{'prec.':>8s}{'recall':>8s}{'F1':>8s}{'AUC':>8s}{'ms/aud':>8s}")
        for f in filas:
            if f["umbral"] == regla:
                print(f"{f['modelo']:7s}{f['condicion']:30s}{f['exactitud']:8.3f}{f['precision']:8.3f}{f['recall']:8.3f}"
                      f"{f['f1']:8.3f}{f['auc']:8.3f}{f['ms_por_audio']:8.1f}")
        print()

    # matrices de confusión
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    for fila_ax, regla in zip(ax, ("0.5", "validación")):
        for a_, (m, e) in zip(fila_ax, [(m, e) for m in MODELOS for e in ETAPAS]):
            r = res[(m, e, regla)]
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]]); a_.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    a_.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            a_.set(title=f"{m.upper()} {e} · umbral {regla}", xticks=[0, 1], yticks=[0, 1],
                   xticklabels=["real", "fake"], yticklabels=["real", "fake"], xlabel="predicho", ylabel="verdadero")
    plt.tight_layout(); plt.savefig(destino / "matrices_confusion.png", dpi=150); plt.close()

    # ROC
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_, m in zip(ax, MODELOS):
        for e, (_, _, nombre) in ETAPAS.items():
            y, p, _, _ = datos[(m, e)]; fpr, tpr, _ = roc_curve(y, p)
            a_.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y, p):.3f})")
        a_.plot([0, 1], [0, 1], "k--", lw=.8)
        a_.set(title=f"ROC {m.upper()} · {n} audios fuera de muestra", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
        a_.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(destino / "roc.png", dpi=150); plt.close()

    k = registro_total(carpetas, destino / "registro_pruebas_total.csv")
    print(f"Escribí en {destino}: metricas_completas.csv / .md, matrices_confusion.png, roc.png y "
          f"registro_pruebas_total.csv ({k} filas).")
    print("ms/audio = tiempo de la red sobre un espectrograma (sin leer el archivo), medido audio por audio.")


if __name__ == "__main__":
    main()