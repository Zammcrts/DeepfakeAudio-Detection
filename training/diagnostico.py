"""Diagnóstico: ¿por qué los resultados parecen demasiado buenos o raros?

Tres revisiones, cada una responde una sospecha distinta:

 1. ATAJOS SIMPLES. Un clasificador lineal con estadísticas triviales del espectrograma (nivel,
    forma del espectro, silencios) intenta separar real de fake. Si lo logra casi perfecto en
    español, la tarea es trivial y las redes pueden estar usando un atajo (canal, micrófono,
    códec, ancho de banda, una voz concreta) y no señales reales de síntesis.
 2. CASI-DUPLICADOS en español (copias con ruido o recortadas, que no son idénticas byte a byte
    y por eso no las detectó verificar_datos.py). Si un audio de prueba tiene un gemelo en
    entrenamiento, la prueba está contaminada.
 3. SESGO DE EVALUACIÓN del preentrenamiento (requiere los modelos entrenados): compara la
    pérdida de entrenamiento SIN dropout ni aumento contra la de validación, y la validación
    según qué parte del clip se mire (inicio, medio, final). Explica por qué la pérdida de
    validación sale mucho menor que la de entrenamiento.

Uso (desde la carpeta del proyecto, con el entorno activado):
    python training/diagnostico.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados
Sin --out (o sin los .pt) solo corren las revisiones 1 y 2.
"""
import argparse, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import list_files


# ----------------------------------------------------------------------------- 1. atajos simples
def features(p):
    s = np.load(p).astype(np.float32)                       # (n_mels, frames)
    z = (s - s.mean()) / (s.std() + 1e-8)                   # como lo ve la red
    fe = z.mean(0)                                          # energía por frame
    raw = [s.mean(), s.std(), s.min(), s.max()]             # nivel absoluto (la red NO lo ve: se normaliza)
    extra = [fe.std(), float((fe < fe.mean() - fe.std()).mean())]
    return np.concatenate([raw, z.mean(1), z.std(1), extra])


def nombres(n_mels):
    return (["nivel_media", "nivel_std", "nivel_min", "nivel_max"] + [f"banda{b}_media" for b in range(n_mels)]
            + [f"banda{b}_std" for b in range(n_mels)] + ["variacion_energia_en_tiempo", "fraccion_silencio"])


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def con_progreso(paths, fn, nombre, cada=500):
    """Aplica fn a cada archivo y avisa cada 'cada' archivos, para que no parezca congelado."""
    out = []
    for i, p in enumerate(paths, 1):
        out.append(fn(p))
        if i % cada == 0 or i == len(paths):
            print(f"   {nombre}: {i}/{len(paths)} archivos leídos", flush=True)
    return np.stack(out)


def cv(X, y, cols, seed):
    p = cross_val_predict(modelo(), X[:, cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return accuracy_score(y, p >= .5), roc_auc_score(y, p)


def revision_atajos(multi, es, max_multi, seed):
    print("\n" + "=" * 70 + "\n1. ¿SE PUEDE SEPARAR REAL DE FAKE CON ESTADÍSTICAS TRIVIALES?\n" + "=" * 70)
    rng = np.random.default_rng(seed)
    mp, my = multi
    idx = rng.choice(len(mp), min(max_multi, len(mp)), replace=False)
    Xm = con_progreso([mp[i] for i in idx], features, "multilingüe"); ym = my[idx]
    Xe = con_progreso(es[0], features, "español"); ye = es[1]
    print("   entrenando clasificadores simples...", flush=True)
    nm = nombres((Xe.shape[1] - 6) // 2)
    nivel, forma, todo = list(range(0, 4)), list(range(4, Xe.shape[1])), list(range(Xe.shape[1]))

    print(f"{'conjunto':42s}{'características':32s}{'exactitud':>10s}{'AUC':>8s}")
    for nom, X, y in [("español (5-fold)", Xe, ye), (f"multilingüe ({len(idx)} audios, 5-fold)", Xm, ym)]:
        for cn, cols in [("solo nivel (la red no lo ve)", nivel), ("forma espectral normalizada", forma)]:
            a, u = cv(X, y, cols, seed)
            print(f"{nom:42s}{cn:32s}{a:10.3f}{u:8.3f}")
    m = modelo().fit(Xm[:, forma], ym)
    p = m.predict_proba(Xe[:, forma])[:, 1]
    print(f"{'entrena multilingüe → prueba español':42s}{'forma espectral normalizada':32s}"
          f"{accuracy_score(ye, p >= .5):10.3f}{roc_auc_score(ye, p):8.3f}   (comparar con la condición A)")

    print("\nCaracterísticas individuales que más separan real de fake en ESPAÑOL (AUC univariado):")
    aucs = np.array([roc_auc_score(ye, Xe[:, j]) for j in range(Xe.shape[1])])
    for j in np.argsort(-np.abs(aucs - .5))[:8]:
        a = max(aucs[j], 1 - aucs[j])
        print(f"   {nm[j]:32s} AUC={a:.3f}  ({'fake más alto' if aucs[j] > .5 else 'real más alto'})")
    print("\nCómo leerlo: exactitud ≥ 0.95 con estas estadísticas simples en español significa que la tarea\n"
          "es casi trivial (la diferencia está en el canal, ancho de banda, silencios o la voz). Una banda\n"
          "con AUC ≈ 1 señala dónde está el atajo (bandas altas suelen indicar distinto muestreo o códec).")


# ----------------------------------------------------------------------------- 2. casi-duplicados
def firma(p, T0=400):
    s = np.load(p).astype(np.float32)
    z = (s - s.mean()) / (s.std() + 1e-8)
    z = z - z.mean(1, keepdims=True)                        # quita el espectro promedio: queda el contenido
    z = z[:, :T0] if z.shape[1] >= T0 else np.pad(z, ((0, 0), (0, T0 - z.shape[1])), constant_values=0)
    m = z.shape[0] // 4 * 4
    v = z[:m].reshape(m // 4, 4, T0 // 8, 8).mean((1, 3)).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


def revision_duplicados(es, out):
    print("\n" + "=" * 70 + "\n2. CASI-DUPLICADOS EN ESPAÑOL (copias con ruido, recortes o recodificadas)\n" + "=" * 70)
    paths, y = es
    V = con_progreso(paths, firma, "español")
    S = V @ V.T
    np.fill_diagonal(S, -1)
    nn, sim = S.argmax(1), S.max(1)
    q = np.quantile(sim, [.5, .9, .99])
    print(f"Similitud con el vecino más parecido: mediana={q[0]:.3f}  p90={q[1]:.3f}  p99={q[2]:.3f}")
    for thr in (0.95, 0.90, 0.80):
        print(f"   archivos con un gemelo de similitud ≥ {thr}: {(sim >= thr).sum()} de {len(paths)}")
    print("   pares más parecidos:")
    for i in np.argsort(-sim)[:8]:
        print(f"      {sim[i]:.3f}  {Path(paths[i]).name}  ~  {Path(paths[nn[i]]).name}")

    sp = Path(out) / "splits.json" if out else None
    if sp and sp.exists():
        sj = json.load(open(sp))
        clave = lambda p: (etiqueta(p), Path(p).name)       # robusto a rutas absolutas/relativas o \ y /
        pos = {clave(p): i for i, p in enumerate(paths)}
        tr = [pos[clave(p)] for p in sj["es_train"] if clave(p) in pos]
        te = [pos[clave(p)] for p in sj["es_test"] if clave(p) in pos]
        if tr and te:
            mx = (V[te] @ V[tr].T).max(1)
            print(f"\nPrueba vs entrenamiento (partición real usada): {len(te)} audios de prueba")
            for thr in (0.95, 0.90, 0.80):
                print(f"   audios de PRUEBA con un gemelo en ENTRENAMIENTO ≥ {thr}: {(mx >= thr).sum()}")
    print("\nCómo leerlo: si hay muchos pares ≥ 0.95, hay copias casi idénticas y la prueba está contaminada.\n"
          "Esta revisión detecta copias cercanas del mismo audio, NO la misma frase leída o sintetizada dos\n"
          "veces ni al mismo hablante: eso hay que revisarlo con los metadatos de la grabación.")


# ----------------------------------------------------------------------------- 3. sesgo de evaluación
def etiqueta(p):
    for part in Path(p).parts:
        if part in ("real", "fake"):
            return int(part == "fake")
    return 0


def cargar(p, pos, n):
    s = np.load(p).astype(np.float32)
    T = s.shape[1]
    if T >= n:
        st = {"inicio": 0, "medio": (T - n) // 2, "final": T - n}[pos]
        s = s[:, st:st + n]
    else:
        s = np.pad(s, ((0, 0), (0, n - T)), constant_values=s.min())
    return (s - s.mean()) / (s.std() + 1e-8)


def evaluar(torch, model, paths, pos, n, device, bs=64):
    probs = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = torch.from_numpy(np.stack([cargar(p, pos, n) for p in paths[i:i + bs]])).to(device)
            probs.append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
    p = np.concatenate(probs); y = np.array([etiqueta(q) for q in paths])
    ll = -np.mean(np.where(y == 1, np.log(np.clip(p, 1e-7, 1)), np.log(np.clip(1 - p, 1e-7, 1))))
    return ll, accuracy_score(y, p >= .5)


def revision_modelos(out, n_frames, n_mels, seed):
    import torch
    from models import build
    print("\n" + "=" * 70 + "\n3. SESGO DE EVALUACIÓN DEL PREENTRENAMIENTO (modelos de la etapa 1)\n" + "=" * 70)
    sp = Path(out) / "splits.json"
    if not sp.exists():
        print("No encontré splits.json en", out, "→ se omite."); return
    sj = json.load(open(sp)); rng = np.random.default_rng(seed)
    tr = [p for p in sj["multi_train"] if Path(p).exists()]; va = [p for p in sj["multi_val"] if Path(p).exists()]
    if not tr or not va:
        print("Las rutas de splits.json no existen en este equipo (¿se entrenó en otro?) → se omite."); return
    tr = list(rng.choice(tr, min(3000, len(tr)), replace=False)); va = va[:3000]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("cnn", "lstm"):
        ck = Path(out) / f"{name}_pre.pt"
        if not ck.exists():
            print(f"{name}: falta {ck}"); continue
        print(f"   evaluando {name.upper()}...", flush=True)
        m = build(name, n_mels).to(device)
        m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
        print(f"\n{name.upper()}  (ventana de {n_frames} frames; pérdida = log-loss sin pesos de clase)")
        print(f"   {'conjunto':38s}{'pérdida':>9s}{'exactitud':>11s}")
        for nom, paths, pos in [("entrenamiento, modo evaluación (inicio)", tr, "inicio"),
                                ("validación, inicio del clip", va, "inicio"),
                                ("validación, medio del clip", va, "medio"),
                                ("validación, final del clip", va, "final")]:
            l, a = evaluar(torch, m, paths, pos, n_frames, device)
            print(f"   {nom:38s}{l:9.4f}{a:11.3f}")
    print("\nCómo leerlo:\n"
          " - Entrenamiento en modo evaluación ≈ validación: la brecha del log (entrenamiento 0.34 vs\n"
          "   validación 0.08) venía del dropout, el aumento y el recorte aleatorio; es normal.\n"
          " - Entrenamiento (evaluación) también ≫ validación: hay audios difíciles o mal etiquetados\n"
          "   que solo están en entrenamiento, o la validación es más fácil por otra razón.\n"
          " - Validación 'inicio' ≪ 'medio'/'final': el comienzo del clip es más fácil (silencio inicial\n"
          "   distinto, entrada característica de un generador): atajo por posición, la validación\n"
          "   sale optimista porque solo mira el inicio.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_dir", required=True)
    ap.add_argument("--es_dir", required=True)
    ap.add_argument("--out", default=None, help="carpeta de resultados con splits.json y los .pt (opcional)")
    ap.add_argument("--n_frames", type=int, default=128, help="el mismo valor con el que se entrenó")
    ap.add_argument("--max_multi", type=int, default=3000, help="audios multilingües a muestrear en la revisión 1")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print("Leyendo la lista de archivos...", flush=True)
    multi, es = list_files(a.multi_dir), list_files(a.es_dir)
    print(f"   multilingüe: {len(multi[0])} archivos · español: {len(es[0])} archivos", flush=True)
    n_mels = np.load(es[0][0]).shape[0]
    revision_atajos(multi, es, a.max_multi, a.seed)
    revision_duplicados(es, a.out)
    if a.out:
        revision_modelos(a.out, a.n_frames, n_mels, a.seed)
    else:
        print("\n(Sin --out: se omite la revisión 3.)")


if __name__ == "__main__":
    main()