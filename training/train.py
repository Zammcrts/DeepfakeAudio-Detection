"""Entrenamiento en dos etapas para CNN y LSTM (transferencia de aprendizaje).

ETAPA 1  Preentrenamiento con los ~15k audios multilingües (9069 real / 7184 fake).
         La red aprende por sí sola qué distingue real de fake directamente del espectrograma.
ETAPA 2  Se quita la ÚLTIMA capa (clasificador), se pone una nueva y se afina con los audios en español.

Se evalúa SIEMPRE en el mismo test en español, que nunca se usa para entrenar ni elegir nada:
  A) preentrenado, sin ver español   -> ¿generaliza a otro idioma?
  B) preentrenado + afinado          -> ¿cuánto ayuda adaptarlo?
  C) entrenado solo con español      -> línea base: ¿sirve de algo el preentrenamiento?

Estructura de datos (.npy = espectrograma (n_mels, frames)):
    datos/multilingue/{real,fake}/**/*.npy
    datos/espanol/{real,fake}/*.npy

Uso:
    python train.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados
"""
import argparse, csv, json, shutil, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build
from preprocesamiento import acondicionar, normalizar, perturbar_canal
from grupos_es import grupo

LABELS = {"real": 0, "fake": 1}


# ----------------------------------------------------------------------------- datos
def list_files(root):
    paths, labels = [], []
    for cls, y in LABELS.items():
        for p in sorted(Path(root, cls).rglob("*.npy")):
            paths.append(str(p)); labels.append(y)
    if not paths:
        raise SystemExit(f"No encontré .npy en {root}/real y {root}/fake")
    return np.array(paths), np.array(labels)


class SpecDataset(Dataset):
    """Carga .npy, lo acondiciona a n_frames y normaliza por muestra con preprocesamiento.py,
    el mismo código que usa la interfaz en inferencia."""

    def __init__(self, paths, labels, n_frames, train=False, relleno="repetir", canal=0.0, estres=False):
        """canal: probabilidad de perturbar el 'canal' en entrenamiento (ver preprocesamiento.perturbar_canal).
        estres: aplica SIEMPRE una perturbación fija por audio (prueba de robustez, sin aleatoriedad entre corridas)."""
        self.paths, self.labels, self.n_frames, self.train, self.relleno = paths, labels, n_frames, train, relleno
        self.canal, self.estres = canal, estres

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        s = np.load(self.paths[i]).astype(np.float32)
        s = acondicionar(s, self.n_frames, self.relleno, aleatorio=self.train)
        if self.estres:
            s = perturbar_canal(s, np.random.default_rng(1000 + i), p=1.0)
        elif self.train and self.canal > 0:
            s = perturbar_canal(s, None, p=self.canal)
        s = normalizar(s)
        if self.train:  # SpecAugment simple
            f = np.random.randint(0, 12); f0 = np.random.randint(0, s.shape[0] - f)
            t = np.random.randint(0, 20); t0 = np.random.randint(0, s.shape[1] - t)
            s[f0:f0 + f, :] = 0; s[:, t0:t0 + t] = 0
        return torch.from_numpy(s), torch.tensor(self.labels[i])


# ----------------------------------------------------------------------------- entrenamiento
def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    """scaler != None activa precisión mixta (AMP) en entrenamiento: más rápido en GPU, mismos resultados."""
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
                out = model(x)
            loss = loss_fn(out.float(), y)
            if opt:
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            total += loss.item() * len(y); n += len(y)
    return total / n


def fit(model, L_tr, L_va, y_tr, device, epochs, lr, patience, ckpt, tag, lr_patience=5, smooth=0.0, amp=False):
    """Entrena con early stopping sobre la pérdida de validación y deja cargado el mejor modelo.
    smooth: label smoothing en la pérdida de entrenamiento (la de validación es siempre la normal)."""
    w = torch.tensor(len(y_tr) / (2 * np.bincount(y_tr, minlength=2)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=smooth)
    loss_val = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda") if (amp and device == "cuda") else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=lr_patience)
    best, bad, hist = 1e9, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tl = run_epoch(model, L_tr, loss_fn, device, opt, scaler)
        vl = run_epoch(model, L_va, loss_val, device)
        sched.step(vl); hist.append((tl, vl))
        print(f"[{tag}] ep {ep:02d}  train {tl:.4f}  val {vl:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        if vl < best:
            best, bad = vl, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stopping"); break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return hist


def replace_last_layer(model, n_classes=2):
    """Quita la última capa (Linear -> 2 salidas) y pone una nueva con pesos aleatorios."""
    last = model.head[-1]
    model.head[-1] = nn.Linear(last.in_features, n_classes)
    return model


def set_backbone_trainable(model, flag):
    for name, p in model.named_parameters():
        if not name.startswith("head.") or not name.endswith(("weight", "bias")):
            p.requires_grad = flag
    for p in model.head[-1].parameters():   # la capa nueva siempre se entrena
        p.requires_grad = True


# ----------------------------------------------------------------------------- evaluación
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    t0 = time.perf_counter()
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs), time.perf_counter() - t0


@torch.no_grad()
def logits_de(model, loader, device):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        zs.append(model(x.to(device)).float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def ajustar_temperatura(z, y):
    """Temperature scaling: busca T que minimiza la entropía cruzada de softmax(z / T) en validación.
    Solo se permite SUAVIZAR (T >= 1): con una validación casi perfecta el ajuste libre pediría T < 1
    (aún más seguridad), que es justo lo contrario de lo que se quiere."""
    d = (z[:, 1] - z[:, 0]).astype(np.float64)
    ys = y.astype(np.float64)
    mejor, mejor_T = 1e18, 1.0
    for T in np.geomspace(1.0, 20.0, 80):
        s = d / T
        nll = np.mean(np.logaddexp(0, s) - ys * s)       # -log p(y)
        if nll < mejor:
            mejor, mejor_T = nll, float(T)
    return mejor_T


def metrics(y, p, elapsed, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"umbral": float(thr), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "ms_por_audio": 1000 * elapsed / len(y)}


def intervalos(y, p, thr=0.5, n_boot=1000, seed=0):
    """Intervalo de confianza 95 % por bootstrap (remuestreo de audios) para accuracy, F1 y AUC."""
    rng = np.random.default_rng(seed)
    acc, f1, auc = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        pred = p[i] >= thr
        acc.append((pred == y[i]).mean()); f1.append(f1_score(y[i], pred, zero_division=0))
        auc.append(roc_auc_score(y[i], p[i]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy": q(acc), "f1": q(f1), "auc": q(auc)}


def diferencia_auc(y, p1, p2, n_boot=1000, seed=0):
    """AUC(modelo 1) - AUC(modelo 2) sobre los MISMOS audios, con IC 95 % por bootstrap pareado.
    Si el intervalo contiene el 0, no se puede afirmar que un modelo sea mejor que el otro."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return {"delta_auc": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
            "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def evaluate(model, L_va, L_te, device, n_boot=1000, L_x=None):
    """Métricas en test a 0.5 y con el umbral elegido en validación en español (nunca en test)."""
    yv, pv, _ = predict(model, L_va, device)
    fpr, tpr, thr = roc_curve(yv, pv)
    thr_v = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
    yt, pt, el = predict(model, L_te, device)
    r = {"test@0.5": metrics(yt, pt, el), "test@umbral_val": metrics(yt, pt, el, thr_v)}
    if n_boot:
        r["test@0.5"]["ic95"] = intervalos(yt, pt, 0.5, n_boot)
        r["test@umbral_val"]["ic95"] = intervalos(yt, pt, thr_v, n_boot)
    if L_x is not None:   # mismo test con otro "canal": si el modelo se apoya en la grabación, aquí se cae
        yx, px, _ = predict(model, L_x, device)
        r["test_estres"] = {"accuracy": accuracy_score(yx, px >= .5), "auc": roc_auc_score(yx, px)}
    return r, (yt, pt)


def evaluate_todo(model, L_all, device, n_boot=1000, L_x=None):
    """Para la condición A (nunca vio español): todo el español sirve de prueba, no solo el 15 %.
    Más audios = márgenes de error más estrechos."""
    y, p, el = predict(model, L_all, device)
    m = metrics(y, p, el); m["ic95"] = intervalos(y, p, 0.5, n_boot) if n_boot else None
    r = {"todo_espanol@0.5": m}
    if L_x is not None:
        yx, px, _ = predict(model, L_x, device)
        r["test_estres"] = {"accuracy": accuracy_score(yx, px >= .5), "auc": roc_auc_score(yx, px)}
    return r, (y, p)


def partir_espanol(paths, labels, por_grupo, seed, fold=0):
    """70/15/15 estratificado al azar, o (por_grupo) separado por hablante/voz: un mismo grupo nunca queda
    en entrenamiento y prueba a la vez. Devuelve (train, val, test) como índices."""
    idx = np.arange(len(paths))
    if not por_grupo:
        tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
        va, te = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=seed)
        return tr, va, te
    from sklearn.model_selection import StratifiedGroupKFold
    g = np.array([grupo(Path(p).name) for p in paths])   # persona real y su clon IA: mismo grupo
    folds = [te for _, te in StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed).split(idx, labels, g)]
    k = fold % len(folds)                                  # fold k es la prueba, k+1 la validación
    te, va = folds[k], folds[(k + 1) % len(folds)]
    tr = np.concatenate([f for j, f in enumerate(folds) if j not in (k, (k + 1) % len(folds))])
    for nombre, ix in (("train", tr), ("val", va), ("test", te)):
        if len(np.unique(labels[ix])) < 2:
            raise SystemExit(f"Con --por_grupo el conjunto {nombre} quedó con una sola clase: hay muy pocos grupos. "
                             "Revisa python training/grupos_es.py o corre sin --por_grupo.")
    print("por grupo -> grupos en test:", sorted(set(g[te])))
    return tr, va, te


@torch.no_grad()
def registro(model, paths, labels, n_frames, device, modelo, condicion, relleno="repetir"):
    """Matriz de registro (4.2): una fila por audio con clase real, predicción, confianza y tiempo.
    El tiempo es solo el de la red (sin leer el .npy), medido audio por audio."""
    model.eval()
    ds = SpecDataset(paths, labels, n_frames, train=False, relleno=relleno)
    rows = []
    for i in range(len(ds)):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        pred = int(p[1] >= 0.5)
        rows.append({"id_audio": Path(paths[i]).stem, "archivo": str(paths[i]),
                     "clase_real": "fake" if int(y) else "real", "modelo": modelo.upper(), "condicion": condicion,
                     "clasificacion": "fake" if pred else "real", "prob_fake": round(float(p[1]), 4),
                     "confianza": round(float(p[pred]), 4), "correcto": int(pred == int(y)),
                     "tiempo_ms": round(ms, 3)})
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_dir", required=True, help="15k audios multilingües")
    ap.add_argument("--es_dir", required=True, help="audios en español (500 real + 500 fake)")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--n_frames", type=int, default=431,
                    help="frames por muestra. 431 = clip completo de los .npy; menos recorta (y puede caer en relleno)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs_pre", type=int, default=100, help="etapa 1 y línea base (tope)")
    ap.add_argument("--lr_pre", type=float, default=1e-3)
    ap.add_argument("--epochs_head", type=int, default=20, help="afinado 2a: cabeza (tope)")
    ap.add_argument("--epochs_ft", type=int, default=100, help="afinado 2b: toda la red (tope)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15,
                    help="early stopping: épocas sin mejorar la validación. Igualarlo a las épocas lo desactiva")
    ap.add_argument("--lr_patience", type=int, default=5,
                    help="ReduceLROnPlateau: épocas sin mejorar antes de bajar el lr a la mitad (< patience)")
    ap.add_argument("--relleno", choices=["repetir", "mantener"], default="repetir",
                    help="repetir: recorta el relleno sin voz y repite el contenido (recomendado); "
                         "mantener: comportamiento anterior (relleno con el mínimo)")
    ap.add_argument("--canal_prob", type=float, default=0.8,
                    help="probabilidad de perturbar el canal (ruido de fondo, respuesta en frecuencia, ancho de banda) "
                         "en cada muestra de entrenamiento. 0 = sin aumento (reproduce el comportamiento anterior)")
    ap.add_argument("--por_grupo", action="store_true",
                    help="separa el español por hablante/voz (ver grupos_es.py): evaluación más exigente")
    ap.add_argument("--n_boot", type=int, default=1000, help="remuestreos bootstrap para los IC 95 %% (0 = sin IC)")
    ap.add_argument("--seed", type=int, default=42, help="semilla de inicialización y de aumentos")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="semilla de las particiones. Fija: así varias --seed comparten los mismos conjuntos de prueba")
    ap.add_argument("--fold", type=int, default=None,
                    help="validación cruzada por persona (7 folds): este fold es la prueba. Implica --por_grupo")
    ap.add_argument("--pre_dir", default=None,
                    help="reutiliza <modelo>_pre.pt de esta carpeta en vez de repetir la etapa 1 (ahorra horas en CV)")
    ap.add_argument("--smooth", type=float, default=0.05, help="label smoothing en entrenamiento (0 = sin)")
    ap.add_argument("--amp", action="store_true", help="precisión mixta en GPU (más rápido)")
    ap.add_argument("--workers", type=int, default=0,
                    help="procesos para leer datos. 0 = en el proceso principal (más estable en Windows)")
    a = ap.parse_args()

    if a.fold is not None:
        a.por_grupo = True
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

    # --- splits (los mismos para CNN y LSTM) ---
    mp, my = list_files(a.multi_dir)
    mtr_p, mva_p, mtr_y, mva_y = train_test_split(mp, my, test_size=0.15, stratify=my, random_state=a.split_seed)
    ep_, ey = list_files(a.es_dir)
    i_tr, i_va, i_te = partir_espanol(ep_, ey, a.por_grupo, a.split_seed, a.fold or 0)
    etr_p, etr_y, eva_p, eva_y, ete_p, ete_y = ep_[i_tr], ey[i_tr], ep_[i_va], ey[i_va], ep_[i_te], ey[i_te]
    np.save(out / "es_test_idx.npy", i_te)               # posición de cada audio de prueba en la lista completa
    json.dump({"multi_train": mtr_p.tolist(), "multi_val": mva_p.tolist(), "es_train": etr_p.tolist(),
               "es_val": eva_p.tolist(), "es_test": ete_p.tolist()}, open(out / "splits.json", "w"), indent=1)
    n_mels = np.load(mp[0]).shape[0]
    print(f"multi: train={len(mtr_p)} val={len(mva_p)} | español: train={len(etr_p)} val={len(eva_p)} test={len(ete_p)}"
          f" | n_mels={n_mels} frames={a.n_frames} | {device}")

    mk = lambda p, l, tr, x=False: DataLoader(SpecDataset(p, l, a.n_frames, tr, a.relleno, a.canal_prob if tr else 0.0, x),
                                              batch_size=a.batch, shuffle=tr,
                                     num_workers=a.workers, persistent_workers=a.workers > 0,
                                     pin_memory=(device == "cuda"))
    L_mtr, L_mva = mk(mtr_p, mtr_y, True), mk(mva_p, mva_y, False)
    L_etr, L_eva, L_ete = mk(etr_p, etr_y, True), mk(eva_p, eva_y, False), mk(ete_p, ete_y, False)
    L_eall = mk(ep_, ey, False)     # todo el español, para la condición A (no entrena con él)
    L_ete_x, L_eall_x = mk(ete_p, ete_y, False, True), mk(ep_, ey, False, True)   # pruebas de robustez al canal
    L_eva_x = mk(eva_p, eva_y, False, True)             # validación con otro canal: para calibrar sin exceso de confianza
    temps = {}

    results, preds, history, rows = {}, {}, {}, []
    for name in ["cnn", "lstm"]:
        results[name], preds[name], history[name] = {}, {}, {}

        # ---------- ETAPA 1: preentrenamiento multilingüe ----------
        print(f"\n=== {name.upper()} · ETAPA 1: preentrenamiento multilingüe ===")
        model = build(name, n_mels).to(device)
        pre_f = Path(a.pre_dir, f"{name}_pre.pt") if a.pre_dir else None
        # el .done se escribe SOLO cuando el preentrenamiento terminó: un .pt a medias de una corrida interrumpida no cuenta
        if pre_f is not None and pre_f.exists() and pre_f.with_suffix(".done").exists():
            print(f"(reutilizo {pre_f})")
            model.load_state_dict(torch.load(pre_f, map_location=device)); history[name]["pre"] = []
            if pre_f.resolve() != (out / f"{name}_pre.pt").resolve():
                shutil.copy(pre_f, out / f"{name}_pre.pt")
        else:
            history[name]["pre"] = fit(model, L_mtr, L_mva, mtr_y, device, a.epochs_pre, a.lr_pre, a.patience,
                                       out / f"{name}_pre.pt", f"{name} pre", a.lr_patience, a.smooth, a.amp)
            (out / f"{name}_pre.done").touch()
        yv, pv, _ = predict(model, L_mva, device)
        results[name]["0_multilingue_val"] = {"auc": roc_auc_score(yv, pv),
                                              "accuracy": accuracy_score(yv, pv >= .5)}
        results[name]["A_preentrenado_sin_ver_es"], preds[name]["A"] = evaluate(model, L_eva, L_ete, device, a.n_boot, L_ete_x)
        results[name]["A_todo_espanol"], preds[name]["A_todo"] = evaluate_todo(model, L_eall, device, a.n_boot, L_eall_x)
        temps[(name, "A")] = ajustar_temperatura(*logits_de(model, L_eva_x, device)[:1], eva_y)
        rows += registro(model, ete_p, ete_y, a.n_frames, device, name, "A_preentrenado", a.relleno)

        # ---------- ETAPA 2: quitar última capa + afinar con español ----------
        print(f"\n=== {name.upper()} · ETAPA 2: nueva capa final + afinado con español ===")
        replace_last_layer(model); model.to(device)
        set_backbone_trainable(model, False)                      # 2a: solo aprende la capa nueva
        fit(model, L_etr, L_eva, etr_y, device, a.epochs_head, a.lr_head, a.patience,
            out / f"{name}_head.pt", f"{name} capa", a.lr_patience, a.smooth, a.amp)
        set_backbone_trainable(model, True)                       # 2b: se afina toda la red con lr bajo
        history[name]["ft"] = fit(model, L_etr, L_eva, etr_y, device, a.epochs_ft, a.lr_ft, a.patience,
                                  out / f"{name}.pt", f"{name} afinado", a.lr_patience, a.smooth, a.amp)
        results[name]["B_preentrenado_y_afinado_es"], preds[name]["B"] = evaluate(model, L_eva, L_ete, device, a.n_boot, L_ete_x)
        temps[(name, "B")] = ajustar_temperatura(*logits_de(model, L_eva_x, device)[:1], eva_y)
        rows += registro(model, ete_p, ete_y, a.n_frames, device, name, "B_afinado_es", a.relleno)

        # ---------- Línea base: solo español, sin preentrenar ----------
        print(f"\n=== {name.upper()} · LÍNEA BASE: solo español ===")
        base = build(name, n_mels).to(device)
        fit(base, L_etr, L_eva, etr_y, device, a.epochs_pre, a.lr_pre, a.patience, out / f"{name}_solo_es.pt",
            f"{name} solo-es", a.lr_patience, a.smooth, a.amp)
        results[name]["C_solo_espanol"], preds[name]["C"] = evaluate(base, L_eva, L_ete, device, a.n_boot, L_ete_x)
        temps[(name, "C")] = ajustar_temperatura(*logits_de(base, L_eva_x, device)[:1], eva_y)
        results[name]["temperaturas"] = {k: temps[(name, k)] for k in "ABC"}
        rows += registro(base, ete_p, ete_y, a.n_frames, device, name, "C_solo_es", a.relleno)
        results[name]["params"] = sum(p.numel() for p in model.parameters())

    # ---------- ¿CNN y LSTM difieren de verdad? (mismos audios, bootstrap pareado) ----------
    comparacion = {}
    if a.n_boot:
        for k, etiqueta in (("A_todo", "A_todo_espanol"), ("B", "B_test"), ("C", "C_test")):
            (y1, p1), (y2, p2) = preds["cnn"][k], preds["lstm"][k]
            comparacion[etiqueta] = diferencia_auc(y1, p1, p2, a.n_boot)

    json.dump({"config": vars(a), "resultados": results, "comparacion_cnn_vs_lstm": comparacion},
              open(out / "metricas.json", "w"), indent=2)
    with open(out / "registro_pruebas.csv", "w", newline="", encoding="utf-8-sig") as f:   # abre bien en Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for name in preds:
        for k, (yt, pt) in preds[name].items():
            np.save(out / f"pred_{name}_{k}.npy", np.stack([yt, pt]))   # tabla de registro (4.2)

    # ---------- Figuras ----------
    names = {"A": "A: preentrenado", "A_todo": "A: preentrenado (todo el español)",
             "B": "B: preentrenado + afinado", "C": "C: solo español"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a_, name in zip(ax, preds):
        for k, (yt, pt) in preds[name].items():
            fpr, tpr, _ = roc_curve(yt, pt)
            a_.plot(fpr, tpr, label=f"{names[k]} (AUC={roc_auc_score(yt, pt):.3f})")
        a_.plot([0, 1], [0, 1], "k--", lw=.8); a_.set(title=f"ROC {name.upper()} · test español", xlabel="FPR", ylabel="TPR")
        a_.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "roc.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.8))
    for a_, name in zip(ax, results):
        m = results[name]["B_preentrenado_y_afinado_es"]["test@0.5"]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]]); a_.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a_.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        a_.set(title=f"{name.upper()} (B)", xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "fake"],
               yticklabels=["real", "fake"], xlabel="predicho", ylabel="verdadero")
    plt.tight_layout(); plt.savefig(out / "matrices_confusion.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a_, name in zip(ax, history):
        for k, h in history[name].items():
            a_.plot([e[1] for e in h], label=f"{k} val"); a_.plot([e[0] for e in h], "--", label=f"{k} train")
        a_.set(title=f"Pérdida {name.upper()}", xlabel="época"); a_.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "perdidas.png", dpi=150)

    print(f"\nRESUMEN (español, umbral 0.5; entre corchetes, IC 95 % bootstrap) · relleno={a.relleno}"
          f" · por_grupo={a.por_grupo}")
    for name in results:
        for k, sub in [("A_todo_espanol", "todo_espanol@0.5"), ("A_preentrenado_sin_ver_es", "test@0.5"),
                       ("B_preentrenado_y_afinado_es", "test@0.5"), ("C_solo_espanol", "test@0.5")]:
            m = results[name][k][sub]
            ic = (m.get("ic95") or {}).get("auc")
            ic = f" [{ic[0]:.3f}, {ic[1]:.3f}]" if ic else ""
            ex = results[name][k].get("test_estres")
            ex = f"  | con otro canal: acc={ex['accuracy']:.3f} auc={ex['auc']:.3f}" if ex else ""
            print(f"{name.upper():5s} {k:28s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}{ic}{ex}")
    for k, d in comparacion.items():
        veredicto = "no hay diferencia clara" if d["ic95"][0] <= 0 <= d["ic95"][1] else \
            ("CNN mejor" if d["delta_auc"] > 0 else "LSTM mejor")
        print(f"AUC CNN - LSTM en {k:15s} = {d['delta_auc']:+.3f}  IC95 [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]"
              f"  -> {veredicto}")


if __name__ == "__main__":
    main()
    # salida inmediata: en Windows, el cierre normal de CUDA/DataLoader a veces falla DESPUÉS de guardar todo
    # y devuelve un código de error aunque el entrenamiento terminó bien
    import os, sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)