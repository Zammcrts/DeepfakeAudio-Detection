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
import argparse, csv, json, time
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
    """Carga .npy, recorta/rellena a n_frames y normaliza por muestra (igual que en inferencia)."""

    def __init__(self, paths, labels, n_frames, train=False):
        self.paths, self.labels, self.n_frames, self.train = paths, labels, n_frames, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        s = np.load(self.paths[i]).astype(np.float32)
        T = s.shape[1]
        if T >= self.n_frames:
            start = np.random.randint(0, T - self.n_frames + 1) if self.train else 0
            s = s[:, start:start + self.n_frames]
        else:
            s = np.pad(s, ((0, 0), (0, self.n_frames - T)), constant_values=s.min())
        s = (s - s.mean()) / (s.std() + 1e-8)
        if self.train:  # SpecAugment simple
            f = np.random.randint(0, 12); f0 = np.random.randint(0, s.shape[0] - f)
            t = np.random.randint(0, 20); t0 = np.random.randint(0, s.shape[1] - t)
            s[f0:f0 + f, :] = 0; s[:, t0:t0 + t] = 0
        return torch.from_numpy(s), torch.tensor(self.labels[i])


# ----------------------------------------------------------------------------- entrenamiento
def run_epoch(model, loader, loss_fn, device, opt=None):
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if opt:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item() * len(y); n += len(y)
    return total / n


def fit(model, L_tr, L_va, y_tr, device, epochs, lr, patience, ckpt, tag):
    """Entrena con early stopping sobre la pérdida de validación y deja cargado el mejor modelo."""
    w = torch.tensor(len(y_tr) / (2 * np.bincount(y_tr, minlength=2)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, bad, hist = 1e9, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tl = run_epoch(model, L_tr, loss_fn, device, opt)
        vl = run_epoch(model, L_va, loss_fn, device)
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


def metrics(y, p, elapsed, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"umbral": float(thr), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "ms_por_audio": 1000 * elapsed / len(y)}


def evaluate(model, L_va, L_te, device):
    """Métricas en test a 0.5 y con el umbral elegido en validación en español (nunca en test)."""
    yv, pv, _ = predict(model, L_va, device)
    fpr, tpr, thr = roc_curve(yv, pv)
    thr_v = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
    yt, pt, el = predict(model, L_te, device)
    return {"test@0.5": metrics(yt, pt, el), "test@umbral_val": metrics(yt, pt, el, thr_v)}, (yt, pt)


@torch.no_grad()
def registro(model, paths, labels, n_frames, device, modelo, condicion):
    """Matriz de registro (4.2): una fila por audio con clase real, predicción, confianza y tiempo.
    El tiempo es solo el de la red (sin leer el .npy), medido audio por audio."""
    model.eval()
    ds = SpecDataset(paths, labels, n_frames, train=False)
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
    ap.add_argument("--n_frames", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs_pre", type=int, default=40)
    ap.add_argument("--lr_pre", type=float, default=1e-3)
    ap.add_argument("--epochs_head", type=int, default=10, help="afinado A: solo la capa nueva")
    ap.add_argument("--epochs_ft", type=int, default=30, help="afinado B: toda la red")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="procesos para leer datos. 0 = en el proceso principal (más estable en Windows)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

    # --- splits (los mismos para CNN y LSTM) ---
    mp, my = list_files(a.multi_dir)
    mtr_p, mva_p, mtr_y, mva_y = train_test_split(mp, my, test_size=0.15, stratify=my, random_state=a.seed)
    ep_, ey = list_files(a.es_dir)
    etr_p, ete_p, etr_y, ete_y = train_test_split(ep_, ey, test_size=0.30, stratify=ey, random_state=a.seed)
    eva_p, ete_p, eva_y, ete_y = train_test_split(ete_p, ete_y, test_size=0.50, stratify=ete_y, random_state=a.seed)
    json.dump({"multi_train": mtr_p.tolist(), "multi_val": mva_p.tolist(), "es_train": etr_p.tolist(),
               "es_val": eva_p.tolist(), "es_test": ete_p.tolist()}, open(out / "splits.json", "w"), indent=1)
    n_mels = np.load(mp[0]).shape[0]
    print(f"multi: train={len(mtr_p)} val={len(mva_p)} | español: train={len(etr_p)} val={len(eva_p)} test={len(ete_p)}"
          f" | n_mels={n_mels} frames={a.n_frames} | {device}")

    mk = lambda p, l, tr: DataLoader(SpecDataset(p, l, a.n_frames, tr), batch_size=a.batch, shuffle=tr,
                                     num_workers=a.workers, persistent_workers=a.workers > 0,
                                     pin_memory=(device == "cuda"))
    L_mtr, L_mva = mk(mtr_p, mtr_y, True), mk(mva_p, mva_y, False)
    L_etr, L_eva, L_ete = mk(etr_p, etr_y, True), mk(eva_p, eva_y, False), mk(ete_p, ete_y, False)

    results, preds, history, rows = {}, {}, {}, []
    for name in ["cnn", "lstm"]:
        results[name], preds[name], history[name] = {}, {}, {}

        # ---------- ETAPA 1: preentrenamiento multilingüe ----------
        print(f"\n=== {name.upper()} · ETAPA 1: preentrenamiento multilingüe ===")
        model = build(name, n_mels).to(device)
        history[name]["pre"] = fit(model, L_mtr, L_mva, mtr_y, device, a.epochs_pre, a.lr_pre, a.patience,
                                   out / f"{name}_pre.pt", f"{name} pre")
        yv, pv, _ = predict(model, L_mva, device)
        results[name]["0_multilingue_val"] = {"auc": roc_auc_score(yv, pv),
                                              "accuracy": accuracy_score(yv, pv >= .5)}
        results[name]["A_preentrenado_sin_ver_es"], preds[name]["A"] = evaluate(model, L_eva, L_ete, device)
        rows += registro(model, ete_p, ete_y, a.n_frames, device, name, "A_preentrenado")

        # ---------- ETAPA 2: quitar última capa + afinar con español ----------
        print(f"\n=== {name.upper()} · ETAPA 2: nueva capa final + afinado con español ===")
        replace_last_layer(model); model.to(device)
        set_backbone_trainable(model, False)                      # 2a: solo aprende la capa nueva
        fit(model, L_etr, L_eva, etr_y, device, a.epochs_head, a.lr_head, a.patience,
            out / f"{name}_head.pt", f"{name} capa")
        set_backbone_trainable(model, True)                       # 2b: se afina toda la red con lr bajo
        history[name]["ft"] = fit(model, L_etr, L_eva, etr_y, device, a.epochs_ft, a.lr_ft, a.patience,
                                  out / f"{name}.pt", f"{name} afinado")
        results[name]["B_preentrenado_y_afinado_es"], preds[name]["B"] = evaluate(model, L_eva, L_ete, device)
        rows += registro(model, ete_p, ete_y, a.n_frames, device, name, "B_afinado_es")

        # ---------- Línea base: solo español, sin preentrenar ----------
        print(f"\n=== {name.upper()} · LÍNEA BASE: solo español ===")
        base = build(name, n_mels).to(device)
        fit(base, L_etr, L_eva, etr_y, device, a.epochs_pre, a.lr_pre, a.patience, out / f"{name}_solo_es.pt",
            f"{name} solo-es")
        results[name]["C_solo_espanol"], preds[name]["C"] = evaluate(base, L_eva, L_ete, device)
        rows += registro(base, ete_p, ete_y, a.n_frames, device, name, "C_solo_es")
        results[name]["params"] = sum(p.numel() for p in model.parameters())

    json.dump({"config": vars(a), "resultados": results}, open(out / "metricas.json", "w"), indent=2)
    with open(out / "registro_pruebas.csv", "w", newline="", encoding="utf-8-sig") as f:   # abre bien en Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for name in preds:
        for k, (yt, pt) in preds[name].items():
            np.save(out / f"pred_{name}_{k}.npy", np.stack([yt, pt]))   # tabla de registro (4.2)

    # ---------- Figuras ----------
    names = {"A": "A: preentrenado", "B": "B: preentrenado + afinado", "C": "C: solo español"}
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

    print("\nRESUMEN (test en español, umbral 0.5)")
    for name in results:
        for k in ["A_preentrenado_sin_ver_es", "B_preentrenado_y_afinado_es", "C_solo_espanol"]:
            m = results[name][k]["test@0.5"]
            print(f"{name.upper():5s} {k:30s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")


if __name__ == "__main__":
    main()