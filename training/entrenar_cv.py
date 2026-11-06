"""Entrenamiento final y confiable: validación cruzada por persona + varias semillas + conjunto de modelos.

Qué hace
  1. Para cada semilla y cada uno de los 7 folds, llama a train.py con --fold k (la prueba son personas que
     el modelo no vio; cada persona real y su clon IA quedan juntas). La etapa 1 (multilingüe) se entrena una
     sola vez por semilla y se reutiliza en los demás folds (--pre_dir).
  2. Junta las predicciones de los folds: cada uno de los 963 audios en español se predice exactamente una vez
     con modelos que no lo vieron ni vieron a su hablante (out-of-fold). Con eso calcula accuracy, F1 y AUC con
     intervalo de confianza 95 %, la media ± desviación entre semillas y el AUC del conjunto de semillas.
  3. Compara CNN y LSTM con bootstrap pareado y reporta la robustez a cambios de canal.
  4. Escribe modelos.json: lista de checkpoints con su temperatura de calibración y umbral. La interfaz promedia
     esos modelos, que es más estable que un solo entrenamiento.

Es reanudable: si se interrumpe, al volver a correr se saltan los folds ya terminados.

Uso (recomendado: una semilla primero, ~1 h; luego las tres, durante la noche):
    python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados_cv --seeds 42
    python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados_cv --seeds 42 43 44
Los argumentos que no reconoce se pasan a train.py (por ejemplo --amp, --canal_prob 0.9, --epochs_ft 60).
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

AQUI = Path(__file__).resolve().parent

ETAPAS = {"A": ("A_preentrenado_sin_ver_es", "{m}_pre.pt"),
          "B": ("B_preentrenado_y_afinado_es", "{m}.pt"),
          "C": ("C_solo_espanol", "{m}_solo_es.pt")}
MODELOS = ("cnn", "lstm")


def entrenar(a, extra):
    for seed in a.seeds:
        base = Path(a.out, f"seed{seed}")
        for f in range(a.folds):
            d = base / f"fold{f}"
            if completo(d):
                print(f"[seed {seed} fold {f}] ya está, lo salto"); continue
            cmd = [sys.executable, "-u", str(AQUI / "train.py"), "--multi_dir", a.multi_dir, "--es_dir", a.es_dir,
                   "--out", str(d), "--seed", str(seed), "--split_seed", str(a.split_seed), "--fold", str(f),
                   "--n_boot", "0"] + extra
            cmd += ["--pre_dir", str(base / "fold0")]   # fold 0: reutiliza lo que ya terminó si se interrumpió; los demás, el de fold 0
            print(f"\n[seed {seed} fold {f}] {' '.join(cmd[2:])}", flush=True)
            d.mkdir(parents=True, exist_ok=True)
            for intento in (1, 2, 3):                       # un fallo suelto (memoria, driver) no debe tirar horas de trabajo
                with open(d / "log.txt", "w", encoding="utf-8") as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if completo(d):
                    if r.returncode != 0:
                        print(f"  el fold terminó y guardó todo, pero el proceso salió con código {r.returncode} "
                              f"(suele pasar al cerrar CUDA en Windows). Lo doy por bueno.", flush=True)
                    break
                print(f"  intento {intento} falló, código {r.returncode} (ver {d / 'log.txt'})", flush=True)
            else:
                print("  FALLÓ 3 veces. Vuelve a correr este comando para reintentar."); sys.exit(1)


def completo(d):
    """Un fold está completo si train.py llegó a escribir todos sus resultados."""
    pedidos = ["metricas.json", "es_test_idx.npy"] + [f"pred_{m}_{k}.npy" for m in MODELOS for k in ("A", "A_todo", "B", "C")]
    return all((d / f).exists() for f in pedidos)


def cargar(a):
    """oof[(modelo, etapa)][seed] = (y, p) con 1 predicción por audio; stress[...] = lista de AUC con otro canal."""
    oof, stress, modelos = {}, {}, {}
    for seed in a.seeds:
        for f in range(a.folds):
            d = Path(a.out, f"seed{seed}", f"fold{f}")
            met = json.load(open(d / "metricas.json", encoding="utf-8"))["resultados"]
            idx = np.load(d / "es_test_idx.npy")
            for m in MODELOS:
                for e, (clave, patron) in ETAPAS.items():
                    y, p = np.load(d / f"pred_{m}_{e}.npy")
                    o = oof.setdefault((m, e), {}).setdefault(seed, {"y": {}, "p": {}})
                    o["y"].update(zip(idx.tolist(), y.astype(int).tolist())); o["p"].update(zip(idx.tolist(), p.tolist()))
                    x = met[m][clave].get("test_estres")
                    if x: stress.setdefault((m, e), []).append(x["auc"])
                    if e == "A" and f > 0:
                        continue                                  # la etapa A es la misma en todos los folds de una semilla
                    modelos.setdefault(m, {}).setdefault(e, []).append({
                        "ckpt": str((d / patron.format(m=m)).relative_to(a.out)).replace("\\", "/"),
                        "T": met[m]["temperaturas"][e],
                        "thr": met[m][clave]["test@umbral_val"]["umbral"]})
    out = {}
    for k, por_seed in oof.items():
        out[k] = {}
        for s, o in por_seed.items():
            idx = sorted(o["y"])
            out[k][s] = (np.array([o["y"][i] for i in idx]), np.array([o["p"][i] for i in idx]))
    return out, stress, modelos


def resumen(a):
    sys.path.insert(0, str(AQUI))
    from train import intervalos, diferencia_auc       # se importa aquí: el proceso que lanza los entrenamientos no necesita torch
    oof, stress, modelos = cargar(a)
    res, ens = {}, {}
    print(f"\nRESULTADOS FUera DE MUESTRA (cada audio predicho por modelos que no vieron a su hablante)".replace("FUera", "fuera"))
    print(f"{len(a.seeds)} semilla(s) × {a.folds} folds · umbral 0.5 · IC 95 % por bootstrap\n")
    print(f"{'mod.':5s}{'etapa':6s}{'acc media±sd':>16s}{'AUC media±sd':>16s}{'AUC conjunto [IC95]':>26s}{'AUC otro canal':>16s}")
    for m in MODELOS:
        for e in "ABC":
            por = oof[(m, e)]
            y = next(iter(por.values()))[0]
            accs = [accuracy_score(yy, pp >= .5) for yy, pp in por.values()]
            aucs = [roc_auc_score(yy, pp) for yy, pp in por.values()]
            p_ens = np.mean([pp for _, pp in por.values()], axis=0)
            ic = intervalos(y, p_ens, 0.5, a.n_boot)
            ens[(m, e)] = (y, p_ens)
            res[f"{m}_{e}"] = {"acc_media": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                               "auc_media": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                               "conjunto": {"accuracy": accuracy_score(y, p_ens >= .5), "f1": f1_score(y, p_ens >= .5),
                                            "auc": roc_auc_score(y, p_ens), "ic95": ic},
                               "auc_otro_canal": float(np.mean(stress[(m, e)])) if (m, e) in stress else None}
            x = res[f"{m}_{e}"]
            print(f"{m.upper():5s}{e:6s}{x['acc_media']:9.3f}±{x['acc_sd']:.3f}{x['auc_media']:9.3f}±{x['auc_sd']:.3f}"
                  f"{x['conjunto']['auc']:11.3f} [{ic['auc'][0]:.3f},{ic['auc'][1]:.3f}]"
                  f"{(x['auc_otro_canal'] if x['auc_otro_canal'] is not None else float('nan')):12.3f}")
    print("\nCNN − LSTM (AUC del conjunto, bootstrap pareado)")
    comp = {}
    for e in "ABC":
        y, p1 = ens[("cnn", e)]; _, p2 = ens[("lstm", e)]
        d = diferencia_auc(y, p1, p2, a.n_boot); comp[e] = d
        v = "sin diferencia clara" if d["ic95"][0] <= 0 <= d["ic95"][1] else ("CNN mejor" if d["delta_auc"] > 0 else "LSTM mejor")
        print(f"  etapa {e}: {d['delta_auc']:+.3f}  IC95 [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]  -> {v}")
    print("\nLectura: 'AUC otro canal' mucho menor que 'AUC conjunto' = el modelo se apoya en la grabación, no en la voz.")
    json.dump({"semillas": a.seeds, "folds": a.folds, "resultados": res, "cnn_vs_lstm": comp},
              open(Path(a.out, "resumen_cv.json"), "w", encoding="utf-8"), indent=2)
    json.dump(modelos, open(Path(a.out, "modelos.json"), "w", encoding="utf-8"), indent=2)
    print(f"\nEscribí {Path(a.out, 'resumen_cv.json')} y {Path(a.out, 'modelos.json')} (lo que usa la interfaz).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_dir", required=True)
    ap.add_argument("--es_dir", required=True)
    ap.add_argument("--out", default="resultados_cv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--solo_resumen", action="store_true", help="no entrena, solo recalcula el resumen")
    a, extra = ap.parse_known_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.solo_resumen:
        entrenar(a, extra)
    resumen(a)


if __name__ == "__main__":
    main()