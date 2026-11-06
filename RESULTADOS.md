# Resultados del entrenamiento

Evaluación de modelos CNN y LSTM mediante representaciones espectrales para la detección de audios sintéticos.

Este documento reporta lo que se midió, con sus márgenes de error, y **lo que se puede y no se puede afirmar** con esos números. El método está en `ENTRENAMIENTO.md`.

---

## 1. Resumen

- Se entrenaron una **CNN** y una **LSTM** con mel-espectrogramas, en dos etapas (preentrenamiento con 16,546 audios multilingües y adaptación con 963 audios en español) y se evaluaron con **validación cruzada por persona**: 21 personas (cada una con su clon de IA), 7 folds de 3 personas. Cada audio en español fue probado una vez por modelos que no vieron a su hablante.
- **Generalización a otro idioma** (modelo preentrenado que nunca vio español): AUC **0.79** (CNN) y **0.82** (LSTM).
- **Dentro del español** (preentrenado y afinado, o entrenado solo con español): AUC **1.000** y accuracy **0.99–1.00**, incluso con personas nuevas y con perturbaciones de canal.
- **CNN y LSTM no se distinguen** de forma demostrable (diferencia de AUC en la condición A: −0.024, IC 95 % [−0.056, +0.009]).
- **El preentrenamiento no mostró beneficio** en español: la línea base entrenada solo con español obtiene lo mismo que el modelo preentrenado y afinado.
- **El 100 % dentro del español no debe presentarse como capacidad general de detección.** Falta la prueba con audios y herramientas de voz externas (sección 7).

---

## 2. Configuración

| Elemento | Valor |
|---|---|
| Datos multilingües | 9,569 reales + 6,977 fake (16,546), tras quitar 246 duplicados |
| Datos en español | 502 reales + 461 fake (963), 21 personas; cada persona tiene un clon de IA |
| Espectrograma | 128 mels × 431 frames (≈ 10 s), dB absolutos; sr 22,050, n_fft 2048, hop 512 |
| Validación | cruzada por persona, 7 folds (3 a 4 personas de prueba por fold); persona real y clon siempre juntos |
| Tamaño por fold | entrenamiento ≈ 690–710, validación ≈ 126–129, prueba ≈ 129–144 audios |
| CNN / LSTM | ≈ 404,900 / ≈ 692,700 parámetros |
| Optimización | AdamW, lote 32, lr 1e-3 (etapa 1) y 1e-4 (afinado completo), early stopping (paciencia 15), `ReduceLROnPlateau` |
| Regularización | dropout 0.4, weight decay 1e-4, label smoothing 0.05, SpecAugment, aumento de canal (prob. 0.8) |
| Semillas | 1 (42). Pendiente: 43 y 44 para medir la variación entre entrenamientos |
| Intervalos de confianza | bootstrap, 1,000 remuestreos, 95 % |

**Condiciones evaluadas:** **A** = preentrenado, nunca vio español · **B** = preentrenado + afinado con español · **C** = entrenado solo con español (línea base).

---

## 3. Resultados principales (fuera de muestra, 963 audios)

Umbral de decisión 0.5. "AUC con otro canal" es el AUC del mismo conjunto de prueba con una perturbación fija de ruido de fondo, respuesta en frecuencia y ancho de banda.

| Modelo | Condición | Accuracy | AUC [IC 95 %] | AUC con otro canal |
|---|---|---|---|---|
| CNN | A · preentrenado | 0.613 | 0.794 [0.764, 0.825] | 0.759 |
| CNN | B · preentrenado + afinado | 0.991 | 1.000 [1.000, 1.000] | 0.996 |
| CNN | C · solo español | 0.999 | 1.000 [1.000, 1.000] | 1.000 |
| LSTM | A · preentrenado | 0.631 | 0.818 [0.789, 0.846] | 0.721 |
| LSTM | B · preentrenado + afinado | 0.989 | 1.000 [1.000, 1.000] | 0.993 |
| LSTM | C · solo español | 0.998 | 1.000 [1.000, 1.000] | 0.999 |

(Con una sola semilla, la desviación entre semillas no está definida. Los valores de F1, intervalos de accuracy y matrices de confusión por fold están en `resultados_cv/resumen_cv.json` y en cada carpeta de fold.)

### Comparación CNN − LSTM (AUC, bootstrap pareado)

| Condición | Diferencia | IC 95 % | Conclusión |
|---|---|---|---|
| A | −0.024 | [−0.056, +0.009] | sin diferencia demostrable |
| B | 0.000 | [0.000, 0.000] | empate (ambos en el techo) |
| C | 0.000 | [0.000, 0.000] | empate (ambos en el techo) |

---

## 4. Interpretación

**4.1 Condición A: generalización a otro idioma.** Es la única condición en que el modelo no vio español durante el entrenamiento, así que su AUC (≈ 0.8) es la medida más directa de qué tanto transfiere lo aprendido de un idioma a otro. Separa real de fake bastante mejor que el azar (0.5), pero lejos de perfecto. Con el umbral 0.5, el accuracy es bajo (0.61–0.63): el modelo tiende a clasificar la mayoría como real. El AUC indica que la separación existe; en la interfaz se compensa con el umbral y la temperatura que cada modelo calibra en validación.

**4.2 Condiciones B y C: dentro del español.** Ambas alcanzan AUC 1.000 y accuracy de 0.99 a 1.00 con personas que no vieron. La perturbación de canal casi no las afecta (AUC 0.99–1.00). Esto descarta que dependan de ruido de fondo, respuesta en frecuencia o ancho de banda, pero **no** descarta otras explicaciones:
- Los falsos en español provienen de pocas herramientas de voz, con huellas muy marcadas que un modelo aprende con unos 700 ejemplos.
- Hay otra pista que el aumento de canal no cubre (pausas, ritmo, duración, códec o el texto leído: los reales son cuentos).

**4.3 El preentrenamiento no aportó en español.** C (solo español, sin preentrenar) iguala a B. Como ambos están en el techo, no hay margen para ver una mejora; con este conjunto no se puede ni confirmar ni descartar que el preentrenamiento ayude. El único indicio es que A, sin ver español, ya separa con AUC ≈ 0.8.

**4.4 CNN contra LSTM.** No hay diferencia demostrable. En una corrida previa con una sola partición, cada modelo "ganó" una vez con los mismos datos; ese comportamiento es la variación entre entrenamientos, no una superioridad real.

---

## 5. Evolución de los resultados (por qué se cambió el método)

| Versión | Cambio | AUC de A sobre los 963 audios (CNN / LSTM) | Lectura |
|---|---|---|---|
| Inicial | ventanas de 128 frames, relleno sin tratar | 0.91 / 0.90 | Los clips multilingües tienen 53–63 % de relleno de ceros; se cae en ventanas sin voz. |
| + clip completo y recorte de relleno | `--relleno repetir`, 431 frames | 0.98–1.00 / 0.98–1.00 (dos corridas) | AUC muy alto y B/C perfectos; el diagnóstico mostró que real y fake en español se separaban por el ruido de fondo y la forma espectral. |
| + aumento de canal y label smoothing (**final**) | `--canal_prob 0.8`, `--smooth 0.05` | **0.79 / 0.82** | Quitar la pista de la grabación bajó el AUC de A: lo que queda es más cercano a lo que transfiere de la voz. |

La caída de A de ≈ 0.98 a ≈ 0.80 es coherente con que antes el modelo se apoyaba parcialmente en las condiciones de grabación. Es una inferencia, no una prueba: no se corrió una versión idéntica que cambie solo una cosa a la vez.

---

## 6. Lo que se puede y no se puede afirmar

**Se puede afirmar:**
- Con voces en español que no vio (y con personas de prueba nuevas), el modelo preentrenado separa real de fake con AUC ≈ 0.8 (0.79 CNN, 0.82 LSTM).
- Dentro de los datos en español, un modelo entrenado con otras personas distingue real de fake con AUC ≈ 1.000 y accuracy ≈ 0.99, y eso no depende de ruido de fondo, respuesta en frecuencia ni ancho de banda.
- CNN y LSTM rinden igual con estos datos (diferencia no demostrable).

**No se puede afirmar:**
- Que el sistema detecte deepfakes de herramientas de voz que no están en el conjunto de datos.
- Que funcione con grabaciones de usuarios reales (micrófonos, compresión, ruido) sin probarlo.
- Que el preentrenamiento mejore el resultado en español.
- Que un modelo sea mejor que el otro.

---

## 7. Pendiente (necesario para llamar "confiable" al sistema)

1. **Prueba externa** con `probar_externo.py`: 20–30 voces reales grabadas por el equipo o de fuentes ajenas al conjunto de datos, y 20–30 voces falsas hechas con **otras herramientas** de síntesis, en español.
   - Si B y C aciertan: generalizan mejor de lo que indica A.
   - Si B y C fallan y A se mantiene ≈ 0.8: B y C aprendieron las herramientas del conjunto; la interfaz debe presentarlos con esa advertencia.
2. **Repetir con 2 semillas más** (`--seeds 42 43 44`) para reportar media ± desviación y comparar CNN y LSTM con mayor certeza.
3. **Documentar el origen** de los audios reales y de los falsos en español (herramientas de síntesis, textos, condiciones de grabación).

*Resultado de la prueba externa:* _(por completar)_

---

## 8. Cómo reproducir

```bash
python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados_cv --seeds 42 --amp
```

Resultados en `resultados_cv/resumen_cv.json` (métricas, intervalos, comparación CNN − LSTM) y `resultados_cv/modelos.json` (modelos que usa la interfaz). Tiempo aproximado: 2 h por semilla en una RTX 4070 portátil.