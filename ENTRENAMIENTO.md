# Entrenamiento de los modelos CNN y LSTM para detección de audio deepfake

Este documento explica, paso a paso y con lenguaje técnico, **cómo se entrenan y evalúan** los dos modelos del proyecto. Está escrito para que alguien que no conoce el proyecto pueda entender qué se hace, en qué orden y por qué. Corresponde al código de `models.py`, `preprocesamiento.py`, `train.py` y `entrenar_cv.py`. Los resultados obtenidos están en la sección 14 y, con más detalle, en `RESULTADOS.md`.

---

## 1. Visión general

**Problema.** Clasificación binaria supervisada: dado un audio, decidir si es **real** (voz humana) o **fake** (voz sintetizada o manipulada con IA).

**Entrada.** No se usa el audio crudo, sino su **espectrograma** (una matriz `n_mels × frames`, guardada como archivo `.npy`).

**Modelos.** Dos redes neuronales con enfoques distintos, que se comparan entre sí:

- **CNN** (red convolucional): analiza el espectrograma como una imagen.
- **LSTM** (red recurrente): analiza el espectrograma como una secuencia de instantes.

**Estrategia de entrenamiento.** Transferencia de aprendizaje (*transfer learning*) en dos etapas:

```
ETAPA 1  Preentrenamiento
         ~16.5k audios multilingües (9,569 reales / 6,977 fake)
         La red aprende, desde cero, qué distingue lo real de lo fake.
                         │
                         ▼
ETAPA 2  Adaptación al español (fine-tuning)
         Se quita la última capa, se pone una nueva y se reajusta
         con los audios en español (502 reales / 461 fake).
                         │
                         ▼
EVALUACIÓN  Validación cruzada por persona (7 folds): cada audio en español
            se predice con modelos que no vieron a su hablante.
```

Los dos modelos reciben **exactamente los mismos datos, las mismas particiones y el mismo procedimiento**. Así, si su desempeño difiere, la diferencia se debe a la arquitectura y no a otra variable.

---

---

## 2. Los datos de entrada

### 2.1 Qué es un archivo `.npy`

Es una matriz de NumPy con el **mel-espectrograma en decibeles** del audio:

- Eje vertical (`n_mels` = 128): bandas de frecuencia en escala Mel, que imita cómo percibimos el sonido.
- Eje horizontal (`frames` = 431): instantes de tiempo (≈ 10 s).
- Valor de cada celda: energía de esa banda en ese instante, en dB **absolutos**.

Los parámetros con los que se generaron (script de conversión del equipo): `sr = 22050`, `n_fft = 2048`, `hop_length = 512`, `n_mels = 128`, `fmin = 20`, `fmax = 11025`, potencia 2, `power_to_db(ref = 1.0, top_db = None)`, y cada audio se **recorta o rellena con ceros a 10 s antes** de calcular el espectrograma. Esos ceros aparecen en el espectrograma como frames "planos" (todas las bandas en −100 dB).

Las voces sintéticas suelen dejar rastros en esta representación (texturas demasiado uniformes, artefactos en frecuencias altas, transiciones poco naturales). Las redes aprenden a detectarlos a partir de ejemplos etiquetados.

### 2.2 Preparación de cada muestra (`SpecDataset` + `preprocesamiento.py`)

Todo el preprocesamiento vive en `preprocesamiento.py`, y **el mismo código lo usan el entrenamiento y la interfaz**. Cada espectrograma pasa por estos pasos:

1. **Acondicionamiento a `n_frames` frames** (`acondicionar`; `--relleno repetir`, por defecto).
   - Se **recortan los frames planos** de los extremos (el relleno de ceros, que no tiene voz).
   - Si lo que queda es más corto que `n_frames`, el contenido con voz se **repite** hasta llenar la ventana. Si es más largo, se toma una ventana (en entrenamiento, en posición aleatoria; en validación y prueba, desde el inicio, para resultados deterministas).
   - *Por qué:* el diagnóstico mostró que, en el conjunto multilingüe, entre 53 % y 63 % de cada clip es relleno de ceros, mientras que en español casi no hay. Con el recorte aleatorio, muchas ventanas caían en puro relleno, y la duración del clip se volvía una pista ajena a la voz. Con este paso toda muestra tiene voz de principio a fin. `--relleno mantener` reproduce el comportamiento anterior.
2. **Aumento de canal** (solo en entrenamiento, `--canal_prob`, 0.8 por defecto; `perturbar_canal`). Simula otra grabación del mismo audio: suma un ruido de fondo de nivel y color aleatorios, aplica una respuesta en frecuencia suave aleatoria y, en 30 % de los casos, limita el ancho de banda.
   - *Por qué:* el diagnóstico mostró que real y fake en español se distinguían por el suelo de ruido, el rango dinámico y la forma espectral (condiciones de grabación). Una red pequeña aprende ese atajo y sacaría ~100 % sin saber detectar voz sintética. Variar esas propiedades al azar le quita ese apoyo. Los artefactos de síntesis no viven en el suelo de ruido ni en la inclinación del espectro.
3. **Normalización por muestra (z-score):** `s = (s − media) / (desviación + 1e-8)`. Elimina diferencias de volumen o ganancia entre grabaciones, de modo que el modelo se fije en la *forma* del espectro. **La interfaz aplica exactamente la misma normalización.**
4. **SpecAugment (solo en entrenamiento):** se enmascaran con cero una franja aleatoria de frecuencias (hasta 11 bandas) y una franja de tiempo (hasta 19 frames). Tras la normalización, cero equivale a la media. Obliga a la red a no depender de una región específica.

En la interfaz, un audio se convierte con `audio_a_mel` (mismos parámetros y relleno de ceros a 10 s) y `preparar` toma ventanas de 431 frames con solape de 50 % (hasta 8); se promedian las probabilidades de las ventanas.

### 2.3 Etiquetas

Se toman del nombre de la carpeta: `real` → clase **0**, `fake` → clase **1**. Por convención, **fake es la clase positiva** al calcular métricas.

---

## 3. Particiones de los datos

Las particiones usan semilla fija (`--split_seed 42`) para que sean reproducibles. Los índices del conjunto de prueba de cada fold quedan en `es_test_idx.npy`.

**Multilingüe** (estratificado): 85 % entrenamiento (≈ 14,064) y 15 % validación (≈ 2,482). Solo se usa en la etapa 1.

**Español: validación cruzada por persona (7 folds).** Los 963 audios en español pertenecen a **21 personas**, y cada persona tiene un **clon de IA** (por ejemplo `karina` y `karinaia`). Se agrupa por persona (`grupos_es.py`, que une a la persona real con su clon) y se forman 7 grupos de 3 personas. En cada fold:

| Conjunto | Contenido | Uso |
|---|---|---|
| **Prueba** | 1 grupo (3 a 4 personas, real y clon) | Solo evaluación final |
| **Validación** | el grupo siguiente | Early stopping, selección de modelo, umbral y calibración |
| **Entrenamiento** | los 5 grupos restantes (≈ 690 a 710 audios) | Etapa 2 y línea base |

Al terminar los 7 folds, cada audio fue probado exactamente una vez, siempre por modelos que **no vieron a esa persona ni a su clon** (*out-of-fold*). Así las métricas miden generalización a voces nuevas y no reconocimiento de personas.

**Para qué sirve cada conjunto:**

- **Entrenamiento:** los datos con los que la red ajusta sus pesos.
- **Validación:** datos que la red no usa para ajustar pesos, pero que sirven para decidir *cuándo parar*, *qué versión conservar*, *qué umbral usar* y *cómo calibrar*.
- **Prueba:** datos que **nunca** influyen en ninguna decisión.

### 3.1 Verificación y limpieza de los datos

Antes de entrenar se revisaron los datos con `training/verificar_datos.py`, que compara el contenido de cada archivo (huella MD5), y se limpiaron con `training/deduplicar_datos.py`:

- Todos los espectrogramas tienen 128 bandas Mel × 431 frames.
- Se encontraron y apartaron **duplicados exactos**: 246 en el conjunto multilingüe (40 reales y 206 fake) y 40 fake en español. Se movieron a carpetas `*_duplicados`, sin borrarlos.
- No hay archivos compartidos entre el conjunto multilingüe y el de español, ni archivos con la misma huella en clases distintas.
- Conteos finales: multilingüe **9,569 reales y 6,977 fake** (16,546); español **502 reales y 461 fake** (963).

La verificación solo detecta copias exactas, no un mismo audio exportado de otra forma. Por eso el español se separa también por persona.

---
## 4. Arquitecturas (`models.py`)

Ambos modelos terminan en **2 salidas** (logits): índice 0 = real, índice 1 = fake. Aplicar *softmax* a esas salidas produce probabilidades que suman 1, que son los porcentajes que se muestran en la interfaz (p. ej. "78 % fake / 22 % real").

### 4.1 CNN (≈ 404,900 parámetros)

La red trata el espectrograma como una imagen de **1 canal**: `(batch, 1, 128, 128)`.

**Un bloque convolucional** se compone de:

| Capa | Qué hace |
|---|---|
| `Conv2d` 3×3 | Desliza filtros de 3×3 sobre la imagen. Cada filtro aprende a detectar un patrón local (bordes, texturas, armónicos). `padding=1` conserva el tamaño y `bias=False` porque la capa siguiente ya incluye un desplazamiento. |
| `BatchNorm2d` | Normaliza las activaciones de cada lote. Estabiliza y acelera el aprendizaje. |
| `ReLU` | No linealidad: `max(0, x)`. Sin ella, apilar capas equivaldría a una sola operación lineal. |
| `MaxPool2d(2)` | Reduce a la mitad alto y ancho, conservando el valor máximo de cada ventana de 2×2. Aporta cierta invarianza a pequeños desplazamientos. |

**Arquitectura completa:**

| Etapa | Forma de salida |
|---|---|
| Entrada | `(B, 1, 128, 128)` |
| Bloque 1 (1 → 32 filtros) | `(B, 32, 64, 64)` |
| Bloque 2 (32 → 64) | `(B, 64, 32, 32)` |
| Bloque 3 (64 → 128) | `(B, 128, 16, 16)` |
| Bloque 4 (128 → 256) | `(B, 256, 8, 8)` |
| `AdaptiveAvgPool2d(1)` | `(B, 256)` |
| Cabeza: `Dropout(0.4)` → `Linear(256→64)` → `ReLU` → `Dropout(0.4)` → `Linear(64→2)` | `(B, 2)` |

Las primeras capas detectan patrones simples (líneas, bordes) y las profundas los combinan en patrones complejos. El **promedio global** resume cada mapa de características en un solo número, lo que además hace que la red funcione con cualquier duración de entrada. Las formas de la tabla corresponden a 128 frames; con 431 frames el último mapa de la CNN es `(B, 256, 8, 26)` y la secuencia de la LSTM pasa de 128 a 431 pasos.

### 4.2 LSTM (≈ 692,700 parámetros)

La red trata el espectrograma como una **secuencia**: cada frame es un vector de 128 números (el espectro de ese instante). Se transpone a `(batch, frames, 128)`.

**Cómo funciona una LSTM.** Lee la secuencia en orden y mantiene una **memoria** (estado de celda). En cada paso, tres compuertas aprendidas deciden qué información nueva guardar (*input gate*), qué olvidar (*forget gate*) y qué exponer como salida (*output gate*). Eso le permite capturar dependencias a largo plazo: ritmo, entonación y la forma en que se enlazan los sonidos, aspectos donde los sintetizadores suelen fallar.

**Arquitectura completa:**

| Etapa | Detalle | Forma de salida |
|---|---|---|
| Entrada | espectrograma transpuesto | `(B, 128, 128)` |
| `LayerNorm(128)` | normaliza cada frame | `(B, 128, 128)` |
| `LSTM` | 2 capas, 128 unidades ocultas, **bidireccional**, dropout 0.4 entre capas | `(B, 128, 256)` |
| Pooling temporal | concatena la **media** y el **máximo** sobre el tiempo | `(B, 512)` |
| Cabeza: `Dropout(0.4)` → `Linear(512→64)` → `ReLU` → `Dropout(0.4)` → `Linear(64→2)` | | `(B, 2)` |

- **Bidireccional:** una LSTM lee la secuencia hacia adelante y otra hacia atrás; sus salidas se concatenan (por eso 128 × 2 = 256). Cada instante queda contextualizado por el pasado y el futuro.
- **Media + máximo:** la media resume el comportamiento general del audio y el máximo captura los eventos más marcados. La combinación es más robusta que usar solo el último estado.

---

## 5. Cómo aprende una red (conceptos base)

Un ciclo de entrenamiento repite estos pasos sobre lotes de 32 espectrogramas:

1. **Propagación hacia adelante (*forward*):** la red produce 2 logits por muestra.
2. **Pérdida (*loss*):** se mide qué tan equivocada está la predicción.
3. **Retropropagación (*backpropagation*):** se calcula cuánto contribuyó cada peso al error (gradiente).
4. **Actualización:** el optimizador modifica los pesos en la dirección que reduce el error.

Una pasada completa por todos los datos de entrenamiento es una **época**. Se repite durante varias épocas.

> Las **métricas** (exactitud, F1, AUC…) **no intervienen en el entrenamiento**. La red solo optimiza la función de pérdida. Las métricas se calculan al final, con datos que la red no vio, para evaluarla.

### 5.1 Función de pérdida: entropía cruzada ponderada

`CrossEntropyLoss` mide la distancia entre las probabilidades predichas y la etiqueta verdadera: penaliza fuertemente cuando el modelo está muy seguro y se equivoca.

**Pesos por clase.** Hay más audios reales que fake (9,569 vs 6,977). Sin corrección, la red tendería a favorecer la clase mayoritaria. Se pondera cada clase de forma inversamente proporcional a su frecuencia:

```
peso_c = N / (2 · N_c)      →   real ≈ 0.86,   fake ≈ 1.19   (etapa 1)
```

Así, equivocarse con un audio fake cuesta algo más que equivocarse con uno real. Los pesos se recalculan con las etiquetas de cada etapa.

---

---

## 6. Hiperparámetros y mecanismos de control

| Elemento | Valor | Para qué sirve |
|---|---|---|
| Optimizador | **AdamW** | Variante de Adam que adapta la tasa de aprendizaje por parámetro y separa correctamente la regularización de los pesos. |
| Weight decay | 1e-4 | Penaliza pesos grandes (regularización L2). |
| Tamaño de lote | 32 | Muestras por actualización. |
| Learning rate (etapa 1) | 1e-3 | Tamaño del paso de aprendizaje. |
| Dropout | 0.4 | Apaga al azar el 40 % de las neuronas en entrenamiento para evitar el sobreajuste. |
| Recorte de gradiente | norma máx. 5.0 | Evita actualizaciones explosivas (importante en redes recurrentes). |
| Scheduler `ReduceLROnPlateau` | factor 0.5, paciencia 5 (`--lr_patience`) | Si la pérdida de validación no mejora en 5 épocas, reduce el learning rate a la mitad. Debe ser menor que la paciencia del early stopping para que el lr baje antes de detener el entrenamiento. |
| **Early stopping** | paciencia 15 (`--patience`) | Detiene el entrenamiento si la validación no mejora en 15 épocas seguidas. |
| Épocas máximas | 100 (etapa 1) · 20 (2a) · 100 (2b) · 100 (línea base) | Tope; normalmente se detiene antes por el early stopping. Siempre se conserva el modelo con menor pérdida de validación, no el de la última época. |
| Label smoothing | 0.05 (`--smooth`) | Evita que la red se sobreconfíe con pocos datos: las etiquetas objetivo son 0.95 / 0.05 en vez de 1 / 0. Solo en la pérdida de entrenamiento; la de validación es la normal. Eleva el piso de la pérdida de entrenamiento, por eso ya no se parece a corridas anteriores. |
| Precisión mixta | `--amp` | Cálculo en media precisión en GPU: más rápido, mismos resultados prácticos. |
| Semilla | `--seed` (inicialización) y `--split_seed` (particiones) | Reproducibilidad. Con `--split_seed` fijo se pueden usar varias semillas sobre las mismas particiones. |

**Selección del mejor modelo.** En cada época se mide la pérdida en validación y se guarda el modelo **solo si mejora**. Al terminar se recarga el mejor, no el último. Esto protege contra el **sobreajuste** (*overfitting*): cuando la red memoriza el entrenamiento y empeora con datos nuevos, lo que se ve como pérdida de entrenamiento bajando mientras la de validación sube.

**Modos `train` y `eval`.** Durante el entrenamiento, *Dropout* está activo y *BatchNorm* usa estadísticas del lote. Al evaluar se desactiva el dropout y BatchNorm usa estadísticas acumuladas, de modo que las predicciones son deterministas.

---

---

## 7. Etapa 1: preentrenamiento multilingüe

**Objetivo.** Que la red aprenda, con muchos ejemplos variados, características generales que separen voz real de sintética, sin depender de un idioma concreto.

**Procedimiento (para cada modelo):**

1. Se crea la red con pesos aleatorios.
2. Se entrena con la partición multilingüe de entrenamiento (≈ 14,064 audios), con pérdida ponderada, AdamW (lr 1e-3), aumento de canal, SpecAugment y ventanas aleatorias.
3. Tras cada época se mide la pérdida en la validación multilingüe; se aplican scheduler, early stopping y guardado del mejor modelo.
4. Resultado: `cnn_pre.pt` / `lstm_pre.pt`.

En este punto se evalúa el modelo en el test en español **sin haberlo visto nunca**. Es la **condición A** (ver sección 9).

---

## 8. Etapa 2: adaptación al español (*fine-tuning*)

### 8.1 Idea

Una red ya preentrenada ha aprendido características útiles (en la CNN, filtros que detectan texturas; en la LSTM, patrones temporales). Reaprender todo con solo ≈ 700 audios en español provocaría sobreajuste. Lo eficiente es **reutilizar** lo aprendido y ajustar solo lo necesario.

### 8.2 Reemplazo de la última capa

Se elimina la capa final de clasificación, `Linear(64 → 2)`, y se coloca una nueva con pesos aleatorios (`replace_last_layer`). Todo lo anterior conserva sus pesos preentrenados.

Dos partes de la red:

- **Extractor de características (*backbone*):** convoluciones (CNN) o LSTM, que convierten el espectrograma en un vector de rasgos.
- **Cabeza de clasificación (*head*):** las dos capas lineales finales, que convierten ese vector en la decisión real/fake.

### 8.3 Dos fases de ajuste

| Fase | Qué se entrena | Parámetros entrenables | Learning rate | Épocas máx. |
|---|---|---|---|---|
| **2a** | Solo la **cabeza** (la capa intermedia `Linear→64` y la capa nueva). El extractor queda **congelado**. | CNN: 16,578 · LSTM: 32,962 | 1e-3 | 20 |
| **2b** | **Toda la red** | CNN: 404,898 · LSTM: 692,674 | 1e-4 | 100 |

**Por qué dos fases.** La capa nueva parte de pesos aleatorios y, al principio, produce gradientes grandes y poco informativos. Si se entrenara toda la red desde el inicio, esos gradientes podrían **dañar** los pesos preentrenados. Primero se deja que la cabeza se acomode (2a) y después se afina todo con un learning rate 10 veces menor (2b), de modo que los ajustes sean pequeños y cuidadosos.

**Detalle técnico.** En la fase 2a las capas `BatchNorm` de la CNN, aunque sus pesos estén congelados, siguen actualizando sus estadísticas internas (media y varianza acumuladas) con los datos en español, porque la red está en modo `train`. En la práctica esto ayuda a adaptar la normalización al nuevo idioma, pero es un comportamiento a tener presente al describirlo.

**Selección.** Igual que en la etapa 1: validación en español, early stopping, y se conserva el mejor modelo. El resultado final es `cnn.pt` y `lstm.pt` (**condición B**).

---

---

## 9. Condiciones experimentales y evaluación

Para responder la pregunta de investigación se evalúan tres condiciones sobre **los mismos audios de prueba en español** (en cada fold):

| Condición | Descripción | Pregunta que responde |
|---|---|---|
| **A** · Preentrenado | Solo etapa 1; nunca vio español | ¿Generaliza a otro idioma sin ayuda? |
| **B** · Preentrenado + afinado | Etapas 1 y 2 | ¿Cuánto mejora al adaptarlo? |
| **C** · Solo español | Red nueva entrenada únicamente con el español de entrenamiento (línea base) | ¿Aporta algo el preentrenamiento? |

Si B supera a C, el preentrenamiento con datos variados aporta conocimiento transferible. Si A es mucho peor que B, hay una brecha de dominio entre idiomas.

### 9.1 Umbral de decisión

La red entrega una probabilidad de fake. Por defecto se clasifica como fake si `prob_fake ≥ 0.5`. Además se calcula un **umbral alternativo** que maximiza `TPR − FPR` (estadístico J de Youden) **sobre la validación en español**, nunca sobre la prueba, para no ajustar el resultado final con los datos de evaluación. Se reportan ambos (`test@0.5` y `test@umbral_val`).

### 9.2 Métricas

Con **fake como clase positiva**: VP = fakes detectados, VN = reales aceptados, FP = reales marcados como fake, FN = fakes que pasaron como reales.

| Métrica | Fórmula | Interpretación |
|---|---|---|
| Exactitud | (VP + VN) / total | Proporción de aciertos. |
| Precisión | VP / (VP + FP) | De los audios marcados como fake, cuántos lo eran. |
| Sensibilidad (recall) | VP / (VP + FN) | De los fake reales, cuántos se detectaron. |
| F1-score | media armónica de precisión y recall | Equilibrio entre ambas. |
| Matriz de confusión | conteos VN, FP, FN, VP | Tipo de errores que comete. |
| Curva ROC y AUC | TPR vs FPR en todos los umbrales | Capacidad de separar clases **independiente del umbral**; 0.5 = azar, 1.0 = perfecto. |
| Tiempo de procesamiento | ms por audio | Costo computacional de cada modelo. |

El **AUC** es especialmente útil para comparar modelos porque no depende del umbral elegido.


### 9.3 Calibración de las probabilidades

Las redes suelen dar probabilidades demasiado extremas. Para que el "80 % fake" de la interfaz se parezca a un 80 % de acierto, a cada modelo se le ajusta una **temperatura** *T* (*temperature scaling*): las probabilidades salen de `softmax(logits / T)`. *T* se ajusta en la validación en español **con otro canal** (el mismo aumento de canal) y solo se permite *T* ≥ 1, es decir, solo **suavizar**: con una validación casi perfecta el ajuste libre pediría más seguridad, lo contrario de lo que se busca. La temperatura queda en `metricas.json` y en `modelos.json`.

### 9.4 Prueba de robustez "con otro canal"

Además de las métricas normales, cada modelo se evalúa sobre el **mismo conjunto de prueba con una perturbación de canal fija por audio** (ruido de fondo, respuesta en frecuencia, ancho de banda). Si el AUC cae mucho respecto al normal, el modelo se apoyaba en la grabación y no en la voz. Se reporta como "AUC otro canal".

---

## 10. Correcciones tras el diagnóstico (para que los resultados sean confiables)

El diagnóstico (`diagnostico.py`) mostró problemas que se corrigieron:

| Problema detectado | Corrección | Dónde |
|---|---|---|
| En el conjunto multilingüe, 53–63 % de cada clip es relleno de ceros sin voz; en español casi no hay. Los recortes al azar caían en relleno puro y la duración del clip se volvía una pista ajena a la voz. | `--relleno repetir`: se recorta el relleno y el contenido con voz se repite hasta llenar la ventana. | `preprocesamiento.py` |
| Real y fake en español se separaban por el suelo de ruido, el rango dinámico y la forma espectral (condiciones de grabación): un atajo que da ~100 % sin detectar voz sintética. | Aumento de canal (`--canal_prob`) y prueba de robustez "con otro canal". | `preprocesamiento.py`, `train.py` |
| Entrenamiento e interfaz podían preparar el audio de forma distinta. | Un único módulo, `preprocesamiento.py`, con los parámetros reales del script de generación de los `.npy`; se verifican con `probar_preprocesamiento.py`. | `preprocesamiento.py`, `probar_preprocesamiento.py` |
| Con ≈ 145 audios de prueba, el margen de error es de ±0.03–0.04 y CNN y LSTM se alternaban como "mejor" con los mismos datos. | Validación cruzada por persona (963 audios probados), intervalos de confianza por bootstrap y comparación CNN − LSTM con bootstrap pareado. | `entrenar_cv.py` |
| Una persona real y su clon IA podían quedar en lados distintos de la partición: el modelo podría aprender la persona y no la autenticidad. | Partición por persona, con la persona real y su clon siempre juntos. | `grupos_es.py`, `train.py` |

---

## 11. Entrenamiento final confiable (`entrenar_cv.py`)

`entrenar_cv.py` coordina el entrenamiento final:

1. **Validación cruzada por persona (7 folds)** y **varias semillas** (`--seeds`). Para cada semilla y fold llama a `train.py --fold k`. La etapa 1 (multilingüe) se entrena **una sola vez por semilla** y se reutiliza en los demás folds (`--pre_dir`), porque no depende del español.
2. **Agregación.** Junta las predicciones de los folds (una por audio), calcula accuracy, F1 y AUC con intervalo de confianza 95 % (bootstrap), media ± desviación entre semillas, el AUC del conjunto de semillas, el AUC "con otro canal" y la comparación CNN − LSTM con bootstrap pareado. Una diferencia solo cuenta si su intervalo no contiene 0.
3. **Conjunto de modelos.** Escribe `modelos.json` con los checkpoints, su temperatura y su umbral. La interfaz **promedia los modelos de cada etapa**, que es más estable que uno solo. No se escoge un "mejor" fold: elegirlo por su resultado de prueba haría que ese número deje de ser honesto.

Es **reanudable**: un fold se da por completo cuando existen todos sus resultados, y el preentrenamiento solo se reutiliza si terminó (archivo `*_pre.done`). Si se interrumpe, al volver a correr el mismo comando continúa donde se quedó.

---

## 12. Archivos que se generan

| Archivo | Contenido |
|---|---|
| `resultados_cv/seedS/foldK/cnn.pt`, `lstm.pt` | Modelos finales de ese fold (condición B). |
| `.../cnn_pre.pt`, `lstm_pre.pt` | Preentrenados (condición A); iguales en todos los folds de una semilla. |
| `.../cnn_solo_es.pt`, `lstm_solo_es.pt` | Líneas base (condición C). |
| `.../metricas.json` | Métricas por modelo y condición de ese fold, temperaturas y configuración. |
| `.../pred_<modelo>_<A, A_todo, B o C>.npy` | Etiquetas y probabilidades de prueba (matriz de registro 4.2). |
| `.../es_test_idx.npy` | Posición de cada audio de prueba en la lista de 963 audios. |
| `.../registro_pruebas.csv` | Una fila por audio, modelo y condición: clase real, clasificación, probabilidad, confianza, acierto y tiempo. |
| `.../roc.png`, `matrices_confusion.png`, `perdidas.png` | Curvas ROC, matrices de confusión y curvas de pérdida de ese fold. |
| `.../log.txt` | Registro del entrenamiento del fold. |
| `.../*_pre.done` | Marca de que el preentrenamiento terminó (permite reanudar). |
| `resultados_cv/resumen_cv.json` | Resumen out-of-fold: métricas, intervalos y comparación CNN − LSTM. |
| `resultados_cv/modelos.json` | Lista de checkpoints, temperatura y umbral: **lo que carga la interfaz**. |

---

## 13. Cómo ejecutarlo

```bash
pip install -r requirements.txt
python training/verificar_datos.py --multi_dir datos/multilingue --es_dir datos/espanol
# entrenamiento final (reanudable); una semilla tarda aprox. 2 h en una RTX 4070 portátil
python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados_cv --seeds 42 --amp
# opcional: más semillas
python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados_cv --seeds 42 43 44 --amp
# prueba externa con audios propios (externo/real y externo/fake)
python probar_externo.py --resultados resultados_cv --dir externo
```

Un entrenamiento suelto, para pruebas rápidas: `python training/train.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados`.

Estructura de datos esperada:

```
datos/
├── multilingue/{real,fake}/**/*.npy
└── espanol/{real,fake}/*.npy
```

Se usa GPU automáticamente si está disponible. Los hiperparámetros se pueden cambiar con argumentos de `train.py`, que `entrenar_cv.py` reenvía (`--epochs_ft`, `--canal_prob`, `--smooth`, etc.).

---

## 14. Resultados (resumen; detalle en `RESULTADOS.md`)

Entrenamiento final: semilla 42, 7 folds por persona, 963 audios en español probados una vez cada uno (*out-of-fold*), umbral 0.5.

| Modelo | Etapa | Accuracy | AUC [IC 95 %] | AUC con otro canal |
|---|---|---|---|---|
| CNN | A · preentrenado (no vio español) | 0.613 | 0.794 [0.764, 0.825] | 0.759 |
| CNN | B · preentrenado + afinado | 0.991 | 1.000 [1.000, 1.000] | 0.996 |
| CNN | C · solo español | 0.999 | 1.000 [1.000, 1.000] | 1.000 |
| LSTM | A · preentrenado (no vio español) | 0.631 | 0.818 [0.789, 0.846] | 0.721 |
| LSTM | B · preentrenado + afinado | 0.989 | 1.000 [1.000, 1.000] | 0.993 |
| LSTM | C · solo español | 0.998 | 1.000 [1.000, 1.000] | 0.999 |

Diferencia de AUC CNN − LSTM (bootstrap pareado): A **−0.024** [−0.056, +0.009]; B y C 0.000. **No hay diferencia demostrable entre los dos modelos.**

**Lectura.**
1. Con voces en español que no vio, el modelo preentrenado separa real de fake con AUC ≈ 0.8: es la medida más honesta de generalización a otro idioma.
2. Dentro del español, B y C llegan a ≈ 100 % incluso con personas nuevas y con otro canal.
3. C iguala a B, así que con estos datos **no se puede afirmar que el preentrenamiento aporte** en español (efecto de techo).
4. El 100 % no demuestra que el modelo detecte deepfakes en general: puede reflejar las huellas de las pocas herramientas de voz del conjunto, o una pista que el aumento de canal no cubre. Solo una prueba externa (`probar_externo.py`) con otras herramientas de voz y grabaciones propias distingue ambas explicaciones.

---

## 15. Limitaciones a tener presentes

- **Pocas herramientas de voz.** Los falsos en español provienen, según los nombres de archivo, de clones de 21 personas, probablemente hechos con pocas herramientas. El modelo solo conoce los métodos de generación presentes en los datos; con técnicas nuevas el desempeño puede caer hacia lo que mide la condición A (AUC ≈ 0.8).
- **Posibles atajos no cubiertos.** El aumento de canal solo varía ruido de fondo, respuesta en frecuencia y ancho de banda. Pausas, ritmo, duración, códec o el contenido leído (los reales son cuentos) no se controlan.
- **Pocas personas por fold.** Con 3 a 4 personas de prueba por fold, las cifras de un fold son muy variables; por eso se reportan los 7 juntos.
- **Una sola semilla** en el resultado actual: no se mide la variación entre entrenamientos. Ver `--seeds`.
- **Umbral.** En la etapa A el accuracy a 0.5 es bajo aunque el AUC sea ≈ 0.8: el umbral y la temperatura calibrados en validación corrigen el sesgo en la interfaz.
- **Probabilidades.** Aun calibradas, los porcentajes de la interfaz son la confianza de los modelos, no una certeza.
- **Coherencia con la interfaz.** El preprocesamiento debe ser idéntico al del entrenamiento; por eso vive en un solo módulo.
- **Sin prueba externa todavía.** Pendiente de realizar con audios propios y otras herramientas de voz.

---

## 16. Glosario

- **Época:** una pasada completa por los datos de entrenamiento.
- **Lote (*batch*):** grupo de muestras procesadas juntas en cada actualización.
- **Logit:** salida cruda de la red antes de convertirla en probabilidad.
- **Softmax:** función que convierte logits en probabilidades que suman 1.
- **Sobreajuste:** el modelo memoriza el entrenamiento y generaliza mal.
- **Regularización:** técnicas (dropout, weight decay, aumento de datos) que reducen el sobreajuste.
- **Backbone / head:** extractor de características / capas finales de clasificación.
- **Congelar capas:** impedir que sus pesos cambien durante el entrenamiento.
- **Transfer learning / fine-tuning:** reutilizar un modelo preentrenado y adaptarlo a una tarea o dominio nuevo.
- **Línea base (*baseline*):** modelo de referencia contra el que se compara.
- **AUC:** área bajo la curva ROC; mide la separabilidad de las clases sin depender del umbral.
- **Fold:** una de las 7 rondas de la validación cruzada; en cada una, un grupo distinto de personas se reserva para la prueba.
- **Out-of-fold:** predicción de un audio hecha por un modelo que no lo vio ni vio a su hablante.
- **Bootstrap:** remuestreo con reemplazo de los audios para estimar un intervalo de confianza.
- **Calibración / temperatura:** ajuste de las probabilidades para que reflejen mejor la frecuencia real de acierto.
- **Aumento de canal:** perturbaciones aleatorias de ruido de fondo, respuesta en frecuencia y ancho de banda durante el entrenamiento.
- **Atajo (*shortcut learning*):** pista fácil que correlaciona con la etiqueta en los datos pero no es lo que se quiere detectar (p. ej., cómo se grabó).