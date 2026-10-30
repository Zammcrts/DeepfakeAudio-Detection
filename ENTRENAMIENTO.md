# Entrenamiento de los modelos CNN y LSTM para detección de audio deepfake

Este documento explica, paso a paso y con lenguaje técnico, **cómo se entrenan y evalúan** los dos modelos del proyecto. Está escrito para que alguien que no conoce el proyecto pueda entender qué se hace, en qué orden y por qué. Todo lo descrito corresponde al código de `models.py` y `train.py`.

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
         ~16k audios multilingües (9069 reales / 7184 fake)
         La red aprende, desde cero, qué distingue lo real de lo fake.
                         │
                         ▼
ETAPA 2  Adaptación al español (fine-tuning)
         Se quita la última capa, se pone una nueva y se reajusta
         con los audios en español (≈502 reales / ≈501 fake).
                         │
                         ▼
EVALUACIÓN  Siempre sobre el mismo conjunto de prueba en español,
            que nunca se usó para entrenar ni para tomar decisiones.
```

Los dos modelos reciben **exactamente los mismos datos, las mismas particiones y el mismo procedimiento**. Así, si su desempeño difiere, la diferencia se debe a la arquitectura y no a otra variable.

---

## 2. Los datos de entrada

### 2.1 Qué es un archivo `.npy`

Es una matriz de NumPy con el **mel-espectrograma en decibeles** del audio:

- Eje vertical (`n_mels`, p. ej. 128): bandas de frecuencia en escala Mel, que imita cómo percibimos el sonido.
- Eje horizontal (`frames`): instantes de tiempo.
- Valor de cada celda: energía de esa banda en ese instante.

Las voces sintéticas suelen dejar rastros en esta representación (texturas demasiado uniformes, artefactos en frecuencias altas, transiciones poco naturales). Las redes aprenden a detectarlos automáticamente a partir de ejemplos etiquetados.

### 2.2 Preparación de cada muestra (`SpecDataset`)

Las redes necesitan entradas de tamaño fijo, pero los audios duran distinto. Cada espectrograma pasa por estos pasos:

1. **Ajuste de longitud a 128 frames** (`--n_frames`).
   - Si es más largo: se **recorta**. En entrenamiento el recorte empieza en una posición aleatoria (cada época la red ve un fragmento distinto, lo que actúa como aumento de datos). En validación y prueba siempre se toma el inicio, para que los resultados sean deterministas.
   - Si es más corto: se **rellena** con el valor mínimo del espectrograma, que equivale aproximadamente a silencio.
   - Referencia: con 16 kHz y `hop_length = 512`, 128 frames ≈ 4.1 s. El valor real depende de los parámetros con los que se generaron los `.npy`.

2. **Normalización por muestra (z-score):** `s = (s − media) / (desviación + 1e-8)`. Elimina diferencias de volumen o ganancia entre grabaciones, de modo que el modelo se fije en la *forma* del espectro y no en lo fuerte que se grabó. **La misma normalización debe aplicarse en la interfaz**; si no, las predicciones no son comparables con las del entrenamiento.

3. **SpecAugment (solo en entrenamiento):** se enmascaran con cero una franja aleatoria de frecuencias (hasta 11 bandas) y una franja de tiempo (hasta 19 frames). Tras la normalización, cero equivale a la media. Obliga a la red a no depender de una región específica y reduce el sobreajuste.

### 2.3 Etiquetas

Se toman del nombre de la carpeta: `real` → clase **0**, `fake` → clase **1**. Por convención, **fake es la clase positiva** al calcular métricas.

---

## 3. Particiones de los datos

Todas las particiones son **estratificadas** (conservan la proporción real/fake en cada subconjunto) y usan semilla fija (`--seed 42`) para que sean reproducibles. Los índices quedan guardados en `splits.json`.

| Conjunto | Origen | Proporción | Tamaño aprox. | Uso |
|---|---|---|---|---|
| Multilingüe · entrenamiento | datos multilingües | 85 % | ≈ 13,800 | Etapa 1 |
| Multilingüe · validación | datos multilingües | 15 % | ≈ 2,440 | Etapa 1: monitoreo y selección |
| Español · entrenamiento | español | 70 % | ≈ 702 | Etapa 2 y línea base |
| Español · validación | español | 15 % | ≈ 150 | Etapa 2: monitoreo, selección y umbral |
| Español · **prueba** | español | 15 % | ≈ 151 | **Solo evaluación final** |

**Para qué sirve cada uno:**

- **Entrenamiento:** los datos con los que la red ajusta sus pesos.
- **Validación:** datos que la red no usa para ajustar pesos, pero que sirven para decidir *cuándo parar*, *qué versión del modelo conservar* y *qué umbral usar*.
- **Prueba:** datos que **nunca** influyen en ninguna decisión. Dan la estimación honesta de cómo se comportará el modelo con audios nuevos.

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

Las primeras capas detectan patrones simples (líneas, bordes) y las profundas los combinan en patrones complejos. El **promedio global** resume cada mapa de características en un solo número, lo que además hace que la red funcione con cualquier duración de entrada.

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

**Pesos por clase.** Hay más audios reales que fake (9069 vs 7184). Sin corrección, la red tendería a favorecer la clase mayoritaria. Se pondera cada clase de forma inversamente proporcional a su frecuencia:

```
peso_c = N / (2 · N_c)      →   real ≈ 0.90,   fake ≈ 1.13   (etapa 1)
```

Así, equivocarse con un audio fake cuesta algo más que equivocarse con uno real. Los pesos se recalculan con las etiquetas de cada etapa.

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
| Scheduler `ReduceLROnPlateau` | factor 0.5, paciencia 2 | Si la pérdida de validación no mejora en 2 épocas, reduce el learning rate a la mitad. |
| **Early stopping** | paciencia 7 | Detiene el entrenamiento si la validación no mejora en 7 épocas seguidas. |
| Épocas máximas | 40 (etapa 1) · 10 (2a) · 30 (2b) | Tope; normalmente se detiene antes. |
| Semilla | 42 | Reproducibilidad. |

**Selección del mejor modelo.** En cada época se mide la pérdida en validación y se guarda el modelo **solo si mejora**. Al terminar se recarga el mejor, no el último. Esto protege contra el **sobreajuste** (*overfitting*): cuando la red memoriza el entrenamiento y empeora con datos nuevos, lo que se ve como pérdida de entrenamiento bajando mientras la de validación sube.

**Modos `train` y `eval`.** Durante el entrenamiento, *Dropout* está activo y *BatchNorm* usa estadísticas del lote. Al evaluar se desactiva el dropout y BatchNorm usa estadísticas acumuladas, de modo que las predicciones son deterministas.

---

## 7. Etapa 1: preentrenamiento multilingüe

**Objetivo.** Que la red aprenda, con muchos ejemplos variados, características generales que separen voz real de sintética, sin depender de un idioma concreto.

**Procedimiento (para cada modelo):**

1. Se crea la red con pesos aleatorios.
2. Se entrena con la partición multilingüe de entrenamiento (≈ 13,800 audios), con pérdida ponderada, AdamW (lr 1e-3), SpecAugment y recorte aleatorio.
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
| **2a** | Solo la **cabeza** (la capa intermedia `Linear→64` y la capa nueva). El extractor queda **congelado**. | CNN: 16,578 · LSTM: 32,962 | 1e-3 | 10 |
| **2b** | **Toda la red** | CNN: 404,898 · LSTM: 692,674 | 1e-4 | 30 |

**Por qué dos fases.** La capa nueva parte de pesos aleatorios y, al principio, produce gradientes grandes y poco informativos. Si se entrenara toda la red desde el inicio, esos gradientes podrían **dañar** los pesos preentrenados. Primero se deja que la cabeza se acomode (2a) y después se afina todo con un learning rate 10 veces menor (2b), de modo que los ajustes sean pequeños y cuidadosos.

**Detalle técnico.** En la fase 2a las capas `BatchNorm` de la CNN, aunque sus pesos estén congelados, siguen actualizando sus estadísticas internas (media y varianza acumuladas) con los datos en español, porque la red está en modo `train`. En la práctica esto ayuda a adaptar la normalización al nuevo idioma, pero es un comportamiento a tener presente al describirlo.

**Selección.** Igual que en la etapa 1: validación en español, early stopping, y se conserva el mejor modelo. El resultado final es `cnn.pt` y `lstm.pt` (**condición B**), los que usará la interfaz.

---

## 9. Condiciones experimentales y evaluación

Para responder la pregunta de investigación se evalúan tres condiciones sobre **el mismo conjunto de prueba en español**:

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

---

## 10. Archivos que genera `train.py`

| Archivo | Contenido |
|---|---|
| `cnn.pt`, `lstm.pt` | Modelos finales (condición B), los que usa la interfaz. |
| `cnn_pre.pt`, `lstm_pre.pt` | Modelos preentrenados (condición A). |
| `cnn_solo_es.pt`, `lstm_solo_es.pt` | Líneas base (condición C). |
| `metricas.json` | Todas las métricas por modelo y condición, más la configuración usada. |
| `registro_pruebas.csv` | Una fila por audio, modelo y condición: clase real, clasificación, probabilidad, confianza, acierto y tiempo. |
| `splits.json` | Qué archivo cayó en cada partición. |
| `roc.png` | Curvas ROC de A, B y C por modelo. |
| `matrices_confusion.png` | Matrices de confusión de la condición B. |
| `perdidas.png` | Curvas de pérdida de entrenamiento y validación. |

---

## 11. Cómo ejecutarlo

```bash
pip install -r requirements.txt
python train.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados
```

Estructura de datos esperada:

```
datos/
├── multilingue/{real,fake}/**/*.npy
└── espanol/{real,fake}/*.npy
```

Se usa GPU automáticamente si está disponible; si no, CPU. Los hiperparámetros se pueden cambiar con argumentos (`--epochs_pre`, `--lr_ft`, `--batch`, `--seed`, etc.).

---

## 12. Limitaciones a tener presentes

- **Tamaño del conjunto de prueba.** Con ≈ 151 audios, el intervalo de confianza al 95 % de una exactitud de 0.90 es de aproximadamente ±4.8 puntos. Diferencias de pocos puntos entre CNN y LSTM **pueden no ser significativas**. Conviene repetir el experimento con varias semillas y reportar media y desviación estándar.
- **Fuga por hablante.** Las particiones son aleatorias por audio. Si un mismo hablante aparece en entrenamiento y en validación o prueba, los resultados pueden ser optimistas porque el modelo reconoce a la persona y no la autenticidad. Lo ideal es separar por hablante cuando se tenga esa información.
- **Tipos de síntesis.** El modelo solo conoce los métodos de generación presentes en los datos. Con técnicas de síntesis nuevas el desempeño puede caer.
- **Probabilidades sobreconfiadas.** Las redes tienden a producir probabilidades extremas (p. ej. 98 %) que no equivalen a certeza real. Los porcentajes de la interfaz deben interpretarse como confianza del modelo, no como probabilidad calibrada.
- **Coherencia con la interfaz.** El preprocesamiento (parámetros del espectrograma, longitud de 128 frames y normalización z-score) debe ser idéntico al del entrenamiento.

---

## 13. Glosario

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