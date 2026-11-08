# Entrenamiento y evaluación de los modelos CNN y LSTM

En este documento se describe el procedimiento mediante el cual se entrenan y evalúan los dos modelos del proyecto, de manera que pueda reproducirse y auditarse. Corresponde al código de `models.py`, `preprocesamiento.py`, `train.py` y `entrenar_cv.py`. Los resultados se resumen en la sección 11 y se analizan en [`RESULTADOS.md`](RESULTADOS.md).

## Contenido

1. [Visión general](#1-visión-general)
2. [Datos y particiones](#2-datos-y-particiones)
3. [Preprocesamiento](#3-preprocesamiento)
4. [Arquitecturas](#4-arquitecturas)
5. [Optimización y control del entrenamiento](#5-optimización-y-control-del-entrenamiento)
6. [Entrenamiento en dos etapas](#6-entrenamiento-en-dos-etapas)
7. [Condiciones experimentales y métricas](#7-condiciones-experimentales-y-métricas)
8. [Correcciones metodológicas](#8-correcciones-metodológicas)
9. [Orquestación y archivos generados](#9-orquestación-y-archivos-generados)
10. [Ejecución](#10-ejecución)
11. [Resumen de resultados](#11-resumen-de-resultados)
12. [Limitaciones](#12-limitaciones)

---

## 1. Visión general

**Tarea.** Clasificación binaria supervisada: dado un audio, determinar si corresponde a voz humana (**real**, clase 0) o a voz sintetizada o manipulada con inteligencia artificial (**fake**, clase 1). Por convención, *fake* es la clase positiva en el cálculo de las métricas.

**Entrada.** El modelo no recibe la señal temporal sino su **mel-espectrograma**, una matriz de energía por banda de frecuencia e instante que hace visibles los artefactos característicos de los sintetizadores.

**Modelos comparados.** Una red convolucional (**CNN**), que interpreta el espectrograma como una imagen, y una red recurrente (**LSTM**), que lo interpreta como una secuencia de instantes. Ambos reciben los mismos datos, las mismas particiones y el mismo procedimiento, de modo que una diferencia de desempeño se atribuye a la arquitectura y no a otra variable.

**Estrategia.** Transferencia de aprendizaje en dos etapas, seguida de una evaluación agrupada por hablante:

```
ETAPA 1  Preentrenamiento multilingüe (≈ 16.5 mil audios)
         La red aprende, desde pesos aleatorios, rasgos generales que distinguen real de fake.
                          │
                          ▼
ETAPA 2  Adaptación al español (ajuste fino)
         Se reemplaza la capa final y se reajusta con el conjunto ampliado en español.
                          │
                          ▼
EVALUACIÓN  Validación cruzada de 7 particiones agrupada por hablante: cada audio se
            clasifica con modelos que no vieron a su hablante.
```

---

## 2. Datos y particiones

### 2.1 Conjuntos de datos

El diseño distingue dos grupos de datos con funciones diferentes. El **conjunto multilingüe** (9,569 audios reales y 6,977 sintéticos) se emplea únicamente en el preentrenamiento y permite estudiar si lo aprendido en otros idiomas se transfiere al español. El **conjunto en español ampliado** (`espanol_v3`) se emplea en el ajuste, en la línea base y en toda la evaluación. Integra cuatro orígenes:

- el conjunto original, de 963 audios y 21 hablantes, cada uno con su clon sintético;
- una muestra balanceada de **HISPASpoof** con voces naturales y sintéticas generadas con ElevenLabs, F5-TTS, XTTS v2 y YourTTS (archivos `hs_<herramienta>_speaker_<n>_sentence_<m>`);
- voces reales de **CIEMPIESS**;
- voces reales de **conferencias** (13 fuentes) y de **otras personas** (14), segmentadas en fragmentos de 10 s.

La ampliación obedece a un hallazgo del diagnóstico: con el conjunto original, ambos modelos alcanzaban un desempeño cercano al 100 % que no se mantenía con audios de otras fuentes, lo que indicaba el aprendizaje de rasgos de la grabación y no de la voz sintética. Los conjuntos nuevos se incorporaron mediante copias, sin modificar el original.

### 2.2 Verificación y depuración

Antes de entrenar se verificó la integridad de los datos con `verificar_datos.py` (huella MD5) y se depuraron con `deduplicar_datos.py`. Todos los espectrogramas tienen forma 128 × 431. Se apartaron, sin eliminarlos, los duplicados exactos: 246 en el conjunto multilingüe (40 reales y 206 sintéticos) y 40 sintéticos en español. No existen archivos compartidos entre ambos conjuntos ni idénticos en clases distintas. La verificación solo detecta copias exactas, no un mismo audio exportado de otra forma; por ello la partición se realiza además por hablante.

### 2.3 Particiones

Las particiones emplean semilla fija (`--split_seed 42`) y los índices de prueba de cada fold se conservan en `es_test_idx.npy`.

- **Conjunto multilingüe.** División estratificada: 85 % para entrenamiento (≈ 14,064 audios) y 15 % para validación (≈ 2,482). El preentrenamiento no depende del español y se reutiliza (`*_pre.pt`, `*_pre.done`) en todos los folds.
- **Conjunto en español.** Validación cruzada de 7 particiones agrupada por hablante (`StratifiedGroupKFold`). El módulo `grupos_es.py` asigna el grupo de cada audio: la persona real y su clon comparten grupo; en HISPASpoof el grupo es el hablante (`hs<n>`); en las fuentes externas, la fuente y la persona (`ext<fuente><persona>`).

En cada fold, los conjuntos cumplen funciones independientes:

| Conjunto | Contenido | Función |
|---|---|---|
| Entrenamiento | cinco grupos de hablantes | Ajuste de los pesos (etapa 2 y línea base) |
| Validación | un grupo adicional | Decidir cuándo detener el entrenamiento, qué versión conservar, qué umbral usar y cómo calibrar |
| Prueba | un grupo adicional | Evaluación final; no interviene en ninguna decisión |

Al concluir los siete folds, cada audio fue evaluado una única vez por modelos que no conocieron a su hablante (*out-of-fold*). Esta propiedad hace que las métricas midan generalización a voces nuevas y no reconocimiento de personas.

---

## 3. Preprocesamiento

El módulo `preprocesamiento.py` concentra todas las transformaciones y es el mismo para el entrenamiento y para la interfaz de prueba. Esta unicidad garantiza que el audio que clasifica la interfaz reciba el mismo tratamiento que el evaluado.

### 3.1 Representación

Cada audio se convierte en un mel-espectrograma en decibeles absolutos con los parámetros `sr = 22,050 Hz`, `n_fft = 2048`, `hop = 512`, `n_mels = 128`, `fmin = 20 Hz` y potencia 2, para una ventana fija de 10 s (431 frames). Los segmentos más cortos se completan con ceros, que aparecen como frames planos.

### 3.2 Pasos aplicados a cada muestra

1. **Limitación de banda (`BANDA = 16000`).** El audio se carga a 16 kHz y se remuestrea a 22,050 Hz, lo que descarta el contenido por encima de 8 kHz. Los datos de entrenamiento poseen aproximadamente ese ancho de banda; los audios con frecuencia de muestreo superior (p. ej., mp3 a 44.1 o 48 kHz) incorporan energía adicional que el modelo interpretaba como indicio de audio no real, lo que producía falsos positivos sobre voces reales de teléfono. Los conjuntos incorporados posteriormente se convirtieron con la misma banda (`--sr_banda 16000`). Con `banda = 0` se reproduce el comportamiento anterior.
2. **Acondicionamiento a `n_frames` (`--relleno repetir`).** Se recortan los frames planos de los extremos y el contenido con voz se repite hasta completar la ventana; si es más largo, se toma una ventana (aleatoria en entrenamiento, desde el inicio en validación y prueba). Con ello se evita que la duración del clip actúe como indicio, ya que entre 53 % y 63 % de cada clip multilingüe es relleno.
3. **Aumento de canal** (solo en entrenamiento, `--canal_prob 0.8`). Simula otra grabación del mismo audio mediante ruido de fondo de nivel y color aleatorios, una respuesta en frecuencia suave aleatoria y, con probabilidad 0.3, limitación del ancho de banda. La opción `--canal_fuerte`, empleada en el entrenamiento final, amplía los rangos: ruido de color de desviación 8 dB, respuesta en frecuencia de desviación 8 dB, inclinación global de ±15 dB, filtro pasa-bajos con probabilidad 0.6 y corte entre 0.35 y 0.95 de la banda, y caída de hasta 40 dB. Su finalidad es impedir que el modelo se apoye en las condiciones de grabación, pues los artefactos de síntesis no residen en el suelo de ruido ni en la inclinación del espectro.
4. **Normalización por muestra (z-score).** Elimina diferencias de ganancia entre grabaciones para que el modelo atienda a la forma del espectro.
5. **SpecAugment** (solo en entrenamiento). Enmascara una franja de frecuencias (hasta 11 bandas) y una de tiempo (hasta 19 frames), de modo que la red no dependa de una región específica.

En inferencia, `audio_a_mel` aplica la limitación de banda y `preparar` toma ventanas de 431 frames con solape del 50 % (hasta 8); las probabilidades de las ventanas se promedian.

### 3.3 Formato de almacenamiento: `.npy` en lugar de imágenes

Los espectrogramas se almacenan como matrices numéricas de NumPy (`.npy`, `float32`) y no como imágenes (PNG o JPG). La decisión se fundamenta en que el espectrograma, aunque se represente visualmente como una imagen, es en esencia una medición física: cada celda contiene la energía, en decibeles, de una banda de frecuencia en un instante. Para un clasificador, la información relevante es ese valor numérico, y cualquier formato de imagen introduce transformaciones que lo alteran.

- **Conservación de la precisión.** Un `.npy` guarda el valor exacto en punto flotante. Una imagen estándar emplea 8 bits por canal, es decir, 256 niveles posibles; representar un intervalo dinámico de más de 100 dB en ese número de niveles obliga a cuantizar, con la consecuente pérdida de detalle en las diferencias de energía finas, que son precisamente las que pueden delatar una síntesis.
- **Ausencia de mapas de color.** Para generar una imagen es necesario asignar cada valor a un color (por ejemplo, *viridis*). Esa asignación es arbitraria, expande la información de un canal a tres y obliga a la red a recuperar una magnitud que ya se conocía. El `.npy` se entrega a las redes como una matriz de un solo canal, sin intermediarios.
- **Compresión sin pérdidas de información útil.** El formato JPG comprime con pérdidas y puede suavizar o crear artefactos en las zonas de alta frecuencia y en los bordes, que coinciden con las huellas que buscan los modelos; el resultado podría enmascarar rasgos de la síntesis o introducir rasgos falsos que dependen del algoritmo de compresión y no del audio. El formato PNG evita esta pérdida, pero mantiene la cuantización a 8 bits.
- **Escala absoluta.** El procesamiento emplea decibeles absolutos y normaliza después por muestra (z-score). Al guardar una imagen, lo habitual es reescalar cada una a su propio mínimo y máximo, lo que elimina el nivel absoluto y vuelve incomparables los espectrogramas entre sí.
- **Resolución tiempo-frecuencia intacta.** Las imágenes suelen redimensionarse (por ejemplo, a 224 × 224) o incluir ejes y márgenes, lo que interpola o recorta la matriz. El `.npy` conserva exactamente 128 bandas × 431 frames, de modo que cada fila mantiene su significado en frecuencia y cada columna, su posición en el tiempo. Esto es indispensable en la LSTM, que lee el espectrograma como una secuencia de instantes.
- **Coherencia entre entrenamiento e inferencia.** Con `.npy`, la interfaz genera la matriz con `audio_a_mel` y la entrega directamente al modelo, de la misma forma que se hizo en el entrenamiento. Con imágenes habría que reproducir en ambos extremos la misma cadena de renderizado, escalado y compresión, con el riesgo de que diverja.
- **Eficiencia.** Un espectrograma de 128 × 431 ocupa aproximadamente 215 KB en `float32`, se lee directamente como arreglo y no requiere decodificar ni convertir, lo que acelera la carga de más de 17 mil muestras en cada época.

Las imágenes se reservan para la **visualización** de los resultados (por ejemplo, las figuras de `metricas_v3/figuras/`), donde su función es la comunicación y no la entrada a los modelos. Las redes convolucionales no requieren una imagen en sentido estricto: operan sobre cualquier matriz bidimensional, por lo que interpretar el espectrograma como "imagen" es una analogía de procesamiento y no un requisito de formato.

---

## 4. Arquitecturas

Ambos modelos terminan en dos salidas (*logits*), una por clase. La función *softmax* las convierte en las probabilidades que presenta la interfaz.

### 4.1 CNN (≈ 404,900 parámetros)

La CNN interpreta el espectrograma como una imagen de un canal. Cada bloque convolucional combina `Conv2d` 3×3 (detección de patrones locales), `BatchNorm2d` (estabilización del aprendizaje), `ReLU` (no linealidad) y `MaxPool2d` (reducción espacial con cierta invarianza a desplazamientos). Cuatro bloques con 32, 64, 128 y 256 filtros extraen rasgos de complejidad creciente; un promedio global los resume y una cabeza `Dropout → Linear(256→64) → ReLU → Dropout → Linear(64→2)` produce la decisión. El promedio global permite, además, procesar entradas de duración distinta.

### 4.2 LSTM (≈ 692,700 parámetros)

La LSTM interpreta el espectrograma como una secuencia de 431 vectores de 128 valores. Tras una `LayerNorm`, una LSTM **bidireccional** de dos capas y 128 unidades ocultas recorre la secuencia en ambos sentidos, de modo que cada instante queda contextualizado por el pasado y el futuro; esto permite capturar dependencias a largo plazo como el ritmo y el enlace entre sonidos. Un *pooling* temporal concatena la media (comportamiento general) y el máximo (eventos más marcados), y una cabeza `Dropout → Linear(512→64) → ReLU → Dropout → Linear(64→2)` produce la decisión.

---

## 5. Optimización y control del entrenamiento

El entrenamiento repite, sobre lotes de 32 espectrogramas, el ciclo de propagación hacia adelante, cálculo de la pérdida, retropropagación y actualización de pesos. Las métricas de evaluación no intervienen en este ciclo: la red optimiza únicamente la función de pérdida.

**Función de pérdida.** Entropía cruzada ponderada por clase, con pesos inversamente proporcionales a la frecuencia (`peso_c = N / (2·N_c)`), de modo que el desbalance entre reales y sintéticos no favorezca a la clase mayoritaria. Los pesos se recalculan en cada etapa.

**Hiperparámetros.**

| Elemento | Valor | Función |
|---|---|---|
| Optimizador | AdamW | Tasa de aprendizaje adaptativa con regularización de pesos desacoplada |
| *Weight decay* | 1×10⁻⁴ | Penaliza pesos grandes |
| Tasa de aprendizaje | 1×10⁻³ (etapa 1 y cabeza); 1×10⁻⁴ (ajuste completo) | Magnitud de cada actualización |
| *Dropout* | 0.4 | Reduce el sobreajuste |
| Recorte de gradiente | norma máxima 5.0 | Evita actualizaciones explosivas, relevantes en redes recurrentes |
| `ReduceLROnPlateau` | factor 0.5, paciencia 5 | Reduce la tasa si la validación no mejora; su paciencia es menor que la del *early stopping* |
| *Early stopping* | paciencia 15 | Detiene el entrenamiento sin mejora sostenida |
| Épocas máximas | 100 (etapa 1), 20 (cabeza), 100 (ajuste y línea base) | Tope de seguridad |
| *Label smoothing* | 0.05 | Evita la sobreconfianza con pocos datos; solo en la pérdida de entrenamiento |
| Precisión mixta | `--amp` | Menor tiempo de cómputo en GPU |

**Selección del modelo.** Tras cada época se mide la pérdida de validación y se guarda el modelo solo si mejora; al finalizar se recarga el mejor y no el último. Esta práctica protege contra el sobreajuste, que se manifiesta como una pérdida de entrenamiento decreciente con una de validación creciente.

---

## 6. Entrenamiento en dos etapas

### 6.1 Etapa 1: preentrenamiento multilingüe

El objetivo es que la red aprenda, a partir de muchos ejemplos variados, rasgos generales de la voz sintética independientes del idioma. Para cada modelo, se parte de pesos aleatorios y se entrena con la partición multilingüe de entrenamiento, con aumento de canal, SpecAugment y ventanas aleatorias. El resultado (`cnn_pre.pt`, `lstm_pre.pt`) se evalúa en español sin haberlo visto nunca, lo que constituye la condición A.

### 6.2 Etapa 2: adaptación al español

Reentrenar toda la red con unos cientos de audios en español produciría sobreajuste; resulta más eficiente reutilizar lo aprendido y ajustar solo lo necesario. Para ello se reemplaza la capa final `Linear(64→2)` por una nueva con pesos aleatorios, conservando el extractor de características (convoluciones o LSTM). El ajuste procede en dos fases:

| Fase | Parámetros entrenables | Tasa de aprendizaje | Épocas máx. |
|---|---|---|---|
| 2a: solo la cabeza (extractor congelado) | CNN 16,578 · LSTM 32,962 | 1×10⁻³ | 20 |
| 2b: toda la red | CNN 404,898 · LSTM 692,674 | 1×10⁻⁴ | 100 |

La capa nueva produce al inicio gradientes grandes y poco informativos que podrían deteriorar los pesos preentrenados; por ello primero se acomoda la cabeza (2a) y luego se afina toda la red con una tasa diez veces menor (2b). En la fase 2a, las capas `BatchNorm` de la CNN, aunque con pesos congelados, actualizan sus estadísticas con los datos en español, lo que adapta la normalización al nuevo idioma. El resultado (`cnn.pt`, `lstm.pt`) corresponde a la condición B.

---

## 7. Condiciones experimentales y métricas

### 7.1 Condiciones

Las tres condiciones se evalúan sobre los mismos audios de prueba de cada fold y responden preguntas distintas:

| Condición | Descripción | Pregunta que responde |
|---|---|---|
| **A** | Solo etapa 1; nunca vio español | ¿Generaliza a otro idioma sin adaptación? |
| **B** | Etapas 1 y 2 | ¿Cuánto mejora al adaptarlo al español? |
| **C** | Red nueva entrenada solo con el español de entrenamiento | ¿Aporta algo el preentrenamiento? |

Si B supera a C, el preentrenamiento aporta conocimiento transferible; si A es muy inferior a B, existe una brecha de dominio entre idiomas.

### 7.2 Umbral y calibración

La red entrega una probabilidad de *fake*, y por defecto se clasifica como tal si `prob ≥ 0.5`. Adicionalmente se calcula un umbral que maximiza `TPR − FPR` (índice J de Youden) **sobre la validación**, nunca sobre la prueba, para no ajustar el resultado final con los datos de evaluación. Se reportan ambos (`test@0.5` y `test@umbral_val`).

Para que un "80 % fake" se aproxime a una frecuencia real de acierto del 80 %, a cada modelo se le ajusta una **temperatura** *T* (`softmax(logits / T)`) sobre la validación con aumento de canal, restringida a *T* ≥ 1 (solo se permite suavizar, pues con una validación casi perfecta el ajuste libre exigiría mayor seguridad). El umbral y la temperatura se guardan en `metricas.json` y `modelos.json`.

### 7.3 Métricas

| Métrica | Fórmula | Información que aporta |
|---|---|---|
| Exactitud | (VP + VN) / total | Proporción global de aciertos |
| Precisión | VP / (VP + FP) | De los marcados como fake, cuántos lo eran |
| Sensibilidad (*recall*) | VP / (VP + FN) | De los fake, cuántos se detectaron |
| Especificidad | VN / (VN + FP) | De los reales, cuántos se aceptaron |
| F1 | media armónica de precisión y recall | Equilibrio entre ambas |
| AUC (ROC) | área bajo TPR vs FPR | Separabilidad independiente del umbral (0.5 = azar) |
| Matriz de confusión | VN, FP, FN, VP | Tipo de errores |
| Tiempo | ms por audio | Costo computacional |

El AUC es la medida principal de comparación porque no depende del umbral. Los intervalos de confianza al 95 % se estiman por *bootstrap* (1,000 remuestreos) y la diferencia CNN − LSTM mediante *bootstrap* pareado sobre los mismos audios, considerándose demostrable solo si el intervalo no contiene el cero.

### 7.4 Prueba de robustez de canal

Cada modelo se evalúa además sobre el mismo conjunto de prueba con una perturbación de canal fija por audio. Una caída marcada del AUC indicaría dependencia de la grabación y no de la voz.

---

## 8. Correcciones metodológicas

El diagnóstico (`diagnostico.py`) y las pruebas externas revelaron problemas que se corrigieron para que los resultados sean confiables:

| Problema detectado | Corrección | Módulo |
|---|---|---|
| Entre 53 % y 63 % de cada clip multilingüe es relleno de ceros; los recortes aleatorios caían en relleno y la duración actuaba como indicio | Relleno `repetir`: se recorta el relleno y se repite el contenido con voz | `preprocesamiento.py` |
| Real y fake en español se separaban por suelo de ruido, rango dinámico y forma espectral (condiciones de grabación) | Aumento de canal y prueba de robustez | `preprocesamiento.py`, `train.py` |
| El entrenamiento y la interfaz podían preparar el audio de forma distinta | Módulo único de preprocesamiento, verificado con `probar_preprocesamiento.py` | `preprocesamiento.py` |
| Con ≈ 145 audios de prueba, el margen de error era de ±0.03–0.04 y los modelos se alternaban como "mejor" | Validación cruzada, intervalos por *bootstrap* y comparación pareada | `entrenar_cv.py` |
| La persona real y su clon podían quedar en lados distintos de la partición | Partición por hablante con la persona y su clon juntos | `grupos_es.py` |
| Voces reales con frecuencia de muestreo mayor a la del entrenamiento se clasificaban como sintéticas; se descartó el códec (Opus a 24 kbps no degradó la aceptación) y se confirmó el ancho de banda | Limitación de toda entrada a 16 kHz y conversión de los datos nuevos con la misma banda | `preprocesamiento.py`, `hispaspoof_a_npy.py`, `audios_a_npy.py` |
| El conjunto original tenía pocas fuentes de audio real y de herramientas de síntesis, lo que permitía aprender la fuente | Ampliación con HISPASpoof, CIEMPIESS, conferencias y otras personas; aumento de canal fuerte; pruebas con herramientas no vistas | `preparar_hispaspoof.py`, `probar_externo.py` |

---

## 9. Orquestación y archivos generados

`entrenar_cv.py` coordina el entrenamiento final en tres pasos. Primero ejecuta `train.py --fold k` para cada semilla y partición, entrenando la etapa 1 una sola vez por semilla y reutilizándola en los demás folds (`--pre_dir`). Segundo, agrega las predicciones de los folds y calcula exactitud, F1 y AUC con intervalos, la media y desviación entre semillas y la comparación CNN − LSTM. Tercero, escribe `modelos.json`, con los puntos de control, la temperatura y el umbral de cada modelo; la interfaz promedia los modelos de cada etapa, lo que resulta más estable que uno solo. No se selecciona un "mejor" fold, pues hacerlo según su resultado de prueba invalidaría la honestidad de la métrica. El proceso es reanudable: un fold se considera completo cuando existen todos sus resultados.

Las métricas completas (tablas e imágenes) se generan con `metricas/reporte_completo.py` a partir de las predicciones fuera de muestra.

| Archivo | Contenido |
|---|---|
| `resultados_v3/seedS/foldK/cnn.pt`, `lstm.pt` | Modelos finales del fold (condición B) |
| `.../cnn_pre.pt`, `lstm_pre.pt` | Modelos preentrenados (condición A) |
| `.../cnn_solo_es.pt`, `lstm_solo_es.pt` | Líneas base (condición C) |
| `.../metricas.json` | Métricas, umbrales, temperaturas y configuración del fold |
| `.../pred_<modelo>_<condición>.npy` | Etiquetas y probabilidades de prueba |
| `.../es_test_idx.npy` | Posición de cada audio de prueba en `espanol_v3` |
| `.../registro_pruebas.csv` | Matriz de registro 4.2: una fila por audio, modelo y condición |
| `.../log.txt` | Registro del entrenamiento |
| `resultados_v3/resumen_cv.json` | Resumen fuera de muestra con intervalos |
| `resultados_v3/modelos.json` | Modelos, umbrales y temperaturas que carga la interfaz |
| `metricas_v3/` | Tablas y figuras generadas por `reporte_completo.py` |

---

## 10. Ejecución

```bash
pip install -r requirements.txt
python training/verificar_datos.py --multi_dir datos/multilingue --es_dir espanol_v3

# entrenamiento final (reanudable; ≈ 2 h por semilla en una RTX 4070 portátil)
python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir espanol_v3 \
       --out resultados_v3 --seeds 42 --amp --canal_fuerte

# semillas adicionales (opcional)
python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir espanol_v3 \
       --out resultados_v3 --seeds 42 43 44 --amp --canal_fuerte

# métricas, tablas y figuras
python metricas/reporte_completo.py --out resultados_v3 --es_dir espanol_v3 --destino metricas_v3

# prueba con audios externos
python probar_externo.py --resultados resultados_v3 --dir externo_nuevas --detalle
```

Se emplea GPU automáticamente si está disponible. Los hiperparámetros pueden modificarse mediante argumentos de `train.py`, que `entrenar_cv.py` reenvía (`--epochs_ft`, `--canal_prob`, `--smooth`, etc.).

---

## 11. Resumen de resultados

Evaluación fuera de muestra (semilla 42, 7 folds por hablante, `espanol_v3`):

| Modelo | A · preentrenado | B · preentrenado + ajustado | C · solo español |
|---|---|---|---|
| CNN (AUC) | 0.825 | 0.998 | 0.998 |
| LSTM (AUC) | 0.628 | 0.973 | 0.989 |

La CNN supera de forma demostrable a la LSTM en las tres condiciones (diferencia de AUC de +0.198, +0.025 y +0.009). La condición C iguala o supera a la B, por lo que el preentrenamiento multilingüe no aporta mejora medible en español. Con herramientas de síntesis no vistas (fish-speech y XTTS v1), la CNN alcanza una exactitud de 0.935 (B) y 0.930 (C), y la LSTM de 0.825 y 0.820. El análisis completo se encuentra en [`RESULTADOS.md`](RESULTADOS.md).

---

## 12. Limitaciones

- **Canal de teléfono.** Con notas de voz de WhatsApp se aceptan como reales cerca del 62 % (CNN y LSTM en C); el umbral requiere recalibración con un conjunto independiente de ese canal.
- **Muestra propia reducida.** Las grabaciones propias son decenas de fragmentos y el resultado depende del decodificador empleado.
- **Herramientas de síntesis.** El modelo solo conoce los métodos presentes en el entrenamiento; XTTS v1 fue la más difícil de detectar (78–82 %).
- **Hablantes compartidos en HISPASpoof** entre sus particiones originales.
- **Atajos no controlados.** El aumento de canal varía ruido, respuesta en frecuencia y ancho de banda; pausas, ritmo, duración y contenido léxico no se controlan.
- **Una sola semilla** en el resultado actual (`--seeds`).
- **Probabilidades.** Aun calibradas, expresan la confianza del modelo y no una certeza.
- **Coherencia con la interfaz.** El preprocesamiento debe ser idéntico al del entrenamiento, incluida la banda de 16 kHz.