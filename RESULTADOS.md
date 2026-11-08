# Resultados experimentales

En este documento se reportan las métricas obtenidas en la evaluación de los modelos CNN y LSTM, su incertidumbre y las conclusiones que sustentan. El procedimiento se describe en [`ENTRENAMIENTO.md`](ENTRENAMIENTO.md); las tablas y figuras completas se generan con `metricas/reporte_completo.py` (carpeta `metricas_v3/`).

## Contenido

1. [Resumen](#1-resumen)
2. [Diseño de la evaluación](#2-diseño-de-la-evaluación)
3. [Validación cruzada por hablante](#3-validación-cruzada-por-hablante)
4. [Acierto por fuente de audio](#4-acierto-por-fuente-de-audio)
5. [Pruebas externas](#5-pruebas-externas)
6. [Sesgo de ancho de banda](#6-sesgo-de-ancho-de-banda)
7. [Evolución del método](#7-evolución-del-método)
8. [Discusión](#8-discusión)
9. [Limitaciones](#9-limitaciones)
10. [Conclusiones](#10-conclusiones)
11. [Reproducción](#11-reproducción)

---

## 1. Resumen

El propósito de la evaluación fue comparar una red convolucional (CNN) y una recurrente (LSTM) en la detección de audio sintético en español, bajo tres condiciones: **A** (preentrenamiento multilingüe sin ver español), **B** (preentrenamiento y ajuste con español) y **C** (entrenamiento únicamente con español, como línea base). Los hallazgos principales son los siguientes:

- Con evaluación fuera de muestra agrupada por hablante, el AUC de la CNN fue 0.825 (A), 0.998 (B) y 0.998 (C); el de la LSTM, 0.628, 0.973 y 0.989.
- La CNN superó a la LSTM de forma demostrable en las tres condiciones.
- El preentrenamiento multilingüe no produjo una mejora medible en español.
- Ante herramientas de síntesis ausentes del entrenamiento (fish-speech y XTTS v1), la CNN alcanzó una exactitud de 0.93 y la LSTM de 0.82.
- Se identificó y corrigió un sesgo de ancho de banda que provocaba que las voces reales de teléfono se clasificaran como sintéticas.
- Persiste una limitación en el canal de teléfono: cerca del 62 % de las voces reales propias se aceptan como tales.

---

## 2. Diseño de la evaluación

### 2.1 Datos

El **conjunto multilingüe** (9,569 audios reales y 6,977 sintéticos) se empleó solo en el preentrenamiento. El **conjunto en español ampliado** integró el conjunto original (963 audios, 21 hablantes con su clon sintético), una muestra de HISPASpoof (ElevenLabs, F5-TTS, XTTS v2 y YourTTS), voces reales de CIEMPIESS y voces reales de 13 fuentes de conferencias y 14 personas adicionales. La diversidad de fuentes reales y sintéticas pretende que los modelos discriminen la autenticidad y no las condiciones de grabación. La composición exacta por fuente se encuentra en `metricas_v3/tablas/acierto_por_fuente.md`.

### 2.2 Configuración

Los espectrogramas tienen 128 bandas mel × 431 frames (≈ 10 s) a 22,050 Hz, tras limitar la entrada a 16 kHz. Los modelos (CNN ≈ 404,900 y LSTM ≈ 692,700 parámetros) se entrenaron con AdamW, lote de 32, *early stopping*, *dropout*, *label smoothing*, SpecAugment y aumento de canal fuerte, con una sola semilla (42).

### 2.3 Protocolo

La evaluación principal fue una **validación cruzada de 7 particiones agrupada por hablante**, de modo que cada audio se clasificó con modelos que no vieron a su hablante; el umbral y la calibración se determinaron únicamente con la validación. Las diferencias entre modelos se estimaron con *bootstrap* pareado (1,000 remuestreos, nivel 95 %) y se consideraron demostrables solo cuando el intervalo excluyó el cero. Como complemento se usaron tres pruebas externas (sección 5) con conjuntos que no participaron en el entrenamiento. La clase positiva es *sintético*.

---

## 3. Validación cruzada por hablante

### 3.1 Desempeño por condición

| Modelo | Condición | AUC fuera de muestra |
|---|---|---|
| CNN | A · preentrenado, sin ver español | 0.825 |
| CNN | B · preentrenado + ajustado | 0.998 |
| CNN | C · solo español | 0.998 |
| LSTM | A · preentrenado, sin ver español | 0.628 |
| LSTM | B · preentrenado + ajustado | 0.973 |
| LSTM | C · solo español | 0.989 |

La condición A mide la transferencia entre idiomas: la CNN conserva una capacidad discriminativa moderada, mientras que la LSTM se acerca al azar. Tras incorporar datos en español (B y C), ambos modelos superan 0.97 de AUC. La exactitud, precisión, sensibilidad, especificidad, F1, matrices de confusión, intervalos de confianza y dispersión entre folds se reportan en `metricas_v3/tablas/` (`metricas_completas.md`, `metricas_por_fold.csv`, `dispersion_folds.md`) y en `figuras/01_matrices_confusion.png`.

### 3.2 Comparación entre arquitecturas

| Condición | Diferencia de AUC (CNN − LSTM) | Conclusión |
|---|---|---|
| A | +0.198 | CNN superior |
| B | +0.025 | CNN superior |
| C | +0.009 | CNN superior |

La ventaja de la CNN es amplia cuando el modelo no vio español y se reduce, sin desaparecer, tras el ajuste o el entrenamiento con español.

---

## 4. Acierto por fuente de audio

El desempeño agregado puede ocultar fuentes con comportamiento deficiente. Por ello se desglosó el acierto (umbral 0.5, evaluación fuera de muestra) según el origen del audio. Los valores corresponden a la CNN en las condiciones B y C.

| Fuente | CNN B | CNN C |
|---|---|---|
| Real · CIEMPIESS | 100 % | 100 % |
| Real · HISPASpoof (natural) | 98 % | 98 % |
| Real · conferencias | 98 % | 96 % |
| Real · otras personas | 88 % | 86 % |
| Sintético · conjunto original | 100 % | 100 % |
| Sintético · HISPASpoof, por herramienta | 93–99 % | 93–99 % |

La condición A detecta solo entre 12 % y 40 % de los sintéticos de la mayoría de las herramientas de HISPASpoof, lo que indica que el preentrenamiento no transfiere a estas voces. La fuente real con menor acierto fue la de *otras personas*, cuyas grabaciones son las más heterogéneas, lo que sugiere que la variabilidad del canal de captura es el factor más difícil de cubrir.

---

## 5. Pruebas externas

Las pruebas externas evalúan la generalización en escenarios que la validación cruzada interna no cubre.

### 5.1 Herramientas de síntesis no vistas

Se evaluaron audios generados con fish-speech y XTTS v1, ausentes del entrenamiento, junto con voces reales naturales.

| Modelo | Condición | Exactitud | AUC |
|---|---|---|---|
| CNN | B | 0.935 | 0.996 |
| CNN | C | 0.930 | 0.993 |
| CNN | A | 0.680 | 0.800 |
| LSTM | B | 0.825 | 0.889 |
| LSTM | C | 0.820 | 0.921 |
| LSTM | A | 0.455 | 0.413 |

La CNN detectó el 92 % (B) y el 94 % (C) de los audios de fish-speech, y el 82 % (B) y el 78 % (C) de los de XTTS v1. La LSTM en la condición A obtuvo un AUC inferior a 0.5, es decir, sin capacidad discriminativa en este conjunto.

### 5.2 Muestra de HISPASpoof (partición de validación)

La exactitud fue 1.000 (CNN C), 0.980 (CNN B), 0.975 (LSTM C) y 0.960 (LSTM B); en la condición A, 0.735 (CNN) y 0.51 (LSTM). Estos valores deben interpretarse con reserva, ya que HISPASpoof comparte 24 hablantes entre sus particiones y la prueba no es independiente por hablante.

### 5.3 Notas de voz grabadas con teléfono

Se evaluaron notas de voz de WhatsApp grabadas por las autoras, tras aplicar la limitación de banda a 16 kHz.

| Modelo | Condición | Exactitud | AUC | Reales aceptados |
|---|---|---|---|---|
| CNN | B | 0.732 | 0.910 | 42 % |
| CNN | C | 0.821 | 0.967 | 62 % |
| LSTM | B | 0.696 | 0.917 | 38 % |
| LSTM | C | 0.804 | 0.945 | 62 % |

Un AUC elevado con una baja aceptación de reales indica que los modelos separan las clases, pero que el umbral de 0.5 no es adecuado para este canal. Al convertir previamente las notas a WAV de 16 kHz con ffmpeg, la LSTM C aceptó el 88 % de los reales (exactitud 0.929) y la CNN C alcanzó 0.839 de exactitud, lo que evidencia sensibilidad al decodificador y al remuestreador. Dado que la muestra comprende decenas de fragmentos, estas cifras son indicativas.

---

## 6. Sesgo de ancho de banda

Las voces reales de teléfono eran clasificadas como sintéticas, por lo que se contrastaron dos hipótesis. La primera atribuía el fenómeno a la compresión con pérdidas (códec Opus); se descartó, pues voces reales de HISPASpoof comprimidas con Opus a 24 kbps siguieron aceptándose (CNN B 88 %, CNN C 86 %, LSTM B 81 %, LSTM C 92 %). La segunda lo atribuía al ancho de banda: los datos de entrenamiento estaban limitados a unos 16 kHz, mientras que los mp3 de WhatsApp conservan contenido por encima de 8 kHz, que el modelo interpretaba como evidencia de audio no real. Esta hipótesis se confirmó.

La corrección consistió en limitar toda entrada a 16 kHz (`BANDA = 16000` en `preprocesamiento.py`) y convertir los conjuntos nuevos con la misma banda. La aceptación de reales propios pasó de 4–38 % a 38–62 % según el modelo, con un máximo de 88 %. Un aumento de canal más intenso y una mayor diversidad de reales no bastaron por sí solos para resolver el problema.

---

## 7. Evolución del método

| Versión | Cambio principal | Efecto observado |
|---|---|---|
| v1 | Entrenamiento con el conjunto original (21 hablantes) y aumento de canal moderado | AUC interno ≈ 1.000 y AUC de A ≈ 0.8; sin prueba externa |
| v2 | Mismos datos; pruebas con notas de voz propias | Reales propios aceptados en 4–38 %: el desempeño interno no se trasladaba a otros canales |
| v3 | Conjunto ampliado, aumento de canal fuerte y banda unificada de 16 kHz | AUC interno de 0.97–0.998 en B y C, generalización a herramientas no vistas (CNN ≈ 0.93) y mejora en reales propios |

La reducción del AUC interno entre v1 y v3 es esperable y deseable: el conjunto ampliado elimina atajos asociados a la grabación del conjunto original, de modo que la medición refleja con mayor fidelidad la capacidad real de detección.

---

## 8. Discusión

**Superioridad de la CNN.** La CNN superó a la LSTM en todas las condiciones y pruebas externas. Una explicación plausible es que los artefactos de síntesis se manifiestan como patrones locales en tiempo y frecuencia, que las convoluciones capturan directamente, mientras que la LSTM procesa cada frame como un vector y depende más de la dinámica temporal. Esta interpretación no se verificó de forma aislada.

**Papel del preentrenamiento.** El preentrenamiento multilingüe generaliza de forma moderada en la CNN y deficiente en la LSTM, y tras el ajuste no supera al entrenamiento solo con español. Con el volumen de datos disponible no se observa ventaja de transferir desde otros idiomas.

**Validez de la medición.** La evaluación por hablante evita que el modelo se limite a reconocer personas, y las pruebas con herramientas no vistas y con grabaciones propias aportan evidencia de generalización que la validación interna no puede ofrecer por sí sola.

---

## 9. Limitaciones

- **Canal de teléfono.** Los reales grabados con teléfono se aceptan en torno al 62 % (CNN y LSTM en C), probablemente por la escasez de ejemplos de ese canal en el entrenamiento.
- **Muestra propia reducida.** Las pruebas con notas de voz comprenden decenas de fragmentos; la variación entre decodificadores muestra que la cifra es inestable.
- **Hablantes compartidos en HISPASpoof** entre sus particiones, por lo que las cifras de esa fuente no son independientes por hablante.
- **Herramientas de síntesis.** El modelo solo conoce las herramientas presentes en el entrenamiento; XTTS v1 fue la menos detectada (78–82 %).
- **Una sola semilla,** por lo que no se estima la variación entre entrenamientos.
- **Probabilidades.** Aun calibradas, constituyen una medida de confianza y no una certeza.

---

## 10. Conclusiones

1. Los mel-espectrogramas permiten detectar audio sintético en español con AUC fuera de muestra de 0.97–0.998 al entrenar con datos en español (condiciones B y C).
2. La CNN es superior a la LSTM en todas las condiciones, con diferencias estadísticamente demostrables.
3. El preentrenamiento multilingüe no aporta una mejora medible respecto del entrenamiento solo con español; la configuración recomendada es la CNN en la condición C (o B).
4. La generalización a herramientas de síntesis no vistas es razonable en la CNN (≈ 93 % de exactitud) y claramente inferior en la LSTM (≈ 82 %).
5. El ancho de banda de entrada es un factor crítico: unificarlo en 16 kHz es necesario para audios de teléfono.
6. Antes de un uso en condiciones reales se recomienda recalibrar el umbral con un conjunto independiente de notas de voz y repetir el entrenamiento con semillas adicionales (`--seeds 42 43 44`).

---

## 11. Reproducción

```bash
python training/entrenar_cv.py --multi_dir datos/multilingue --es_dir espanol_v3 \
       --out resultados_v3 --seeds 42 --amp --canal_fuerte
python metricas/reporte_completo.py --out resultados_v3 --es_dir espanol_v3 --destino metricas_v3
python probar_externo.py --resultados resultados_v3 --dir externo_nuevas --detalle
```