# Evaluación de modelos CNN y LSTM mediante representaciones espectrales para la detección de audios sintéticos

**Autoras:** Cortés Carrisoza, Edna Samantha · García Vázquez, Arantza

Este repositorio contiene el código y la documentación del proyecto de evaluación comparativa de modelos CNN y LSTM aplicados a espectrogramas de audio para distinguir entre habla genuina y habla generada o manipulada mediante inteligencia artificial (audios *deepfake*).

> El interés del proyecto **no** es desarrollar un sistema de producción, sino poner a prueba y contrastar el rendimiento de ambos modelos.

---

## Contenido

1. [Planteamiento del problema](#planteamiento-del-problema)
2. [Pregunta de investigación](#pregunta-de-investigación)
3. [Objetivos](#objetivos)
4. [Justificación](#justificación)
5. [Alcances y limitaciones](#alcances-y-limitaciones)
6. [Metodología](#metodología)
7. [Materiales y herramientas](#materiales-y-herramientas)
8. [Procedimiento general](#procedimiento-general)
9. [Marco teórico](#marco-teórico)
10. [Implementación en este repositorio](#implementación-en-este-repositorio)
11. [Estructura del repositorio](#estructura-del-repositorio)
12. [Cómo ejecutar el entrenamiento](#cómo-ejecutar-el-entrenamiento)

---

## Planteamiento del problema

El desarrollo de la inteligencia artificial (IA) ha transformado la manera en que se genera, procesa y distribuye contenido digital. La generación de audio mediante modelos de aprendizaje profundo ha alcanzado un nivel de precisión que permite producir voces sintéticas con características similares a las de una persona real, dando origen a lo que se conoce como **audios deepfake**. El término proviene de la combinación de *deep learning* (aprendizaje profundo) y *fake* (falsificación de contenido) [1]. Estas manipulaciones pueden generarse mediante redes generativas adversarias (GANs), redes neuronales convolucionales (CNNs) y redes neuronales profundas (DNNs), capaces de replicar la voz de una persona con alta precisión en timbre, entonación y patrones de habla [2].

Aunque esta tecnología tiene aplicaciones legítimas en asistentes de voz y accesibilidad, también ha sido explotada con fines ilícitos. La facilidad de acceso a herramientas de generación de deepfakes ha reducido significativamente la barrera técnica para producir contenido falso convincente [3], y los ciberdelincuentes recurren cada vez con mayor frecuencia a la clonación de voz mediante IA para ejecutar estafas de suplantación de identidad, pues basta con segundos de audio para generar una réplica convincente [4].

Los audios deepfake han incrementado el riesgo de suplantación de identidad, fraudes financieros y desinformación:

- En 2021, un grupo de estafadores utilizó un deepfake de voz para engañar a un gerente bancario y sustraer 35 millones de dólares [1].
- Los ataques de suplantación de voz mediante IA (*vishing*) se incrementaron 442 % en 2025, facilitando alrededor de 40 mil millones de dólares en fraude a nivel global, y más del 10 % de las instituciones bancarias ha perdido más de un millón de dólares por una sola llamada con audio falsificado [5].
- En Estados Unidos, el FBI reportó en 2025 más de 22,000 denuncias de fraude relacionado con IA, con pérdidas superiores a 893 millones de dólares, cifra que se considera subestimada pues se calcula que menos del 5 % de las víctimas reporta sus pérdidas.
- Una voz puede clonarse a partir de tan solo tres segundos de audio disponible públicamente [6].

La capacidad humana para reconocer estas manipulaciones también se ha visto superada: la calidad de la clonación de voz ha cruzado el "umbral de indistinguibilidad" [7]. Alrededor del 80 % de las personas confunde voces generadas por IA con voces reales en clips cortos, y 1 de cada 10 reportó haber recibido un mensaje con una voz clonada, de las cuales el 77 % terminó perdiendo dinero [9].

Los mecanismos tradicionales de verificación de identidad por voz (reconocimiento auditivo humano y sistemas biométricos basados en reglas fijas) han demostrado ser insuficientes: asumen que una voz sintética conservará artefactos detectables, pero los sistemas de clonación actuales evaden estos filtros, y los ciclos de mejora de los modelos de síntesis se miden en meses, mientras que el desarrollo de herramientas de detección se mide en años [8].

Esta brecha entre la evolución de las técnicas de generación y la de los métodos de verificación constituye el núcleo del problema. Además, para el usuario común no existen herramientas accesibles que le permitan verificar por su cuenta si un audio recibido es auténtico, quedando su única defensa en la percepción auditiva, ya demostrada como poco confiable.

**Antecedente directo.** Se realizó previamente una comparación de técnicas de aprendizaje profundo para la detección de audio deepfake, evaluando CNN, LSTM, un enfoque híbrido (CNN+LSTM) y Audio Transformers (AST) mediante exactitud, precisión, recall, F1-score y robustez [10]. El modelo híbrido CNN+LSTM obtuvo el mejor desempeño global, las CNN simples destacaron por su eficiencia computacional y las LSTM por su sensibilidad a la dinámica del habla, aunque con una mayor tasa de falsos positivos [10]. Este antecedente confirma la viabilidad de comparar arquitecturas de este tipo sobre representaciones espectrales y aporta el marco metodológico que se retoma en este proyecto.

**Problema que se aborda.** No siempre resulta claro qué tan bien se comportan los modelos propuestos en la literatura frente a audios variados, fuera de las condiciones controladas de un conjunto de entrenamiento. Se plantea, por tanto, evaluar y comparar el desempeño de CNN y LSTM, aplicadas sobre representaciones espectrales del audio, frente a un escenario cercano al uso real, en el que un usuario proporciona un audio a través de una interfaz y recibe una retroalimentación sobre su autenticidad.

> Nota: las referencias [1]–[10] corresponden a la lista bibliográfica del protocolo completo.

---

## Pregunta de investigación

**¿Qué diferencias existen en el desempeño de los modelos CNN y LSTM para identificar audios deepfake y voz humana al evaluarse en un entorno real?**

---

## Objetivos

### Objetivo general

Evaluar los modelos CNN y LSTM mediante representaciones espectrales para la detección de audios sintéticos.

### Objetivos específicos

1. Compilar un conjunto de audios, tanto auténticos como generados o manipulados mediante IA.
2. Reentrenar los modelos CNN y LSTM para adaptarlos a las representaciones espectrales del nuevo conjunto de audios, a fin de evaluar su desempeño.
3. Diseñar una interfaz de prueba que permita a un usuario cargar un audio y obtener como resultado la clasificación (auténtico / deepfake).
4. Comparar el desempeño de cada modelo mediante métricas cuantitativas (exactitud, precisión, sensibilidad, F1-score, matriz de confusión, curva ROC) al procesar los audios de prueba a través de la interfaz.

---

## Justificación

El crecimiento acelerado del fraude mediante audios deepfake, evidenciado por el incremento de más de 400 % en ataques de suplantación de voz durante 2025 y por pérdidas económicas que se proyectan en miles de millones de dólares a nivel global [5], [6], hace necesario contar con evidencia confiable sobre qué tan efectivos son los modelos de detección disponibles cuando se enfrentan a condiciones más cercanas a la realidad, y no únicamente a datasets curados y controlados.

Investigaciones previas han identificado a CNN y LSTM como las técnicas con mejor desempeño frente a otras alternativas evaluadas sobre datasets estándar. Sin embargo, un modelo que funciona adecuadamente sobre un conjunto de referencia no garantiza el mismo nivel de precisión ante audios con condiciones de grabación variables, ruido de fondo distinto o técnicas de síntesis no vistas durante el entrenamiento. Evaluar estos modelos en un entorno más realista, a través de una interfaz que simule el uso por parte de un usuario final, permite generar evidencia empírica sobre su desempeño práctico y sus limitaciones.

**Beneficiarios:**

- **Usuarios individuales** que reciben llamadas o mensajes de voz y no cuentan con herramientas para verificar su autenticidad, en particular adultos mayores y personas con menor familiaridad tecnológica, identificados como uno de los grupos más vulnerables a este tipo de fraude [2].
- **Empresas y organizaciones** que dependen de comunicaciones por voz para autorizar transacciones o instrucciones internas [8].
- **La comunidad académica y de investigación** en seguridad digital y aprendizaje profundo, al contar con evidencia sobre el comportamiento de estos modelos fuera de condiciones controladas de laboratorio.
- De forma indirecta, **periodistas y verificadores de hechos**, expuestos a la difusión de audios falsos atribuidos a figuras públicas [1].

---

## Alcances y limitaciones

### Alcances

- Evaluación comparativa del desempeño de los modelos CNN y LSTM, previamente identificados como los de mejor rendimiento en una investigación anterior sobre detección de audios deepfake.
- Se trabaja únicamente con representaciones espectrales del audio (espectrogramas, mel-espectrogramas y/o MFCC) como entrada a ambos modelos.
- Se diseña una interfaz de prueba que permite cargar audios y visualizar el resultado de la clasificación y las métricas de desempeño de cada modelo, con fines exclusivamente de evaluación.
- Los audios de prueba corresponden a un conjunto acotado y viable de audios reales (auténticos y generados o manipulados mediante IA), recolectados o generados específicamente para esta etapa de evaluación.
- El análisis se centra en métricas cuantitativas de desempeño (exactitud, precisión, sensibilidad, F1-score, etc.) y en la comparación directa entre ambos modelos.

### Limitaciones

- La interfaz tiene un propósito exclusivamente de prueba y demostración; no está pensada para un despliegue real ni para uso masivo por parte de usuarios finales.
- Los resultados están condicionados por el tamaño y la diversidad del conjunto de audios de prueba, por lo que no necesariamente son generalizables a todos los escenarios posibles de audios deepfake (distintos idiomas, acentos, técnicas de síntesis más recientes o condiciones de grabación no contempladas).
- No se evalúa el desempeño de los modelos frente a ataques adversariales diseñados específicamente para evadir sistemas de detección, ni se abordan aspectos de seguridad informática más allá de la clasificación del audio.

---

## Metodología

La metodología se organiza de acuerdo con los objetivos específicos. Para cada objetivo se establecen actividades concretas que permiten llevar a cabo el desarrollo experimental, desde la recopilación y preparación de los audios hasta la evaluación y comparación de los modelos CNN y LSTM.

### Objetivo específico 1

*Compilar un conjunto de audios, tanto auténticos como generados o manipulados mediante IA, que permitan someter a prueba a los modelos CNN y LSTM en condiciones distintas a las de un dataset controlado.*

1.1 Realizar una búsqueda de conjuntos de datos de audio que contengan muestras de voz humana auténtica y audios generados o manipulados mediante inteligencia artificial.
1.2 Seleccionar las fuentes de audio que cumplan con las características necesarias para el experimento, considerando disponibilidad, formato, duración, calidad y diversidad de las muestras.
1.3 Complementar el conjunto de datos con audios auténticos y/o generados mediante IA cuando sea necesario, con el propósito de representar condiciones de grabación diferentes a las presentes en el conjunto utilizado durante el entrenamiento original.
1.4 Clasificar y etiquetar los audios en las categorías auténtico y deepfake, asignando un identificador único a cada muestra.
1.5 Registrar las características principales de cada audio, incluyendo clase, duración, formato, frecuencia de muestreo y procedencia.
1.6 Realizar el preprocesamiento de los audios mediante herramientas de análisis de señales, utilizando bibliotecas como Librosa y NumPy, con el propósito de establecer condiciones homogéneas para su procesamiento.
1.7 Dividir el conjunto de datos en subconjuntos de entrenamiento, validación y prueba, manteniendo el conjunto de prueba separado para la evaluación final de los modelos.

### Objetivo específico 2

*Reentrenar los modelos CNN y LSTM para adaptarlos a las representaciones espectrales del nuevo conjunto de audios, a fin de evaluar su desempeño.*

2.1 Analizar las características de los modelos CNN y LSTM utilizados como referencia en el antecedente de investigación.
2.2 Definir el proceso de transformación de los audios a representaciones espectrales, considerando espectrogramas, mel-espectrogramas y/o MFCC.
2.3 Generar las representaciones espectrales de los audios utilizando bibliotecas especializadas en procesamiento de señales de audio.
2.4 Establecer las mismas clases de salida para ambos modelos: auténtico y deepfake.
2.5 Implementar y configurar los modelos CNN y LSTM utilizando un framework de aprendizaje profundo, como TensorFlow/Keras o PyTorch.
2.6 Reentrenar ambos modelos utilizando el mismo conjunto de entrenamiento y aplicar el conjunto de validación para supervisar su desempeño durante el aprendizaje.
2.7 Registrar los parámetros utilizados durante el entrenamiento, incluyendo número de épocas, tamaño de lote, tasa de aprendizaje y función de pérdida.
2.8 Seleccionar la configuración de cada modelo con base en su desempeño durante la validación y conservar los modelos entrenados para la etapa de evaluación.

### Objetivo específico 3

*Diseñar una interfaz de prueba que permita a un usuario cargar un audio y obtener como resultado la clasificación (auténtico / deepfake) junto con una retroalimentación del nivel de confianza de cada modelo.*

3.1 Definir los requerimientos funcionales de la interfaz de prueba, considerando la carga de archivos, procesamiento del audio y presentación de resultados.
3.2 Diseñar la estructura de la interfaz para permitir la selección y carga de archivos de audio.
3.3 Integrar el proceso de preprocesamiento y transformación espectral dentro del flujo de evaluación de la interfaz.
3.4 Integrar los modelos CNN y LSTM previamente entrenados para procesar el audio ingresado por el usuario.
3.5 Configurar la interfaz para mostrar la clasificación obtenida por cada modelo como auténtico o deepfake.
3.6 Incorporar la visualización del nivel de confianza o probabilidad asociada a la clasificación realizada por cada modelo.
3.7 Realizar pruebas funcionales de la interfaz utilizando diferentes audios para comprobar que el proceso de carga, procesamiento y clasificación se ejecute correctamente.

### Objetivo específico 4

*Comparar el desempeño de cada modelo mediante métricas cuantitativas (exactitud, precisión, sensibilidad, F1-score, matriz de confusión, curva ROC) al procesar los audios de prueba a través de la interfaz.*

4.1 Definir las métricas que serán utilizadas para evaluar y comparar el desempeño de CNN y LSTM: exactitud (*accuracy*), precisión (*precision*), sensibilidad (*recall*), F1-score, matriz de confusión, curva ROC y área bajo la curva (AUC).
4.2 Diseñar una matriz de registro para almacenar los resultados obtenidos durante las pruebas, incluyendo identificador del audio, clase real, modelo utilizado, clasificación obtenida, nivel de confianza y tiempo de procesamiento.
4.3 Procesar el conjunto de audios de prueba mediante CNN y registrar las predicciones obtenidas.
4.4 Procesar el mismo conjunto de audios de prueba mediante LSTM y registrar las predicciones obtenidas.
4.5 Calcular las métricas de desempeño de cada modelo utilizando herramientas como Scikit-learn.
4.6 Generar las matrices de confusión de CNN y LSTM para identificar verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
4.7 Generar las curvas ROC y calcular el AUC correspondiente para cada modelo.
4.8 Registrar y comparar el tiempo de procesamiento requerido por CNN y LSTM para generar sus predicciones.
4.9 Elaborar tablas y gráficos comparativos de las métricas obtenidas por ambos modelos.
4.10 Analizar los resultados obtenidos para identificar las diferencias de desempeño entre CNN y LSTM y determinar cuál presenta mejores resultados bajo las condiciones establecidas para el experimento.

---

## Materiales y herramientas

- **Python**, como lenguaje de programación.
- **Jupyter Notebook**, para el desarrollo y ejecución de los experimentos.
- **TensorFlow/Keras o PyTorch**, para la implementación y reentrenamiento de los modelos CNN y LSTM.
- **Librosa**, para la carga, procesamiento y extracción de características de los audios.
- **NumPy y Pandas**, para el procesamiento y organización de los datos.
- **Matplotlib y/o Seaborn**, para la representación gráfica de los resultados.
- **Scikit-learn**, para el cálculo de métricas, matriz de confusión y curva ROC.
- Una computadora con capacidad suficiente para ejecutar los modelos y procesar los audios.

Los parámetros definitivos de hardware y software utilizados durante los experimentos se registran para facilitar la reproducibilidad de las pruebas.

---

## Procedimiento general

```
Recopilación de audios → Preprocesamiento → Extracción de características →
División de datos → Reentrenamiento de CNN y LSTM → Validación →
Diseño e integración de la interfaz → Pruebas con audios →
Registro de resultados → Cálculo de métricas → Comparación de modelos →
Análisis de resultados
```

Esta metodología permite relacionar cada actividad con los objetivos específicos y proporciona los datos necesarios para responder la pregunta de investigación sobre las diferencias de desempeño entre CNN y LSTM en la identificación de audios auténticos y deepfake en condiciones cercanas a un entorno real.

---

## Marco teórico

1. **Inteligencia artificial:** concepto de IA · aprendizaje automático · aprendizaje profundo
2. **Audios generados mediante IA:** generación de audio mediante IA · síntesis y clonación de voz · deepfakes de audio · características de los audios sintéticos
3. **Detección de deepfakes de audio:** técnicas de detección · detección mediante aprendizaje profundo · clasificación de voz humana y audio generado por IA
4. **Representaciones espectrales del audio:** señal de audio · espectrograma · mel-espectrograma · coeficientes cepstrales en las frecuencias de Mel (MFCC)
5. **Redes neuronales utilizadas para la detección:** CNN · LSTM · aplicación de CNN y LSTM en el análisis de audio · diferencias entre CNN y LSTM
6. **Evaluación de modelos de clasificación:** exactitud · precisión · sensibilidad · F1-score · matriz de confusión · curva ROC · AUC · tiempo de procesamiento
7. **Herramientas para el desarrollo:** Python · Librosa · NumPy · TensorFlow/Keras · PyTorch · Scikit-learn · Matplotlib
8. **Trabajos relacionados:** estudios sobre detección de audio deepfake · comparación de modelos CNN y LSTM · antecedentes del proyecto

---

## Implementación en este repositorio

Esta sección describe cómo se concretó el protocolo en el código.

**Framework:** PyTorch. **Representación de entrada:** mel-espectrogramas en decibeles, almacenados como archivos `.npy` (matriz `n_mels × frames`).

**Datos**

| Conjunto | Reales | Fake | Uso |
|---|---|---|---|
| Multilingüe (varios idiomas) | 9,569 | 6,977 | Etapa 1: preentrenamiento |
| Español | 502 | 461 | Etapa 2: adaptación, y evaluación final |

Los datos no se versionan en el repositorio por su tamaño (ver `.gitignore`). Cada espectrograma tiene 128 bandas Mel × 431 frames (≈ 10 s). Antes de entrenar se verificaron los datos y se apartaron los duplicados exactos (246 en el conjunto multilingüe y 40 en español) con `verificar_datos.py` y `deduplicar_datos.py`; los conteos de la tabla son los finales.

**Modelos** (`training/models.py`)

- **CNN:** trata el espectrograma como una imagen de un canal. Cuatro bloques *Conv2d → BatchNorm → ReLU → MaxPool* (32, 64, 128 y 256 filtros), promedio global y cabeza densa.
- **LSTM:** trata el espectrograma como una secuencia de frames. LSTM bidireccional de 2 capas y 128 unidades, *pooling* temporal (media y máximo) y la misma cabeza densa.
- Ambos tienen las mismas clases de salida: **real (0)** y **fake (1)**.

**Entrenamiento en dos etapas** (`training/train.py`)

1. **Preentrenamiento** con los audios multilingües, para que la red aprenda a distinguir real de fake directamente del espectrograma.
2. **Adaptación al español:** se reemplaza la última capa de clasificación y se afina con los audios en español, primero la cabeza y después toda la red con una tasa de aprendizaje menor.

Ambos modelos usan las mismas particiones y el mismo procedimiento. La evaluación se hace siempre sobre el mismo conjunto de prueba en español, que no se usa para entrenar ni para seleccionar modelos, en tres condiciones: **A** (preentrenado, sin ver español), **B** (preentrenado y afinado) y **C** (línea base entrenada solo con español).

El detalle completo del entrenamiento, los hiperparámetros y las métricas está en [`training/ENTRENAMIENTO.md`](training/ENTRENAMIENTO.md).

**Resultados.** Se incluirán al concluir la evaluación sobre el conjunto de prueba.

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── training/
    ├── models.py            # arquitecturas CNN y LSTM
    ├── train.py             # entrenamiento en dos etapas y evaluación
    ├── verificar_datos.py   # revisa conteos, formas, duplicados y fugas entre conjuntos
    ├── deduplicar_datos.py  # aparta duplicados exactos (simulacro por defecto)
    └── ENTRENAMIENTO.md     # explicación detallada del entrenamiento
```

Estructura esperada de los datos (fuera del control de versiones):

```
datos/
├── multilingue/{real,fake}/**/*.npy
└── espanol/{real,fake}/*.npy
```

---

## Cómo ejecutar el entrenamiento

```bash
pip install -r requirements.txt
python training/verificar_datos.py --multi_dir datos/multilingue --es_dir datos/espanol
python training/train.py --multi_dir datos/multilingue --es_dir datos/espanol --out resultados --n_frames 431
```

El script usa GPU si está disponible. Genera en `resultados/`: los modelos entrenados (`.pt`), `metricas.json`, `registro_pruebas.csv` (matriz de registro del punto 4.2), `splits.json`, y las figuras `roc.png`, `matrices_confusion.png` y `perdidas.png`.