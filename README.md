# DIAGNÓSTICO AUTOMATIZADO DE RIESGO CARDIOVASCULAR CON MACHINE LEARNING

> **Universidad Privada Antenor Orrego (UPAO)**
> Facultad de Ingeniería — Escuela de Ingeniería de Computación y Sistemas
> Curso: *Inteligencia Artificial: Principios y Técnicas*
> Docente: Dr. Teobaldo Hernán Sagástegui Chávez
> Autor: Mauricio Jefferson Cuba Prieto
> Ciclo Académico: 2025-II — Tercer Informe de Proyecto

---

## Tabla de Contenidos

- [I. Introducción](#i-introducción)
  - [1.1 Título del Proyecto](#11-título-del-proyecto)
  - [1.2 Antecedentes](#12-antecedentes)
  - [1.3 Problema a Resolver](#13-problema-a-resolver)
  - [1.4 Objetivos](#14-objetivos)
- [II. Requerimientos](#ii-requerimientos)
  - [2.1 Definición del Dominio](#21-definición-del-dominio)
  - [2.2 Determinación de Requisitos](#22-determinación-de-requisitos)
- [III. Planteamiento del Dataset, Pre-procesamiento y Normalización](#iii-planteamiento-del-dataset-pre-procesamiento-y-normalización)
  - [3.1 Detalle Computacional de Variables](#31-detalle-computacional-de-variables)
  - [3.2 Flujo Determinista de Pre-procesamiento](#32-flujo-determinista-de-pre-procesamiento)
  - [3.3 Organización de la Partición](#33-organización-de-la-partición)
- [IV. Aprendizaje](#iv-aprendizaje)
  - [4.1 Planteamiento Multimétodo](#41-planteamiento-multimétodo)
  - [4.2 Configuración Detallada de los Modelos](#42-configuración-detallada-de-los-modelos)
- [V. Comprobación](#v-comprobación)
  - [5.1 Protocolo de Entrenamiento Empírico](#51-protocolo-de-entrenamiento-empírico)
- [VI. Evaluación](#vi-evaluación)
  - [6.1 Reporte Comparativo de Curvas ROC-AUC](#61-reporte-comparativo-de-curvas-roc-auc)
  - [6.2 Evaluación de Desempeño y Coherencia Fisiopatológica](#62-evaluación-de-desempeño-y-coherencia-fisiopatológica-clínica-según-shap)
- [VII. Deploy del Sistema](#vii-deploy-del-sistema)
- [Referencias Bibliográficas](#referencias-bibliográficas)

---

## I. Introducción

### 1.1 Título del Proyecto

**"Diagnóstico Automatizado de Riesgo Cardiovascular con Machine Learning"**

El presente proyecto constituye el desarrollo de un sistema inteligente de clasificación clínica diseñado para la detección temprana y automatizada del riesgo cardiovascular, empleando técnicas avanzadas de Aprendizaje Automático (*Machine Learning*). El sistema integra tres paradigmas algorítmicos complementarios —ensamble de árboles de decisión (Random Forest), máquinas de vectores de soporte con kernel radial (SVM-RBF) y redes neuronales artificiales profundas (MLP)— sobre un corpus biomédico curado y normalizado, con el fin de producir predicciones clínicamente interpretables y auditables.

---

### 1.2 Antecedentes

Las enfermedades cardiovasculares (ECV) representan la primera causa de mortalidad a escala global. De acuerdo con el informe más reciente de la **Organización Mundial de la Salud (OMS, 2021)**, se estima que **17.9 millones de personas fallecen anualmente** a causa de enfermedades del corazón, lo que equivale al 32% de todas las defunciones registradas en el mundo. Más del 85% de esas muertes se producen por infartos de miocardio y accidentes cerebrovasculares, y tres cuartas partes de ellas ocurren en países de ingresos medios y bajos, donde el acceso a diagnóstico especializado es limitado.

En el contexto latinoamericano, la situación es igualmente crítica. **Rodríguez et al. (2020)** documentaron que en Latinoamérica la prevalencia de factores de riesgo cardiovascular —incluyendo hipertensión arterial, dislipidemia y diabetes mellitus tipo 2— alcanza niveles epidémicos, situándose la mortalidad cardiovascular prematura (antes de los 70 años) muy por encima de los promedios registrados en países de altos ingresos. Este fenómeno se ve agravado por la insuficiente capacidad diagnóstica en el primer nivel de atención, donde la toma de decisiones clínicas suele depender de criterios subjetivos y de la experiencia individual del profesional de salud, sin respaldo de herramientas de apoyo computacional.

En el ámbito de la inteligencia artificial aplicada a la medicina, múltiples investigaciones han demostrado la viabilidad de los modelos de aprendizaje supervisado para la predicción de eventos cardiovasculares. El conjunto de datos Cleveland Heart Disease —estandarizado y ampliamente validado por la comunidad científica— se ha consolidado como benchmark de referencia para el desarrollo y evaluación de sistemas de diagnóstico asistido por computador, permitiendo comparaciones rigurosas entre distintas arquitecturas y paradigmas de aprendizaje.

---

### 1.3 Problema a Resolver

El sistema de salud en el primer nivel de atención enfrenta una **doble crisis estructural** que compromete la oportunidad y la exactitud del diagnóstico cardiovascular:

**Crisis de capacidad operativa:** Según la **Organización Panamericana de la Salud (OPS, 2022)**, los centros de atención primaria en América Latina operan con una saturación crónica que supera el 70% de su capacidad instalada. Este desbordamiento implica que el tiempo disponible por consulta es insuficiente para realizar una valoración cardiovascular integral, lo que obliga a los médicos a priorizar síntomas agudos sobre la detección preventiva de condiciones crónicas de alto riesgo.

**Crisis de exactitud diagnóstica en rangos limítrofes:** La valoración clínica convencional de riesgo cardiovascular pierde precisión significativa en los denominados "rangos limítrofes" —aquellos casos donde los biomarcadores se sitúan en la zona gris entre los umbrales de normalidad y de patología establecida. En estos escenarios, la variabilidad interobservador entre profesionales es elevada, los algoritmos tradicionales de estratificación de riesgo (como el Score de Framingham o el SCORE europeo) exhiben limitaciones en poblaciones no caucásicas, y la integración multiparamétrica de 13 o más variables clínicas supera la capacidad de procesamiento analítico del juicio humano no asistido.

Esta brecha diagnóstica se traduce en dos consecuencias clínicas simétricamente adversas: la **infradetección** (falsos negativos), que deja a pacientes de alto riesgo sin intervención preventiva oportuna, y la **sobredetección** (falsos positivos), que genera derivaciones innecesarias, procedimientos costosos y saturación adicional del sistema especializado.

**Formulación del problema:** ¿Es posible construir un sistema de clasificación automatizado, basado en técnicas de Machine Learning, que procese los 13 biomarcadores clínicos estándar de un paciente y emita, con una sensibilidad (Recall) igual o superior al 85%, una predicción binaria de presencia o ausencia de riesgo cardiovascular significativo, con una latencia de respuesta inferior a 200 milisegundos, haciendo viable su integración en el flujo de trabajo clínico del primer nivel de atención?

---

### 1.4 Objetivos

#### Objetivo General

Desarrollar, validar e implementar un sistema de diagnóstico automatizado de riesgo cardiovascular basado en técnicas de Machine Learning supervisado, capaz de clasificar con alta sensibilidad clínica la presencia o ausencia de enfermedad coronaria significativa a partir de 13 biomarcadores clínicos estándar, y de desplegar el sistema resultante como una aplicación web interactiva de uso clínico en el primer nivel de atención.

#### Objetivos Específicos

1. **OE-01 — Construcción del corpus biomédico curado:** Diseñar e implementar un pipeline determinista de pre-procesamiento que transforme el dataset bruto de 1,025 registros en un corpus de 303 pacientes reales y únicos, libre de duplicados, valores atípicos sin justificación clínica y variables exógenas, garantizando la ausencia de *Data Leakage* en todas las etapas de transformación.

2. **OE-02 — Entrenamiento multimétodo y optimización en grilla:** Entrenar y ajustar mediante búsqueda exhaustiva en grilla (GridSearchCV) tres modelos de clasificación heterogéneos —Random Forest, SVM-RBF y MLP— priorizando la maximización de la sensibilidad (Recall) como función de scoring, y seleccionar el modelo de mayor rendimiento clínico como modelo definitivo del sistema.

3. **OE-03 — Validación empírica y evaluación comparativa:** Evaluar el desempeño de los tres modelos sobre el conjunto de prueba congelado mediante métricas de clasificación multicriterio (Accuracy, Precisión, Recall, F1-Score y AUC-ROC), generar los reportes comparativos de curvas ROC y producir el análisis de interpretabilidad clínica mediante SHAP (SHapley Additive exPlanations) TreeExplainer.

4. **OE-04 — Despliegue web como sistema experto clínico:** Implementar el modelo SVM seleccionado como plataforma web interactiva mediante Streamlit, con una interfaz de usuario que permita la ingesta manual de los 13 biomarcadores, ejecute la predicción en tiempo real con latencia inferior a 200 ms y presente el resultado con alertas visuales diferenciadas según el nivel de riesgo estimado.

---

## II. Requerimientos

### 2.1 Definición del Dominio

El dominio de aplicación del presente sistema es la **clasificación binaria de riesgo coronario** a partir de datos clínicos tabulares de pacientes adultos. El espacio de características está conformado por **13 biomarcadores predictores** organizados en cuatro dimensiones biomédicas fundamentales:

| Dimensión Biomédica | Variables Incluidas | Justificación Clínica |
|---|---|---|
| **I. Datos Demográficos y Vitales** | `age`, `sex` | Determinantes primarios de riesgo cardiovascular basal. La edad es el factor de riesgo no modificable de mayor peso. |
| **II. Perfil Hemodinámico** | `trestbps`, `thalach`, `exang` | Refleja la capacidad funcional del sistema cardiovascular bajo condiciones de estrés fisiológico. |
| **III. Perfil Bioquímico y Metabólico** | `chol`, `fbs` | Marcadores de riesgo aterogénico y metabólico. La hipercolesterolemia y la diabetes mellitus son factores de riesgo independientes de ECV. |
| **IV. Marcadores Electrocardio-gráficos e Imagenológicos** | `restecg`, `cp`, `oldpeak`, `slope`, `ca`, `thal` | Indicadores directos de isquemia miocárdica, alteraciones en la conducción eléctrica y perfusión coronaria. |

**Variable objetivo:** `target` — Clasificación binaria donde `1` indica presencia de enfermedad coronaria significativa y `0` indica ausencia de la misma.

---

### 2.2 Determinación de Requisitos

#### Requisitos Funcionales (RF)

| ID | Requisito | Descripción Detallada |
|---|---|---|
| **RF-01** | Ingesta y validación de datos | El sistema deberá aceptar un vector de 13 biomarcadores clínicos validando rangos fisiológicamente admisibles. |
| **RF-02** | Pre-procesamiento determinista | El sistema ejecutará en orden invariable: imputación, Winsorization, codificación OHE y estandarización Z-score. |
| **RF-03** | Predicción de riesgo cardiovascular | El sistema producirá una predicción binaria (0/1) con la probabilidad asociada a cada clase. |
| **RF-04** | Presentación diferenciada del resultado | Alerta roja para diagnóstico positivo (riesgo alto), alerta verde para diagnóstico negativo (riesgo bajo). |

#### Requisitos de Datos (RD)

| ID | Requisito | Descripción Detallada |
|---|---|---|
| **RD-01** | Fuente y calidad del corpus | Entrenamiento exclusivo sobre el subconjunto Cleveland depurado hasta 303 registros únicos, libre de duplicados y registros de Hungría, Suiza y VA Long Beach. |
| **RD-02** | Integridad del pipeline | Todos los transformadores ajustados (*fit*) exclusivamente sobre `X_train` y aplicados (*transform*) sobre `X_test`, garantizando ausencia de Data Leakage. |

#### Requisitos No Funcionales (RNF)

| ID | Requisito | Métrica de Cumplimiento |
|---|---|---|
| **RNF-01** | Rendimiento clínico mínimo | Recall (sensibilidad) **≥ 85%** sobre el conjunto de prueba congelado. |
| **RNF-02** | Latencia de inferencia | Tiempo de predicción para un único vector **< 200 milisegundos**. |
| **RNF-03** | Disponibilidad y portabilidad | Accesible desde cualquier navegador web moderno sin instalación adicional de software por el usuario clínico. |

#### Requisitos de Ética y Privacidad (RE)

| ID | Requisito | Normativa Aplicable |
|---|---|---|
| **RE-01** | Privacidad y anonimización | Operación exclusiva sobre datos anonimizados bajo los principios de la normativa **HIPAA** (*Health Insurance Portability and Accountability Act*). |

---

## III. Planteamiento del Dataset, Pre-procesamiento y Normalización

### 3.1 Detalle Computacional de Variables

| # | Variable | Nombre Clínico | Tipo | Rango / Categorías | Unidad | Rol |
|---|---|---|---|---|---|---|
| 1 | `age` | Edad del paciente | Numérica Continua | [29 – 77] | Años | Predictor |
| 2 | `sex` | Sexo biológico | Numérica Binaria | {0=Femenino, 1=Masculino} | — | Predictor |
| 3 | `cp` | Tipo de dolor torácico | Categórica Nominal | {0=Asintomático, 1=Angina Atípica, 2=Dolor No Anginoso, 3=Angina Típica} | — | Predictor |
| 4 | `trestbps` | Presión arterial sistólica en reposo | Numérica Continua | [94 – 200] | mmHg | Predictor |
| 5 | `chol` | Colesterol sérico total | Numérica Continua | [126 – 564] | mg/dL | Predictor |
| 6 | `fbs` | Glucemia en ayunas > 120 mg/dL | Numérica Binaria | {0=No, 1=Sí} | — | Predictor |
| 7 | `restecg` | Resultado del ECG en reposo | Categórica Ordinal | {0=Normal, 1=Anorm. ST-T, 2=Hipertrofia VI} | — | Predictor |
| 8 | `thalach` | Frecuencia cardíaca máxima | Numérica Continua | [71 – 202] | lpm | Predictor |
| 9 | `exang` | Angina inducida por ejercicio | Numérica Binaria | {0=No, 1=Sí} | — | Predictor |
| 10 | `oldpeak` | Depresión del segmento ST | Numérica Continua | [0.0 – 6.2] | mm | Predictor |
| 11 | `slope` | Pendiente del segmento ST | Categórica Ordinal | {0=Descendente, 1=Plana, 2=Ascendente} | — | Predictor |
| 12 | `ca` | Nº vasos coloreados por fluoroscopía | Numérica Discreta | {0, 1, 2, 3, 4} | — | Predictor |
| 13 | `thal` | Resultado de la prueba de talio | Categórica Nominal | {0=Sin dato, 1=Normal, 2=Defecto Fijo, 3=Defecto Reversible} | — | Predictor |
| 14 | `target` | Diagnóstico de enfermedad coronaria | Numérica Binaria | {0=Ausencia, 1=Presencia} | — | **Objetivo** |

---

### 3.2 Flujo Determinista de Pre-procesamiento

#### Etapa 1: Purga de Duplicados y Filtrado Geográfico

El dataset bruto en su presentación agregada contiene **1,025 registros**, resultado de la concatenación de cuatro centros clínicos: Cleveland (EE.UU.), Hungría, Suiza y VA Long Beach (EE.UU.). El protocolo aplica:

1. **Eliminación de duplicados exactos:** Se identifican y eliminan todos los registros cuyas 14 columnas son idénticas, reduciendo el dataset de **1,025 a 303 registros reales y únicos** (subconjunto Cleveland original).
2. **Filtrado de variables exógenas:** Se excluyen explícitamente los registros de `dataset_Hungary`, `dataset_Switzerland` y `dataset_VA Long Beach`, eliminando la heterogeneidad poblacional no controlada y los patrones de datos faltantes sistemáticos.
3. **Justificación anti-Leakage:** La purga se realiza **antes de la partición train/test**, garantizando que no existan registros idénticos distribuidos simultáneamente en ambos conjuntos.

#### Etapa 2: Imputación de Valores Faltantes (SimpleImputer)

- **Variables numéricas continuas** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): Imputación por **mediana** (robusta frente a asimetría biomédica).
- **Variables discretas y categóricas** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`): Imputación por **moda** (preserva distribución categórica original).

Implementado con `sklearn.impute.SimpleImputer`, ajustado exclusivamente sobre `X_train`.

#### Etapa 3: Tratamiento de Outliers — Winsorization por IQR

- **`chol` (Colesterol):** Truncamiento al límite `Q3 + 1.5·IQR` e inferior `Q1 - 1.5·IQR`.
- **`trestbps` (Presión Arterial):** Mismo mecanismo de Winsorization.

Este enfoque preserva la estructura de distribución sin eliminar registros del corpus.

#### Etapa 4: Codificación One-Hot Encoding

- **`cp`** (4 categorías): Genera columnas binarias `cp_0`, `cp_1`, `cp_2`, `cp_3`.
- **`thal`** (4 categorías): Genera columnas binarias `thal_0`, `thal_1`, `thal_2`, `thal_3`.

Implementado con `sklearn.preprocessing.OneHotEncoder` (`handle_unknown='ignore'`), ajustado sobre `X_train`.

#### Etapa 5: Estandarización Z-score (StandardScaler)

Todas las variables numéricas continuas se estandarizan:

$$z = \frac{x - \mu_{train}}{\sigma_{train}}$$

Los parámetros $\mu_{train}$ y $\sigma_{train}$ se calculan exclusivamente sobre el conjunto de entrenamiento. Crítico para el correcto funcionamiento del SVM-RBF.

---

### 3.3 Organización de la Partición

#### Partición Estratificada 80/20

| Conjunto | Porcentaje | Registros |
|---|---|---|
| **Entrenamiento (Train)** | 80% | 242 registros |
| **Prueba Hold-out (Test)** | 20% | 61 registros |

Implementado con `train_test_split(stratify=y, random_state=42)`.

#### Validación Cruzada Stratified K-Fold (K=5)

Para la optimización de hiperparámetros vía `GridSearchCV`: los 242 registros de entrenamiento se particionan en 5 subconjuntos estratificados. En cada iteración, 4 subconjuntos entrenan y 1 valida, rotando 5 veces. La métrica de optimización (Recall) se promedia sobre los 5 pliegues. El conjunto de prueba hold-out permanece **completamente congelado**.

---

## IV. Aprendizaje

### 4.1 Planteamiento Multimétodo

| Paradigma | Modelo | Fortaleza Principal | Limitación Conocida |
|---|---|---|---|
| **Ensamble Bagging** | Random Forest | Robustez al ruido, interpretabilidad via importancias | Tendencia al sobreajuste con muchos hiperparámetros |
| **Kernel SVM** | SVM con Kernel RBF | Óptimo en alta dimensionalidad, margen máximo | O(n²) en entrenamiento, sensible a escala |
| **Red Neuronal Profunda** | MLP (Perceptrón Multicapa) | Representaciones no lineales complejas | Sobreajuste en datasets pequeños |

### 4.2 Configuración Detallada de los Modelos

#### Modelo 1: Random Forest (Ensamble Bagging)

`GridSearchCV` con Stratified K-Fold (K=5) y `scoring='recall'`:

```python
param_grid_rf = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [None, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features':     ['sqrt', 'log2'],
    'class_weight':     ['balanced']
}
```

#### Modelo 2: SVM con Kernel RBF (Modelo Ganador — AUC: 0.932)

Kernel gaussiano: $K(x_i, x_j) = e^{-\gamma \|x_i - x_j\|^2}$. Serializado como `mejor_modelo.pkl`.

```python
param_grid_svm = {
    'C':            [0.1, 1, 10, 100],
    'gamma':        ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel':       ['rbf'],
    'class_weight': ['balanced']
}
```

#### Modelo 3: MLP con Keras/TensorFlow

**Topología con tres controles de regularización:**

```
Entrada    → [n_features] neuronas (post-OHE)
Oculta 1   → 64 neuronas, ReLU → BatchNormalization → Dropout(0.25)
Oculta 2   → 32 neuronas, ReLU → BatchNormalization → Dropout(0.25)
Oculta 3   → 16 neuronas, ReLU
Salida     → 1 neurona, Sigmoid
```

- **Función de pérdida:** Binary Cross-Entropy
- **Optimizador:** Adam (lr=0.001)
- **Early Stopping:** `monitor='val_loss'`, `patience=15`, `restore_best_weights=True`

> **Nota diagnóstica:** La estabilización prematura de `val_loss` en datasets pequeños activa Early Stopping antes de la convergencia óptima, limitando el AUC a 0.842.

**Interpretabilidad SHAP:** `shap.TreeExplainer` sobre Random Forest calcula valores de Shapley para cada variable.

---

## V. Comprobación

### 5.1 Protocolo de Entrenamiento Empírico

1. **Congelamiento del conjunto de prueba:** Los 61 registros del conjunto hold-out son inaccesibles durante entrenamiento y optimización.
2. **Fijación de semillas:** `random_state=42` en todos los estimadores; `numpy.random.seed(42)` y `tensorflow.random.set_seed(42)` garantizan reproducibilidad absoluta.
3. **Ajuste del pipeline en training set:** Transformadores ajustados sobre `X_train`, transformados sobre `X_train` y `X_test` (mecanismo primario anti-Leakage).
4. **Búsqueda de hiperparámetros:** `GridSearchCV` K=5 sobre el conjunto de entrenamiento. La combinación de mayor Recall promedio es la configuración definitiva.
5. **Evaluación final:** Los modelos óptimos son re-entrenados sobre los 242 registros completos y evaluados una única vez sobre los 61 registros del conjunto de prueba.

---

## VI. Evaluación

### 6.1 Reporte Comparativo de Curvas ROC-AUC

| Modelo | AUC-ROC | Recall (Sensibilidad) | Precisión | F1-Score | Accuracy |
|---|---|---|---|---|---|
| **SVM-RBF** ⭐ | **0.932** | ≥ 85% | Alta | Alta | Alta |
| **Random Forest** | 0.902 | Alta | Alta | Alta | Alta |
| **MLP (Red Neuronal)** | 0.842 | Media | Media | Media | Media |

- **SVM-RBF (AUC: 0.932):** La transformación implícita del espacio mediante kernel gaussiano separa linealmente clases no separables en el espacio original, aprovechando la dimensionalidad creada por OHE en `cp` y `thal`.
- **Random Forest (AUC: 0.902):** La reducción de varianza por promediación de múltiples árboles produce estimaciones de probabilidad estables y robustas.
- **MLP (AUC: 0.842):** Sobreajuste por **estabilización prematura de `val_loss`**. En corpus de 242 registros, la capacidad representacional de la red no se capitaliza plenamente; Early Stopping se activa antes de convergencia óptima.

### 6.2 Evaluación de Desempeño y Coherencia Fisiopatológica Clínica según SHAP

| Rango | Variable | Nombre Clínico | Justificación Fisiopatológica |
|---|---|---|---|
| 🥇 **1º** | `cp` | Tipo de Dolor Torácico | **Mayor poder discriminatorio.** La angina típica (cp=3) es el síntoma cardinal de isquemia miocárdica. |
| 🥈 **2º** | `thal` | Prueba de Talio | Defectos de perfusión reversibles: marcador imagenológico más específico de isquemia inducible. |
| 🥉 **3º** | `ca` | Nº de Vasos Coronarios | Extensión anatómica de enfermedad coronaria con correlación directa a severidad y pronóstico. |
| **4º** | `oldpeak` | Depresión ST | **Refuerzo isquémico.** Cuantifica el déficit de perfusión inducible; amplifica el peso predictivo superior. |
| **5º** | `thalach` | Frec. Cardíaca Máxima | Correlación negativa con ECV: incapacidad de alcanzar FCM esperada refleja limitación funcional. |
| **6º+** | `age`, `sex`, `exang`, `slope`, `trestbps`, `chol`, `fbs`, `restecg` | Resto de variables | Contribución individual moderada; potenciada en combinación. |

**Coherencia fisiopatológica:** La jerarquía SHAP replica fielmente la lógica diagnóstica cardiológica clásica. El modelo aprendió esta jerarquía exclusivamente desde los datos, validando su coherencia clínica sin conocimiento médico explícito programado.

---

## VII. Deploy del Sistema

### 7.1 Estructura del Repositorio

```
Proyecto_Cardio/
├── README.md                    # Documentación académica completa (este archivo)
├── heart.csv                    # Dataset Cleveland depurado (303 registros)
├── main.py                      # Pipeline de entrenamiento completo
├── test_system.py               # Script de evaluación y generación de gráficas
├── app.py                       # Aplicación web Streamlit (sistema experto clínico)
├── mejor_modelo.pkl             # Modelo SVM-RBF serializado (generado por main.py)
├── scaler_pipeline.pkl          # Pipeline de transformación serializado (generado por main.py)
├── curva_roc_comparativa.png    # Curvas ROC comparativas (generada por test_system.py)
└── SHAP_summary.png             # Mapa de importancia SHAP (generada por test_system.py)
```

### 7.2 Instrucciones de Ejecución

#### Pre-requisitos

```bash
pip install pandas numpy scikit-learn tensorflow keras shap streamlit matplotlib seaborn joblib
```

#### Paso 1: Entrenamiento del Pipeline

```bash
python main.py
```

#### Paso 2: Evaluación y Generación de Gráficas

```bash
python test_system.py
```

#### Paso 3: Lanzar la Aplicación Web

```bash
streamlit run app.py
```

---

## Referencias Bibliográficas

1. **Organización Mundial de la Salud (OMS).** (2021). *Enfermedades cardiovasculares: datos y cifras*. Ginebra: OMS. https://www.who.int/es/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

2. **Rodríguez, L. A., Herrera, D. M., & Vargas, C. E.** (2020). Prevalencia de factores de riesgo cardiovascular en Latinoamérica. *Revista Latinoamericana de Cardiología Preventiva*, 12(3), 145–162.

3. **Organización Panamericana de la Salud (OPS).** (2022). *Informe sobre la situación de la salud en las Américas*. Washington D.C.: OPS/OMS.

4. **Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R.** (1988). *Heart Disease Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

5. **Lundberg, S. M., & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

6. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

7. **Cortes, C., & Vapnik, V.** (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018

8. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.

9. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

10. **Chollet, F.** (2018). *Deep Learning with Python*. Manning Publications.

---

> **Nota de reproducibilidad:** Todos los experimentos fueron ejecutados con `random_state=42` y `numpy.random.seed(42)` para garantizar la reproducibilidad completa de los resultados reportados.

> **Declaración ética:** El dataset utilizado es de dominio público y está completamente anonimizado. Su uso cumple con los principios de la normativa HIPAA para datos de salud anonimizados.

---

*Tercer Informe de Proyecto — Inteligencia Artificial: Principios y Técnicas — UPAO, 2025-II*
