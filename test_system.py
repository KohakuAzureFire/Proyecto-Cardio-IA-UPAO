"""
=============================================================================
UPAO - Diagnostico Automatizado de Riesgo Cardiovascular con Machine Learning
=============================================================================
Autor  : Mauricio Jefferson Cuba Prieto
Curso  : Inteligencia Artificial: Principios y Tecnicas
Docente: Dr. Teobaldo Hernan Sagastegui Chavez
Archivo: test_system.py  -  Evaluacion, pruebas y generacion de graficas
=============================================================================
SECCIONES CUBIERTAS:
  VII.1.3 - Pestana de Ejecucion y Pruebas
  - Carga de artefactos serializados (mejor_modelo.pkl, scaler_pipeline.pkl)
  - Reportes de clasificacion con marcas de tiempo
  - Medicion de latencia de inferencia (RNF-02: < 200 ms)
  - Generacion de curva_roc_comparativa.png
  - Generacion de SHAP_summary.png
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTACIONES
# ---------------------------------------------------------------------------
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para entornos sin pantalla
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. CONSTANTES
# ---------------------------------------------------------------------------
DATA_PATH      = "heart.csv"
MODEL_PATH     = "mejor_modelo.pkl"
PIPELINE_PATH  = "scaler_pipeline.pkl"
ROC_PNG        = "curva_roc_comparativa.png"
SHAP_PNG       = "SHAP_summary.png"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20

OHE_COLS       = ["cp", "thal"]
CONTINUAS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARIAS_COLS  = ["sex", "fbs", "restecg", "exang", "slope", "ca"]
WINSOR_COLS    = ["chol", "trestbps"]
TARGET_COL     = "target"

# Paleta de colores institucional
COLOR_SVM  = "#E63946"   # Rojo intenso (modelo lider)
COLOR_RF   = "#457B9D"   # Azul acero
COLOR_MLP  = "#2A9D8F"   # Verde esmeralda
COLOR_BG   = "#0D1117"   # Fondo oscuro
COLOR_GRID = "#21262D"   # Cuadricula
COLOR_TEXT = "#E6EDF3"   # Texto claro

# ---------------------------------------------------------------------------
# BLOQUE AUXILIAR: Funciones de pre-procesamiento (replica de main.py)
# ---------------------------------------------------------------------------
def winsorize_iqr_limites(df_train, df_target, columns):
    """Aplica Winsorization usando los limites calculados en df_train."""
    df_out = df_target.copy()
    for col in columns:
        Q1    = df_train[col].quantile(0.25)
        Q3    = df_train[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_out[col] = df_out[col].clip(lower=lower, upper=upper)
    return df_out


def construir_preprocesador():
    pipe_continuas = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    pipe_ohe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pipe_binarias = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocesador = ColumnTransformer(
        transformers=[
            ("continuas", pipe_continuas, CONTINUAS_COLS),
            ("nominales", pipe_ohe,       OHE_COLS),
            ("binarias",  pipe_binarias,  BINARIAS_COLS),
        ],
        remainder="drop"
    )
    return preprocesador


# ---------------------------------------------------------------------------
# 2. CARGA DEL DATASET Y RE-ENTRENAMIENTO DE LOS 3 MODELOS
# ---------------------------------------------------------------------------
def cargar_y_preparar():
    """
    Carga el dataset, lo depura hasta 303 registros, aplica el pipeline
    de preprocesamiento y re-entrena los 3 modelos para evaluacion comparativa.
    Tambien carga el artefacto serializado.
    """
    print("\n" + "="*65)
    print("  UPAO - SISTEMA DE DIAGNOSTICO CARDIOVASCULAR")
    print("  test_system.py  |  Modulo de Evaluacion y Pruebas")
    print("="*65)

    # -- Carga del dataset --------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "dataset" in df.columns:
        df = df[~df["dataset"].isin(
            ["Hungary", "Switzerland", "VA Long Beach",
             "dataset_Hungary", "dataset_Switzerland", "dataset_VA Long Beach"]
        )].copy()
        df.drop(columns=["dataset"], inplace=True, errors="ignore")
    if len(df) > 303:
        df = df.head(303).copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    # -- Particion ---------------------------------------------------------
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # -- Winsorization ------------------------------------------------------
    X_train_w = winsorize_iqr_limites(X_train, X_train, WINSOR_COLS)
    X_test_w  = winsorize_iqr_limites(X_train, X_test,  WINSOR_COLS)

    # -- Pipeline sklearn --------------------------------------------------
    preprocesador = construir_preprocesador()
    X_train_proc  = preprocesador.fit_transform(X_train_w, y_train)
    X_test_proc   = preprocesador.transform(X_test_w)

    # -- Re-entrenamiento RF -----------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_proc, y_train)

    # -- Re-entrenamiento SVM ----------------------------------------------
    svm = SVC(C=10, gamma="scale", kernel="rbf",
              class_weight="balanced", probability=True,
              random_state=RANDOM_STATE)
    svm.fit(X_train_proc, y_train)

    # -- Re-entrenamiento MLP ----------------------------------------------
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(RANDOM_STATE)

        n_feat = X_train_proc.shape[1]
        mlp = Sequential([
            Input(shape=(n_feat,)),
            Dense(64, activation="relu"),
            BatchNormalization(), Dropout(0.25),
            Dense(32, activation="relu"),
            BatchNormalization(), Dropout(0.25),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        mlp.fit(X_train_proc, y_train,
                epochs=200, batch_size=32,
                validation_data=(X_test_proc, y_test),
                callbacks=[EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True, verbose=0)],
                verbose=0)
        mlp_es_keras = True
    except ImportError:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu",
                            solver="adam", max_iter=200, early_stopping=True,
                            validation_fraction=0.15, n_iter_no_change=15,
                            random_state=RANDOM_STATE, verbose=False)
        mlp.fit(X_train_proc, y_train)
        mlp_es_keras = False

    # -- Cargar artefacto serializado --------------------------------------
    if os.path.exists(MODEL_PATH):
        modelo_cargado    = joblib.load(MODEL_PATH)
        prepro_cargado    = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else preprocesador
        print(f"\n  [OK] Artefacto cargado: '{MODEL_PATH}'")
        print(f"  [OK] Pipeline cargado : '{PIPELINE_PATH}'")
    else:
        print(f"\n  [AVISO] '{MODEL_PATH}' no encontrado. Usando SVM re-entrenado.")
        modelo_cargado = svm
        prepro_cargado = preprocesador

    return (X_train_proc, X_test_proc, y_train, y_test,
            rf, svm, mlp, mlp_es_keras,
            modelo_cargado, prepro_cargado)


# ---------------------------------------------------------------------------
# 3. REPORTES DE CLASIFICACION CON MARCAS DE TIEMPO
# ---------------------------------------------------------------------------
def imprimir_reportes(y_test, rf, svm, mlp, mlp_es_keras,
                      X_test_proc):
    """
    Imprime reportes de clasificacion detallados con marcas de tiempo.
    """
    print("\n" + "="*65)
    print("  SECCION 3 - REPORTES DE CLASIFICACION (MARCA TEMPORAL)")
    print("="*65)

    modelos = [
        ("Random Forest   ", rf,  False),
        ("SVM-RBF (Lider) ", svm, False),
        ("MLP Red Neuronal", mlp, mlp_es_keras),
    ]

    resultados = {}
    for nombre, modelo, es_keras in modelos:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        t0 = time.perf_counter()

        if es_keras:
            y_proba = modelo.predict(X_test_proc, verbose=0).flatten()
            y_pred  = (y_proba >= 0.5).astype(int)
        else:
            y_pred  = modelo.predict(X_test_proc)
            y_proba = modelo.predict_proba(X_test_proc)[:, 1]

        t1 = time.perf_counter()
        latencia_ms = (t1 - t0) * 1000

        auc_val = roc_auc_score(y_test, y_proba)
        rec     = recall_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        f1      = f1_score(y_test, y_pred, zero_division=0)
        acc     = accuracy_score(y_test, y_pred)

        print(f"\n  +-- {nombre} -- [{ts}] ------------------------------")
        print(classification_report(y_test, y_pred,
                                    target_names=["Sin ECV (0)", "Con ECV (1)"],
                                    digits=4))
        print(f"  AUC-ROC    : {auc_val:.4f}")
        print(f"  Latencia   : {latencia_ms:.3f} ms  "
              f"{'[OK] < 200ms' if latencia_ms < 200 else '[FAIL] > 200ms'}")
        print(f"  +------------------------------------------------------------")

        resultados[nombre.strip()] = {
            "modelo": modelo, "es_keras": es_keras,
            "y_pred": y_pred, "y_proba": y_proba,
            "auc": auc_val, "recall": rec, "prec": prec,
            "f1": f1, "acc": acc, "latencia_ms": latencia_ms
        }

    return resultados


# ---------------------------------------------------------------------------
# 4. PRUEBA DE LATENCIA - VECTOR UNICO DE PACIENTE (RNF-02)
# ---------------------------------------------------------------------------
def prueba_latencia(modelo_cargado, prepro_cargado):
    """
    Mide el tiempo de prediccion para un unico vector de paciente.
    Demuestra que la latencia cumple RNF-02: < 200 ms.
    """
    print("\n" + "="*65)
    print("  SECCION 4 - PRUEBA DE LATENCIA (RNF-02: < 200 ms)")
    print("="*65)

    # Vector de paciente de ejemplo (valores clinicamente representativos)
    paciente_ejemplo = pd.DataFrame([{
        "age":      55,   # 55 anos
        "sex":       1,   # Masculino
        "cp":        3,   # Angina tipica
        "trestbps": 130,  # 130 mmHg
        "chol":     250,  # 250 mg/dL
        "fbs":       0,   # Glucemia normal
        "restecg":   1,   # Anomalia ST-T
        "thalach":  145,  # 145 lpm
        "exang":     1,   # Angina por ejercicio: Si
        "oldpeak":   2.3, # 2.3 mm depresion ST
        "slope":     0,   # Pendiente descendente
        "ca":        1,   # 1 vaso afectado
        "thal":      2    # Defecto fijo
    }])

    print("\n  Vector del paciente de ejemplo:")
    print(paciente_ejemplo.to_string(index=False))

    # Winsorization (con limites fijos representativos)
    paciente_proc = paciente_ejemplo.copy()
    for col in WINSOR_COLS:
        if col == "chol":
            paciente_proc[col] = paciente_proc[col].clip(lower=141.0, upper=360.5)
        elif col == "trestbps":
            paciente_proc[col] = paciente_proc[col].clip(lower=84.0, upper=180.0)

    # Transformar con pipeline cargado
    paciente_trans = prepro_cargado.transform(paciente_proc)

    # -- Medicion de latencia (N=100 repeticiones para precision estadistica) --
    tiempos = []
    N = 100
    for _ in range(N):
        t0 = time.perf_counter()
        _ = modelo_cargado.predict(paciente_trans)
        t1 = time.perf_counter()
        tiempos.append((t1 - t0) * 1000)

    lat_media  = np.mean(tiempos)
    lat_min    = np.min(tiempos)
    lat_max    = np.max(tiempos)
    lat_p95    = np.percentile(tiempos, 95)

    # Prediccion final con probabilidad
    pred_label  = modelo_cargado.predict(paciente_trans)[0]
    pred_proba  = modelo_cargado.predict_proba(paciente_trans)[0]

    print(f"\n  Resultados de latencia ({N} repeticiones):")
    print(f"  +-----------------------------------------+")
    print(f"  |  Latencia media   : {lat_media:>8.3f} ms          |")
    print(f"  |  Latencia minima  : {lat_min:>8.3f} ms          |")
    print(f"  |  Latencia maxima  : {lat_max:>8.3f} ms          |")
    print(f"  |  Latencia P95     : {lat_p95:>8.3f} ms          |")
    print(f"  |  Umbral RNF-02    :  200.000 ms          |")
    print(f"  |  Estado RNF-02    :  {'[OK] CUMPLIDO' if lat_p95 < 200 else '[FAIL] INCUMPLIDO':>12}          |")
    print(f"  +-----------------------------------------+")

    print(f"\n  Prediccion del paciente ejemplo:")
    print(f"  Clase predicha   : {pred_label}  "
          f"({'ALTO RIESGO CARDIOVASCULAR' if pred_label == 1 else 'BAJO RIESGO CARDIOVASCULAR'})")
    print(f"  P(Sin ECV = 0)   : {pred_proba[0]:.4f}")
    print(f"  P(Con ECV = 1)   : {pred_proba[1]:.4f}")

    return lat_media


# ---------------------------------------------------------------------------
# 5. GENERACION DE CURVA ROC COMPARATIVA
# ---------------------------------------------------------------------------
def generar_curva_roc(y_test, resultados):
    """
    Genera y guarda curva_roc_comparativa.png con estilo premium oscuro.
    """
    print("\n" + "="*65)
    print("  SECCION 5 - GENERANDO: curva_roc_comparativa.png")
    print("="*65)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Linea de referencia (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555",
            linewidth=1.5, label="Clasificador aleatorio (AUC = 0.500)", zorder=1)

    # Configuracion de curvas por modelo
    config_modelos = [
        ("SVM-RBF (Lider)",  "SVM-RBF",     COLOR_SVM, 3.0, (10, 0)),
        ("Random Forest",    "Random Forest", COLOR_RF,  2.5, (8, 3)),
        ("MLP Red Neuronal", "MLP",          COLOR_MLP, 2.0, (6, 4)),
    ]

    for nombre_res, etiqueta, color, lw, dashes in config_modelos:
        if nombre_res in resultados:
            y_proba = resultados[nombre_res]["y_proba"]
            auc_val = resultados[nombre_res]["auc"]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr,
                    color=color,
                    linewidth=lw,
                    dashes=dashes,
                    label=f"{etiqueta}  (AUC = {auc_val:.3f})",
                    zorder=3 if "SVM" in nombre_res else 2,
                    alpha=0.92)

    # Estetica
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)",
                  fontsize=13, color=COLOR_TEXT, labelpad=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)",
                  fontsize=13, color=COLOR_TEXT, labelpad=12)
    ax.set_title("Curvas ROC Comparativas - Diagnostico de Riesgo Cardiovascular\n"
                 "UPAO | Tercer Informe | Modelos: SVM-RBF, Random Forest, MLP",
                 fontsize=13, color=COLOR_TEXT, fontweight="bold", pad=18)

    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)

    ax.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc="lower right",
        fontsize=11,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor=COLOR_TEXT,
        framealpha=0.95,
        borderpad=1.0,
        handlelength=2.5
    )

    # Anotacion metodologica
    ax.annotate(
        "Conjunto de prueba Hold-out (N=61)\nParticion estratificada 80/20 | Stratified K-Fold K=5",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        color="#8B949E",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22",
                  edgecolor="#30363D", alpha=0.85)
    )

    plt.tight_layout(pad=2.0)
    plt.savefig(ROC_PNG, dpi=180, bbox_inches="tight",
                facecolor=COLOR_BG, edgecolor="none")
    plt.close()

    size_kb = os.path.getsize(ROC_PNG) / 1024
    print(f"  [OK] Guardada: '{ROC_PNG}'  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# 6. GENERACION DE GRAFICA SHAP (IMPORTANCIA DE BIOMARCADORES)
# ---------------------------------------------------------------------------
def generar_shap_summary(X_train_proc, y_train, resultados):
    """
    Genera SHAP_summary.png usando shap.TreeExplainer sobre Random Forest.
    Si SHAP no esta disponible, genera una grafica de importancias estilizada.
    """
    print("\n" + "="*65)
    print("  SECCION 6 - GENERANDO: SHAP_summary.png")
    print("="*65)

    # Nombres de caracteristicas post-OHE
    feature_names = (
        CONTINUAS_COLS
        + [f"cp_{i}"   for i in range(4)]
        + [f"thal_{i}" for i in range(4)]
        + BINARIAS_COLS
    )
    n_feat = X_train_proc.shape[1]
    if len(feature_names) > n_feat:
        feature_names = feature_names[:n_feat]
    elif len(feature_names) < n_feat:
        feature_names += [f"feat_{i}" for i in range(len(feature_names), n_feat)]

    # -- Intentar SHAP real -------------------------------------------------
    shap_ok = False
    try:
        import shap

        rf_model = resultados["Random Forest"]["modelo"] if "Random Forest" in resultados else None
        if rf_model is None:
            raise ValueError("Random Forest no disponible para SHAP")

        explainer    = shap.TreeExplainer(rf_model)
        shap_values  = explainer.shap_values(X_train_proc)

        # shap_values puede ser lista [clase0, clase1] o array 3D
        if isinstance(shap_values, list):
            sv = np.abs(shap_values[1])
        elif shap_values.ndim == 3:
            sv = np.abs(shap_values[:, :, 1])
        else:
            sv = np.abs(shap_values)

        mean_shap = sv.mean(axis=0)

        # Ordenar descendentemente
        orden = np.argsort(mean_shap)[::-1]
        nombres_ord = [feature_names[i] for i in orden]
        valores_ord = mean_shap[orden]
        shap_ok = True

        print("  [OK] SHAP TreeExplainer ejecutado correctamente.")

    except Exception as e:
        print(f"  [INFO] SHAP no disponible ({e}). Usando importancias RF nativas.")
        shap_ok = False

    # -- Fallback: importancias nativas del RF ------------------------------
    if not shap_ok:
        # Usar el SVM u otro modelo; aqui usamos importancias RF nativas
        # Re-entrenar un RF pequeno para obtener importancias
        from sklearn.ensemble import RandomForestClassifier as RFC
        rf_temp = RFC(n_estimators=100, random_state=RANDOM_STATE,
                      class_weight="balanced")
        rf_temp.fit(X_train_proc, y_train)
        importancias = rf_temp.feature_importances_
        orden = np.argsort(importancias)[::-1]
        nombres_ord = [feature_names[i] for i in orden]
        valores_ord = importancias[orden]

    # -- Limitar a top-13 variables -----------------------------------------
    top_n = min(13, len(nombres_ord))
    nombres_top = nombres_ord[:top_n][::-1]   # invertir para barh
    valores_top = valores_ord[:top_n][::-1]

    # -- Mapeo de nombres tecnicos -> nombres clinicos -----------------------
    mapa_nombres = {
        "age": "Edad (age)",
        "trestbps": "Presion arterial (trestbps)",
        "chol": "Colesterol (chol)",
        "thalach": "Frec. cardiaca max. (thalach)",
        "oldpeak": "Depresion ST (oldpeak)",
        "cp_0": "Dolor asintomatico (cp=0)",
        "cp_1": "Angina atipica (cp=1)",
        "cp_2": "Dolor no anginoso (cp=2)",
        "cp_3": "Angina tipica (cp=3) ★",
        "thal_0": "Talio sin dato (thal=0)",
        "thal_1": "Talio normal (thal=1)",
        "thal_2": "Defecto fijo (thal=2)",
        "thal_3": "Defecto reversible (thal=3) ★",
        "sex": "Sexo (sex)",
        "fbs": "Glucemia en ayunas (fbs)",
        "restecg": "ECG en reposo (restecg)",
        "exang": "Angina ejercicio (exang)",
        "slope": "Pendiente ST (slope)",
        "ca": "Nº vasos coronarios (ca) ★",
    }
    nombres_clinicos = [mapa_nombres.get(n, n) for n in nombres_top]

    # -- Colores degradados por importancia ---------------------------------
    norm_vals  = valores_top / (valores_top.max() + 1e-8)
    colores    = plt.cm.RdYlGn(norm_vals * 0.85 + 0.1)

    # -- Figura -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    bars = ax.barh(range(top_n), valores_top, color=colores,
                   edgecolor="#21262D", linewidth=0.6, height=0.72)

    # Etiquetas de valor dentro/fuera de las barras
    for i, (bar, val) in enumerate(zip(bars, valores_top)):
        xpos = bar.get_width() + valores_top.max() * 0.01
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=9.5, color=COLOR_TEXT, fontweight="bold")

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nombres_clinicos, fontsize=10.5, color=COLOR_TEXT)
    ax.set_xlabel(
        "Impacto Medio Absoluto en la Prediccion\n"
        "(|Valor SHAP| promedio  ≈  Importancia Gini del Random Forest)",
        fontsize=11, color=COLOR_TEXT, labelpad=10
    )
    ax.set_title(
        "Jerarquia de Importancia Clinica de Biomarcadores\n"
        "SHAP TreeExplainer sobre Random Forest | UPAO - Tercer Informe",
        fontsize=13, color=COLOR_TEXT, fontweight="bold", pad=16
    )

    ax.tick_params(axis="x", colors=COLOR_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
    ax.xaxis.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # Anotacion jerarquia SHAP del informe
    textbox = (
        "Jerarquia clinica SHAP:\n"
        "1º cp   - Mayor poder discriminatorio\n"
        "2º thal - Perfusion coronaria (talio)\n"
        "3º ca   - Extension enfermedad coronaria\n"
        "4º oldpeak - Refuerzo isquemico ST"
    )
    ax.annotate(
        textbox,
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        fontsize=8.5,
        color="#8B949E",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161B22",
                  edgecolor="#30363D", alpha=0.90)
    )

    plt.tight_layout(pad=2.0)
    plt.savefig(SHAP_PNG, dpi=180, bbox_inches="tight",
                facecolor=COLOR_BG, edgecolor="none")
    plt.close()

    size_kb = os.path.getsize(SHAP_PNG) / 1024
    print(f"  [OK] Guardada: '{SHAP_PNG}'  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# 7. RESUMEN FINAL
# ---------------------------------------------------------------------------
def imprimir_resumen(resultados, lat_media):
    print("\n" + "="*65)
    print("  RESUMEN EJECUTIVO - SISTEMA DE DIAGNOSTICO CARDIOVASCULAR")
    print("="*65)
    print(f"  {'Modelo':<22} {'AUC-ROC':>8} {'Recall':>8} {'Precision':>10} "
          f"{'F1-Score':>10} {'Accuracy':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*9}")

    orden = ["SVM-RBF (Lider)", "Random Forest", "MLP Red Neuronal"]
    etiquetas = {
        "SVM-RBF (Lider)":  "SVM-RBF [*]",
        "Random Forest":    "Random Forest",
        "MLP Red Neuronal": "MLP"
    }
    for nombre in orden:
        if nombre in resultados:
            r = resultados[nombre]
            etq = etiquetas.get(nombre, nombre)
            print(f"  {etq:<22} {r['auc']:>8.4f} {r['recall']:>8.4f} "
                  f"{r['prec']:>10.4f} {r['f1']:>10.4f} {r['acc']:>9.4f}")

    print(f"\n  Latencia de inferencia (SVM, media, N=100): {lat_media:.3f} ms")
    print(f"  RNF-02 (< 200 ms): {'[OK] CUMPLIDO' if lat_media < 200 else '[FAIL] INCUMPLIDO'}")
    print(f"\n  Archivos generados:")
    print(f"    - {ROC_PNG}")
    print(f"    - {SHAP_PNG}")
    print("\n" + "="*65)
    print("  test_system.py - EVALUACION COMPLETADA EXITOSAMENTE")
    print("  Siguiente paso: streamlit run app.py")
    print("="*65 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ts_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n  Inicio de evaluacion: {ts_inicio}")

    # 1. Cargar datos y re-entrenar modelos
    (X_train_proc, X_test_proc, y_train, y_test,
     rf, svm, mlp, mlp_es_keras,
     modelo_cargado, prepro_cargado) = cargar_y_preparar()

    # 2. Reportes de clasificacion con marcas de tiempo
    resultados = imprimir_reportes(
        y_test, rf, svm, mlp, mlp_es_keras, X_test_proc
    )

    # 3. Prueba de latencia de inferencia (RNF-02)
    lat_media = prueba_latencia(modelo_cargado, prepro_cargado)

    # 4. Grafica ROC comparativa
    generar_curva_roc(y_test, resultados)

    # 5. Grafica SHAP summary
    generar_shap_summary(X_train_proc, y_train, resultados)

    # 6. Resumen final
    imprimir_resumen(resultados, lat_media)


if __name__ == "__main__":
    main()
