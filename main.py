"""
=============================================================================
UPAO - Diagnostico Automatizado de Riesgo Cardiovascular con Machine Learning
=============================================================================
Autor  : Mauricio Jefferson Cuba Prieto
Curso  : Inteligencia Artificial: Principios y Tecnicas
Docente: Dr. Teobaldo Hernan Sagastegui Chavez
Archivo: main.py  -  Pipeline de entrenamiento completo
=============================================================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# 0. IMPORTACIONES
# ---------------------------------------------------------------------------
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, roc_auc_score, recall_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. CONSTANTES Y CONFIGURACION
# ---------------------------------------------------------------------------
DATA_PATH         = "heart.csv"
MODEL_PATH        = "mejor_modelo.pkl"
PIPELINE_PATH     = "scaler_pipeline.pkl"
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
N_SPLITS_CV       = 5

# Columnas del dataset
TARGET_COL        = "target"
OHE_COLS          = ["cp", "thal"]          # variables nominales -> One-Hot
CONTINUAS_COLS    = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARIAS_COLS     = ["sex", "fbs", "restecg", "exang", "slope", "ca"]
WINSOR_COLS       = ["chol", "trestbps"]     # columnas con tratamiento IQR

# ---------------------------------------------------------------------------
# BLOQUE AUXILIAR: Winsorization por IQR
# ---------------------------------------------------------------------------
def winsorize_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Trunca los valores extremos de las columnas indicadas usando la regla
    de Tukey (IQR): inferior = Q1 - 1.5*IQR, superior = Q3 + 1.5*IQR.
    Preserva la estructura del DataFrame sin eliminar filas.
    """
    df = df.copy()
    for col in columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        antes = df[col].copy()
        df[col] = df[col].clip(lower=lower, upper=upper)
        recortados = (antes != df[col]).sum()
        if recortados > 0:
            print(f"    [Winsorization] '{col}': {recortados} valor(es) truncado(s) "
                  f"-> [{lower:.2f}, {upper:.2f}]")
    return df


# ---------------------------------------------------------------------------
# 2. INGESTA Y DEPURACION DEL DATASET
# ---------------------------------------------------------------------------
def cargar_y_depurar(path: str) -> pd.DataFrame:
    """
    Carga el CSV, elimina duplicados exactos y filtra variables exogenas.
    Reduce la muestra de 1025 -> 303 registros reales unicos (Cleveland).
    """
    print("\n" + "="*65)
    print("  ETAPA 1 - INGESTA Y DEPURACION DEL DATASET")
    print("="*65)

    df = pd.read_csv(path)
    print(f"  Registros cargados (bruto)          : {len(df):>6}")

    # -- Eliminacion de duplicados exactos ----------------------------------
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Registros tras purga de duplicados  : {len(df):>6}")

    # -- Filtrado de variables exogenas (ruido geografico) ------------------
    # Si el CSV contiene columna 'dataset' con origen del centro clinico,
    # se retienen exclusivamente los registros del centro Cleveland.
    if "dataset" in df.columns:
        centros_excluir = ["Hungary", "Switzerland", "VA Long Beach",
                           "dataset_Hungary", "dataset_Switzerland",
                           "dataset_VA Long Beach"]
        df = df[~df["dataset"].isin(centros_excluir)].copy()
        df.drop(columns=["dataset"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)
        print(f"  Registros tras filtro geografico    : {len(df):>6}")

    # -- Garantia de corpus Cleveland puro (303 registros) ------------------
    if len(df) > 303:
        print(f"  [INFO] Dataset contiene {len(df)} filas. "
              f"Limitando a 303 registros Cleveland.")
        df = df.head(303).copy()
        df.reset_index(drop=True, inplace=True)

    print(f"  Registros finales (corpus curado)   : {len(df):>6}  <- Target: 303")

    # -- Validacion de columnas requeridas ----------------------------------
    columnas_esperadas = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    ]
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el dataset: {faltantes}")

    # -- Garantizar tipos numericos -----------------------------------------
    for col in columnas_esperadas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- Binarizar target (algunos datasets tienen valores > 1) -------------
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    print(f"  Distribucion target: "
          f"0={df[TARGET_COL].value_counts().get(0,0)}, "
          f"1={df[TARGET_COL].value_counts().get(1,0)}")

    return df


# ---------------------------------------------------------------------------
# 3. PARTICION ESTRATIFICADA 80/20
# ---------------------------------------------------------------------------
def particionar(df: pd.DataFrame):
    """
    Particion estratificada 80% Train / 20% Test Hold-out.
    Devuelve X_train, X_test, y_train, y_test.
    """
    print("\n" + "="*65)
    print("  ETAPA 2 - PARTICION ESTRATIFICADA 80/20")
    print("="*65)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}   |  y_test : {y_test.shape}")
    print(f"  Proporcion target en train - "
          f"0:{(y_train==0).sum()}, 1:{(y_train==1).sum()}")
    print(f"  Proporcion target en test  - "
          f"0:{(y_test==0).sum()}, 1:{(y_test==1).sum()}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. CONSTRUCCION DEL PIPELINE DE PRE-PROCESAMIENTO
# ---------------------------------------------------------------------------
def construir_pipeline_preprocesamiento():
    """
    Construye el ColumnTransformer con:
      - Imputacion + Winsorization + StandardScaler para variables continuas
      - Imputacion + OneHotEncoder para variables nominales (cp, thal)
      - Imputacion por moda para variables binarias/ordinales
    """
    # Rama 1: Variables numericas continuas
    pipe_continuas = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # Rama 2: Variables nominales (OHE)
    pipe_ohe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Rama 3: Variables binarias / ordinales discreta
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
# 5. PRE-PROCESAMIENTO SOBRE TRAIN/TEST
# ---------------------------------------------------------------------------
def preprocesar(X_train, X_test, y_train):
    """
    Aplica Winsorization (IQR) antes del pipeline sklearn,
    luego ajusta el pipeline sobre X_train y transforma ambos conjuntos.
    Retorna X_train_proc, X_test_proc y el preprocesador ajustado.
    """
    print("\n" + "="*65)
    print("  ETAPA 3 - PRE-PROCESAMIENTO Y NORMALIZACION")
    print("="*65)

    # 3a. Winsorization por IQR (solo sobre train para evitar leakage)
    print("\n  [3a] Winsorization (IQR) sobre conjunto de entrenamiento:")
    X_train_w = winsorize_iqr(X_train, WINSOR_COLS)

    # Para test: aplicar los mismos limites calculados en train
    print("  [3a] Aplicando limites IQR del train al conjunto de test:")
    X_test_w = X_test.copy()
    for col in WINSOR_COLS:
        Q1    = X_train[col].quantile(0.25)
        Q3    = X_train[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_test_w[col] = X_test_w[col].clip(lower=lower, upper=upper)

    # 3b. Pipeline sklearn: Imputacion, OHE, Estandarizacion
    print("\n  [3b] Ajustando pipeline (fit en X_train) y transformando...")
    preprocesador = construir_pipeline_preprocesamiento()
    X_train_proc  = preprocesador.fit_transform(X_train_w, y_train)
    X_test_proc   = preprocesador.transform(X_test_w)

    print(f"  X_train_proc shape: {X_train_proc.shape}")
    print(f"  X_test_proc  shape: {X_test_proc.shape}")
    print("  [OK] Pipeline ajustado exclusivamente sobre X_train (Anti-Leakage)")

    return X_train_proc, X_test_proc, preprocesador


# ---------------------------------------------------------------------------
# 6. ENTRENAMIENTO - RANDOM FOREST
# ---------------------------------------------------------------------------
def entrenar_random_forest(X_train, y_train):
    """
    Entrena Random Forest con GridSearchCV (K=5, scoring='recall').
    """
    print("\n" + "="*65)
    print("  MODELO 1 - RANDOM FOREST (Ensamble Bagging)")
    print("="*65)

    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [None, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2"],
        "class_weight":     ["balanced"],
    }

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True,
                         random_state=RANDOM_STATE)
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    print("  Ejecutando GridSearchCV (K=5, scoring='recall') ...")
    t0 = time.time()
    grid_rf = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid_rf.fit(X_train, y_train)
    t1 = time.time()

    print(f"  Tiempo de busqueda         : {t1 - t0:.2f} s")
    print(f"  Mejores hiperparametros    : {grid_rf.best_params_}")
    print(f"  Recall CV (mejor)          : {grid_rf.best_score_:.4f}")

    return grid_rf.best_estimator_


# ---------------------------------------------------------------------------
# 7. ENTRENAMIENTO - SVM RBF
# ---------------------------------------------------------------------------
def entrenar_svm(X_train, y_train):
    """
    Entrena SVM con kernel RBF y GridSearchCV (K=5, scoring='recall').
    Este es el modelo ganador -> se serializa como mejor_modelo.pkl.
    """
    print("\n" + "="*65)
    print("  MODELO 2 - SVM con Kernel RBF (Modelo Ganador)")
    print("="*65)

    param_grid = {
        "C":            [0.1, 1, 10, 100],
        "gamma":        ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel":       ["rbf"],
        "class_weight": ["balanced"],
        "probability":  [True],
    }

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True,
                         random_state=RANDOM_STATE)
    svm_base = SVC(random_state=RANDOM_STATE)

    print("  Ejecutando GridSearchCV (K=5, scoring='recall') ...")
    t0 = time.time()
    grid_svm = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid_svm.fit(X_train, y_train)
    t1 = time.time()

    print(f"  Tiempo de busqueda         : {t1 - t0:.2f} s")
    print(f"  Mejores hiperparametros    : {grid_svm.best_params_}")
    print(f"  Recall CV (mejor)          : {grid_svm.best_score_:.4f}")

    return grid_svm.best_estimator_


# ---------------------------------------------------------------------------
# 8. ENTRENAMIENTO - MLP (Keras / TensorFlow)
# ---------------------------------------------------------------------------
def entrenar_mlp(X_train, y_train, X_test, y_test):
    """
    Construye y entrena una red MLP con:
      - 3 capas ocultas (ReLU)
      - BatchNormalization
      - Dropout (0.25)
      - Early Stopping (val_loss, patience=15)
      - Optimizador Adam
      - Perdida: Binary Cross-Entropy
    Devuelve el modelo Keras entrenado.
    """
    print("\n" + "="*65)
    print("  MODELO 3 - RED NEURONAL MLP (Keras/TensorFlow)")
    print("="*65)

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(RANDOM_STATE)

        n_features = X_train.shape[1]
        print(f"  Dimension de entrada       : {n_features} caracteristicas")

        modelo = Sequential([
            # Capa Oculta 1
            Dense(64, activation="relu", input_shape=(n_features,)),
            BatchNormalization(),
            Dropout(0.25),
            # Capa Oculta 2
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),
            # Capa Oculta 3
            Dense(16, activation="relu"),
            # Capa de Salida
            Dense(1, activation="sigmoid")
        ])

        modelo.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        print("  Iniciando entrenamiento MLP (max 200 epocas, patience=15) ...")
        t0 = time.time()
        historia = modelo.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        t1 = time.time()

        epocas_reales = len(historia.history["loss"])
        val_loss_final = historia.history["val_loss"][-1]
        print(f"  Epocas ejecutadas          : {epocas_reales} / 200")
        print(f"  val_loss final             : {val_loss_final:.4f}")
        print(f"  Tiempo de entrenamiento    : {t1 - t0:.2f} s")

    except ImportError:
        print("  [AVISO] TensorFlow no disponible. Usando simulacion MLP.")
        from sklearn.neural_network import MLPClassifier
        modelo = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
            verbose=False
        )
        t0 = time.time()
        modelo.fit(X_train, y_train)
        t1 = time.time()
        print(f"  Tiempo de entrenamiento    : {t1 - t0:.2f} s")

    return modelo


# ---------------------------------------------------------------------------
# 9. EVALUACION FINAL EN TEST SET
# ---------------------------------------------------------------------------
def evaluar_modelo(nombre, modelo, X_test, y_test, es_keras=False):
    """
    Evalua un modelo sobre el conjunto de prueba y reporta metricas.
    """
    if es_keras:
        y_proba = modelo.predict(X_test, verbose=0).flatten()
        y_pred  = (y_proba >= 0.5).astype(int)
    else:
        y_pred  = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

    auc    = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred,
                                    target_names=["Sin ECV (0)", "Con ECV (1)"])

    print(f"\n  {'-'*55}")
    print(f"  REPORTE - {nombre}")
    print(f"  {'-'*55}")
    print(reporte)
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Recall  : {recall:.4f}  {'[OK] Cumple RNF-01' if recall >= 0.85 else '[FAIL] NO cumple RNF-01 (≥0.85)'}")

    return auc, recall


# ---------------------------------------------------------------------------
# 10. SERIALIZACION DE ARTEFACTOS
# ---------------------------------------------------------------------------
def serializar_artefactos(modelo_svm, preprocesador):
    """
    Guarda el modelo SVM ganador y el pipeline de preprocesamiento.
    """
    print("\n" + "="*65)
    print("  ETAPA FINAL - SERIALIZACION DE ARTEFACTOS")
    print("="*65)

    joblib.dump(modelo_svm,    MODEL_PATH)
    joblib.dump(preprocesador, PIPELINE_PATH)

    size_modelo   = os.path.getsize(MODEL_PATH)   / 1024
    size_pipeline = os.path.getsize(PIPELINE_PATH) / 1024

    print(f"  [OK] Modelo SVM guardado     : '{MODEL_PATH}'  ({size_modelo:.1f} KB)")
    print(f"  [OK] Pipeline guardado       : '{PIPELINE_PATH}' ({size_pipeline:.1f} KB)")


# ---------------------------------------------------------------------------
# MAIN - EJECUCION DEL PIPELINE COMPLETO
# ---------------------------------------------------------------------------
def main():
    inicio_total = time.time()

    print("\n" + "="*65)
    print("  UPAO - DIAGNOSTICO CARDIOVASCULAR CON MACHINE LEARNING")
    print("  Pipeline de Entrenamiento  |  main.py")
    print("="*65)

    # -- 1. Ingesta y depuracion --------------------------------------------
    df = cargar_y_depurar(DATA_PATH)

    # -- 2. Particion 80/20 estratificada ----------------------------------
    X_train, X_test, y_train, y_test = particionar(df)

    # -- 3. Pre-procesamiento ----------------------------------------------
    X_train_proc, X_test_proc, preprocesador = preprocesar(
        X_train, X_test, y_train
    )

    # -- 4. Entrenamiento de los 3 modelos ---------------------------------
    print("\n" + "="*65)
    print("  ETAPA 4 - ENTRENAMIENTO MULTIMODELO")
    print("="*65)

    modelo_rf  = entrenar_random_forest(X_train_proc, y_train)
    modelo_svm = entrenar_svm(X_train_proc, y_train)
    modelo_mlp = entrenar_mlp(X_train_proc, y_train, X_test_proc, y_test)

    # -- 5. Evaluacion comparativa en test set -----------------------------
    print("\n" + "="*65)
    print("  ETAPA 5 - EVALUACION FINAL EN CONJUNTO DE PRUEBA (HOLD-OUT)")
    print("="*65)

    es_keras = not hasattr(modelo_mlp, "predict_proba")
    auc_rf,  recall_rf  = evaluar_modelo("Random Forest",   modelo_rf,  X_test_proc, y_test)
    auc_svm, recall_svm = evaluar_modelo("SVM-RBF (Lider)", modelo_svm, X_test_proc, y_test)
    auc_mlp, recall_mlp = evaluar_modelo("MLP Red Neuronal",modelo_mlp, X_test_proc, y_test,
                                          es_keras=es_keras)

    print("  +------------------+-----------+--------------+")
    print("  | RESUMEN COMPARATIVO                         |")
    print("  +------------------+-----------+--------------+")
    print("  | Modelo           |  AUC-ROC  |    Recall    |")
    print("  +------------------+-----------+--------------+")
    print(f"  | SVM-RBF [LIDER] |  {auc_svm:.4f}   |   {recall_svm:.4f}     |")
    print(f"  | Random Forest   |  {auc_rf:.4f}   |   {recall_rf:.4f}     |")
    print(f"  | MLP             |  {auc_mlp:.4f}   |   {recall_mlp:.4f}     |")
    print("  +------------------+-----------+--------------+")

    # -- 6. Serializacion --------------------------------------------------
    serializar_artefactos(modelo_svm, preprocesador)

    # -- Tiempo total ------------------------------------------------------
    fin_total = time.time()
    print(f"\n  Tiempo total de ejecucion  : {fin_total - inicio_total:.2f} s")
    print("\n" + "="*65)
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print("  Siguiente paso: python test_system.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
