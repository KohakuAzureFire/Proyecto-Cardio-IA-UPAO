"""
=============================================================================
UPAO - Diagnóstico Automatizado de Riesgo Cardiovascular con Machine Learning
=============================================================================
Autor  : Mauricio Jefferson Cuba Prieto
Curso  : Inteligencia Artificial: Principios y Técnicas
Docente: Dr. Teobaldo Hernán Sagástegui Chávez
Archivo: app.py  —  Aplicación Web Streamlit (Sistema Experto Clínico)
=============================================================================
SECCIÓN CUBIERTA: VII.2 — Deploy de la Aplicación de Predicción
Ejecutar con: streamlit run app.py
=============================================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA (debe ser la primera instrucción de Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UPAO — Sistema Experto de Diagnóstico Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    None,
        "Report a bug": None,
        "About": (
            "**Sistema Experto de Diagnóstico de Riesgo Cardiovascular**\n\n"
            "Desarrollado por Mauricio Jefferson Cuba Prieto\n"
            "UPAO — Inteligencia Artificial: Principios y Técnicas\n"
            "Dr. Teobaldo Hernán Sagástegui Chávez — 2025-II"
        )
    }
)

# ---------------------------------------------------------------------------
# CONSTANTES Y RUTAS
# ---------------------------------------------------------------------------
MODEL_PATH    = "mejor_modelo.pkl"
PIPELINE_PATH = "scaler_pipeline.pkl"

CONTINUAS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
OHE_COLS       = ["cp", "thal"]
BINARIAS_COLS  = ["sex", "fbs", "restecg", "exang", "slope", "ca"]
WINSOR_LIMITES = {
    "chol":     (141.0, 360.5),
    "trestbps": (84.0,  180.0),
}

# ---------------------------------------------------------------------------
# CSS PERSONALIZADO — Interfaz Premium Dark Mode
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Fuente global ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Fondo principal ──────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0D1117 0%, #0A0F1A 50%, #0D1117 100%);
    min-height: 100vh;
}

/* ── Header institucional ─────────────────────────────────────────── */
.header-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #0D1117 100%);
    border: 1px solid #30363D;
    border-left: 4px solid #E63946;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(230, 57, 70, 0.12);
}

.header-title {
    font-size: 2.0rem;
    font-weight: 700;
    color: #E6EDF3;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
    line-height: 1.2;
}

.header-subtitle {
    font-size: 1.0rem;
    color: #8B949E;
    font-weight: 400;
    margin: 0;
}

.header-badge {
    display: inline-block;
    background: rgba(230, 57, 70, 0.15);
    border: 1px solid rgba(230, 57, 70, 0.4);
    color: #E63946;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Tarjetas de secciones ────────────────────────────────────────── */
.section-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Sliders y selectboxes ────────────────────────────────────────── */
/* Hace que el círculo deslizante (thumb) sea rojo */
div[role="slider"] {
    background-color: #E63946 !important;
}

/* Hace que la barra activa (track) sea roja */
div[data-baseweb="slider"] > div > div > div {
    background: #E63946 !important;
}

/* Cambia el color del número encima del círculo a blanco brillante para que resalte */
[data-testid="stThumbValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ── Botón principal ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #E63946 0%, #C1121F 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 32px !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(230, 57, 70, 0.35) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(230, 57, 70, 0.55) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Alertas de resultado ─────────────────────────────────────────── */
.resultado-alto {
    background: linear-gradient(135deg, rgba(230, 57, 70, 0.15) 0%,
                                         rgba(193, 18, 31, 0.08) 100%);
    border: 2px solid #E63946;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    animation: pulse-red 2s ease-in-out infinite;
}

.resultado-bajo {
    background: linear-gradient(135deg, rgba(35, 134, 54, 0.15) 0%,
                                         rgba(26, 127, 55, 0.08) 100%);
    border: 2px solid #2DA44E;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    animation: pulse-green 2s ease-in-out infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(230, 57, 70, 0.0); }
    50%       { box-shadow: 0 0 0 8px rgba(230, 57, 70, 0.15); }
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(35, 134, 54, 0.0); }
    50%       { box-shadow: 0 0 0 8px rgba(35, 134, 54, 0.15); }
}

.resultado-icon { font-size: 3.5rem; margin-bottom: 8px; }
.resultado-titulo-alto {
    font-size: 1.5rem; font-weight: 700; color: #F85149;
    margin-bottom: 6px;
}
.resultado-titulo-bajo {
    font-size: 1.5rem; font-weight: 700; color: #3FB950;
    margin-bottom: 6px;
}
.resultado-desc {
    font-size: 1.0rem; color: #8B949E; line-height: 1.5;
}

/* ── Métricas de probabilidad ─────────────────────────────────────── */
.prob-card {
    background: #1C2128;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.prob-valor {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.prob-label {
    font-size: 0.8rem;
    color: #8B949E;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* ── Info chips ───────────────────────────────────────────────────── */
.chip {
    display: inline-block;
    background: #1C2128;
    border: 1px solid #30363D;
    color: #8B949E;
    font-size: 0.78rem;
    padding: 4px 10px;
    border-radius: 16px;
    margin: 3px 2px;
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D !important;
}

/* ── Labels de inputs ─────────────────────────────────────────────── */
.stSelectbox label, .stSlider label {
    color: #C9D1D9 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── Expanders ────────────────────────────────────────────────────── */
details {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CARGA LAZY DE ARTEFACTOS (cacheada)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_artefactos():
    """Carga el modelo SVM y el pipeline de preprocesamiento desde disco."""
    try:
        modelo    = joblib.load(MODEL_PATH)
        pipeline  = joblib.load(PIPELINE_PATH)
        return modelo, pipeline, True, None
    except FileNotFoundError as e:
        return None, None, False, str(e)


# ---------------------------------------------------------------------------
# FUNCIÓN DE PRE-PROCESAMIENTO PARA INFERENCIA
# ---------------------------------------------------------------------------
def preprocesar_vector(datos_raw: dict, pipeline) -> np.ndarray:
    """
    Aplica Winsorization + pipeline de transformación a un vector de paciente.
    Devuelve el array pre-procesado listo para el modelo.
    """
    df = pd.DataFrame([datos_raw])

    # Winsorization con límites del entrenamiento
    for col, (lower, upper) in WINSOR_LIMITES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    # Transformar con pipeline ajustado (imputer + OHE + scaler)
    X_trans = pipeline.transform(df)
    return X_trans


# ---------------------------------------------------------------------------
# BARRA LATERAL — INFORMACIÓN DEL SISTEMA
# ---------------------------------------------------------------------------
def renderizar_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size: 3rem;">🫀</div>
            <div style="color:#E63946; font-weight:700; font-size:1.1rem;
                        letter-spacing:0.5px;">CardioAI UPAO</div>
            <div style="color:#8B949E; font-size:0.8rem;">Sistema Experto Clínico</div>
        </div>
        <hr style="border-color:#21262D; margin:12px 0;">
        """, unsafe_allow_html=True)

        st.markdown("**📊 Modelo en Producción**")
        st.markdown("""
        <div class="chip">SVM — Kernel RBF</div>
        <div class="chip">AUC: 0.932</div>
        <div class="chip">Recall ≥ 85%</div>
        <div class="chip">Latencia &lt; 200 ms</div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#21262D; margin:14px 0;'>",
                    unsafe_allow_html=True)

        st.markdown("**📋 Biomarcadores**")
        st.markdown("""
        | Dimensión | Variables |
        |---|---|
        | Demog. | `age`, `sex` |
        | Hemodin. | `trestbps`, `thalach`, `exang` |
        | Bioquím. | `chol`, `fbs` |
        | Cardiol. | `cp`, `restecg`, `oldpeak`, `slope`, `ca`, `thal` |
        """)

        st.markdown("<hr style='border-color:#21262D; margin:14px 0;'>",
                    unsafe_allow_html=True)

        st.markdown("**⚖️ Marco Normativo**")
        st.caption("Datos anonimizados — Normativa HIPAA")
        st.caption("Dataset Cleveland UCI — Dominio público")
        st.caption("Pipeline anti-Leakage (Fit solo en Train)")

        st.markdown("<hr style='border-color:#21262D; margin:14px 0;'>",
                    unsafe_allow_html=True)

        st.markdown("**👨‍💻 Autor**")
        st.caption("Mauricio J. Cuba Prieto")
        st.caption("UPAO — IA: Principios y Técnicas")
        st.caption(f"Versión 1.0 — 2025-II")


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    renderizar_sidebar()

    # ── HEADER INSTITUCIONAL ───────────────────────────────────────────────
    st.markdown("""
    <div class="header-card">
        <div style="margin-bottom:10px;">
            <span class="header-badge">UPAO</span>
            <span class="header-badge">ML Clínico</span>
            <span class="header-badge">Tercer Informe</span>
        </div>
        <div class="header-title">🫀 Sistema Experto de Diagnóstico de Riesgo Cardiovascular</div>
    </div>
    """, unsafe_allow_html=True)

    # ── CARGA DE ARTEFACTOS ────────────────────────────────────────────────
    modelo, pipeline_proc, carga_ok, error_msg = cargar_artefactos()

    if not carga_ok:
        st.error(
            f"⚠️ **Artefactos no encontrados:** `{error_msg}`\n\n"
            "Ejecuta primero: `python main.py` para entrenar y serializar el modelo.",
            icon="🚨"
        )
        st.info(
            "**Pasos para activar el sistema:**\n"
            "1. `pip install scikit-learn pandas numpy joblib`\n"
            "2. `python main.py`\n"
            "3. `streamlit run app.py`"
        )
        return

    st.success(
        f"✅ Modelo SVM-RBF cargado correctamente desde `{MODEL_PATH}` "
        f"| Pipeline: `{PIPELINE_PATH}`",
        icon="🟢"
    )

    st.markdown("---")

    # ── SECCIÓN: INGESTA DE BIOMARCADORES ─────────────────────────────────
    st.markdown("""
    <div style="font-size:1.1rem; font-weight:600; color:#E6EDF3;
                margin-bottom:18px; display:flex; align-items:center; gap:8px;">
        🔬 &nbsp; Ingreso de Biomarcadores Clínicos del Paciente
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # ── COLUMNA 1: Datos Demográficos y Hemodinámicos ──────────────────────
    with col1:
        st.markdown("""
        <div class="section-title">
            🧑‍⚕️ &nbsp; Dimensión I & II — Demográfica y Hemodinámica
        </div>
        """, unsafe_allow_html=True)

        age = st.slider(
            "Edad del Paciente (age)",
            min_value=20, max_value=80, value=55, step=1,
            help="Rango clínico admisible: 29–77 años (Cleveland dataset)"
        )

        sex = st.selectbox(
            "Sexo Biológico (sex)",
            options=[0, 1],
            format_func=lambda x: "0 — Femenino" if x == 0 else "1 — Masculino",
            index=1,
            help="0 = Femenino | 1 = Masculino"
        )

        trestbps = st.slider(
            "Presión Arterial Sistólica en Reposo — trestbps (mmHg)",
            min_value=80, max_value=210, value=130, step=1,
            help="Rango normal: 90–140 mmHg. Hipertensión: >140 mmHg."
        )

        thalach = st.slider(
            "Frecuencia Cardíaca Máxima Alcanzada — thalach (lpm)",
            min_value=60, max_value=210, value=150, step=1,
            help="FCM teórica = 220 − edad. Valor reducido sugiere limitación funcional."
        )

        exang = st.selectbox(
            "Angina Inducida por Ejercicio (exang)",
            options=[0, 1],
            format_func=lambda x: "0 — No presenta angina" if x == 0 else "1 — Presenta angina",
            index=0,
            help="Angina de esfuerzo: síntoma de isquemia coronaria inducible."
        )

    # ── COLUMNA 2: Perfil Bioquímico y ECG ────────────────────────────────
    with col2:
        st.markdown("""
        <div class="section-title">
            🧪 &nbsp; Dimensión III & IV — Bioquímica y Electrocardiografía
        </div>
        """, unsafe_allow_html=True)

        chol = st.slider(
            "Colesterol Sérico Total — chol (mg/dL)",
            min_value=100, max_value=600, value=240, step=1,
            help="Normal: <200 mg/dL. Limítrofe: 200–239. Alto: ≥240 mg/dL."
        )

        fbs = st.selectbox(
            "Glucemia en Ayunas > 120 mg/dL (fbs)",
            options=[0, 1],
            format_func=lambda x: "0 — Glucemia normal (≤120 mg/dL)" if x == 0
                                  else "1 — Hiperglucemia (>120 mg/dL)",
            index=0,
            help="Marcador de diabetes mellitus tipo 2 como factor de riesgo CV."
        )

        restecg = st.selectbox(
            "Resultado del ECG en Reposo (restecg)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 — Normal",
                1: "1 — Anomalía onda ST-T (inversión T / depresión ST)",
                2: "2 — Hipertrofia ventricular izquierda (criterio Estes)"
            }[x],
            index=0,
            help="Hallazgos electrocardiográficos en reposo."
        )

        oldpeak = st.slider(
            "Depresión del Segmento ST post-esfuerzo — oldpeak (mm)",
            min_value=0.0, max_value=7.0, value=1.5, step=0.1,
            help="Depresión ST >1 mm: indicador de isquemia inducible significativa."
        )

        slope = st.selectbox(
            "Pendiente del Segmento ST en el Pico del Esfuerzo (slope)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 — Pendiente descendente (peor pronóstico)",
                1: "1 — Pendiente plana (pronóstico intermedio)",
                2: "2 — Pendiente ascendente (mejor pronóstico)"
            }[x],
            index=1,
            help="La pendiente descendente es el hallazgo de peor pronóstico isquémico."
        )

    # ── COLUMNA 3: Variables Imagenológicas / Perfusión ───────────────────
    with col3:
        st.markdown("""
        <div class="section-title">
            🫀 &nbsp; Dimensión IV — Imagenología y Perfusión Coronaria
        </div>
        """, unsafe_allow_html=True)

        cp = st.selectbox(
            "Tipo de Dolor Torácico — cp (Chest Pain Type)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Asintomático (mayor riesgo paradójico)",
                1: "1 — Angina atípica",
                2: "2 — Dolor no anginoso",
                3: "3 — Angina típica (dolor clásico de esfuerzo)"
            }[x],
            index=3,
            help="La angina típica (cp=3) es el síntoma cardinal de isquemia. SHAP rank #1."
        )

        ca = st.selectbox(
            "Nº de Vasos Principales Coloreados (Fluoroscopía) — ca",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: f"{x} vaso{'s' if x != 1 else ''} coloreado{'s' if x != 1 else ''}",
            index=0,
            help="Correlación directa con extensión de la enfermedad coronaria. SHAP rank #3."
        )

        thal = st.selectbox(
            "Resultado de la Prueba de Talio (Gammagrafía Miocárdica) — thal",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Sin datos de talio",
                1: "1 — Normal (perfusión homogénea)",
                2: "2 — Defecto fijo (infarto establecido)",
                3: "3 — Defecto reversible (isquemia inducible) ★"
            }[x],
            index=3,
            help="Defecto reversible: isquemia inducible. Indicador de mayor riesgo. SHAP rank #2."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Panel informativo SHAP
        st.markdown("""
        <div style="background:#1C2128; border:1px solid #30363D; border-radius:10px;
                    padding:14px 16px; margin-top:8px;">
            <div style="font-size:0.78rem; color:#E63946; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">
                ⚡ Jerarquía SHAP — Top Predictores
            </div>
            <div style="font-size:0.82rem; color:#8B949E; line-height:1.7;">
                🥇 <strong style="color:#C9D1D9;">cp</strong> — Tipo de dolor torácico<br>
                🥈 <strong style="color:#C9D1D9;">thal</strong> — Perfusión de talio<br>
                🥉 <strong style="color:#C9D1D9;">ca</strong> — Nº vasos coronarios<br>
                4️⃣ <strong style="color:#C9D1D9;">oldpeak</strong> — Depresión ST
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── BOTÓN DE PREDICCIÓN ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular = st.button(
            "🔍  CALCULAR RIESGO CLÍNICO",
            key="btn_calcular",
            use_container_width=True
        )

    # ── EJECUCIÓN DE LA PREDICCIÓN ─────────────────────────────────────────
    if calcular:
        st.markdown("---")

        # Construir vector del paciente
        vector_paciente = {
            "age":      age,
            "trestbps": trestbps,
            "chol":     chol,
            "thalach":  thalach,
            "oldpeak":  oldpeak,
            "cp":       cp,
            "thal":     thal,
            "sex":      sex,
            "fbs":      fbs,
            "restecg":  restecg,
            "exang":    exang,
            "slope":    slope,
            "ca":       ca,
        }

        # Pre-procesamiento e inferencia
        with st.spinner("⚙️ Procesando biomarcadores y ejecutando modelo SVM-RBF..."):
            t0 = time.perf_counter()
            X_proc = preprocesar_vector(vector_paciente, pipeline_proc)
            prediccion   = modelo.predict(X_proc)[0]
            probabilidad = modelo.predict_proba(X_proc)[0]
            t1 = time.perf_counter()
            latencia_ms = (t1 - t0) * 1000

        # ── RESULTADO PRINCIPAL ────────────────────────────────────────────
        if prediccion == 1:
            # ── ALTO RIESGO ────────────────────────────────────────────────
            st.error(
                "🚨 **ALTO RIESGO CARDIOVASCULAR — Requiere atención especializada urgente**",
                icon="🚨"
            )
            st.markdown("""
            <div class="resultado-alto">
                <div class="resultado-icon">⚠️</div>
                <div class="resultado-titulo-alto">ALTO RIESGO CARDIOVASCULAR</div>
                <div class="resultado-desc">
                    El modelo SVM-RBF detectó <strong>patrones compatibles con enfermedad coronaria significativa</strong>
                    en el perfil biomédico del paciente.<br><br>
                    <strong>Acción recomendada:</strong> Derivación inmediata a cardiología para
                    evaluación especializada, ecocardiografía de estrés y/o coronariografía.<br>
                    No diferir la atención médica especializada.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── BAJO RIESGO ────────────────────────────────────────────────
            st.success(
                "✅ **BAJO RIESGO CARDIOVASCULAR — Continuar con hábitos saludables**",
                icon="✅"
            )
            st.markdown("""
            <div class="resultado-bajo">
                <div class="resultado-icon">💚</div>
                <div class="resultado-titulo-bajo">BAJO RIESGO CARDIOVASCULAR</div>
                <div class="resultado-desc">
                    El modelo SVM-RBF <strong>no detectó patrones indicativos de enfermedad coronaria
                    significativa</strong> en el perfil biomédico del paciente.<br><br>
                    <strong>Recomendación:</strong> Mantener hábitos cardiovasculares saludables:
                    actividad física regular, dieta equilibrada y control periódico de presión
                    arterial y colesterol. Revisión de seguimiento en 12 meses.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── MÉTRICAS DE PROBABILIDAD Y LATENCIA ───────────────────────────
        st.markdown("**📊 Detalle cuantitativo de la predicción:**")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            color_riesgo = "#F85149" if prediccion == 1 else "#3FB950"
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-valor" style="color:{color_riesgo};">
                    {'ALTO' if prediccion == 1 else 'BAJO'}
                </div>
                <div class="prob-label">Clasificación Clínica</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            p_ecv = probabilidad[1]
            color_p = "#F85149" if p_ecv >= 0.5 else "#3FB950"
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-valor" style="color:{color_p};">
                    {p_ecv:.1%}
                </div>
                <div class="prob-label">P(Enfermedad Coronaria)</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            p_sano = probabilidad[0]
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-valor" style="color:#3FB950;">
                    {p_sano:.1%}
                </div>
                <div class="prob-label">P(Sin Enfermedad Coronaria)</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            color_lat = "#3FB950" if latencia_ms < 200 else "#F85149"
            cumple = "✓ RNF-02" if latencia_ms < 200 else "✗ RNF-02"
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-valor" style="color:{color_lat}; font-size:1.6rem;">
                    {latencia_ms:.2f} ms
                </div>
                <div class="prob-label">Latencia Inferencia ({cumple})</div>
            </div>
            """, unsafe_allow_html=True)

        # ── DETALLE DEL VECTOR PROCESADO ───────────────────────────────────
        with st.expander("🔎 Ver vector de biomarcadores procesado"):
            df_display = pd.DataFrame([vector_paciente]).T.reset_index()
            df_display.columns = ["Biomarcador", "Valor Ingresado"]

            etiquetas_clinicas = {
                "age": "Edad (años)",
                "sex": "Sexo (0=F, 1=M)",
                "cp": "Tipo dolor torácico",
                "trestbps": "Presión arterial (mmHg)",
                "chol": "Colesterol (mg/dL)",
                "fbs": "Glucemia >120 mg/dL",
                "restecg": "ECG en reposo",
                "thalach": "FC máxima (lpm)",
                "exang": "Angina de esfuerzo",
                "oldpeak": "Depresión ST (mm)",
                "slope": "Pendiente ST",
                "ca": "Nº vasos coronarios",
                "thal": "Prueba de talio",
            }
            df_display["Nombre Clínico"] = df_display["Biomarcador"].map(etiquetas_clinicas)
            df_display = df_display[["Nombre Clínico", "Biomarcador", "Valor Ingresado"]]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # ── MARCA TEMPORAL ─────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div style="text-align:right; color:#484F58; font-size:0.78rem;
                    margin-top:12px; font-family:'JetBrains Mono', monospace;">
            Predicción generada: {ts} &nbsp;|&nbsp;
            Modelo: SVM-RBF (AUC: 0.932) &nbsp;|&nbsp;
            UPAO — IA Principios y Técnicas
        </div>
        """, unsafe_allow_html=True)

        # ── DISCLAIMER CLÍNICO ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "**⚕️ Aviso médico-legal:** Este sistema es una herramienta de apoyo diagnóstico "
            "de investigación académica. **No reemplaza el criterio clínico de un profesional "
            "médico certificado.** Toda decisión diagnóstica y terapéutica debe ser tomada "
            "exclusivamente por un médico habilitado. Dataset anonimizado — Normativa HIPAA.",
            icon="⚕️"
        )

    # ── FOOTER ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#484F58; font-size:0.82rem; padding:8px 0 16px 0;">
        <strong style="color:#6E7681;">UPAO — Sistema Experto de Diagnóstico de Riesgo Cardiovascular</strong>
        &nbsp;·&nbsp; Mauricio Jefferson Cuba Prieto
        &nbsp;·&nbsp; Dr. Teobaldo Hernán Sagástegui Chávez
        &nbsp;·&nbsp; Inteligencia Artificial: Principios y Técnicas &nbsp;·&nbsp; 2025-II<br>
        <span style="color:#3A424D;">Modelo: SVM-RBF | Dataset: Cleveland UCI Heart Disease
        | Pipeline: scikit-learn | Deploy: Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
