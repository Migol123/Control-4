# ============================================================
# APP STREAMLIT – PREDICTOR DE RIESGO CREDITICIO (SAFE UI)
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Predictor de Riesgo Crediticio",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
LOGS_DIR = APP_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ESTRUCTURA FIJA (según tu entrega/)
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH  = MODELS_DIR / "model_metadata.json"

PRED_LOG_PATH = LOGS_DIR / "predictions_log.csv"

# ============================================================
# PLACEHOLDER + VALORES CATEGÓRICOS
# ============================================================
PLACEHOLDER = "— Seleccione —"

CATEGORICAL_OPTIONS = {
    "person_home_ownership": [PLACEHOLDER, "RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        PLACEHOLDER,
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
    "loan_grade": [PLACEHOLDER, "A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": [PLACEHOLDER, "N", "Y"],
}

# ============================================================
# UTILS
# ============================================================
def log_prediction(
    path: Path,
    model_filename: str,
    threshold: float,
    proba: float,
    pred_class: int,
    X_row: pd.DataFrame
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = {
        "timestamp": ts,
        "model_file": model_filename,
        "threshold": float(threshold),
        "proba_default": float(proba),
        "pred_class": int(pred_class),
    }
    out = {**base, **X_row.iloc[0].to_dict()}
    df_out = pd.DataFrame([out])

    write_header = not path.exists()
    df_out.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8"
    )

# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def load_pipeline(path: Path):
    return joblib.load(path)

@st.cache_data
def load_metadata(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# VALIDACIONES DE ARCHIVOS + CARGA
# ============================================================
if not MODELS_DIR.exists():
    st.error("No existe la carpeta `models/` dentro de `app/`.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("No se encontró el archivo del modelo: `app/models/model.joblib`.")
    st.stop()

pipeline = load_pipeline(MODEL_PATH)
metadata = load_metadata(META_PATH) if META_PATH.exists() else {}

try:
    pipeline.set_output(transform="pandas")
except Exception:
    pass

# ============================================================
# SIDEBAR – SOLO MODELO + METADATA RESUMIDA + LOGGING
# ============================================================
with st.sidebar:
    st.markdown("### Modelo en uso")

    st.write("**Archivo:**", f"`{MODEL_PATH.name}`")

    # Resumen de metadatos (solo lo importante)
    st.divider()
    st.markdown("### Información del modelo")

    if metadata:
        st.write("**Nombre:**", metadata.get("model_name", "N/A"))
        st.write("**Versión:**", metadata.get("version", "N/A"))
        st.write("**Algoritmo:**", metadata.get("algorithm", "N/A"))
        st.write("**Contexto:**", metadata.get("business_context", "N/A"))

        auc_test = metadata.get("metrics", {}).get("auc_test", None)
        if auc_test is not None:
            st.metric("AUC (test)", f"{auc_test:.3f}")
        else:
            st.write("**AUC (test):** N/A")

        thr = metadata.get("threshold", None)
        if thr is not None:
            st.metric("Umbral operativo", f"{float(thr):.2f}")
        else:
            st.write("**Umbral operativo:** N/A")

    else:
        st.info("No se encontró `models/model_metadata.json` (opcional).")

    st.divider()
    st.markdown("### Logging")
    enable_logging = st.toggle("Guardar predicciones", value=True)

    if PRED_LOG_PATH.exists():
        with open(PRED_LOG_PATH, "rb") as f:
            st.download_button(
                "Descargar predictions_log.csv",
                data=f,
                file_name="predictions_log.csv",
                mime="text/csv"
            )
    else:
        st.caption("Aún no hay log generado.")

# ============================================================
# HEADER
# ============================================================
st.title("Predictor de Riesgo Crediticio")
st.markdown(
    "Ingrese los datos del cliente para estimar la **probabilidad de default** "
    "y visualizar una explicación local mediante **SHAP**."
)

# ============================================================
# EXTRAER PREPROCESS + MODELO
# ============================================================
try:
    preprocess = pipeline.steps[0][1]
    model = pipeline.steps[-1][1]
except Exception as e:
    st.error("El objeto cargado no parece ser un Pipeline válido.")
    st.exception(e)
    st.stop()

# Columnas esperadas
if hasattr(pipeline, "feature_names_in_"):
    expected_cols = list(pipeline.feature_names_in_)
elif hasattr(preprocess, "feature_names_in_"):
    expected_cols = list(preprocess.feature_names_in_)
else:
    expected_cols = metadata.get("features", None)

if expected_cols is None:
    st.error("No se pudieron determinar las columnas esperadas por el modelo.")
    st.stop()

# Feature names post-transform (para SHAP)
try:
    post_feature_names = list(preprocess.get_feature_names_out())
except Exception:
    post_feature_names = None

# ============================================================
# INPUTS
# ============================================================
st.subheader("Datos del cliente")

c1, c2 = st.columns(2)
inputs = {}

for i, col in enumerate(expected_cols):
    container = c1 if i % 2 == 0 else c2

    if col in CATEGORICAL_OPTIONS:
        inputs[col] = container.selectbox(
            col,
            CATEGORICAL_OPTIONS[col],
            index=0
        )
    elif any(k in col.lower() for k in [
        "age", "income", "emp", "amnt",
        "rate", "percent", "length", "hist"
    ]):
        inputs[col] = container.number_input(
            col,
            value=0.0,
            step=1.0
        )
    else:
        inputs[col] = container.text_input(col, "")

X_new = pd.DataFrame([inputs], columns=expected_cols)

# Casting explícito
cat_cols = [c for c in X_new.columns if X_new[c].dtype == object]
num_cols = [c for c in X_new.columns if c not in cat_cols]

for c in cat_cols:
    X_new[c] = X_new[c].astype(str)

for c in num_cols:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

# ============================================================
# VALIDACIÓN DE PLACEHOLDER
# ============================================================
invalid_cats = [
    c for c in CATEGORICAL_OPTIONS
    if c in X_new.columns and X_new[c].iloc[0] == PLACEHOLDER
]

if invalid_cats:
    st.warning("Debe seleccionar un valor válido para: " + ", ".join(invalid_cats))

# ============================================================
# PREDICCIÓN
# ============================================================
st.divider()
st.subheader("Predicción")

threshold = float(metadata.get("threshold", 0.5))

if st.button("Calcular Riesgo"):
    if invalid_cats:
        st.error("Complete todas las variables categóricas antes de predecir.")
        st.stop()

    try:
        proba = float(pipeline.predict_proba(X_new)[0, 1])
    except Exception as e:
        st.error("No se pudo predecir. Revisa columnas/tipos vs entrenamiento.")
        st.exception(e)
        st.stop()

    pred_class = int(proba >= threshold)

    if enable_logging:
        try:
            log_prediction(
                path=PRED_LOG_PATH,
                model_filename=MODEL_PATH.name,
                threshold=threshold,
                proba=proba,
                pred_class=pred_class,
                X_row=X_new
            )
        except Exception as e:
            st.warning("No se pudo escribir el log de predicciones.")
            st.exception(e)

    left, right = st.columns([1, 2])

    with left:
        st.metric("Probabilidad de Default", f"{proba:.1%}")
        st.write(f"Umbral: **{threshold:.2f}**")
        st.write(f"Clase predicha: **{pred_class}** (1=Default, 0=No default)")

        if proba >= threshold:
            st.error("ALTO RIESGO")
        else:
            st.success("BAJO RIESGO")

    # ========================================================
    # SHAP LOCAL
    # ========================================================
    with right:
        st.markdown("### Explicación local (SHAP)")

        try:
            X_tr = preprocess.transform(X_new)
            if hasattr(X_tr, "toarray"):
                X_tr = X_tr.toarray()

            explainer = shap.TreeExplainer(model)
            shap_out = explainer(X_tr)

            shap_exp = shap_out[1] if isinstance(shap_out, list) else shap_out

            ax = shap.plots.waterfall(
                shap_exp[0],
                max_display=15,
                show=False
            )
            st.pyplot(ax.figure, clear_figure=True)

            vals = shap_exp[0].values
            names = (
                np.array(post_feature_names)
                if post_feature_names is not None
                else np.array([f"f{i}" for i in range(len(vals))])
            )

            top_pos = names[np.argsort(vals)[-5:][::-1]]
            top_neg = names[np.argsort(vals)[:5]]

            st.markdown(
                f"""
**Interpretación:**
- Factores que más **incrementan** el riesgo: **{", ".join(top_pos)}**
- Factores que más **reducen** el riesgo: **{", ".join(top_neg)}**
"""
            )

        except Exception as e:
            st.warning(
                "No se pudo generar SHAP para este input. "
                "Suele ocurrir si hay categorías nuevas/no vistas o si el modelo no es tipo árbol."
            )
            st.exception(e)
