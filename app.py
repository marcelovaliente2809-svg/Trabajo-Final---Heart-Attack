import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

@st.cache_resource
def load_models():
    base   = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base, "best_model_baggingclassifier_weighted_recall.joblib"))
    ohe    = joblib.load(os.path.join(base, "one_hot_encoder.joblib"))
    scaler = joblib.load(os.path.join(base, "standard_scaler.joblib"))
    le     = LabelEncoder()
    le.fit(["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    return model, ohe, scaler, le

model, ohe, scaler, le = load_models()

st.set_page_config(page_title="Riesgo de Ataque Cardiaco", page_icon="🫀", layout="centered")
st.title("🫀 Predictor de Riesgo de Ataque Cardiaco")
st.markdown("Completa los datos del paciente para estimar su riesgo.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age       = st.number_input("Edad", min_value=1, max_value=120, value=50)
    diastolic = st.number_input("Presión Diastólica (mmHg)", min_value=30, max_value=200, value=85)
    sedentary = st.number_input("Horas sedentarias por día", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
with col2:
    income   = st.number_input("Ingreso anual (USD)", min_value=0, max_value=1_000_000, value=50000, step=1000)
    activity = st.number_input("Días de actividad física por semana", min_value=0, max_value=7, value=3)

continent = st.selectbox("Continente", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
st.divider()

if st.button("🔍 Predecir Riesgo", use_container_width=True):
    num_df = pd.DataFrame([{
        "Age": age,
        "Diastolic": diastolic,
        "Sedentary Hours Per Day": sedentary,
        "Income": income,
        "Physical Activity Days Per Week": activity
    }])
    scaled     = scaler.transform(num_df)
    cont_label = np.array([[le.transform([continent])[0]]])
    cont_df    = pd.DataFrame([[continent]], columns=["Continent"])
    encoded    = ohe.transform(cont_df)
    X          = np.hstack([scaled, cont_label, encoded])

    prediction = model.predict(X)[0]
    proba      = model.predict_proba(X)[0][1]

    st.markdown("### Resultado")
    if prediction == 1:
        st.error("⚠️ **ALTO RIESGO** de ataque cardiaco")
    else:
        st.success("✅ **BAJO RIESGO** de ataque cardiaco")

    st.metric(label="Probabilidad de riesgo", value=f"{proba:.1%}")
    st.progress(float(proba))
    st.caption("⚕️ Este resultado es orientativo. Consulta siempre a un médico.")

    #algo de información adicional
    #creando nuevo dataframe con los datos de entrada para mostrarlo
    if prediction == 1:
        st.markdown("""
        *holi ugu**""")