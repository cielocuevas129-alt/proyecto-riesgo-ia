import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Risk Score AI",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# 🎨 ESTILO PROFESIONAL (UI EMPRESA)
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
    color: #111827;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Botones */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    transform: scale(1.02);
}

/* Métricas */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODELO
# =====================================================

modelo = joblib.load("modelo_riesgo.pkl")

# =====================================================
# HEADER
# =====================================================

st.title("🏦 Sistema de Evaluación de Riesgo Crediticio")
st.caption("Modelo de Machine Learning para análisis de clientes")

st.divider()

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "🤖 Predicción",
    "📊 Dashboard",
    "📂 Datos CSV"
])

# =====================================================
# TAB 1 - PREDICCIÓN
# =====================================================

with tab1:

    st.header("Evaluación del Cliente")

    edad = st.number_input("Edad", 18, 100)
    prestamos = st.number_input("Préstamos retrasados", 0, 20)
    retraso60 = st.number_input("Retrasos 60 días", 0, 20)
    retraso3 = st.number_input("Retrasos últimos 3 años", 0, 20)
    dependientes = st.number_input("Dependientes", 0, 15)

    if st.button("Analizar cliente"):

        datos = np.array([[edad, prestamos, retraso60, retraso3, dependientes]])

        pred = modelo.predict(datos)
        prob = modelo.predict_proba(datos)[0][1] * 100

        st.metric("📊 Riesgo estimado", f"{prob:.2f}%")

        if prob < 30:
            st.success("🟢 Cliente de bajo riesgo (aprobable)")
        elif prob < 70:
            st.warning("🟡 Riesgo medio (revisión manual)")
        else:
            st.error("🔴 Alto riesgo (rechazado automático)")

# =====================================================
# TAB 2 - DASHBOARD
# =====================================================

with tab2:

    st.header("📊 Análisis general del portafolio")

    import random

    datos_grafico = pd.DataFrame({
        "Categoría": ["Alto Riesgo", "Bajo Riesgo", "En Revisión"],
        "Cantidad": [
            random.randint(20, 60),
            random.randint(40, 90),
            random.randint(10, 30)
        ]
    })

    fig = px.bar(
        datos_grafico,
        x="Categoría",
        y="Cantidad",
        color="Categoría",
        text="Cantidad"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 - CSV
# =====================================================

with tab3:

    st.header("📂 Análisis de datos externos")

    archivo = st.file_uploader("Sube archivo CSV", type=["csv"])

    if archivo is not None:

        df = pd.read_csv(archivo)

        st.dataframe(df)

        num = df.select_dtypes(include="number")

        if not num.empty:
            st.bar_chart(num)
        else:
            st.warning("No hay columnas numéricas")

# =====================================================
# FOOTER
# =====================================================

st.divider()
st.caption("🚀 Proyecto de Machine Learning")