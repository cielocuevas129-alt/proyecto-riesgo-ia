import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Proyecto IA",
    page_icon="🤖",
    layout="wide"
)
modelo = joblib.load("modelo_riesgo.pkl")

# SIDEBAR
st.sidebar.title("🚀 Menú")
st.sidebar.info("""
Proyecto desarrollado con:
- Python
- Machine Learning
- Streamlit
""")

# TITULO
st.title("🤖 Proyecto de Machine Learning")

st.markdown("""
### Plataforma de análisis y predicción de datos
Proyecto profesional desarrollado con Inteligencia Artificial.
""")

# COLUMNAS
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Tecnologías")
    
    st.success("Python")
    st.success("Pandas")
    st.success("Scikit-learn")
    st.success("Streamlit")

with col2:
    st.subheader("📈 Resultados")
    
    st.metric("Precisión del modelo", "95%")
    st.metric("Datos analizados", "10K")
    st.metric("Tiempo de entrenamiento", "2 min")

# SEPARADOR
st.divider()

# DESCRIPCION
st.header("🧠 Sobre el proyecto")

st.write("""
Este proyecto utiliza Machine Learning para analizar datos
y generar predicciones inteligentes utilizando Python.
""")

# INPUTS

edad = st.number_input("Edad", step=1)

prestamos = st.number_input("Número de préstamos retrasados", step=1)

retraso60 = st.number_input("Número de retrasos de 60 días", step=1)

retraso3 = st.number_input("Retrasos últimos 3 años", step=1)

dependientes = st.number_input("Número de dependientes", step=1)

# BOTON

if st.button("Analizar riesgo"):

    datos = np.array([[edad, prestamos, retraso60, retraso3, dependientes]])

    prediccion = modelo.predict(datos)

    if prediccion[0] == 1:
        st.error("⚠️ Cliente con alto riesgo")

    else:
        st.success("✅ Cliente con riesgo bajo")
        st.balloons()

# FOOTER

st.divider()

st.caption("Desarrollado con ❤️ usando Streamlit")

# GRAFICO

st.header("📊 Análisis de Riesgo")

grafico = pd.DataFrame({
    "Riesgo": ["Alto", "Bajo"],
    "Cantidad": [40, 60]
})

st.bar_chart(grafico.set_index("Riesgo"))
