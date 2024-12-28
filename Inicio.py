import streamlit as st
import os

# Configuración inicial de la página
st.set_page_config(layout="wide")

# Introducción del proyecto
st.markdown("""
## Bienvenido
Este proyecto incluye las siguientes páginas:
""")

# Sección de páginas con título e imágenes
col1, col2 = st.columns([2, 2])

with col1:
    st.image("utils/eda.png", width=210)  # Imagen para EDA

with col2:
    st.subheader("EDA: Análisis exploratorio de datos")
    st.markdown("Analiza el dataset sintético para identificar patrones y tendencias.")

col3, col4 = st.columns([2, 2])

with col3:
    st.image("utils/hipotesis.jpg", width=250)  # Imagen para hipótesis

with col4:
    st.subheader("Hipótesis: Visualización de hipótesis propuestas")
    st.markdown("Explora hipótesis relacionadas al riesgo financiero y su validez.")

col5, col6 = st.columns([2, 2])

with col5:
    ruta_imagen = "utils/modelo.png"  # Imagen para modelo
    if os.path.exists(ruta_imagen):
        st.image(ruta_imagen, width=250)
    else:
        st.error(f"No se encontró el archivo en la ruta: {ruta_imagen}")

with col6:
    st.subheader("Modelo: Predicciones con un modelo de árbol de decisiones")
    st.markdown("Realiza predicciones sobre la aprobación de préstamos usando un modelo supervisado.")
