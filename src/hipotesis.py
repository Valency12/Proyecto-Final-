import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis de Préstamos", layout="wide")

# Título de la app
st.title("Análisis de Préstamos - Hipótesis")

# Cargar el dataset
@st.cache_data
def cargar_datos():
    return pd.read_csv("loan_data.csv")

df = cargar_datos()

# Mostrar primeras filas del dataset
st.write("Vista preliminar del dataset:")
st.dataframe(df.head())

# Hipótesis 1
st.header("Hipótesis 1")
st.write("""
1. El 90% de préstamos se desaprueban si la relación préstamo entre ingresos supera el 0.30.

Los análisis indican que existe una correlación positiva entre la proporción préstamo/ingresos y la probabilidad de rechazo de un préstamo.
Esto podría indicar parcialmente la veracidad de la hipótesis. Sin embargo, se hace notar que no cumple el umbral de la hipótesis, por lo cual se descarta.
""")

df['relacion_prestamo_ingreso'] = df['loan_amnt'] / df['person_income']
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="relacion_prestamo_ingreso",
    y="loan_status",
    hue="loan_status",
    palette="coolwarm",
    alpha=0.7
)
plt.axvline(0.30, color="red", linestyle="--", label="Límite 0.30")
plt.title("Relación préstamo/ingreso vs Estado del Préstamo")
plt.legend()
st.pyplot(plt)

if df is not None:
    # Filtrar datos por loan_percent_income > 0.30
    df_mayor_relacion = df[df["loan_percent_income"] > 0.30]

    # Calcular proporciones de aprobación/rechazo
    resultados = df_mayor_relacion["loan_status"].value_counts(normalize=True) * 100

    # Mostrar título y descripción
    st.markdown("### Proporción de Resultados para `loan_percent_income > 0.30`")
    st.markdown(
        """
        Este análisis muestra la proporción de solicitudes aprobadas y desaprobadas 
        para los casos en los que el porcentaje del ingreso destinado al préstamo es mayor al 30%.
        """
    )

    # Crear gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 6))
    resultados.plot(kind="bar", color=["red", "green"], alpha=0.7, ax=ax)
    ax.axhline(90, color="blue", linestyle="--", linewidth=1, label="Umbral 90%")
    ax.set_title("Proporción de Resultados (Proporción mayor a 0.30)")
    ax.set_ylabel("Porcentaje")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Desaprobado", "Aprobado"], rotation=0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)
else:
    st.warning("Por favor, carga un archivo de datos válido para continuar con el análisis.")
    

# Hipótesis 2
st.header("Hipótesis 2")
st.write("""
2. Más del 75% de préstamos se desaprueban para personas con un puntaje crediticio menor a 580.

Es correcta, ya que el análisis muestra que un 78% de los préstamos son desaprobados para personas con un puntaje crediticio inferior a 580. 
Esto respalda la afirmación de que las personas con puntajes bajos tienen una probabilidad significativamente alta de que sus préstamos sean rechazados.
""")

df_menor_580 = df[df["credit_score"] < 580]
resultados = df_menor_580["loan_status"].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 6))
resultados.plot(kind="bar", color=["red", "green"], alpha=0.7)
plt.title("Proporción de Resultados (Puntaje < 580)")
plt.ylabel("Porcentaje")
plt.xticks([0, 1], ["Desaprobado", "Aprobado"], rotation=0)
plt.grid(axis="y")
st.pyplot(plt)

# Hipótesis 3
st.header("Hipótesis 3")
st.write("""
3. Más del 50% solicitan un préstamo de más de $10,000 por motivos de educación o médicos.

El análisis muestra que una proporción significativa de los préstamos superiores a $10,000 se solicita con fines de educación o médicos. 
Sin embargo, no alcanzan a superar el umbral del 50%, sugiriendo que otros factores también motivan estas solicitudes.
""")

df_10k = df[df["loan_amnt"] > 10000]
motivos = df_10k["loan_intent"].value_counts()

plt.figure(figsize=(8, 6))
motivos.plot(kind="bar", color="skyblue", alpha=0.7)
plt.title("Motivos de préstamos > $10,000")
plt.ylabel("Frecuencia")
plt.xlabel("Motivo")
plt.xticks(rotation=45)
plt.grid(axis="y")
st.pyplot(plt)

# Hipótesis 4
st.header("Hipótesis 4")
st.write("""
4. Más del 85% de préstamos destinados a propósitos personales se rechazan.

Aunque un porcentaje significativo de los préstamos personales es rechazado (80%), la hipótesis de que más del 85% de estos préstamos serían rechazados no se confirma.
""")

prestamos_personales = df[df["loan_intent"] == "PERSONAL"]
if not prestamos_personales.empty:
    resultados_personales = prestamos_personales["loan_status"].value_counts(normalize=True) * 100

    plt.figure(figsize=(8, 6))
    ax = resultados_personales.plot(kind="bar", color=["red", "green"], alpha=0.7)
    plt.axhline(85, color="blue", linestyle="--", linewidth=1, label="Umbral 85%")
    plt.title("Proporción de resultados (Préstamos personales)")
    plt.ylabel("Porcentaje")
    plt.xticks([0, 1], ["Desaprobado", "Aprobado"], rotation=0)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(plt)

# Hipótesis 5
st.header("Hipótesis 5")
st.write("""
5. Las personas con puntajes superiores a 800 y más de dos años de historial crediticio tienen tasas menores al 5%.

El análisis muestra que las tasas de interés para este grupo de personas se encuentran por encima del 10%, lo que sugiere que, 
a pesar de tener un historial crediticio sólido y un puntaje alto, estas personas no necesariamente obtienen tasas de interés bajas.
""")

df_filtrado = df[(df["cb_person_cred_hist_length"] >= 2) & (df["credit_score"] > 800)]
if not df_filtrado.empty:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_filtrado, x="credit_score", y="loan_int_rate", hue="loan_int_rate", palette="viridis", alpha=0.7)
    plt.axhline(5, color="red", linestyle="--", label="Límite 5%")
    plt.title("Tasas de interés vs Puntaje crediticio")
    plt.xlabel("Puntaje crediticio")
    plt.ylabel("Tasa de interés (%)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
else:
    st.write("No se encontraron datos para esta hipótesis.")