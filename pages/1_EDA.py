import streamlit as st
import pandas as pd
from src.eda import crear_boxplot_ingresos, crear_boxplot_loan_status_vs_income, crear_countplot_defaults_vs_loan_status, crear_countplot_educacion_vs_loan_status, crear_piechart_educacion, crear_scatterplot_credit_vs_interest, crear_violinplot_credit_range_vs_interest, crear_scatterplot_interest_vs_loan_amount

# Carga el DataFrame
@st.cache_data
def load_data():
    return pd.read_csv("data/loan_data.csv")  
# Cargar datos
df = load_data()


# Información básica del conjunto de datos
st.header("Aspectos Básicos del Conjunto de Datos")
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Número de Filas", value=df.shape[0], border=True)
    with col2:
        st.metric(label="Número de Columnas", value=df.shape[1], border=True)
    with col3:
        missing_values = df.isnull().any().sum()
        st.metric(label="Valores Perdidos", value="Sí" if missing_values > 0 else "No", border=True)


# Mostrar gráficos
st.header("Diagrama de caja para ingresos")
boxplot_income_fig= crear_boxplot_ingresos(df)
st.pyplot(boxplot_income_fig)

# Mostrar gráficos
st.header("Diagrama de caja para ingresos por resultado de solicitud")
boxplot_loan_income_fig= crear_boxplot_loan_status_vs_income(df)
st.pyplot(boxplot_loan_income_fig)

# Mostrar gráficos
st.header("Countplot de deudas pendientes por resultado de solicitud")
countplot_loan_defaults_fig= crear_countplot_defaults_vs_loan_status(df)
st.pyplot(countplot_loan_defaults_fig)

# Mostrar gráficos
st.header("Countplot de educación por resultado de solicitud")
countplot_loan_education_fig= crear_countplot_educacion_vs_loan_status(df)
st.pyplot(countplot_loan_education_fig)

# Mostrar gráficos
st.header("Piechart de educación")
piechart_education_fig= crear_piechart_educacion(df)
st.pyplot(piechart_education_fig)

# Mostrar gráficos
st.header("Diagrama de dispersión de tasas de interés por puntaje crediticio")
scatterplot_credit_fig= crear_scatterplot_credit_vs_interest(df)
st.pyplot(scatterplot_credit_fig)

# Mostrar gráficos
st.header("Diagrama de violín de tasas de interés por puntaje crediticio")
violinplot_credit_fig= crear_violinplot_credit_range_vs_interest(df)
st.pyplot(violinplot_credit_fig)

# Mostrar gráficos
st.header("Diagrama de dispersión de tasas de interés por monto del préstamo")
scatterplot_amnt_fig= crear_scatterplot_interest_vs_loan_amount(df)
st.pyplot(scatterplot_amnt_fig)













