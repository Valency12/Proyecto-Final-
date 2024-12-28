import streamlit as st
import pandas as pd
import matplot.pyplot as plt
import matplotlib.pyplot as plt 

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

#Mostrar graficos---------------------------------------------------------------------------------------------------

#Loan_status 
def analizar_loan_status(df):
    loan_status_counts = df['loan_status'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        loan_status_counts, 
        labels=['Rechazado (6)', 'Aprobado (1)'], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=['#66b3ff', '#ff9999']
    )
    ax.set_title('Distribución de Loan Status')
    st.pyplot(fig)

#Analisis univariado 
def mostrar_histogramas(df):
    columnas_a_incluir = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
    ]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
    axes = axes.flatten()  

    for i, col in enumerate(columnas_a_incluir):
        df[col].hist(ax=axes[i], bins=30, alpha=0.7, color='skyblue')
        axes[i].set_title(f'Histograma de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
    plt.tight_layout()
    st.pyplot(fig)

def mostrar_pairplot(df):
    fig = plt.figure(figsize=(15, 15))
    sns.pairplot(data=df, kind='scatter', height=2.5, aspect=1.5)
    plt.suptitle('Distribución de características y clases', y=1.02, fontsize=20)
    st.pyplot(fig)

#Analisis Multivariados

def mostrar_pairplot_con_loan_status(df):
    df['loan_status'] = df['loan_status'].astype('category')
    fig = plt.figure(figsize=(15, 15))
    sns.pairplot(df[['person_age', 'person_emp_exp', 'person_income', 'loan_amnt', 
                     'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                     'credit_score', 'loan_status']], 
                 hue='loan_status', plot_kws={'alpha': 0.10})
    plt.suptitle('Gráfico de Pares para Variables Numéricas', y=1.02)

    st.pyplot(fig)

#Matriz de correlacion 
def crear_heatmap_correlacion(df, title='Matriz de Correlación', cmap='coolwarm', annot=True):
    correlation_matrix = df[['person_age', 'person_emp_exp', 'person_income', 'loan_amnt', 
                             'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                             'credit_score']].corr()
    fig, ax = plt.subplots(figsize=(10, 6)) 
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, ax=ax)
    plt.title(title)
    st.pyplot(fig)


# Distribucion de los montos de prestamos segun la respuesta a la solicitud
def crear_boxplot_prestamos(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y='loan_amnt', data=df)
    plt.title('Distribución de los montos de préstamos según la respuesta a la solicitud')
    plt.xlabel('Aprobación de préstamo')
    plt.ylabel('Monto del préstamo')
    st.pyplot(plt.gcf())

#Boxplot para diferentes puntajes crediticios segun estatus de prestamos
def crear_boxplot_puntaje_credito(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y='credit_score', data=df, palette="coolwarm")  
    plt.title('Distribución de los puntajes crediticios según el estatus de préstamo')
    plt.xlabel('Aprobación del préstamo')
    plt.ylabel('Puntaje crediticio de la persona')
    plt.xticks(rotation=45)  
    st.pyplot(plt.gcf())

#Grafico de relacion aprobacion de prestamos vs puntaje crediticio
def crear_scatterplot_credito(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='loan_status', y='credit_score', data=df, hue="loan_status", alpha=0.1, palette="deep")
    plt.title('Relación entre aprobación de préstamos y puntaje crediticio')
    plt.xlabel('Estado del préstamo')
    plt.ylabel('Puntaje crediticio de la persona')
    st.pyplot(plt.gcf())

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








