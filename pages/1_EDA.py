import streamlit as st
import pandas as pd
from src.eda import *
import matplot.pyplot as plt
import matplotlib.pyplot as plt 

#cargar el DataFrame
@st.cache_data
def load_data():
    return pd.read_csv("data/loan_data.csv")

#Configuracion de la pagina
st.title("Analisis Exploratorio de Datos (EDA)")

#cargar datos
df = load_data()

#Informacion basica del conjunto de datos


#Mostrar graficos

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
    





