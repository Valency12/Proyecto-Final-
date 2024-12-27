import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

df = pd.read_csv('loan_data.csv')

#analisis exploratorio de datos

#primeras filas
df.head()
print(df.head())

#informacion general
print(df.info())

#estadisticas descriptivas
print(df.describe())

#contar las categorias de loan status
loan_status_counts = df['loan_status'].value_counts()
print(loan_status_counts)

#crear un grafico de pastel para loan_status
def analizar_loan_status(df):
    loan_status_counts = df['loan_status'].value_counts()
    print(loan_status_counts)

    plt.figure(figsize=(8, 8))
    plt.pie(loan_status_counts, labels=['Rechazado (6)', 'Aprobado (1)'], autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
    plt.title('Distribución de Loan Status')
    plt.show()

    return plt.gcf()

#graficos del analisis unvariado
def analisis_univariados(df):
    columnas_a_incluir = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
    fig.suptitle('Histogramas de Variables Numéricas')

    # Iterar sobre las columnas y crear los histogramas
    for i, column in enumerate(columnas_a_incluir):
        row = i // 2
        col = i % 2
        axes[row, col].hist(df[column], bins=20)
        axes[row, col].set_title(column)

    # Ajustar el espaciado entre subplots
    plt.tight_layout()
    df[columnas_a_incluir].hist(figsize=(15, 15))

    return fig 
