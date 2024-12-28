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

#graficos del analisis multivariado
def crear_pairplot(df, hue_column='loan_status', alpha=0.10, title='Gráfico de Pares para Variables Numéricas'):
      numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
      df[hue_column] = df[hue_column].astype('category')

      fig = sns.pairplot(df[numeric_cols + [hue_column]], hue=hue_column, plot_kws={'alpha': alpha})
      fig.fig.suptitle(title, y=1.02)

      return fig

#Matriz de correlacion
def crear_heatmap_correlacion(df, title='Matriz de Correlación', cmap='coolwarm', annot=True):

  # Calcular la matriz de correlación
  correlation_matrix = df[['person_age', 'person_emp_exp', 'person_income', 'loan_amnt', 
                            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                            'credit_score']].corr()

  # Crear el heatmap
  fig, ax = plt.subplots(figsize=(10, 6)) 
  sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, ax=ax)
  plt.title(title)

  return fig

#Distribucion de los montos de prestamos segun la respuesta a la solicitud
def crear_boxplot_prestamos(df):

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y='loan_amnt', data=df)
    plt.title('Distribución de los montos de préstamos según la respuesta a la solicitud')
    plt.xlabel('Aprobación de préstamo')
    plt.ylabel('Monto del préstamo')

    return plt.gcf()  

#Boxplot para diferentes puntajes crediticios segun estatus de prestamos
def crear_boxplot_puntaje_credito(df):

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y='credit_score', data=df, palette="coolwarm")  
    plt.title('Distribución de los puntajes crediticios según el estatus de préstamo')
    plt.xlabel('Aprobación del préstamo')
    plt.ylabel('Puntaje crediticio de la persona')
    plt.xticks(rotation=45)  

    return plt.gcf()

#Grafico de relacion aprobacion de prestamos vs puntaje crediticio
def crear_scatterplot_credito(df):

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='loan_status', y='credit_score', data=df, hue="loan_status", alpha=0.1, palette="deep")
    plt.title('Relación entre aprobación de préstamos y puntaje crediticio')
    plt.xlabel('Estado del préstamo')
    plt.ylabel('Puntaje crediticio de la persona')

    return plt.gcf()

#Boxplot para comparar estatus de prestamos para diferentes salarios
def crear_boxplot_ingresos(df):

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y='person_income', data=df, palette="coolwarm")
    plt.title('Distribución de los ingresos según el estatus de préstamo')
    plt.xlabel('Aprobación del préstamo')
    plt.ylabel('Ingreso de la persona')
    plt.xticks(rotation=45)  # Rota las etiquetas del eje x para mejor legibilidad

    return plt.gcf()

#Definiendo un rango menor en el eje y
def crear_boxplot_loan_status_vs_income(df):

    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y='person_income', data=df)
    plt.title('Boxplot para comparar estatus de préstamos para diferentes salarios')
    plt.xlabel('Aprobación de préstamo')
    plt.ylabel('Salario de la persona')
    plt.ylim(0, 200000)
    return fig

#Comparacion de Loan Status segun Defaults en prestamos previos
def crear_countplot_defaults_vs_loan_status(df):
   
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='previous_loan_defaults_on_file', hue='loan_status')
    plt.title('Comparación de Loan Status según Defaults en Préstamos Previos')
    plt.xlabel('Defaults en Préstamos Previos (Yes/No)')
    plt.ylabel('Cantidad')
    plt.legend(title='Loan Status', labels=['Rechazado (0)', 'Aprobado (1)'])
    return fig

#Comparacion de Loan Status segun nivel de educacion
def crear_countplot_educacion_vs_loan_status(df):

    educacion_orden = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]

    fig = plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='person_education', hue='loan_status', order=educacion_orden)
    plt.title('Comparación de Loan Status según nivel de educación')
    plt.xlabel('Nivel de educación')
    plt.ylabel('Cantidad')
    plt.legend(title='Loan Status', labels=['Rechazado (0)', 'Aprobado (1)'])
    return fig

#Distribucion para loan_status
def crear_piechart_educacion(df):
  
    education_level_counts = df['person_education'].value_counts()
    educacion_orden = ["Bachelor", "Associate", "High School", "Master", "Doctorate"]

    fig = plt.figure(figsize=(8, 8))
    plt.pie(
        education_level_counts[educacion_orden],
        labels=[f'{level} ({education_level_counts[level]})' for level in educacion_orden],
        autopct='%1.1f%%', startangle=140,
        colors=['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0']
    )
    plt.title('Distribución de Nivel de Educación')
    return fig

    
