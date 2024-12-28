import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,  ConfusionMatrixDisplay
import pickle

try:
    @st.cache_data
    def load_data():
    # Cargar los datos de entrenamiento desde el archivo predefinido
        data = pd.read_csv("data/loan_data.csv")
        st.write("Vista previa del conjunto de datos cargado desde el archivo local:")
        st.dataframe(data.head())
        return data

    train_data = load_data()

    if 'loan_status' not in train_data.columns:
        st.error("El archivo debe contener una columna 'loan_status' para realizar el entrenamiento.")
    else:
        y = train_data['loan_status']
        df_encoded = pd.get_dummies(train_data, columns=['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file' ], drop_first=True)
        X = df_encoded.drop([ 'loan_status'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
        
        #Importando el modelo de RandomForestClassifier
        model = RandomForestClassifier(n_estimators = 100)
        model.fit(X_train,y_train)

        # Guardar el modelo entrenado
        model_path = "models/random_forest.pkl"
        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calcular métricas relevantes
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Mostrar métricas utilizando st.metric
        st.title("Métricas del Modelo Entrenado")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}", border= True)
        col2.metric("F1 Score", f"{f1:.2f}", border= True)
        col3.metric("Precisión", f"{precision:.2f}", border= True)
        col4.metric("Recall", f"{recall:.2f}", border=True)



except FileNotFoundError:
    st.error(f"El archivo no se encontró. Verifica que exista en el folder especificado.")
except Exception as e:
    st.error(f"Ocurrió un error al procesar el archivo de entrenamiento: {e}")

# Sección para realizar predicciones
st.title("Realizar predicciones")
prediction_file = st.file_uploader("Sube un archivo CSV para realizar predicciones", type=["csv"])

if prediction_file is not None and model is not None:
    try:
         # Cargar los datos del CSV de predicciones
        prediction_data = pd.read_csv(prediction_file)
        st.write("Vista previa del archivo cargado para predicciones:")
        st.dataframe(prediction_data.head())

        # Validar si 'Class' está presente y eliminarla
        if 'loan_status' in prediction_data.columns:
            st.warning("La columna 'loan_status' será ignorada para las predicciones.")
            prediction_data_no_target = prediction_data.drop(columns=['loan_status'])
        

        # Especificar las columnas a codificar (puedes personalizar esta lista según tu conjunto de datos)
        columns_to_encode = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file' ]
        
        # Crear las variables dummy para las columnas específicas
        prediction_data_dummies = pd.get_dummies(prediction_data_no_target, columns=columns_to_encode , drop_first=True)




        # Validar si las columnas coinciden con las del modelo
        expected_features = model.feature_names_in_
        if not all(feature in prediction_data_dummies.columns for feature in expected_features):
            missing_features = set(expected_features) - set(prediction_data_dummies.columns)
            st.error(f"El archivo de predicciones no contiene las siguientes columnas esperadas: {missing_features}")
        else:
            # Realizar predicciones
            y_pred = model.predict(prediction_data_dummies)


            st.write("Predicciones realizadas exitosamente:")
            st.dataframe(pd.DataFrame({
                'Predicción (Numérica)': y_pred,
                
            }))

            # Si el archivo contiene la columna 'loan_data', calcular métricas
            if 'loan_status' in prediction_data.columns:
                y_true = prediction_data['loan_status']
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')

                st.write("### Métricas del modelo en las predicciones:")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.2f}")
                col2.metric("F1 Score", f"{f1:.2f}")
                col3.metric("Precisión", f"{precision:.2f}")
                col4.metric("Recall", f"{recall:.2f}")
            else:
                st.warning("El archivo no contiene una columna 'loan_status'. Solo se mostrarán las predicciones.")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo de predicción: {e}")