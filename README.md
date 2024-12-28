# Data Visualization and Prediction App

Esta aplicación permite visualizar datos y hacer predicciones utilizando un modelo de aprendizaje automático. Los usuarios pueden cargar conjuntos de datos, explorar las visualizaciones generadas y obtener predicciones basadas en un modelo previamente entrenado.

### Requisitos
* Python 3.7 o superior
* pip para la gestión de paquetes

### Instrucciones para correr la aplicación localmente
Sigue estos pasos para ejecutar la aplicación en tu máquina local.
1. Crear el ambiente virtual
    ```
    python -m venv venv
    ``` 
2. Activar el ambiente virtual
    En Windows:
    ```
   .\venv\Scripts\activate
    ```

    En Linux/Mac:
    ```
   source venv/bin/activate
    ``` 
3.  Instalar dependencias

    Una vez activado el entorno virtual, instala las dependencias necesarias para el proyecto ejecutando:

    ```
    pip install -r requirements.txt
    ``` 
4. Ejecutar la aplicación
    ```
   streamlit run app.py
    ``` 
5. Salir del ambiente virtual (opcional)

    Cuando termines de trabajar en la aplicación, puedes salir del entorno virtual con el siguiente comando:
    ```
   deactivate
    ``` 

### Estructura del proyecto

```
.   
└── data-visualization-app/
    ├── data/
    │   ├── loan_data.csv
    ├── models/
    │   ├── random_forest.pkl
    ├── pages/
    │   ├── 1_Inicio
    │   ├── 2_EDA
    │   ├── 3_Hipotesis
    │   ├── 4_Modelo
    ├── src/
    │   ├── eda.py
    │   ├── eda.py   
    ├── utils/
    │   ├── eda.png
    │   ├── hipotesis.jpg
    │   ├── modelo.png   
    ├── README.md
    |
    ├──Inicio.py
    |
    ├──.gitignore
    |
    └──requirements.txt
```









