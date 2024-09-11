import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import numpy as np
import os

# Título 
st.title("Clasificación de Iris con Regresión Logística")

# Cargar el dataset de Iris
iris = load_iris()

# DataFrame 
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Mostrar el dataset
st.write("### Primeras filas del dataset Iris")
st.write(iris_df.head())

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

pipeline_filename = 'pipeline_model.sav'

if os.path.exists(pipeline_filename):
    # Cargar el pipeline si ya existe
    st.write("### Modelo cargado desde el archivo existente")
    pipeline = joblib.load(pipeline_filename)
else:
    # Crear un pipeline con StandardScaler y LogisticRegression
    st.write("### Entrenando el modelo...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Paso 1: Escalado de los datos
        ('logreg', LogisticRegression())    # Paso 2: Modelo de Regresión Logística
    ])

    # Entrenar el pipeline 
    pipeline.fit(X_train, y_train)

  
    joblib.dump(pipeline, pipeline_filename)
    st.write("### Modelo entrenado y guardado en el archivo pipeline_model.sav")

# Predecir con los datos de prueba
y_pred = pipeline.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Metricas
st.write("### Métricas del modelo")
st.write(f"Precisión: {accuracy:.4f}")
st.write(f"Precisión (Precision): {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# Matriz de confusion
st.write("### Matriz de Confusión")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Mostrar el informe de clasificación
st.write("### Informe de Clasificación")
st.write(classification_report(y_test, y_pred))

# Añadir funcionalidad para predicciones con datos ingresados por el usuario
st.write("### Realiza una Predicción")

# Inputs de usuario
sepal_length = st.number_input('Longitud del sépalo (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Ancho del sépalo (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Longitud del pétalo (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Ancho del pétalo (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Botón para realizar predicciones
if st.button("Predecir"):
    # Crear los datos a predecir
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar predicción
    prediction = pipeline.predict(input_data)
    predicted_class = iris.target_names[prediction[0]]

    # Mostrar la predicción
    st.write(f"### Clase Predicha: {predicted_class}")

