# Iris Classification App

Esta aplicación de **Streamlit** clasifica las flores del dataset **Iris** utilizando un modelo de **Regresión Logística**. El modelo es parte de un pipeline que incluye preprocesamiento de datos, y ha sido entrenado y optimizado utilizando **GridSearchCV** para encontrar los mejores hiperparámetros. La aplicación permite realizar predicciones en tiempo real ingresando características de las flores, y proporciona métricas detalladas del rendimiento del modelo.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Métricas del Modelo](#métricas-del-modelo)

## Descripción

El dataset de Iris contiene mediciones de flores de tres especies diferentes: **Setosa**, **Versicolor** y **Virginica**. Las mediciones incluyen:

- Longitud del sépalo (cm)
- Ancho del sépalo (cm)
- Longitud del pétalo (cm)
- Ancho del pétalo (cm)

La aplicación carga un modelo preentrenado que clasifica estas flores basado en sus características. El pipeline utilizado incluye:
1. **Escalado** de las características usando `StandardScaler`.
2. **Clasificación** mediante **Regresión Logística**.
3. **Optimización de hiperparámetros** con **GridSearchCV**.

