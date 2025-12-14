# Control 04 - Miguel Antonio Daza Moscoso

# Modelo de Predicción de Riesgo Crediticio

## Descripción
Este proyecto tiene como objetivo **predecir la probabilidad de incumplimiento (default)** de un solicitante de crédito utilizando un **modelo de clasificación LightGBM**.  

La solución abarca el ciclo completo de un proyecto de Machine Learning: **preprocesamiento de datos, entrenamiento y evaluación del modelo, interpretabilidad, serialización y despliegue** mediante una aplicación web en Streamlit.

El enfoque principal es proporcionar un sistema de **scoring crediticio transparente, interpretable y confiable**, alineado a buenas prácticas analíticas y de negocio.

---

## Características Principales
- **Preprocesamiento de Datos**
  - Tratamiento de valores faltantes
  - Escalado de variables numéricas
  - Codificación de variables categóricas
- **Modelo LightGBM**
  - Algoritmo de Gradient Boosting optimizado para datos tabulares
- **Evaluación del Modelo**
  - Validación cruzada estratificada
  - Métrica principal: AUC
- **Interpretabilidad del Modelo**
  - Interpretación global y local con **SHAP**
  - Explicaciones locales por observación con **LIME**
- **Despliegue**
  - Modelo serializado con `joblib`
  - Aplicación **Streamlit** para predicciones en tiempo real

---

## Requisitos
Las dependencias necesarias se encuentran especificadas en el archivo `requirements.txt` dentro del directorio `app/`.

Para instalarlas, ejecutar:

```bash
pip install -r app/requirements.txt
