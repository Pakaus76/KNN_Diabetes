import streamlit as st
import joblib
import pandas as pd

def predict_diabetes(age, bmi, glucose, diabetes_pedigree, blood_pressure):
    # Cargar scaler_info desde el archivo
    scaler_info = joblib.load("scaler.pkl")
    # Extraer el scaler y el orden de columnas del diccionario
    scaler = scaler_info["scaler"]
    columns_order = scaler_info["columns"]
    
    # Crear el DataFrame de entrada usando el orden de columnas correcto
    input_data = pd.DataFrame([[glucose, bmi, age, diabetes_pedigree, blood_pressure]], 
                              columns=columns_order)
    
    # Aplicar la transformación con el scaler extraído
    input_data_scaled = scaler.transform(input_data)
    
    # Cargar el modelo entrenado
    model = joblib.load("knn_diabetes_model.pkl")
    prediction = model.predict(input_data_scaled)
    return prediction[0]

st.title("Predicción de Diabetes")

# Solicitar entradas al usuario
glucose = st.number_input("Nivel de Glucosa", min_value=0, value=100)
bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, value=25.0, format="%.2f")
age = st.number_input("Edad", min_value=0, value=25)
diabetes_pedigree = st.number_input("Función del Pedigrí de Diabetes", min_value=0.0, value=0.5, format="%.2f")
blood_pressure = st.number_input("Presión Arterial", min_value=0, value=70)

if st.button("Predecir"):
    result = predict_diabetes(age, bmi, glucose, diabetes_pedigree, blood_pressure)
    if result == 1:
        st.write("La persona tiene diabetes.")
    else:
        st.write("La persona NO tiene diabetes.")
