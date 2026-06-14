# Trabajando con el dataset Iris ML Supervisado - Clasificación
# Importar las librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset

# Ruta utilizada en ámbito local:
# df_iris = pd.read_csv(r'C:\Users\gero_\OneDrive\Escritorio\INA\2 CUATRIMESTRE\Desarrollo de IA\Clase 26\archive\Iris.csv')

# Ruta Normalizada
df_iris = pd.read_csv('Iris.csv')

# Verificar la cantidad de filas y columnas
# print(df_iris.shape)

# # Verificar los tipos de datos
# print(df_iris.dtypes)

# # Verificar los valores nulos
# print(df_iris.isnull().sum())

# # Verificar los valores duplicados
# print(df_iris.duplicated().sum())

# Datos de entrada
X = df_iris[[
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]]
# Datos de salida
y = df_iris[[
    'Species'
]]

#Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
    )

# Creamos el modelo
modelo = DecisionTreeClassifier()

# Entrenamos el modelo
modelo.fit(X_train, y_train)

# Predicciones
predicciones = modelo.predict(X_test)
# print(predicciones)

flor1 = [5, 3, 5, 1]
flor2 = [6, 6, 7, 4]
flor3 = [1, 5, 1, 2]
flor4 = [3, 4.1, 2.3, 3.4]

flores = np.array([flor1, flor2, flor3, flor4])

nuevas_flores = pd.DataFrame(
    flores, 
    columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
)
print(nuevas_flores)

resultado = modelo.predict(nuevas_flores)
print("Las predicciones para las 4 flores nuevas son:")
print(resultado)

# Precision del modelo - 0 y 1
precision = accuracy_score(y_test, predicciones)
print(f"La precision del modelo es: {round(precision * 100, 2)}%")