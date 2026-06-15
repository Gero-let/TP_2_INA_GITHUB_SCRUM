# Modelo de Machine Learning Supervisado: Clasificación del Dataset Iris

## Descripción General del Proyecto
Este proyecto consiste en el desarrollo de un modelo de Machine Learning supervisado orientado a la clasificación. Utilizando el famoso dataset "Iris", el script entrena un modelo basado en Árboles de Decisión (`DecisionTreeClassifier`) capaz de predecir la especie de una flor Iris basándose en cuatro características morfológicas: longitud y ancho del sépalo, y longitud y ancho del pétalo. Es un ejercicio práctico para comprender el flujo de trabajo en la ciencia de datos: carga de datos, separación en conjuntos de entrenamiento y prueba, entrenamiento del modelo y predicción de nuevos valores.

## Tecnologías Utilizadas
* **Lenguaje:** Python 3
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`DecisionTreeClassifier`, `train_test_split`, `accuracy_score`)
* **Visualización:** Matplotlib (importada para futuras implementaciones)

## Instrucciones para Ejecutar el Código
1. Clonar este repositorio en tu máquina local.
2. Asegurarte de tener instalado Python 3.
3. Instalar las dependencias necesarias. Puedes hacerlo ejecutando en tu terminal:
   `pip install pandas numpy scikit-learn matplotlib`
4. Verificar que el archivo `Iris.csv` se encuentre en el mismo directorio que el script `ML_Supervisado_Iris.py`.
5. Ejecutar el script desde la terminal o tu IDE favorito:
   `python ML_Supervisado_Iris.py`

## Resultados y Capturas de Pantalla
Al ejecutar el script, el modelo evalúa su precisión con los datos de prueba y luego realiza predicciones sobre 4 flores nuevas ingresadas manualmente. 

**Ejemplo de salida en consola:**
Las predicciones para las 4 flores nuevas son:
['Iris-setosa' 'Iris-virginica' 'Iris-setosa' 'Iris-versicolor']
La precision del modelo es: 100.0%

<img width="745" height="190" alt="image" src="https://github.com/user-attachments/assets/02164705-cf5c-464d-8ae9-1e9393f6d835" />

## Conclusiones Personales
Este proyecto sirve para comprender y experimentar el funcionamiento de los modelos de aprendizaje supervisado de Machine Learninig, en particular mediante el modeleo de Árbol de Decision. También apreciamos cómo se trabaja un dataset en un entorno similar a la práctica real, así como el trabajo con librerías específicas para el procesamiento de datos.
