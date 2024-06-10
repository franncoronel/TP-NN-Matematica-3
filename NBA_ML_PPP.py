# -*- coding: utf-8 -*-
"""
SEGUNDO TP MATEMATICA III: MACHINE LEARNING NBA

Grupo TFG
Líder del proyecto: Francisco Coronel
Integrantes: Tomás Aragusuku, Gabriel Tarquini

El objetivo del trabajo es genererar un modelo que prediga los puntos por partido (PTS) que
obtiene un jugador de NBA segun diversos parámetros.
Nos preguntamos cuales son las variales que mas afectan a los PTS y en un analisis previo suponemos
que la posicion de un jugador, los lanzamientos que tiene, la cantidad de participacion en partidos y
cualquier otra estadistica de orden ofensiva son variables claves a la hora de anotar.
Los datos con los que el modelo se nutrirá seran obtenidos de un archivo csv que contiene las
estadisticas personales de los jugadores que participaron de la temporada 2022-2023 de la NBA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

# Paso 0: Importar archivo CSV

df = pd.read_csv('nba.csv')
# df.head()

# Fase 1: EDA (Exploratory Data Analysis)

"""
El dataset con el trabajaremos tiene ciertas irregularidades y datos irrelevantes en nuestra investigacion que debemos
corregir antes de comenzar a entrenar nuestro modelo.
Datos erroneos del dataset: Como el dataset toma la temporada 2022-2023 hay jugadores que jugaron en varios equipos
y en distintas posiciones por lo que las estadisticas estan divididas segun el equipo, en otras palabras hay jugadores duplicados.
Tendremos que decidir si eliminamos a estos o si unificamos datos.
Tambien tenemos datos nulos que corresponden a las variables de efectivad para aquellos jugadores que no tomaron ningun lanzamiento.
"""

# LIMPIEZA DEL DATASET

# Eliminamos las apariciones duplicadas de los jugadores que jugaron en dos equipos
# en una misma temporada. Solo nos quedamos con la fila que contiene la suma del
# desempeño total en ambos equipos. En el dataset aparece el TOTAL primero y luego
# los datos de cada equipo, por eso utilizamos keep="first" para quedarnos solo con
# esa fila.

df = df.drop_duplicates(subset="Player", keep="first")

# También es necesario eliminar las posiciones combinadas en los jugadores que jugaron en más de un equipo

def borrarPosDuplicada(dataframe):
    dataframe["Pos"] = dataframe["Pos"].str.replace(r'-(.*)', '', regex=True)
    return dataframe

df = borrarPosDuplicada(df)

# Corroboramos si el dataframe contiene datos nulos

print ("\n\n-----> HAY DATOS NULOS? EN QUE COLUMNA? <-----\n")
print(df.isnull().any()) # Ubicamos en que columna se ubican los datos nulos si es que los hay

print ("\n\n-----> CUANTOS DATOS NULOS HAY? QUE DECISION TOMAMOS? <-----\n")
print(pd.isnull(df).sum()) # Si los hay, averiguamos cuantos son para tomar dimension al aplicar la solucion

# Como vemos que los tiene ya que hay jugadores que no lanzaron dobles/triples/tiros libres, tienen
# el casillero de efectividad vacio. Por eso lo completamos con 0.

df = df.fillna(0)

# Realizamos la matriz de correlacion con las columnas númericas para evaluar de manera grafica la relacion entre
# las variables y ver si lo planteado en el inicio corresponde o tenemos que redefinir cuales son las variables que afectan a PTS

# MATRIZ DE CORRELACION

df_correlacion = df.drop('Player', axis=1)
# df_correlacion = df_correlacion.drop('Pos', axis=1)
df_correlacion = df_correlacion.drop('Tm', axis=1)
df_correlacion = pd.get_dummies(df_correlacion, columns=['Pos'])

corr = df_correlacion.corr()
plt.subplots(figsize=(25,15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.0%',
            cmap=sns.diverging_palette(240, 10, as_cmap=True))

print ("\n\n-----> CUALES SON LAS VARIABLES QUE TIENEN MAYOR CORRELATIVIDAD CON LA OBTENCION DE PUNTOS POR PARTIDO <-----\n")
print(corr['PTS'].abs().sort_values(ascending=False))

# Eliminamos las columnas que creemos irrelevantes para el modelo de ML
columnas_irrelevantes = ['Player', 'Pos', 'Tm', 'GS', 'FG', 'FGA', 'FG%', '3P%', '3P', '2P%', '2P', 'eFG%', 'FT%', 'FT', 'DRB',
                          'ORB', 'BLK', 'TOV', 'PF']

for columna in columnas_irrelevantes:
    df = df.drop(columna, axis=1)

print ("\n\n-----> FINALMENTE NOS QUEDAMOS CON ESTAS VARIABLES <-----\n")

for i in df.columns:
    print (i, end=" - ")

print ("\n\nReferencias:\nAge = Edad\nG = Partidos jugados\nMP = Minutos jugador por partido\n3PA = Tiros triples intentados\n2PA = Tiros dobles intentados\nFTA = Tiros libres lanzados\nTRB = Rebotes\nAST = Asistencias\nSTL = Robos de balon\nPTS = Puntos por partido (valor a predecir)")

# HACEMOS UN CORTE DEL DATA SET PARA PREPARAR EL ENTRENAMIENTO Y TESTEO DEL MODELO

X = df.iloc[:, :-1].values # Realizamos el corte de nuestro dataframe, en X tendremos las variables independientes

y = df.iloc[:, 9].values # Realizamos el corte de nuestro dataframe, en y tendremos la variable dependiente (lo que queremos predecir)
y = y.reshape(-1, 1)

# Dividimos el dataset en grupos de entrenamiento y testeo

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# ESCALAMOS LOS DATOS PARA UN MEJOR CRITERIO

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print ("\n\n-----> CORROBORAMOS COMO SE ESCALARON LOS DATOS <-----\n")
print("X TRAIN\n", X_train[:8][0:3], "\n")
print("X TEST\n", X_test[:8][0:3], "\n")

# EL TIPO DE ENTRENAMIENTO QUE APLICAREMOS A NUESTRO MODELO ES EL DE REGRESIONES CON VARIABLES MULTIPLES

regressor = LinearRegression() 
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df_aux.head(25)

print ("\n\n-----> REALIZAMOS UNA PEQUEÑA PRUEBA DEL MODELO <-----\n")
print(df1.head())

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.plot(y_test, y_test, color='red', linestyle='-')
plt.xlabel("Actual")
plt.ylabel("Predicción")
plt.title("Actual vs Predicción")
plt.show()

# METRICAS

print ("\n\n-----> M E T R I C A S <-----\n")
print('Error Absoluto Medio (MAE): ', metrics.mean_absolute_error(y_test, y_pred))
print('Error Cuadrático Medio (MSE): ', metrics.mean_squared_error(y_test, y_pred))
print('Raíz del Error Cuadrático Medio (RMSE): ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coeficiente de Determinación R^2: ', r2_score(y_test, y_pred))

print ("\n\n-----> PROBAMOS EL MODELO CON JUGADORES AJENOS A LA NBA <-----\n")

class jugadorNBA:
    def __init__(self, nombre, stats):
        self.nombre = nombre
        self.stats = stats
    
# EL MODELO PERMITE CALCULAR LO PUNTOS QUE HARIA UN JUGADOR DE OTRA LIGA SI MANTIENE SUS NUMEROS EN NBA
facundoCampazzo = jugadorNBA("Facundo Campazzo",[[29, 65, 21.9, 3.3, 1.5, 1.4, 2.1, 3.6, 1.2]])
# Escalar las stats
facundoCampazzo.stats = sc.transform(facundoCampazzo.stats)
predic_Campazzo = regressor.predict(facundoCampazzo.stats)
print ("Con las estadisticas previstas,", facundoCampazzo.nombre, " deberia hacer", round(predic_Campazzo[0][0], 2) , "puntos por partido en una temporada de NBA.")


# TAMBIEN PODRIA PREDECIR LOS PUNTOS QUE HARIAN JUGADORES DE OTRAS EPOCA COMO POR EJEMPLO MICHAEL JORDAN SI TOMAMOS SUS ESTADISTICAS DE LA TEMPORADA
# 87/88 QUE FUE DE LAS MEJORES QUE TUVO.
michaelJordan = jugadorNBA ("Michael Jordan", [[23, 82, 40, 0.8, 27, 11.9, 5.2, 4.6, 2.9]])
#Escalar las stats
michaelJordan.stats = sc.transform(michaelJordan.stats)
predic_Jordan = regressor.predict(michaelJordan.stats)
print ("Con las estadisticas previstas,", michaelJordan.nombre, " del 87/88 deberia hacer", round(predic_Jordan[0][0], 2) , "puntos por partido en una temporada de NBA actual.\nSi vamos a la fuente, Michael Jordan hizo 37.12 puntos por partido. La predicción fue bastante aproximada.\n")

print("CONCLUSIÓN")
print("Las conclusiones que pudimos obtener al efectuar el trabajo fueron las siguientes:")
print("Mediante el análisis de correlatividad de las variables, nos dimos cuenta que debíamos", end="")
print("utilizar, no solo las stats ofensivas, sino también las defensivas que aportan para que la", end="")
print("predicción sea más precisa.\nSuponemos que esto se debe a que es un deporte muy dinámico y los ataques",end="")
print("son constantes durante todos los partidos.\nTambién notamos que las variables de posición no influían en",end="")
print("el resultado final como pensábamos en nuestra hipótesis.")
