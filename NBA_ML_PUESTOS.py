# -*- coding: utf-8 -*-
"""
SEGUNDO TP MATEMATICA III: MACHINE LEARNING NBA

Grupo TFG
Líder del proyecto: Francisco Coronel
Integrantes: Tomás Aragusuku, Gabriel Tarquini

El objetivo de este modelo es predecir cual es puesto mas acorde de un jugador en base a sus estadisticas
personales por partido a lo largo de una temporada.
Los datos con los que el modelo se nutrirá seran obtenidos de un archivo csv que contiene las
estadisticas personales de los jugadores que participaron de la temporada 2022-2023 de la NBA.
"""
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Importación del archivo CSV

df = pd.read_csv('nba.csv')
df.head()

#Fase 1: EDA (Exploratory Data Analysis) y limpieza de los datos

#Datos estadísticos del conjunto de datos

df.describe()

"""
El dataset con el trabajaremos tiene ciertas irregularidades y datos irrelevantes en nuestra investigacion que debemos
corregir antes de comenzar a entrenar nuestro modelo.
Datos erroneos del dataset: Como el dataset toma la temporada 2022-2023 hay jugadores que jugaron en varios equipos
por lo que las estadisticas estan divididas segun el equipo, en otras palabras hay jugadores duplicados.
Tendremos que decidir si eliminamos a estos o si unificamos datos.

"""

#Eliminamos las columnas irrelevantes del dataset

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


# Eliminamos las columnas irrelevantes para el modelo de ML. Para este caso solo eliminaremos las variables
# categoricas ya que creemos que el conjunto de todos los datos numericos nos dara una mayor especificidad sobre
# las caracteristicas principales de cada puesto.

columnas_irrelevantes = ['Player', 'Tm']

for columna in columnas_irrelevantes:
    df = df.drop(columna, axis=1)
    
    
#Balanceo de los datos.
#Antes de seguir, tenemos que balancear los datos para que todas las posiciones estén correctamente representadas
#Y el algoritmo pueda predecir correctamente todas las posiciones y no solo las que más aparecen

rus = RandomUnderSampler(random_state=0)
df, df["Pos"] = rus.fit_resample(df[['Age', 'G', 'GS', 'MP', 'FG', 'FGA',
                                     'FG%', '3P', '3PA', '3P%', '2P', '2PA',
                                     '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
                                     'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
                                     'PF', 'PTS']],
                                 df["Pos"])


print(df.value_counts('Pos'))

#Búsqueda de datos nulos
# Corroboramos si el dataframe contiene datos nulos. En caso afirmativo deberiamos
# decidir que hacer con ellos. Si eliminarlos o obtener un valor de reemplazo por
# ejemplo obteniendo la media de los datos de dicha columna

# Corroboramos si el dataframe contiene datos nulos

print ("\n\n-----> HAY DATOS NULOS? EN QUE COLUMNA? <-----\n")
print(df.isnull().any()) # Ubicamos en que columna se ubican los datos nulos si es que los hay

print ("\n\n-----> CUANTOS DATOS NULOS HAY? QUE DECISION TOMAMOS? <-----\n")
print(pd.isnull(df).sum()) # Si los hay, averiguamos cuantos son para tomar dimension al aplicar la solucion

# Como vemos que los tiene ya que hay jugadores que no lanzaron dobles/triples/tiros libres, tienen
# el casillero de efectividad vacio. Por eso lo completamos con 0.

df = df.fillna(0)


#Codificación de los datos para poder analizarlos

onehotencoder = make_column_transformer((OneHotEncoder(),[26]), remainder="passthrough")

print("Una matriz de correlación nos ayuda a visualizar qué tan relacionados están nuestras variables independientes con la variable dependiente:")
#Para poder estudiar la relación entre cada posición se transforman con codificación One-Hot a números, agregando 4 columnas al DataFrame
df_correlacion = pd.get_dummies(df, columns=['Pos'])
corr = df_correlacion.corr()
plt.subplots(figsize=(18,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.0%',
            cmap=sns.diverging_palette(240, 10, as_cmap=True))
plt.show()
print("\nLos valores obtenidos nos demuestran que exceptuando la posición de center o pivot y la posición de small forward o alero en menor medida,la correlación entre las estadísticas de un jugador y su posición en el campo de juego es baja.\n")


#División de variables independientes y dependiente y partición en conjuntos de entrenamiento y de prueba

X = df.iloc[:,0:25].values # Realizamos el corte de nuestro dataframe, en X tendremos las variables independientes

y = df.iloc[:,26].values # Realizamos el corte de nuestro dataframe, en y tendremos la variable dependiente (la posición)

#Separar las variables dependientes e independientes

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=1)


#Escalado de los datos de nuestro conjunto de variables independientes para que esten dentro de un rango

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Entrenamiento del modelo

########## EL TIPO DE ENTRENAMIENTO QUE APLICAREMOS A NUESTRO MODELO ES EL DE REGRESIONES CON VARIABLES MULTIPLES

svc = SVC(kernel="linear")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
df_pred = pd.DataFrame({'Real': y_test.flatten(), 'Predicción': y_pred.flatten()})
print(df_pred.head(30),"\n")

plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.title("Realidad vs. Predicción")
plt.show()

print(f"\nPuntaje SVC: {svc.score(X_test,y_test)}\n")

print("Estadísticas generales del modelo de machine learning:")
print(classification_report(y_test,
                            svc.predict(X_test),
                            labels=["C","SG","PF","PG","SF"]))

print("\nPodemos observar que este modelo de machine learning tiene casi tantos aciertos como fallas")
print("La conclusión que sacamos no solo tras analizar los resultados sino también leer analisis de expertos")
print("es que las estadísticas de un jugador, aunque pueden predecir si jugará de pivot (center) o alero (small forward),")
print("no son suficientes para posicionar a cinco jugadores en la posición optima del campo de juego.")
print("También suponemos que si el dataset contuviera datos sobre la estatura y peso de cada jugador, la precisión del modelo podría aumentar.")