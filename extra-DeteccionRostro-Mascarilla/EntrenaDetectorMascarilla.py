##PERMITE ENTRENAR EL MODELO QUE SE USARÁ PARA IDENTIFICAR SI TIENE MASCARILLA O NO TIENE MASCARILLA

#PARA USAR
# python EntrenaDetectorMascarilla.py --dataset dataset
#donde dataset es la carpeta que contiene las iimagenes a clasificar en dos carpetas que contengan imagenes según la clasificación

# importa los paquetes de Keras que sirven para generar el modelo
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
#preprocesamiento, selección de modelos y reporte de clasificación generado con Sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="ruta para el dataset de entrada")
ap.add_argument("-p", "--plot", type=str, default="grafico.png",
	help="ruta para mostrar el gráfico de certeza/perdida modelo")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="ruta para la salida del detector de rostro")
args = vars(ap.parse_args())

# inicializa los valores de learning rate (INIT_LR), número de epochs para entrenar (EPOCHS) y el tamaño del conjunto de datos (BS)
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# toma la lista de imagenes en el directorio del dataset
print("[INFO] cargando imagenes...")
rutaImagenes = list(paths.list_images(args["dataset"]))
#inicializa las listas de datos y etiquetas
datos = []
etiquetas = []

# ciclo en la dirección de imagenes
for rutaImagen in rutaImagenes:
	# extrae la clase a la que pertenece la imagen de la etiqueta
	etiqueta = rutaImagen.split(os.path.sep)[-2]

	# carga la imagen de entrada en dimensiones de 224x224, la convierte en arreglo y la preprocesa con Keras
	imagen = load_img(rutaImagen, target_size=(224, 224))
	imagen = img_to_array(imagen)
	imagen = preprocess_input(imagen)

	# agrega la imagen y la etiqueta a la lista correspondiente
	datos.append(imagen)
	etiquetas.append(etiqueta)

# convierte los datos y etiquetas a arreglos de Numpy
datos = np.array(datos, dtype="float32")
etiquetas = np.array(etiquetas)

# realiza one-hot encoding a las etiquetas
lb = LabelBinarizer()
etiquetas = lb.fit_transform(etiquetas)
etiquetas = to_categorical(etiquetas)

# particiona los datos en dos: de entrenamiento y pruebas usando 75% como datos de entrenamiento y el 25% restante para pruebas
(entrenaX, pruebaX, entrenaY, pruebaY) = train_test_split(datos, etiquetas,
	test_size=0.20, stratify=etiquetas, random_state=42)

# construye el generador de imagenes de entrenamiento para acercamiento de datos
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# carga la red MobileNetV2, asegurando que la capa inicial Fully Connected se dejan apagados
modeloBase = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construye la cabecera del modelo que será puesto al principio del modelo base
cabeceraModelo = modeloBase.output
cabeceraModelo = AveragePooling2D(pool_size=(7, 7))(cabeceraModelo)
cabeceraModelo = Flatten(name="flatten")(cabeceraModelo)
cabeceraModelo = Dense(128, activation="relu")(cabeceraModelo)
cabeceraModelo = Dropout(0.5)(cabeceraModelo)
cabeceraModelo = Dense(2, activation="softmax")(cabeceraModelo)

# colocando la cabeza del modelo fully-connected en la base del modelo (esto será el modelo a entrenar)
modelo = Model(inputs=modeloBase.input, outputs=cabeceraModelo)

# ciclo sobre todas las capas del modelo base y congelación, para que no se actualicen durante el primer proceso de entrenamiento
for layer in modeloBase.layers:
	layer.trainable = False

# compilación del modelo
print("[INFO] compilando modelo...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
modelo.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# entrenando la raiz de la red neural
print("[INFO] entrenando encabezado...")
H = modelo.fit(
	aug.flow(entrenaX, entrenaY, batch_size=BS),
	steps_per_epoch=len(entrenaX) // BS,
	validation_data=(pruebaX, pruebaY),
	validation_steps=len(pruebaX) // BS,
	epochs=EPOCHS)

# haciendo predicciones con el test de pruebas
print("[INFO] evaluando la red neural...")
predIdxs = modelo.predict(pruebaX, batch_size=BS)

# para cada imagen en el conjunto de imagenes de pruebas, se necesita encontrar el índice de la etiqueta con la probabilidad más grande predicha
predIdxs = np.argmax(predIdxs, axis=1)

# muestra un reporte de la clasificación en el modelo
print(classification_report(pruebaY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# guardando el modelo en disco
print("[INFO] guardano modelo de detección de mascarilla...")
modelo.save(args["model"], save_format="h5")

# graficando la pérdida y certeza del modelo
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Entrenamiento - Perdida y Certeza")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
