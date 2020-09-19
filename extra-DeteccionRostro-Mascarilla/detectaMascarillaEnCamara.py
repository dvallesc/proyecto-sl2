## ARCHIVO QUE SIRVE PARA DETECTAR MASCARILLAS EN DISPOSITIVO DE CAPTURA DE VIDEO USANDO EL MODELO ENTRENADO
# 
# # Para usar
# python detectaMascarillaEnCamara.py

#medidor de tiempo
from datetime import datetime
from datetime import timedelta
# importación de los paquetes de Keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#importación de paquete video
from imutils.video import VideoStream
#importación de otras librerías de funciones
import numpy as np
import imutils
import time
import cv2
import os

#funcion que detecta y predice si está usando mascarilla
#PARAMETROS: la imagen, la red de cara y la red de mascarilla
def detectaMascarilla(frame, caraNet, maskNet):
	# obtiene las dimensiones del marco (frame) que se visualiza y construye un blob de este
	(h, w) = frame.shape[:2]
	# el blob se construye usando la red neural profunda de cv2
	#sus parámetros son el marco, el factor de escalado (se mantiene la proporcion), las dimensiones  que espera la red neural convolucional
	#también se tiene la media en tres canales que se resta a la imagen.
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# analizar el blob usando la red preentrenada para obtener detecciones de rostros
	caraNet.setInput(blob)
	detecciones = caraNet.forward()

	# inicializando los arreglos que contendrán los rostros, su ubicación y la lista de predicciones de la red de mascarillas
	caras = []
	localizaciones = []
	predicciones = []

	# ciclo para las detecciones que devuelve el modelo pre-entrenado
	for i in range(0, detecciones.shape[2]):
		# se extrae la probabilidad de confianza (confidence) que está asociada a la detección
		confianza = detecciones[0, 0, i, 2]

		# se verifica que sólo las detecciones que tengan una mayor confianza que la confianza mínima se procesen
		if confianza > 0.5:
			# opera las coordenadas de la caja de predicción que rodean al objeto
			cajaMarco = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
			(Xinicial, Yinicial, Xfinal, Yfinal) = cajaMarco.astype("int")

			# ajusta las cajas de predicción en caso de que hayan coordenadas fuera del marco de la ventana de visualización
			# si salen las ajusta para que estén dentro de las coordenadas permitidas.
			(Xinicial, Yinicial) = (max(0, Xinicial), max(0, Yinicial))
			(Xfinal, Yfinal) = (min(w - 1, Xfinal), min(h - 1, Yfinal))

			# extrae 
			cara = frame[Yinicial:Yfinal, Xinicial:Xfinal]
			cara = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
			#redimensiona la imagen a 224 x 224 
			cara = cv2.resize(cara, (224, 224))
			#convierte la imagen a un arreglo
			cara = img_to_array(cara)
			#aplica preprocesamiento a la imagen usando la funcion de Keras
			cara = preprocess_input(cara)

			# agrega la cara y la localización de la cara a las listas que se generaron previamente
			caras.append(cara)
			localizaciones.append((Xinicial, Yinicial, Xfinal, Yfinal))

	# verifica que haya por lo menos un rostro para realizar el análisis de mascarilla sobre los rostros
	if len(caras) > 0:
		# a diferencia del caso anterior en este caso se realiza una predicción aplicada al vector de una sola vez
		# este proceso es más rápido
		caras = np.array(caras, dtype="float32")
		# se envían las caras al modelo para obtener la probabilidad de que tengan mascarilla o no la tengan
		predicciones = maskNet.predict(caras, batch_size=32)

	#retorna las localizaciones de las cajas con su par que es el resultado de la predicción
	return (localizaciones, predicciones)

# cargando el modelo de detección de rostros desde el disco
print("[INFO] cargando modelo de detección de rostros...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
caraNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargando el modelo de detección de mascarillas en disco
print("[INFO] cargando modelo de detección de mascarillas...")
maskNet = load_model("mask_detector.model")

# inicializa la transmisión de video y permite que comience a funcionar el sensor de la cámara
print("[INFO] iniciando transmisión de video...")
vs = VideoStream(src=0).start()
tiempoGlobalInicial = datetime.now()
print("[INFO] Iniciando conteo de tiempo "+str(tiempoGlobalInicial))
time.sleep(2.0)


tiempoMascarilla=0
tiempoSinMascarilla=0
tiempoGlobalAcumulado=0
tiempoFinalCiclo=datetime.now()
tiempoInicioCiclo=datetime.now()
# ciclo entre los diferentes cuadros (frames) de la transmisión del video
while True:
	
	# obtiene el cuadro de video y lo redimensiona para que tenga un tamaño máximo de 400 pixeles para que no sea tan pesado
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detecta rostros en el cuadro y determina si están usando mascarilla o no
	(localizaciones, predicciones) = detectaMascarilla(frame, caraNet, maskNet)

	# loop sobre las ubicaciones de los rostros detectados y sus respectivas ubicaciones
	for (cajaMarco, prediccion) in zip(localizaciones, predicciones):
		tiempoInicioCiclo=datetime.now()
		# obtiene las coordenadas de las esquinas del cuadro y la predicción del rostro
		(Xinicial, Yinicial, Xfinal, Yfinal) = cajaMarco
		(mask, withoutMask) = prediccion

		# determina la etiqueta de clasificación y el color que se usará para la caja de ubicación, ademas de las etiquetas de tiempo
		label = "Mask" if mask > withoutMask else "No Mask"
		tiempo=tiempoInicioCiclo-tiempoFinalCiclo
		tiempoTotal=tiempoInicioCiclo-tiempoGlobalInicial
		if mask > withoutMask:
			tiempoMascarilla+=tiempo / timedelta(seconds=1)
		else:
			tiempoSinMascarilla+=tiempo / timedelta(seconds=1)
		tiempoGlobalAcumulado=tiempoTotal / timedelta(seconds=1)
		porcentajeMascarilla=(tiempoMascarilla/tiempoGlobalAcumulado)*100
		porcentajeSinMascarilla=(tiempoSinMascarilla/tiempoGlobalAcumulado)*100
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# se incluye la probabilidad de que sea mascarilla o no en la etiqueta con dos decimales
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		labelTiempo1 = "{}: {:.2f} {}  ({:.2f}%)".format("Acumulado",tiempoMascarilla,"seg",porcentajeMascarilla)
		labelTiempo2 = "{}: {:.2f} {} ({:.2f}%)".format("Acumulado",tiempoSinMascarilla,"seg",porcentajeSinMascarilla)
		labelTiempoGlobal="{}: {:.2f} {}".format("T. Ejecucion: ",tiempoGlobalAcumulado," segundos")
		
		# muestra la etiqueta, el recuadro y el tiempo en el marco
		cv2.putText(frame, label, (Xinicial, Yinicial - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.putText(frame, labelTiempo1, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
		cv2.putText(frame, labelTiempo2, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
		cv2.putText(frame, labelTiempoGlobal, (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
		cv2.rectangle(frame, (Xinicial, Yinicial), (Xfinal, Yfinal), color, 2)
		tiempoFinalCiclo=datetime.now()
	# muestra el marco de salida
	cv2.imshow("PROYECTO SR - DV", frame)
	key = cv2.waitKey(1) & 0xFF

	# si se presiona la letra q minúscula finaliza el programa, rompiendo el ciclo de video
	if key == ord("q"):
		instanteFinal = datetime.now()
		print("[INFO] Finalizando ejecucion a las "+str(instanteFinal))
		print("[INFO] Tiempo total de ejecución "+str(tiempoGlobalAcumulado))
		break

# limpia un poco el entorno destruyendo las ventanas generadas y finaliza el programa
cv2.destroyAllWindows()
vs.stop()
