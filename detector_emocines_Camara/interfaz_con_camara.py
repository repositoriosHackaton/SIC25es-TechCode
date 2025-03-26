import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import tkinter as tk
from tkinter import messagebox

# Definir las clases de emociones
CLASES_EMOCIONES = ['amusement', 'excitement', 'anger', 'fear', 'sadness', 'contentment', 'awe', 'disgust']
EMOCIONES_ESPAÑOL = ['diversión', 'emocion', 'enojo', 'miedo', 'tristeza', 'satisfaccion', 'asombro', 'disgusto']

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo entrenado
def cargar_modelo(ruta_modelo='mejor_modelo_emociones.h5'):
    try:
        modelo = load_model(ruta_modelo)
        print(f"Modelo cargado correctamente desde {ruta_modelo}")
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Función para obtener el tamaño de entrada del modelo
def obtener_tamaño_modelo(modelo):
    input_shape = modelo.input_shape
    if input_shape is not None and len(input_shape) >= 3:
        altura = input_shape[1]
        anchura = input_shape[2]
        return (altura, anchura)
    else:
        return (224, 224)

# Función para preprocesar la imagen
def preprocesar_imagen(imagen, tamaño):
    if imagen is None or imagen.size == 0:
        return None
    try:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen_redimensionada = cv2.resize(imagen_gris, tamaño)
        imagen_normalizada = imagen_redimensionada / 255.0
        imagen_con_canal = np.expand_dims(imagen_normalizada, axis=-1)
        imagen_con_batch = np.expand_dims(imagen_con_canal, axis=0)
        return imagen_con_batch
    except Exception as e:
        print(f"Error en preprocesar_imagen: {e}")
        return None

# Función para mostrar la emoción
def mostrar_emocion(frame, emocion_idx, probabilidad, x, y, w, h):
    emocion_esp = EMOCIONES_ESPAÑOL[emocion_idx]
    puntos = int(probabilidad * 10) + 1
    color = (0, 255, 0) if CLASES_EMOCIONES[emocion_idx] in ['amusement', 'contentment', 'awe', 'excitement'] else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{emocion_esp}: {puntos} pts", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Función para iniciar el reconocimiento de emociones
def reconocer_emociones_camara(modelo, umbral_confianza=0.5):
    tamaño_modelo = obtener_tamaño_modelo(modelo)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            roi_procesada = preprocesar_imagen(roi, tamaño=tamaño_modelo)
            if roi_procesada is None:
                continue

            try:
                prediccion = modelo.predict(roi_procesada, verbose=0)
                clase_predicha = np.argmax(prediccion)
                probabilidad = prediccion[0][clase_predicha]

                if probabilidad > umbral_confianza:
                    mostrar_emocion(frame, clase_predicha, probabilidad, x, y, w, h)
                else:
                    cv2.putText(frame, "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error al realizar predicción: {e}")

        cv2.imshow('Reconocimiento de Emociones', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función para visualizar los resultados en una ventana de Tkinter
def iniciar_reconocimiento_emociones():
    modelo = cargar_modelo()
    if modelo is None:
        messagebox.showerror("Error", "No se pudo cargar el modelo. Verifique el archivo.")
        return
    reconocer_emociones_camara(modelo)

# Crear la ventana de Tkinter
ventana = tk.Tk()
ventana.title("Reconocimiento de Emociones")
ventana.geometry("400x250")
ventana.config(bg="#F0F8FF")

# Estilo de botones
estilo_boton = {
    "bg": "#4CAF50",  # Verde
    "fg": "white",
    "font": ("Helvetica", 14, "bold"),
    "width": 20,
    "height": 2,
    "relief": "raised",
    "bd": 5,
    "activebackground": "#45a049",
    "activeforeground": "white"
}

# Funciones para los botones
def reiniciar():
    messagebox.showinfo("Reiniciar", "El reconocimiento de emociones se reiniciará.")
    iniciar_reconocimiento_emociones()

def pausar():
    messagebox.showinfo("Pausar", "El reconocimiento de emociones se ha pausado.")
    cv2.destroyAllWindows()

def salir():
    ventana.quit()

# Agregar título
titulo = tk.Label(ventana, text="Sistema de Reconocimiento de Emociones", font=("Helvetica", 16, "bold"), bg="#F0F8FF")
titulo.pack(pady=20)

# Botones
boton_iniciar = tk.Button(ventana, text="Iniciar Reconocimiento", command=iniciar_reconocimiento_emociones, **estilo_boton)
boton_iniciar.pack(pady=10)

boton_reiniciar = tk.Button(ventana, text="Reiniciar", command=reiniciar, **estilo_boton)
boton_reiniciar.pack(pady=10)

boton_pausar = tk.Button(ventana, text="Pausar", command=pausar, **estilo_boton)
boton_pausar.pack(pady=10)

boton_salir = tk.Button(ventana, text="Salir", command=salir, **estilo_boton)
boton_salir.pack(pady=10)

# Ejecutar la interfaz
ventana.mainloop()
