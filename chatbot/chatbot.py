#Librerias para instalar de manera local
#opencv-python numpy tensorflow torch transformers tkinter

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    MarianMTModel,
    MarianTokenizer
)
import time

# Clases de emociones predefinidas
CLASES_EMOCIONES = ['amusement', 'excitement', 'anger', 'fear', 'sadness', 'contentment', 'awe', 'disgust']
EMOCIONES_ESPAÑOL = ['diversión', 'emocion', 'enojo', 'miedo', 'tristeza', 'satisfaccion', 'asombro', 'disgusto']

class EmotionalChatbot:
    def __init__(self, modelo_chatbot_path='chatbot_emocional_modelo',
                 modelo_emociones_path='mejor_modelo_emociones.h5'):
        # Configuración del dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cargar modelo de traducción español a inglés
        self.traductor_es_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
        self.tokenizer_es_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')

        # Cargar modelo de traducción inglés a español
        self.traductor_en_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
        self.tokenizer_en_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

        # Cargar modelo de chatbot
        self.tokenizer_chatbot = GPT2Tokenizer.from_pretrained(modelo_chatbot_path)
        self.modelo_chatbot = GPT2LMHeadModel.from_pretrained(modelo_chatbot_path).to(self.device)

        # Cargar modelo de emociones
        self.modelo_emociones = load_model(modelo_emociones_path)

        # Configuración de detección de rostros
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def traducir(self, texto, traductor, tokenizer):
        """Traducir texto usando MarianMT"""
        inputs = tokenizer(texto, return_tensors="pt", padding=True)
        outputs = traductor.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def obtener_emocion_camara(self, frame):
        """Detectar emoción en un frame de video"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Tomar el primer rostro detectado
            (x, y, w, h) = faces[0]
            roi = frame[y:y+h, x:x+w]

            # Preprocesar imagen
            roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gris, (224, 224))
            roi_normalizado = roi_resized / 255.0
            roi_input = np.expand_dims(roi_normalizado, axis=[0, -1])

            # Predecir emoción
            prediccion = self.modelo_emociones.predict(roi_input, verbose=0)
            emocion_idx = np.argmax(prediccion)

            return CLASES_EMOCIONES[emocion_idx]

        return None

    def generar_respuesta(self, texto_entrada, emocion_camara=None):
        """Generar respuesta del chatbot considerando la emoción"""
        # Traducir entrada al inglés
        texto_ingles = self.traducir(texto_entrada,
                                     self.traductor_es_en,
                                     self.tokenizer_es_en)

        # Preparar contexto considerando la emoción
        if emocion_camara:
            contexto = f"[{emocion_camara.upper()} EMOTION] {texto_ingles}"
        else:
            contexto = texto_ingles

        # Tokenizar y generar respuesta
        inputs = self.tokenizer_chatbot(contexto, return_tensors='pt').to(self.device)

        output = self.modelo_chatbot.generate(
            inputs.input_ids,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7
        )

        # Decodificar respuesta
        respuesta_ingles = self.tokenizer_chatbot.decode(output[0], skip_special_tokens=True)

        # Traducir respuesta a español
        respuesta_espanol = self.traducir(respuesta_ingles,
                                          self.traductor_en_es,
                                          self.tokenizer_en_es)

        return respuesta_espanol
