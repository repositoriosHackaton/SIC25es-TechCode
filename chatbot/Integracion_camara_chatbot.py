import tkinter as tk
from tkinter import scrolledtext, messagebox
import cv2
import numpy as np
import threading
import queue

# Importar la clase EmotionalChatbot, si esta se encuentra en el mismo directorio
from emotional_chatbot import EmotionalChatbot

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot Emocional")
        master.geometry("600x700")
        master.configure(bg='#f0f0f0')

        # Inicializar el chatbot
        try:
            self.chatbot = EmotionalChatbot()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el chatbot: {str(e)}")
            return

        # Cola para manejar comunicación entre hilos
        self.message_queue = queue.Queue()

        # Crear elementos de la interfaz
        self.create_widgets()

        # Iniciar captura de video en un hilo separado
        self.video_thread = threading.Thread(target=self.capturar_video, daemon=True)
        self.video_thread.start()

    def create_widgets(self):
        # Frame de video
        self.video_frame = tk.Label(self.master, bg='black')
        self.video_frame.pack(pady=10, padx=10, fill='x')

        # Etiqueta de emoción detectada
        self.emocion_label = tk.Label(
            self.master,
            text="Emoción detectada: Ninguna",
            font=('Arial', 12),
            bg='#f0f0f0'
        )
        self.emocion_label.pack(pady=5)

        # Área de chat
        self.chat_area = scrolledtext.ScrolledText(
            self.master,
            wrap=tk.WORD,
            width=70,
            height=20,
            font=('Arial', 10)
        )
        self.chat_area.pack(pady=10, padx=10)
        self.chat_area.config(state=tk.DISABLED)

        # Frame de entrada
        input_frame = tk.Frame(self.master, bg='#f0f0f0')
        input_frame.pack(pady=10, padx=10, fill='x')

        # Entrada de texto
        self.entry = tk.Entry(
            input_frame,
            font=('Arial', 12),
            width=50
        )
        self.entry.pack(side=tk.LEFT, expand=True, fill='x', padx=(0, 10))
        self.entry.bind('<Return>', self.enviar_mensaje)

        # Botón de enviar
        send_button = tk.Button(
            input_frame,
            text="Enviar",
            command=self.enviar_mensaje,
            font=('Arial', 12)
        )
        send_button.pack(side=tk.RIGHT)

    def capturar_video(self):
        cap = cv2.VideoCapture(0)
        emocion_actual = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detectar emoción
            emocion = self.chatbot.obtener_emocion_camara(frame)

            if emocion != emocion_actual:
                emocion_actual = emocion
                self.message_queue.put(('emocion', emocion))

            # Convertir frame para mostrar en Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(data=cv2.imencode('.png', frame_rgb)[1].tobytes())
            self.message_queue.put(('video', img))

            # Procesar eventos de la GUI
            self.master.update()

        cap.release()

    def enviar_mensaje(self, event=None):
        # Obtener texto de entrada
        mensaje = self.entry.get()
        if not mensaje:
            return

        # Limpiar entrada
        self.entry.delete(0, tk.END)

        # Actualizar área de chat
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"Tú: {mensaje}\n", 'user')

        # Generar respuesta del chatbot
        emocion_camara = getattr(self, 'emocion_actual', None)
        respuesta = self.chatbot.generar_respuesta(mensaje, emocion_camara)

        # Mostrar respuesta
        self.chat_area.insert(tk.END, f"Chatbot: {respuesta}\n", 'bot')
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def actualizar_gui(self):
        try:
            while True:
                tipo, valor = self.message_queue.get_nowait()

                if tipo == 'video':
                    self.video_frame.configure(image=valor)
                    self.video_frame.image = valor

                elif tipo == 'emocion':
                    emocion_texto = valor if valor else "Ninguna"
                    self.emocion_label.config(text=f"Emoción detectada: {emocion_texto}")
                    self.emocion_actual = valor

        except queue.Empty:
            pass

        # Programar la próxima actualización
        self.master.after(50, self.actualizar_gui)

def main():
    root = tk.Tk()
    gui = ChatbotGUI(root)

    # Configurar estilos de texto
    gui.chat_area.tag_config('user', foreground='blue')
    gui.chat_area.tag_config('bot', foreground='green')

    # Iniciar actualización de la GUI
    gui.actualizar_gui()

    root.mainloop()

if __name__ == "__main__":
    main()