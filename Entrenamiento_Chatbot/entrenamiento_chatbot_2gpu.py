# -*- coding: utf-8 -*-
"""Entrenamiento_Chatbot_2GPU.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/137gOWe7pguk0cv6W_OjWoLHxGAbhhgE5
"""

# Pensado para Kaggle noteboks
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        texto = self.texts[idx]
        emocion = self.labels[idx]

        encoding = self.tokenizer(
            texto,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }

class EmotionalChatbotTrainer:
    def __init__(self, model_name='gpt2-medium', learning_rate=5e-5):
        # Cargar modelo y tokenizador
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Configuración del dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Si hay más de una GPU, utilizar DataParallel
        if torch.cuda.device_count() > 1:
            print(f'Usando {torch.cuda.device_count()} GPUs')
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        # Configuración de hiperparámetros
        self.learning_rate = learning_rate

    def preparar_dataloaders(self, train_file, test_file, batch_size=8, max_length=128):
        # Cargar los CSVs
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Obtener los textos y las emociones
        X_train = train_df['texto'].tolist()
        y_train = train_df['emocion'].tolist()
        X_test = test_df['texto'].tolist()
        y_test = test_df['emocion'].tolist()

        # Crear datasets
        train_dataset = ChatbotDataset(X_train, y_train, self.tokenizer, max_length)
        test_dataset = ChatbotDataset(X_test, y_test, self.tokenizer, max_length)

        # Crear dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    def entrenar(self, train_dataloader, test_dataloader, epochs=3):
        # Configurar el modelo para entrenamiento
        self.model.train()

        # Configurar el optimizador
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)

        # Configurar el scheduler de learning rate
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Usar FP16 (16-bit precision) si es posible
        scaler = torch.cuda.amp.GradScaler(init_scale=2.**15)

        for epoch in range(epochs):
            print(f'Época {epoch + 1}/{epochs}')
            total_train_loss = 0

            for batch in tqdm(train_dataloader, desc='Entrenando'):
                # Mover datos al dispositivo
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Limpiar gradientes
                optimizer.zero_grad()

                # Forward pass con FP16
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

                    # Manejar la pérdida para DataParallel
                    if isinstance(outputs, tuple):
                        loss = outputs.loss.mean()
                    else:
                        loss = outputs[0].mean()

                # Acumular pérdida
                total_train_loss += loss.item()

                # Backward pass con FP16
                scaler.scale(loss).backward()

                # Clip de gradientes
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Actualizar parámetros
                scaler.step(optimizer)
                scaler.update()

                # Actualizar scheduler
                scheduler.step()

            # Calcular pérdida promedio de entrenamiento
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f'Pérdida de entrenamiento: {avg_train_loss}')

            # Evaluación
            self._evaluar(test_dataloader)

    def _evaluar(self, test_dataloader):
        # Configurar el modelo para evaluación
        self.model.eval()
        total_eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc='Evaluando'):
                # Mover datos al dispositivo
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Evaluar con FP16
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

                    # Manejar la pérdida para DataParallel
                    if isinstance(outputs, tuple):
                        loss = outputs.loss.mean()
                    else:
                        loss = outputs[0].mean()

                total_eval_loss += loss.item()

        # Calcular pérdida promedio de evaluación
        avg_eval_loss = total_eval_loss / len(test_dataloader)
        print(f'Pérdida de evaluación: {avg_eval_loss}')

    def guardar_modelo(self, ruta='chatbot_emocional_modelo'):
        # Extraer el modelo base si está en DataParallel
        modelo_guardar = self.model.module if hasattr(self.model, 'module') else self.model

        modelo_guardar.save_pretrained(ruta)
        self.tokenizer.save_pretrained(ruta)
        print(f'Modelo guardado en {ruta}')

if __name__ == "__main__":
    # Definir los archivos CSV
    # Solo se suben los archivos a kaggle y si no estoy mal no cambian las rutas
    train_file = '/kaggle/input/chatbot/train_dataset.csv'
    test_file = '/kaggle/input/chatbot/test_dataset.csv'

    # Inicializar entrenador
    trainer = EmotionalChatbotTrainer(model_name='gpt2-medium')

    # Preparar dataloaders
    train_dataloader, test_dataloader = trainer.preparar_dataloaders(train_file, test_file)

    # Entrenar modelo
    trainer.entrenar(train_dataloader, test_dataloader)

    # Guardar modelo
    trainer.guardar_modelo()