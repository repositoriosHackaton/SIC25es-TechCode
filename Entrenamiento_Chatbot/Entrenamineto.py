import os
import torch
import psutil
import pandas as pd
import numpy as np
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
        # Configuración de memoria compartida
        self._configurar_memoria_compartida()
        
        # Cargar modelo y tokenizador
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Configuración del dispositivo
        self.device = self._seleccionar_mejor_dispositivo()
        
        # Configuración de múltiples GPUs
        if torch.cuda.device_count() > 1:
            print(f'Usando {torch.cuda.device_count()} GPUs')
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # Configuración de hiperparámetros
        self.learning_rate = learning_rate
    
    def _configurar_memoria_compartida(self):
        """Configurar memoria compartida entre GPU y RAM"""
        # Configurar shared memory para PyTorch
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Aumentar el límite de memoria virtual
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, hard))
        except Exception as e:
            print(f"No se pudo ajustar el límite de memoria: {e}")
    
    def _seleccionar_mejor_dispositivo(self):
        """Seleccionar el mejor dispositivo considerando GPU y RAM"""
        # Verificar memoria disponible en GPU y RAM
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        ram_memory = psutil.virtual_memory().total
        
        print(f"Memoria GPU disponible: {gpu_memory / (1024**3):.2f} GB")
        print(f"Memoria RAM disponible: {ram_memory / (1024**3):.2f} GB")
        
        # Elegir dispositivo con más memoria
        if torch.cuda.is_available() and gpu_memory > ram_memory:
            device = torch.device('cuda')
            print("Usando GPU como dispositivo principal")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Usando dispositivo: {device}")
        
        return device
    
    def preparar_dataloaders(self, train_file, test_file, batch_size=16, max_length=128):
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
        
        # Crear dataloaders con prefetch y pin_memory
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,  # Acelerar transferencia de datos a GPU
            num_workers=os.cpu_count() // 2  # Usar múltiples núcleos
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count() // 2
        )
        
        return train_dataloader, test_dataloader
    
    def entrenar(self, train_dataloader, test_dataloader, epochs=3):
        # Configurar el modelo para entrenamiento 
        self.model.train()
        
        # Configurar el optimizador
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            eps=1e-8, 
            weight_decay=0.01  # Regularización
        )
        
        # Configurar el scheduler de learning rate
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Usar FP16 (16-bit precision) si es posible
        scaler = torch.cuda.amp.GradScaler(init_scale=2.**15)
        
        # Acumulación de gradientes para simular batch más grandes
        accumulation_steps = 2
        
        for epoch in range(epochs):
            print(f'Época {epoch + 1}/{epochs}')
            total_train_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='Entrenando')):
                # Mover datos al dispositivo
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass con FP16
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids, 
                        attention_mask=attention_mask, 
                        labels=input_ids
                    )
                    
                    # Manejar la pérdida para DataParallel
                    if isinstance(outputs, tuple):
                        loss = outputs.loss.mean() / accumulation_steps
                    else:
                        loss = outputs[0].mean() / accumulation_steps
                
                # Acumular pérdida
                total_train_loss += loss.item()
                
                # Backward pass con FP16 y acumulación de gradientes
                scaler.scale(loss).backward()
                
                # Actualizar solo cada ciertos pasos
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Clip de gradientes
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Actualizar parámetros
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Actualizar scheduler
                    scheduler.step()
                    
                    # Limpiar gradientes
                    optimizer.zero_grad()
            
            # Liberar memoria de GPU
            torch.cuda.empty_cache()
            
            # Calcular pérdida promedio de entrenamiento
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f'Pérdida de entrenamiento: {avg_train_loss}')
            
            # Monitoreo de memoria
            self._monitorear_memoria()
            
            # Evaluación
            self._evaluar(test_dataloader)
    
    def _monitorear_memoria(self):
        """Monitorear el uso de memoria de GPU y RAM"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(self.device) / (1024**3)
            gpu_max = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            print(f"Memoria GPU actual: {gpu_memory:.2f} GB")
            print(f"Memoria GPU máxima: {gpu_max:.2f} GB")
        
        ram_memory = psutil.virtual_memory()
        print(f"Memoria RAM ocupada: {ram_memory.percent}%")
    
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
                    outputs = self.model(
                        input_ids, 
                        attention_mask=attention_mask, 
                        labels=input_ids
                    )
                    
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
