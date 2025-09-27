# ml/lstm_integration.py
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class LSTMIntegration:
    def __init__(self, model_path: str, scaler_path: str):
        """Integración de modelos LSTM externos para predicción de llenado"""
        try:
            # Cargar modelo y scaler
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.sequence_length = 24  # Ajustar según tu modelo
            self.is_loaded = True
            print("✅ Modelos LSTM cargados exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelos LSTM: {e}")
            self.is_loaded = False
    
    def prepare_sequence_data(self, historical_data: pd.DataFrame, container_id: int) -> Optional[np.ndarray]:
        """Preparar datos en formato de secuencia para LSTM"""
        try:
            # Filtrar datos del contenedor específico
            container_data = historical_data[
                historical_data['container_id'] == container_id
            ].sort_values('timestamp')
            
            if len(container_data) < self.sequence_length:
                return None
            
            # Usar fill_percentage como feature
            values = container_data['fill_percentage'].tail(self.sequence_length).values
            values_scaled = self.scaler.transform(values.reshape(-1, 1))
            
            return values_scaled.reshape(1, self.sequence_length, 1)
            
        except Exception as e:
            print(f"Error preparando datos para contenedor {container_id}: {e}")
            return None
    
    def predict_fill_trend(self, historical_data: pd.DataFrame, container_id: int, 
                          hours_ahead: int = 24) -> Dict[str, float]:
        """Predecir tendencia de llenado usando LSTM"""
        if not self.is_loaded:
            return {"error": "Modelo LSTM no disponible"}
        
        try:
            sequence_data = self.prepare_sequence_data(historical_data, container_id)
            if sequence_data is None:
                return {"error": "Datos insuficientes para predicción"}
            
            # Hacer predicción
            prediction_scaled = self.model.predict(sequence_data, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
            
            # Calcular tasa de llenado por hora (simulada basada en predicción)
            current_fill = historical_data[
                historical_data['container_id'] == container_id
            ]['fill_percentage'].iloc[-1]
            
            fill_rate = max(0.1, (prediction - current_fill) / hours_ahead)
            
            return {
                "predicted_fill_24h": round(prediction, 2),
                "current_fill": round(current_fill, 2),
                "hourly_fill_rate": round(fill_rate, 3),
                "confidence": 0.85  # Puedes calcular esto basado en error del modelo
            }
            
        except Exception as e:
            print(f"Error en predicción LSTM para contenedor {container_id}: {e}")
            return {"error": str(e)}

# Instancia global
lstm_predictor = LSTMIntegration(
    model_path="models/lstm_model.keras", 
    scaler_path="models/lstm_scaler.joblib"
)