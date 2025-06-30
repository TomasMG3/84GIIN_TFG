import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Optional, Any

class WastePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42)
        }
        self.trained_models = {}
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features para predicción"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Features temporales
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Features cíclicas
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Codificar zona (one-hot encoding)
        if 'zone' in df.columns:
            zone_dummies = pd.get_dummies(df['zone'], prefix='zone')
            df = pd.concat([df, zone_dummies], axis=1)
        
        # Eliminar columnas no numéricas
        df = df.drop(columns=['container_id', 'timestamp', 'zone'], errors='ignore')
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Entrena modelos predictivos"""
        # Preparar datos
        df_features = self.create_features(df)
        
        # Seleccionar features
        feature_cols = [col for col in df_features.columns if col not in ['container_id', 'timestamp', 'fill_percentage']]
        
        X = df_features[feature_cols].fillna(0)
        y = df_features['fill_percentage']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.feature_columns = feature_cols
        results = {}
        
        # Entrenar cada modelo
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.trained_models[name] = model
            results[name] = {
                'mae': round(mae, 3),
                'r2': round(r2, 3),
                'accuracy': round((1 - mae/100) * 100, 1)
            }
        
        return results
    
    def predict_fill_time(self, container_id: int, current_fill: float, zone: Optional[str] = None) -> Dict[str, Any]:
        """Predice cuándo se llenará un contenedor"""
        if not self.trained_models:
            return {"error": "Models not trained"}
        
        # Usar zona por defecto si no se proporciona
        zone = zone or 'default'
        
        # Crear datos sintéticos para predicción
        future_hours = 48  # Predecir próximas 48 horas
        now = datetime.now()
        
        future_data = []
        for h in range(future_hours):
            future_time = now + timedelta(hours=h)
            future_data.append({
                'container_id': container_id,
                'timestamp': future_time,
                'fill_percentage': current_fill,  # Valor base
                'zone': zone
            })
        
        df_future = pd.DataFrame(future_data)
        df_features = self.create_features(df_future)
        
        missing_cols = set(self.feature_columns) - set(df_features.columns)
        for col in missing_cols:
            df_features[col] = 0
        df_features = df_features[self.feature_columns]  # Reordenar columnas como en entrenamiento
        
        # Usar mejor modelo (Random Forest por defecto)
        model = self.trained_models.get('random_forest')
        if not model:
            return {"error": "Random Forest model not available"}
        
        X_future = df_features[self.feature_columns].fillna(0)
        predictions = model.predict(X_future)
        
        # Encontrar cuándo alcanzará 100%
        fill_threshold = 95.0  # Umbral de alerta
        
        for i, pred_fill in enumerate(predictions):
            if pred_fill >= fill_threshold:
                hours_to_full = i
                full_time = now + timedelta(hours=hours_to_full)
                
                return {
                    "container_id": container_id,
                    "current_fill": current_fill,
                    "predicted_full_percentage": round(pred_fill, 1),
                    "hours_to_full": hours_to_full,
                    "predicted_full_time": full_time.isoformat(),
                    "priority": "HIGH" if hours_to_full < 24 else "MEDIUM" if hours_to_full < 48 else "LOW",
                    "recommendation": f"Schedule collection within {hours_to_full} hours"
                }
        
        return {
            "container_id": container_id,
            "current_fill": current_fill,
            "predicted_full_percentage": round(predictions[-1], 1),
            "hours_to_full": None,
            "priority": "LOW",
            "recommendation": "No immediate collection needed"
        }
    
    def get_daily_predictions(self, containers_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Obtiene predicciones para todos los contenedores"""
        predictions = []
        
        for _, container in containers_df.iterrows():
            # Obtener valores seguros del DataFrame
            container_id = container.get('id') or container.get('container_id')
            fill_percentage = container.get('fill_percentage', 0.0)
            zone = container.get('zone')
            
            # Verificar que tenemos los datos necesarios
            if container_id is not None and fill_percentage is not None:
                pred = self.predict_fill_time(
                    container_id=int(container_id),
                    current_fill=float(fill_percentage),
                    zone=zone
                )
                predictions.append(pred)
        
        # Ordenar por prioridad
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        predictions.sort(key=lambda x: priority_order.get(x.get('priority', 'LOW'), 2))
        
        return predictions