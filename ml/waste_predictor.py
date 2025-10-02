import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Any
from time import time
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


class WastePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'neural_network': MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
        }
        self.trained_models = {}
        self.feature_importances = {}
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
        
        # Don't drop categorical columns here - we'll handle them in train_models
        # Just remove non-feature columns
        df = df.drop(columns=['container_id', 'timestamp'], errors='ignore')
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Entrena modelos predictivos con validación cruzada"""
        # Preprocesamiento
        df_features = self.create_features(df)
        
        # One-Hot Encoding
        categorical_cols = ['area_type', 'zone']
        for col in categorical_cols:
            if col in df_features.columns:
                dummies = pd.get_dummies(df_features[col], prefix=col)
                df_features = pd.concat([df_features, dummies], axis=1)
                df_features = df_features.drop(columns=[col], errors='ignore')
        
        # Selección de features
        feature_cols = [col for col in df_features.columns 
                    if col not in ['container_id', 'timestamp', 'fill_percentage']]
        X = df_features[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df_features['fill_percentage']
        
        # Inicializar atributos
        self.feature_columns = feature_cols
        self.feature_importances = {}  # Asegurar que está inicializado
        cv_metrics = {}
        
        for name, model in self.models.items():
            print(f"Entrenando {name}...")
            start_time = time()
            
            # Validación cruzada
            kf = KFold(n_splits=5)
            maes, r2s, train_times = [], [], []
            
            # Listas para almacenar importancias por fold
            fold_importances = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Entrenamiento
                train_start = time()
                model.fit(X_train, y_train)
                train_times.append(time() - train_start)
                
                # Predicción
                preds = model.predict(X_test)
                maes.append(mean_absolute_error(y_test, preds))
                r2s.append(r2_score(y_test, preds))
                
                # Recolectar importancias por fold (si aplica)
                if hasattr(model, 'feature_importances_'):
                    fold_importances.append(model.feature_importances_)
            
            # Guardar métricas
            cv_metrics[name] = {
                'MAE_promedio': round(np.mean(maes), 2),
                'MAE_std': round(np.std(maes), 2),
                'R2_promedio': round(np.mean(r2s), 3),
                'Tiempo_entrenamiento_promedio': round(np.mean(train_times), 2),
                'n_features': len(feature_cols)
            }
            
            # Guardar modelo y promediar importancias
            self.trained_models[name] = model
            
            if fold_importances:
                self.feature_importances[name] = {
                    'features': feature_cols,
                    'importance': np.mean(fold_importances, axis=0).tolist(),
                    'std': np.std(fold_importances, axis=0).tolist()
                }
            elif hasattr(model, 'coef_'):  # Para modelos lineales
                self.feature_importances[name] = {
                    'features': feature_cols,
                    'importance': abs(model.coef_[0]).tolist(),  # Valor absoluto
                    'std': None
                }
        
        return cv_metrics
    
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