import pytest
import pandas as pd
from datetime import datetime, timedelta
from ml.waste_predictor import WastePredictor
import numpy as np

@pytest.fixture
def sample_data():
    # Generar datos de prueba más realistas
    base_date = datetime.now()
    hours = 24*7
    timestamps = [base_date - timedelta(hours=h) for h in range(hours, 0, -1)]
    
    data = {
        'container_id': ['TEST-001'] * hours,
        'timestamp': timestamps,
        'fill_percentage': np.clip(np.linspace(0, 100, hours) + np.random.normal(0, 5, hours), 0, 100),
        'zone': ['ZONE_A'] * hours,
        'temperature': 20 + 5 * np.sin(np.linspace(0, 2*np.pi, hours))
    }
    
    # Crear variación entre zonas
    data['zone'][hours//2:] = ['ZONE_B'] * (hours//2)
    data['fill_percentage'][hours//2:] += 10  # Zona B se llena más rápido
    
    return pd.DataFrame(data)

def test_waste_predictor_train(sample_data):
    predictor = WastePredictor()
    results = predictor.train_models(sample_data)
    
    assert 'random_forest' in results
    assert results['random_forest']['mae'] > 0
    assert results['random_forest']['r2'] <= 1

def test_waste_predictor_predict(sample_data):
    predictor = WastePredictor()
    predictor.train_models(sample_data)
    
    prediction = predictor.predict_fill_time(
        container_id=1,
        current_fill=50.0,
        zone="ZONE_A"
    )
    
    assert 'hours_to_full' in prediction
    assert 'priority' in prediction
    assert prediction['priority'] in ['HIGH', 'MEDIUM', 'LOW']