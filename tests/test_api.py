import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)

def test_create_container(client):
    response = client.post(
        "/api/v1/containers/",
        json={
            "container_id": "TEST-001",
            "latitude": -33.4119,
            "longitude": -70.5241,
            "address": "Test Address",
            "zone": "TEST_ZONE",
            "temperature": 22.5,
            "battery_level": 85.0
        }
    )
    assert response.status_code == 200
    assert response.json()["container_id"] == "TEST-001"

def test_get_containers(client):
    response = client.get("/api/v1/containers/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_optimize_routes(client):
    # Crear un contenedor con los campos requeridos
    create_resp = client.post(
        "/api/v1/containers/",
        json={
            "container_id": "TEST-ROUTE",
            "latitude": -33.4119,
            "longitude": -70.5241,
            "address": "Sample Address",
            "zone": "Sample Zone",
            "temperature": 25.0,
            "battery_level": 90.0
        }
    )
    assert create_resp.status_code == 200
    container_data = create_resp.json()
    container_id = container_data["id"]

    # Actualizar el contenedor para establecer un porcentaje de llenado >= 70
    update_resp = client.put(
        f"/api/v1/containers/{container_id}",
        json={
            "current_level": 800.0,
            "fill_percentage": 80.0
        }
    )
    assert update_resp.status_code == 200

    # 3. Ejecutar optimización
    response = client.post("/api/v1/routes/optimize")
    assert response.status_code == 200
    data = response.json()

    # 4. Verificar que se incluyó al menos una ruta
    assert "routes" in data
    assert len(data["routes"]) > 0
    assert "route_id" in data