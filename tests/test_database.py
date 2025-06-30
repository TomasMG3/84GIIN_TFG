import pytest
from app.database import Base, get_db
from app.models import Container
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy.exc import IntegrityError

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="module")
def test_db():
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

def test_container_model(test_db):
    container = Container(
        container_id="TEST-001",
        latitude=-33.4119,
        longitude=-70.5241,
        address="Test Address",
        zone="TEST_ZONE",
        capacity=1000.0,
        current_level=0.0,
        fill_percentage=0.0,
        temperature=20.0,
        battery_level=100.0,
        is_active=True,
        last_emptied=None,
        last_update=datetime.utcnow()
    )
    
    test_db.add(container)
    try:
        test_db.commit()
    except IntegrityError:
        test_db.rollback()
    
    assert container.id is not None
    assert container.is_active is True
    assert container.fill_percentage == 0.0

def test_container_relationships(test_db):
    container = Container(
        container_id="TEST-002",
        latitude=-33.4119,
        longitude=-70.5241
    )
    test_db.add(container)
    try:
        test_db.commit()
    except IntegrityError:
        test_db.rollback()
    
    assert container.sensor_readings == []
    assert container.collection_events == []