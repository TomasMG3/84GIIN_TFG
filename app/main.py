from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import pandas as pd
from contextlib import asynccontextmanager
import csv
from datetime import datetime
from dotenv import load_dotenv

from app.database import engine, get_db
from app import models, schemas, physics_simulator
from app.api import containers, routes, predictions
from app.api import economic_impact
from app.api import physics_routes



# Crear tablas
models.Base.metadata.create_all(bind=engine)
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar la app
    db = next(get_db())
    cargar_contenedores_desde_csv(db)

    yield 

app = FastAPI(
    title="Sistema de Gestión de Residuos - Las Condes",
    description="Sistema IoT para optimización de recolección de residuos urbanos",
    version="1.0.0",
    lifespan=lifespan
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(containers.router, prefix="/api/v1/containers", tags=["containers"])
app.include_router(routes.router, prefix="/api/v1/routes", tags=["routes"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(economic_impact.router, prefix="/api")
app.include_router(physics_routes.router, prefix="/api/v1/physics", tags=["physics-simulation"])

@app.get("/")
async def root():
    return {
        "message": "Sistema de Gestión de Residuos - Las Condes",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "OK"
    }

def cargar_contenedores_desde_csv(db: Session):
    db.query(models.Container).delete()
    db.commit()
    
    with open("data/containers.csv", mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        contenedores = []
        for row in reader:
            # CALCULAR CORRECTAMENTE EL PORCENTAJE
            capacity = float(row["capacity"])
            current_level = float(row["current_level"])
            fill_percentage = (current_level / capacity) * 100
            
            # Limitar a máximo 100%
            if fill_percentage > 100:
                fill_percentage = 100.0
                
            contenedor = models.Container(
                container_id=row["container_id"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                address=row["address"],
                zone=row.get("zone"),
                capacity=capacity,
                current_level=current_level,
                fill_percentage=fill_percentage,  # VALOR CORREGIDO
                temperature=float(row["temperature"]) if row.get("temperature") else None,
                battery_level=float(row["battery_level"]) if row.get("battery_level") else None,
                is_active=row["is_active"].strip().lower() == "true",
                last_emptied=None,
                last_update=datetime.utcnow(),
            )
            contenedores.append(contenedor)

        db.add_all(contenedores)
        db.commit()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

@app.get("/containers")
def get_containers():
    df = pd.read_csv("data/containers.csv")
    return df.to_dict(orient="records")

@app.post("/api/v1/routes/optimize-with-ai")
async def ai_optimization_endpoint(
    min_fill: float = 70.0,
    use_ai: bool = True,
    db: Session = Depends(get_db)
):
    """Endpoint para optimización con IA"""
    return await optimize_routes_with_ai(
        min_fill_threshold=min_fill,
        use_lstm=use_ai,
        db=db
    )

@app.get("/api/v1/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Estadísticas generales del sistema"""
    total_containers = db.query(models.Container).count()
    active_containers = db.query(models.Container).filter(models.Container.is_active == True).count()
    high_fill_containers = db.query(models.Container).filter(models.Container.fill_percentage > 80).count()
    
    return {
        "total_containers": total_containers,
        "active_containers": active_containers,
        "high_fill_containers": high_fill_containers,
        "system_status": "operational"
    }

    
app.include_router(containers.router, prefix="/api/v1/containers", tags=["containers"])
app.include_router(routes.router, prefix="/api/v1/routes", tags=["routes"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(economic_impact.router, prefix="/api/v1/economic-impact", tags=["economic-impact"])
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
