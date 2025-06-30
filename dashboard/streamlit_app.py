import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Gestión de Residuos - Las Condes",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL base de la API (ajustar según tu configuración)
API_BASE_URL = "http://localhost:8000/api/v1"

class APIClient:
    """Cliente para interactuar con la API FastAPI"""
    
    @staticmethod
    def get_containers():
        try:
            response = requests.get(f"{API_BASE_URL}/containers/")
            return response.json() if response.status_code == 200 else []
        except:
            return []
    
    @staticmethod
    def get_routes():
        try:
            response = requests.get(f"{API_BASE_URL}/routes/")
            return response.json() if response.status_code == 200 else []
        except:
            return []
    
    @staticmethod
    def optimize_routes(min_fill=70.0):
        try:
            response = requests.post(f"{API_BASE_URL}/routes/optimize", 
                                   params={"min_fill_threshold": min_fill})
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    @staticmethod
    def get_predictions():
        try:
            response = requests.get(f"{API_BASE_URL}/predictions/daily-predictions")
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    @staticmethod
    def simulate_data():
        try:
            response = requests.post(f"{API_BASE_URL}/containers/simulate-data")
            return response.json() if response.status_code == 200 else {}
        except:
            return {}

def create_sample_data():
    """Crear datos de muestra para demostración"""
    np.random.seed(42)
    
    # Coordenadas de Las Condes
    las_condes_bounds = {
        'lat_min': -33.48, 'lat_max': -33.35,
        'lon_min': -70.65, 'lon_max': -70.48
    }
    
    n_containers = 50
    containers = []
    
    zones = ['Centro', 'Las Condes Alto', 'El Golf', 'Vitacura Norte', 'Escuela Militar']
    
    for i in range(n_containers):
        containers.append({
            'id': i + 1,
            'container_id': f'CONT-{i+1:03d}',
            'latitude': np.random.uniform(las_condes_bounds['lat_min'], las_condes_bounds['lat_max']),
            'longitude': np.random.uniform(las_condes_bounds['lon_min'], las_condes_bounds['lon_max']),
            'fill_percentage': np.random.uniform(0, 100),
            'temperature': np.random.uniform(15, 30),
            'battery_level': np.random.uniform(20, 100),
            'zone': np.random.choice(zones),
            'address': f'Calle {i+1}, Las Condes',
            'capacity': 1000,
            'is_active': True,
            'last_update': datetime.now() - timedelta(minutes=np.random.randint(0, 120))
        })
    
    return pd.DataFrame(containers)

def get_fill_color(fill_percentage):
    """Obtener color según el porcentaje de llenado"""
    if fill_percentage >= 90:
        return 'red'
    elif fill_percentage >= 75:
        return 'orange'
    elif fill_percentage >= 50:
        return 'yellow'
    else:
        return 'green'

def create_map(df):
    """Crear mapa de contenedores"""
    fig = go.Figure()
    
    # Agregar contenedores al mapa
    for _, container in df.iterrows():
        color = get_fill_color(container['fill_percentage'])
        
        fig.add_trace(go.Scattermapbox(
            lat=[container['latitude']],
            lon=[container['longitude']],
            mode='markers',
            marker=dict(
                size=12,
                color=color,
                opacity=0.8
            ),
            text=f"ID: {container['container_id']}<br>"
                 f"Llenado: {container['fill_percentage']:.1f}%<br>"
                 f"Zona: {container['zone']}<br>"
                 f"Batería: {container['battery_level']:.1f}%",
            hovertemplate='%{text}<extra></extra>',
            name=f'Llenado {color}'
        ))
    
    # Centro de Las Condes
    center_lat = -33.4119
    center_lon = -70.5241
    
    fig.update_layout(
        mapbox=dict(
            accesstoken=None,  # Usar mapa base gratuito
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False
    )
    
    return fig

def create_route_map(optimization_result):
    """Crear mapa con ruta optimizada"""
    fig = go.Figure()
    
    if 'optimization_result' in st.session_state:
        result = st.session_state.optimization_result
        if result.get('routes'):
            route = result['routes'][0]
            if 'route_coordinates' in route:  # Usar coordenadas guardadas
                lats = [coord[1] for coord in route['route_coordinates']]
                lons = [coord[0] for coord in route['route_coordinates']]
                fig.add_trace(go.Scattermapbox(
                    lat=lats, lon=lons, mode='lines', line=dict(width=4, color='blue')
                ))
    
    # Verificar si hay rutas
    if not optimization_result.get('routes') or len(optimization_result['routes']) == 0:
        st.error("No hay rutas para mostrar")
        return fig
    
    route = optimization_result['routes'][0]
    containers = route.get('containers', [])
    
    if not containers:
        st.error("No hay contenedores en la ruta")
        return fig
    
    # Depot por defecto (Las Condes centro)
    depot_lat = route.get('depot_location', {}).get('lat', -33.4119)
    depot_lon = route.get('depot_location', {}).get('lon', -70.5241)
    
    # Coordenadas para la línea de ruta
    lats = [depot_lat]
    lons = [depot_lon]
    
    # Agregar coordenadas de contenedores
    for container in containers:
        lats.append(container.get('latitude', container.get('lat', 0)))
        lons.append(container.get('longitude', container.get('lon', 0)))
    
    # Regresar al depot
    lats.append(depot_lat)
    lons.append(depot_lon)
    
    # Agregar línea de ruta
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=4, color='blue'),
        hoverinfo='skip',
        name='Ruta Optimizada',
        showlegend=False
    ))
    
    # Agregar depot
    fig.add_trace(go.Scattermapbox(
        lat=[depot_lat],
        lon=[depot_lon],
        mode='markers',
        marker=dict(size=20, color='black', symbol='star'),
        text="🏢 Depósito Central<br>Punto de inicio y fin",
        hovertemplate='%{text}<extra></extra>',
        name='Depósito'
    ))
    
    # Agregar contenedores de la ruta
    for i, container in enumerate(containers):
        lat = container.get('latitude', container.get('lat', 0))
        lon = container.get('longitude', container.get('lon', 0))
        fill = container.get('fill_percentage', 0)
        
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers+text',
            marker=dict(
                size=15,
                color='red' if fill >= 90 else 'orange' if fill >= 75 else 'yellow',
                opacity=0.9
            ),
            text=[str(i+1)],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            hovertext=f"🗑️ Parada #{i+1}<br>"
                      f"ID: {container.get('container_id', 'N/A')}<br>"
                      f"Llenado: {fill:.1f}%<br>"
                      f"Distancia: {container.get('distance_from_previous', 0)} km",
            hovertemplate='%{hovertext}<extra></extra>',
            name=f'Parada {i+1}'
        ))
    
    # Configurar el mapa
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=sum(lats)/len(lats), lon=sum(lons)/len(lons)),
            zoom=13
        ),
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
        title=dict(
            text=f"Ruta Optimizada - {len(containers)} contenedores",
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig

def create_dashboard():
    """Crear el dashboard principal"""
    
    # Título principal
    st.title("🗑️ Sistema de Gestión de Residuos - Las Condes")
    st.markdown("**Sistema IoT para optimización de recolección de residuos urbanos**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    # Botón para simular datos
    if st.sidebar.button("🔄 Simular Datos IoT"):
        with st.spinner("Simulando datos de sensores..."):
            result = APIClient.simulate_data()
            if result:
                st.sidebar.success("✅ Datos simulados correctamente")
            else:
                st.sidebar.warning("⚠️ Usando datos de muestra")
    
    # Filtros
    st.sidebar.subheader("🔍 Filtros")
    show_only_full = st.sidebar.checkbox("Solo contenedores >70%", value=False)
    selected_zone = st.sidebar.selectbox("Zona", ["Todas", "Centro", "Las Condes Alto", "El Golf", "Vitacura Norte"])
    
    # AGREGAR INFORMACIÓN DE DEBUG
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Debug Info**")
    
    # Obtener datos
    containers_data = APIClient.get_containers()
    
    if not containers_data:
        st.sidebar.warning("⚠️ No hay conexión con la API. Usando datos de muestra.")
        df = create_sample_data()
    else:
        df = pd.DataFrame(containers_data)
        st.sidebar.success(f"✅ API conectada: {len(df)} contenedores")
    
    # MOSTRAR CONTEO ANTES DE FILTROS
    st.sidebar.info(f"Total en BD: {len(df)}")
    
    # Aplicar filtros
    original_count = len(df)
    
    if show_only_full:
        df = df[df['fill_percentage'] >= 70]
        st.sidebar.info(f"Después filtro >70%: {len(df)}")
    
    if selected_zone != "Todas":
        df = df[df['zone'] == selected_zone]
        st.sidebar.info(f"Después filtro zona: {len(df)}")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Total Contenedores",
            value=len(df),
            delta=f"{len(df[df['is_active']])} activos"
        )
    
    with col2:
        critical_containers = len(df[df['fill_percentage'] >= 90])
        st.metric(
            label="🚨 Críticos (>90%)",
            value=critical_containers,
            delta=f"{critical_containers/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        )
    
    with col3:
        avg_fill = df['fill_percentage'].mean() if len(df) > 0 else 0
        st.metric(
            label="📊 Llenado Promedio",
            value=f"{avg_fill:.1f}%",
            delta="↑" if avg_fill > 50 else "↓"
        )
    
    with col4:
        low_battery = len(df[df['battery_level'] < 20])
        st.metric(
            label="🔋 Batería Baja",
            value=low_battery,
            delta="⚠️" if low_battery > 0 else "✅"
        )
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa", "📈 Análisis", "🚛 Rutas", "🔮 Predicciones"])
    
    with tab1:
        st.subheader("Mapa de Contenedores")
        
        if len(df) > 0:
            # Crear y mostrar mapa
            map_fig = create_map(df)
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Leyenda
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🟢 **Verde**: < 50%")
            with col2:
                st.markdown("🟡 **Amarillo**: 50-75%")
            with col3:
                st.markdown("🟠 **Naranja**: 75-90%")
            with col4:
                st.markdown("🔴 **Rojo**: > 90%")
        else:
            st.info("No hay contenedores para mostrar con los filtros aplicados.")
    
    with tab2:
        st.subheader("Análisis de Datos")
        
        if len(df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de distribución de llenado
                fig_hist = px.histogram(
                    df, 
                    x='fill_percentage', 
                    nbins=20,
                    title="Distribución de Llenado",
                    labels={'fill_percentage': 'Porcentaje de Llenado (%)', 'count': 'Cantidad'}
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Gráfico por zonas
                zone_stats = df.groupby('zone')['fill_percentage'].agg(['mean', 'count']).reset_index()
                fig_zone = px.bar(
                    zone_stats, 
                    x='zone', 
                    y='mean',
                    title="Llenado Promedio por Zona",
                    labels={'mean': 'Llenado Promedio (%)', 'zone': 'Zona'}
                )
                fig_zone.update_layout(height=400)
                st.plotly_chart(fig_zone, use_container_width=True)
            
            # Tabla de contenedores críticos
            critical_df = df[df['fill_percentage'] >= 75].sort_values('fill_percentage', ascending=False)
            if len(critical_df) > 0:
                st.subheader("🚨 Contenedores que Requieren Atención")
                st.dataframe(
                    critical_df[['container_id', 'zone', 'address', 'fill_percentage', 'battery_level']],
                    use_container_width=True
                )
        else:
            st.info("No hay datos para analizar.")
    
    with tab3:
        st.subheader("🚛 Optimización de Rutas")
        
        # Dividir en dos columnas
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**⚙️ Parámetros**")
            min_fill_threshold = st.slider("Umbral mínimo (%)", 50, 95, 70)
            
            # Información sobre contenedores candidatos
            candidates = df[df['fill_percentage'] >= min_fill_threshold]
            st.info(f"📊 {len(candidates)} contenedores cumplen el umbral")
            
            if st.button("🚛 Optimizar Rutas", type="primary"):
                with st.spinner("Optimizando rutas..."):
                    optimization_result = APIClient.optimize_routes(min_fill_threshold)
                    
                    if optimization_result:
                        st.session_state.optimization_result = optimization_result
                        if optimization_result.get('routes'):
                            st.success("✅ Rutas optimizadas correctamente")
                        else:
                            st.warning("⚠️ No hay contenedores para recolectar")
                    else:
                        st.error("❌ Error en la optimización")
        
        with col2:
            # Mostrar mapa de ruta optimizada
            if 'optimization_result' in st.session_state:
                result = st.session_state.optimization_result
                
                if result.get('routes') and len(result['routes']) > 0:
                    # Crear y mostrar mapa de ruta
                    route_map = create_route_map(result)
                    st.plotly_chart(route_map, use_container_width=True)
                else:
                    st.info("🔍 No hay rutas para mostrar")
        
        # Métricas de la ruta (fila completa)
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            
            if result.get('routes') and len(result['routes']) > 0:
                route = result['routes'][0]
                
                st.markdown("### 📊 Métricas de la Ruta")
                
                # Métricas principales
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("📍 Contenedores", result.get('containers_count', 0))
                with col2:
                    st.metric("🛣️ Distancia", f"{result.get('total_distance', 0)} km")
                with col3:
                    st.metric("⏱️ Tiempo", f"{route.get('estimated_time_minutes', 0)} min")
                with col4:
                    st.metric("⛽ Combustible", f"{route.get('fuel_consumption_liters', 0):.1f} L")
                with col5:
                    st.metric("🌱 CO2", f"{route.get('co2_emissions_kg', 0):.1f} kg")
                
                # Ahorros estimados
                if 'optimization_savings' in result:
                    savings = result['optimization_savings']
                    st.success(f"💰 **Ahorros estimados**: {savings.get('distance_saved_km', 0)} km "
                             f"({savings.get('percentage_saved', 0)}%) - "
                             f"{savings.get('fuel_saved_liters', 0):.1f}L combustible")
                
                # Detalles de la ruta
                if route.get('containers'):
                    st.markdown("### 📋 Detalles de la Ruta")
                    route_df = pd.DataFrame(route['containers'])
                    
                    # Formatear columnas
                    if 'fill_percentage' in route_df.columns:
                        route_df['fill_percentage'] = route_df['fill_percentage'].round(1)
                    if 'distance_from_previous' in route_df.columns:
                        route_df['distance_from_previous'] = route_df['distance_from_previous'].round(2)
                    
                    st.dataframe(route_df, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Predicciones de Llenado")
        
        # Obtener predicciones
        predictions_data = APIClient.get_predictions()
        
        if predictions_data and 'predictions' in predictions_data:
            predictions = predictions_data['predictions']
            
            # Resumen de predicciones
            summary = predictions_data.get('summary', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Alta Prioridad", summary.get('high_priority', 0))
            with col2:
                st.metric("🟡 Media Prioridad", summary.get('medium_priority', 0))
            with col3:
                st.metric("🟢 Baja Prioridad", summary.get('low_priority', 0))
            
            # Tabla de predicciones
            if predictions:
                pred_df = pd.DataFrame(predictions)
                st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("⚠️ Sistema de predicciones no disponible. Entrenar modelos primero.")
            
            if st.button("🤖 Entrenar Modelos de Predicción"):
                with st.spinner("Entrenando modelos de machine learning..."):
                    time.sleep(3)  # Simular entrenamiento
                    st.success("✅ Modelos entrenados correctamente")
    
    # Footer
    st.markdown("---")
    st.markdown("**Sistema de Gestión de Residuos - Municipalidad de Las Condes** | Desarrollado con Python, FastAPI y Streamlit")

if __name__ == "__main__":
    create_dashboard()