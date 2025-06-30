import plotly.graph_objects as go
from typing import List, Dict, Union, Optional, Any
import requests
import os


def create_route_map(route_data: Union[Dict[str, Any], List[Dict[str, Any]]], containers: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
    """
    Crea un mapa con una ruta de recolección optimizada usando OpenRouteService para rutas reales
    Soporta múltiples formatos de entrada
    
    Args:
        route_data: Datos de la ruta (puede ser dict directo o resultado de optimización)
        containers: Lista de contenedores (opcional si ya están en route_data)
    
    Returns:
        Figura de Plotly con la ruta
    """
    fig = go.Figure()
    
    # Inicializar variables por defecto
    route_containers: List[Dict[str, Any]] = []
    depot_lat, depot_lon = -33.4119, -70.5241
    
    # Extraer datos según el formato
    if isinstance(route_data, dict) and 'routes' in route_data:
        print("Procesando formato de optimizador avanzado")
        # Formato del optimizador avanzado
        route = route_data['routes'][0]
        
        # Verificar que route sea un diccionario
        if not isinstance(route, dict):
            raise ValueError("Expected route to be a dictionary")
            
        route_containers = route.get('containers', [])
        
        # Verificar si ya tenemos coordenadas de ruta de OpenRouteService
        if 'route_coordinates' in route and route['route_coordinates']:
            print("Usando coordenadas pre-calculadas de OpenRouteService")
            # Usar coordenadas reales de OpenRouteService
            route_coords = route['route_coordinates']
            lats = [coord[1] for coord in route_coords]  # OpenRouteService devuelve [lon, lat]
            lons = [coord[0] for coord in route_coords]
            
            # Agregar línea de ruta real
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='none',
                name='Ruta Optimizada (OpenRouteService)'
            ))
        else:
            print("Obteniendo ruta de OpenRouteService en tiempo real")
            # Fallback: obtener ruta de OpenRouteService en tiempo real
            route_coords = get_openrouteservice_route_realtime(route_containers)
            if route_coords:
                print("Ruta OpenRouteService obtenida exitosamente")
                lats = [coord[1] for coord in route_coords]
                lons = [coord[0] for coord in route_coords]
                
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(width=4, color='blue'),
                    hoverinfo='none',
                    name='Ruta Optimizada (OpenRouteService)'
                ))
            else:
                print("OpenRouteService falló, usando línea recta")
                # Línea recta como último recurso
                add_straight_line_route(fig, route_containers)
        
    elif isinstance(route_data, dict) and 'container_ids' in route_data:
        # Formato del mapa original
        if containers is None:
            raise ValueError("containers parameter is required for this route format")
            
        route = route_data
        route_containers = sorted(
            [c for c in containers if c['container_id'] in route['container_ids']],
            key=lambda x: route['container_ids'].index(x['container_id'])
        )
        depot_lat, depot_lon = route['depot_lat'], route['depot_lon']
        
        # Obtener ruta de OpenRouteService
        route_coords = get_openrouteservice_route_realtime(route_containers, depot_lat, depot_lon)
        if route_coords:
            lats = [coord[1] for coord in route_coords]
            lons = [coord[0] for coord in route_coords]
            
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='none',
                name='Ruta Optimizada (OpenRouteService)'
            ))
        else:
            add_straight_line_route(fig, route_containers, depot_lat, depot_lon)
    
    else:
        # Formato del optimizador simple (routes.py actual)
        if isinstance(route_data, dict) and 'routes' in route_data:
            route = route_data['routes'][0]
        else:
            route = route_data
            
        # Asegurar que route es un diccionario
        if isinstance(route, dict):
            route_containers = route.get('containers', [])
            depot_info = route.get('depot_location', {})
            depot_lat = depot_info.get('lat', -33.4119)
            depot_lon = depot_info.get('lon', -70.5241)
        else:
            # Si route es una lista, usarlo directamente como contenedores
            route_containers = route if isinstance(route, list) else []
            depot_lat, depot_lon = -33.4119, -70.5241
        
        # Obtener ruta de OpenRouteService en tiempo real
        route_coords = get_openrouteservice_route_realtime(route_containers, depot_lat, depot_lon)
        if route_coords:
            lats = [coord[1] for coord in route_coords]
            lons = [coord[0] for coord in route_coords]
            
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='none',
                name='Ruta Optimizada (OpenRouteService)'
            ))
        else:
            add_straight_line_route(fig, route_containers, depot_lat, depot_lon)

    # Agregar marcador de depot
    fig.add_trace(go.Scattermapbox(
        lat=[depot_lat],
        lon=[depot_lon],
        mode='markers',
        marker=dict(size=20, color='black', symbol='star'),
        text="Depósito Central",
        hoverinfo='text',
        name='Depósito'
    ))

    # Agregar contenedores con numeración de paradas
    for i, container in enumerate(route_containers, 1):
        # Manejar diferentes formatos de contenedor
        lat = container.get('lat', container.get('latitude'))
        lon = container.get('lon', container.get('longitude'))
        fill_pct = container.get('fill_percentage', 0)
        container_id = container.get('container_id', container.get('id'))
        
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers+text',
            marker=dict(
                size=15,
                color='red' if fill_pct >= 90 else 'orange',
                opacity=0.9
            ),
            text=str(i),
            textposition="middle center",
            textfont=dict(color="white", size=10),
            hoverinfo='text',
            hovertext=f"Parada {i}: {container_id} ({fill_pct}%)",
            name=f"Parada {i}"
        ))

    # Configuración del mapa
    if route_containers:
        center_lat = sum(container.get('lat', container.get('latitude', 0)) for container in route_containers) / len(route_containers)
        center_lon = sum(container.get('lon', container.get('longitude', 0)) for container in route_containers) / len(route_containers)
    else:
        center_lat, center_lon = depot_lat, depot_lon
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=13
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600,
        showlegend=False,
        title=f"Ruta Optimizada - {len(route_containers)} contenedores"
    )
    
    return fig


def get_openrouteservice_route_realtime(containers: List[Dict[str, Any]], depot_lat: float = -33.4119, depot_lon: float = -70.5241) -> Optional[List[List[float]]]:
    """
    Obtiene coordenadas de ruta real desde OpenRouteService Directions API
    
    Args:
        containers: Lista de contenedores
        depot_lat: Latitud del depósito
        depot_lon: Longitud del depósito
    
    Returns:
        Lista de coordenadas [lon, lat] o None si hay error
    """
    if not containers:
        print("No hay contenedores para crear ruta")
        return None
    
    # Obtener API key de OpenRouteService
    openroute_api_key = os.getenv('OPENROUTESERVICE_API_KEY')
    if not openroute_api_key:
        print("OPENROUTESERVICE_API_KEY no está configurado")
        return None
        
    # Construir coordenadas (OpenRouteService usa formato [lon, lat])
    coordinates = [[depot_lon, depot_lat]]  # Inicio
    
    for container in containers:
        lat = container.get('lat', container.get('latitude'))
        lon = container.get('lon', container.get('longitude'))
        if lat and lon:
            coordinates.append([lon, lat])
        else:
            print(f"Contenedor sin coordenadas válidas: {container}")
    
    coordinates.append([depot_lon, depot_lat])  # Regreso
    
    print(f"Coordenadas para OpenRouteService: {coordinates}")
    
    # Verificar límite de waypoints (OpenRouteService permite hasta 50 waypoints en el plan gratuito)
    if len(coordinates) > 50:
        print(f"Demasiados waypoints ({len(coordinates)}). OpenRouteService permite máximo 50 en plan gratuito.")
        return None
    
    # Configurar llamada a OpenRouteService Directions API
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': openroute_api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    body = {
        "coordinates": coordinates,
        "format": "geojson",
        "instructions": False,  # No necesitamos instrucciones detalladas
        "geometry": True        # Queremos la geometría de la ruta
    }
    
    print(f"URL OpenRouteService: {url}")
    print(f"Payload: {body}")
    
    try:
        response = requests.post(url, json=body, headers=headers, timeout=15)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"OpenRouteService Response keys: {response_data.keys() if isinstance(response_data, dict) else 'No dict'}")
            
            # OpenRouteService devuelve un GeoJSON
            if 'features' in response_data and response_data['features']:
                geometry = response_data['features'][0]['geometry']
                if geometry['type'] == 'LineString':
                    route_coords = geometry['coordinates']
                    print(f"Coordenadas obtenidas de OpenRouteService: {len(route_coords)} puntos")
                    return route_coords
                else:
                    print(f"Tipo de geometría inesperada: {geometry['type']}")
            else:
                print("No se encontraron features en la respuesta de OpenRouteService")
                print(f"Respuesta completa: {response_data}")
        
        elif response.status_code == 401:
            print("API key de OpenRouteService inválida o expirada")
        elif response.status_code == 403:
            print("Acceso denegado - verifica tu API key y cuota")
        elif response.status_code == 404:
            print("Endpoint no encontrado - verifica la URL")
        elif response.status_code == 413:
            print("Payload demasiado grande - reduce el número de waypoints")
        elif response.status_code == 429:
            print("Límite de rate exceeded - espera antes de hacer otra solicitud")
        elif response.status_code == 500:
            print("Error interno del servidor de OpenRouteService")
        else:
            print(f"Error HTTP de OpenRouteService: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Timeout al conectar con OpenRouteService")
    except requests.exceptions.ConnectionError:
        print("Error de conexión con OpenRouteService")
    except requests.exceptions.RequestException as e:
        print(f"Error de solicitud con OpenRouteService: {e}")
    except Exception as e:
        print(f"Error inesperado al obtener ruta de OpenRouteService: {e}")
    
    return None


def add_straight_line_route(fig: go.Figure, containers: List[Dict[str, Any]], depot_lat: float = -33.4119, depot_lon: float = -70.5241) -> None:
    """
    Agrega línea recta como fallback cuando OpenRouteService no está disponible
    
    Args:
        fig: Figura de Plotly
        containers: Lista de contenedores
        depot_lat: Latitud del depósito
        depot_lon: Longitud del depósito
    """
    lats = [depot_lat]
    lons = [depot_lon]
    
    for container in containers:
        lat = container.get('lat', container.get('latitude'))
        lon = container.get('lon', container.get('longitude'))
        if lat and lon:
            lats.append(lat)
            lons.append(lon)
    
    lats.append(depot_lat)
    lons.append(depot_lon)
    
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=3, color='blue', dash='dash'),
        hoverinfo='none',
        name='Ruta (línea recta - aproximada)'
    ))


def check_openrouteservice_status() -> bool:
    """
    Verifica si OpenRouteService está disponible y funcionando
    
    Returns:
        True si está disponible, False en caso contrario
    """
    openroute_api_key = os.getenv('OPENROUTESERVICE_API_KEY')
    if not openroute_api_key:
        return False
    
    try:
        # Hacer una prueba simple con una ruta corta
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {
            'Accept': 'application/json',
            'Authorization': openroute_api_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        body = {
            "coordinates": [[-70.5241, -33.4119], [-70.5200, -33.4100]],
            "format": "geojson"
        }
        
        response = requests.post(url, json=body, headers=headers, timeout=5)
        return response.status_code == 200
        
    except Exception:
        return False


def get_openrouteservice_info() -> Dict[str, Any]:
    """
    Obtiene información sobre el estado y uso de OpenRouteService
    
    Returns:
        Diccionario con información del servicio
    """
    info = {
        "service": "OpenRouteService",
        "api_configured": bool(os.getenv('OPENROUTESERVICE_API_KEY')),
        "status": "unknown",
        "max_waypoints": 50,  # Límite del plan gratuito
        "rate_limit": "40 requests/minute",  # Límite del plan gratuito
        "website": "https://openrouteservice.org"
    }
    
    if info["api_configured"]:
        info["status"] = "available" if check_openrouteservice_status() else "unavailable"
    else:
        info["status"] = "not_configured"
    
    return info