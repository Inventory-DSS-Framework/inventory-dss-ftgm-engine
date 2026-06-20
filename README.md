# Inventory DSS FTGM Engine

Servicio analítico externo de la plataforma **Inventory Optimization DSS Platform**.

Este repositorio contiene el motor de predicción basado en el **Fourier Time-Varying Grey Model (FTGM)**. Su objetivo es recibir series temporales preparadas desde el backend, ejecutar el modelo predictivo y devolver pronósticos de demanda junto con métricas de evaluación.

## Rol dentro del sistema

| Repositorio                 | Responsabilidad                            |
| --------------------------- | ------------------------------------------ |
| `inventory-dss-web`         | Frontend Next.js                           |
| `inventory-dss-api`         | Backend FastAPI monolito modular hexagonal |
| `inventory-dss-ftgm-engine` | Motor analítico FTGM desacoplado           |
| `inventory-dss-infra`       | Infraestructura y despliegue               |
| `inventory-dss-docs`        | Documentación académica y arquitectónica   |

## Responsabilidad principal

Este servicio se encarga únicamente de:

* Recibir series temporales preparadas.
* Ejecutar FTGM.
* Aplicar componente Fourier.
* Manejar parámetros dinámicos.
* Ejecutar validación rolling-origin futura.
* Calcular métricas MAE, RMSE y MAPE.
* Comparar contra modelos base futuros.
* Devolver resultados al backend.

## Lo que este servicio no debe hacer

Este servicio no debe manejar:

* Usuarios.
* Empresas.
* Roles.
* Permisos.
* Inventario transaccional.
* Ventas como entidad de negocio.
* Pagos.
* Suscripciones.
* Dashboards.
* Reportes del sistema.
* Notificaciones del sistema.

## Relación con el backend

El backend `inventory-dss-api` consume este servicio mediante HTTP/REST/JSON a través de un FTGM Adapter.

```text
inventory-dss-api
        ↓
FTGM Adapter
        ↓
inventory-dss-ftgm-engine
        ↓
Forecast + Metrics
```

## Endpoints conceptuales

```text
GET  /api/v1/health
POST /api/v1/forecast
```

## Estado actual

Este repositorio contiene una estructura base funcional y un placeholder del modelo FTGM. La implementación matemática real será desarrollada en fases posteriores.
