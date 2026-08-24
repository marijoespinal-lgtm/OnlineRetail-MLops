# Online Retail MLOps Pipeline 🚀

Infraestructura *end-to-end* de MLOps para la segmentación de clientes (Clustering RFM) y monitoreo de deriva de datos (*Data Drift*) en tiempo real con el dataset Online Retail.

---

## 🛠️ Estructura del Proyecto

```text
OnlineRetail-MLops/
├── .github/
│   └── workflows/
│       └── ci.yml             # Pipeline de CI/CD (GitHub Actions)
├── src/
│   ├── api/
│   │   └── app.py             # Servidor de inferencia FastAPI (/health, /predict, /drift)
│   └── monitoring/
│       └── drift_detector.py  # Motor de monitoreo y cálculo de PSI
├── tests/
│   ├── test_api.py            # Pruebas unitarias para los endpoints
│   └── test_monitoring.py     # Pruebas unitarias para la detección de drift
├── Dockerfile                 # Contenedorización de la aplicación
├── pytest.ini                 # Configuración de pytest
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación

## Pasos para iniciar 
1. Clonar e instalar dependencias
git clone [https://github.com/marijoespinal-lgtm/OnlineRetail-MLops.git](https://github.com/marijoespinal-lgtm/OnlineRetail-MLops.git)
cd OnlineRetail-MLops
pip install -r requirements.txt

2. Ejecutar la API
uvicorn src.api.app:app --reload --port 8000

🐳 Despliegue con Docker
# Construir la imagen
docker build -t online-retail-api .

# Ejecutar el contenedor
docker run -p 8000:8000 online-retail-api

🧪 Pruebas Unitarias y CI/CD
Para ejecutar las pruebas locales con pytest:
pytest

Método,Endpoint,Descripción
GET,/health,Verifica el estado del servicio y la carga del modelo/baseline
POST,/predict,Recibe métricas RFM y clasifica al cliente en un clúster
POST,/drift,Recibe registros recientes y calcula el PSI contra el dataset baseline

## 📌 Endpoints de la API

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| **GET** | `/health` | Verifica el estado del servicio y la carga del modelo/baseline |
| **POST** | `/predict` | Recibe métricas RFM y clasifica al cliente en un clúster |
| **POST** | `/drift` | Recibe registros recientes y calcula el PSI contra el dataset baseline |

## 📊 Monitoreo del Sistema (3 Dimensiones)

1. **Data Drift (Covariate Shift):** Monitoreado en tiempo real a través del endpoint `/drift` utilizando el cálculo del **Population Stability Index (PSI)** sobre las variables de entrada.
2. **Concept Drift:** Se evalúa periódicamente re-calculando la métrica de **Silhouette Score / Inercia** sobre lotes de datos de producción retenidos para detectar si los clústeres han perdido separación o significancia de negocio.
3. **System Performance:** Monitoreado a través del endpoint `/health`, verificando tiempo de respuesta (latencia) y disponibilidad del contenedor Docker.
