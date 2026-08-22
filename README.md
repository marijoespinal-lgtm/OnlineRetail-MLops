# Online Retail MLOps Architecture & Deployment

Este repositorio contiene la infraestructura end-to-end de MLOps para la segmentación de clientes y detección de deriva de datos (*Data Drift*) utilizando el dataset Online Retail.

---

## 🛠️ Estructura del Proyecto

```text
OnlineRetail-MLops/
├── .github/
│   └── workflows/
│       └── ci.yml             # Pipeline de Integración Continua (GitHub Actions)
├── src/
│   ├── api/
│   │   └── app.py             # Servidor de inferencia FastAPI
│   └── monitoring/
│       └── drift_detector.py  # Motor de monitoreo y cálculo de PSI
├── tests/
│   ├── test_api.py            # Pruebas unitarias para endpoints
│   └── test_monitoring.py     # Pruebas unitarias para detección de drift
├── Dockerfile                 # Contenedorización de la aplicación
├── pytest.ini                 # Configuración del entorno de pruebas
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación