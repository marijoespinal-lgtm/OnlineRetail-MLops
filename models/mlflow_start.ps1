# mlflow_start.ps1
# Script para activar entorno y abrir MLflow UI

# Activar entorno virtual
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# Levantar MLflow en puerto 5500 con SQLite como backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5500

# Abrir navegador automáticamente
Start-Process "http://127.0.0.1:5500"
