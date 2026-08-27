# run_pipeline.ps1
# ------------------
# Automatiza lo que antes hacias a mano en 2 terminales:
#   1. Levanta el servidor de MLflow en una ventana nueva
#   2. Espera a que este listo para recibir conexiones
#   3. Corre train.py automaticamente
#
# Uso (desde la raiz del repo, con el venv ya activado o no -- no importa):
#   .\run_pipeline.ps1

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

Write-Host "Levantando el servidor de MLflow en una ventana nueva..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "& '$venvActivate'; mlflow server --host 127.0.0.1 --port 5000 --default-artifact-root ./mlartifacts"
) -WorkingDirectory $repoRoot

Write-Host "Esperando a que el servidor responda en http://127.0.0.1:5000 ..." -ForegroundColor Cyan

$maxIntentos = 30
$listo = $false
for ($i = 0; $i -lt $maxIntentos; $i++) {
    Start-Sleep -Seconds 2
    try {
        $respuesta = Invoke-WebRequest -Uri "http://127.0.0.1:5000" -UseBasicParsing -TimeoutSec 2
        if ($respuesta.StatusCode -eq 200) {
            $listo = $true
            break
        }
    } catch {
        # Todavia no esta listo, seguimos intentando
    }
}

if (-not $listo) {
    Write-Host "El servidor no respondio despues de $($maxIntentos * 2) segundos. Revisa la ventana del servidor por errores." -ForegroundColor Red
    exit 1
}

Write-Host "Servidor listo. Corriendo train.py ..." -ForegroundColor Green

& $venvPython (Join-Path $repoRoot "src\models\train.py")

Write-Host "`nListo. El servidor de MLflow sigue corriendo en la otra ventana -- abre http://127.0.0.1:5000 para revisar los resultados." -ForegroundColor Green