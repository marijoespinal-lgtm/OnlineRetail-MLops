import os
import requests
import pandas as pd

RAW_DATA_DIR = "data/raw/"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

def download_excel(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    file_path = os.path.join(dest_dir, os.path.basename(url))

    print(f"Descargando archivo desde {url}...")
    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"Archivo guardado en {file_path}")
    return file_path

def validate_excel(file_path: str):
    print(f"Validando archivo: {file_path}")
    try:
        df = pd.read_excel(file_path, nrows=5)
        print("✅ Archivo válido, primeras filas:")
        print(df.head())
    except Exception as e:
        raise ValueError(f"El archivo no se pudo leer correctamente: {e}")

if __name__ == "__main__":
    file_path = download_excel(DATA_URL, RAW_DATA_DIR)
    validate_excel(file_path)
