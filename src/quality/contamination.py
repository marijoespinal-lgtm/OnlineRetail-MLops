import pandas as pd
import numpy as np
from pathlib import Path

def pollute_dataset(input_path: str, output_path: str) -> None:
    path_in = Path(input_path)
    if not path_in.exists():
        raise FileNotFoundError(f"❌ No se encontró el dataset limpio en: {input_path}")

    df = pd.read_csv(path_in)
    print(f"📊 Dataset limpio cargado ({len(df)} filas). Contaminando...")

    # Muestra representativa para la simulación
    df_corrupted = df.sample(n=min(1000, len(df)), random_state=42).copy()

    # Convertir UnitPrice a tipo objeto para permitir cadenas de texto
    df_corrupted["UnitPrice"] = df_corrupted["UnitPrice"].astype(object)

    # 1. Inyección de valores inválidos en UnitPrice (negativos y cadenas)
    df_corrupted.loc[df_corrupted.index[0:5], "UnitPrice"] = -15.50
    df_corrupted.loc[df_corrupted.index[5:10], "UnitPrice"] = "PRECIO_INVALIDO"

    # 2. Inyección de cantidades negativas
    df_corrupted.loc[df_corrupted.index[10:15], "Quantity"] = -50

    # 3. Fechas fuera del rango esperado (año 2005)
    df_corrupted.loc[df_corrupted.index[15:20], "InvoiceDate"] = "2005-01-01 10:00:00"

    # 4. Nulos en identificador de cliente
    df_corrupted.loc[df_corrupted.index[20:30], "CustomerID"] = np.nan

    # 5. Inducir exceso de duplicados (> 15% del total del lote)
    duplicated_rows = df_corrupted.iloc[0:200].copy()
    df_corrupted = pd.concat([df_corrupted, duplicated_rows], ignore_index=True)

    # Guardar lote contaminado
    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df_corrupted.to_csv(path_out, index=False)

    print(f"⚠️ Dataset contaminado guardado con éxito en: {output_path}")

if __name__ == "__main__":
    pollute_dataset(
        input_path="data/clean/online_retail_clean.csv",
        output_path="data/corrupted/online_retail_corrupted.csv"
    )
    Dataset limpio cargado (392557 filas). Contaminando...
⚠️ Dataset contaminado guardado con éxito en: data/corrupted/online_retail_corrupted.csv