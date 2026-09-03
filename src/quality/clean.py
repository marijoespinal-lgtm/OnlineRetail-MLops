import pandas as pd
from pathlib import Path

def clean_online_retail():
    # Ruta del archivo crudo
    raw_path = Path("data/raw/Online_Retail.xlsx")

    # Cargar dataset
    df = pd.read_excel(raw_path)

    print(f"Dimensiones iniciales: {df.shape}")

    # --- Conversión de fechas ---
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    invalid_dates = df["InvoiceDate"].isnull().sum()
    if invalid_dates > 0:
        print(f"Registros con fechas inválidas eliminados: {invalid_dates}")
        df = df[df["InvoiceDate"].notnull()]

    # --- Reglas de limpieza ---
    # 1. Eliminar duplicados
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicados eliminados: {dup_count}")

    # 2. Eliminar registros con Quantity <= 0
    qty_invalid = (df["Quantity"] <= 0).sum()
    df = df[df["Quantity"] > 0]
    print(f"Registros con Quantity <= 0 eliminados: {qty_invalid}")

    # 3. Eliminar registros con UnitPrice <= 0
    price_invalid = (df["UnitPrice"] <= 0).sum()
    df = df[df["UnitPrice"] > 0]
    print(f"Registros con UnitPrice <= 0 eliminados: {price_invalid}")

    # 4. Eliminar outliers extremos (ejemplo: Quantity > 10000 o UnitPrice > 10000)
    outliers = ((df["Quantity"] > 10000) | (df["UnitPrice"] > 10000)).sum()
    df = df[(df["Quantity"] <= 10000) & (df["UnitPrice"] <= 10000)]
    print(f"Outliers eliminados: {outliers}")

    # 5. Eliminar registros sin CustomerID
    no_customer = df["CustomerID"].isnull().sum()
    df = df[df["CustomerID"].notnull()]
    print(f"Registros sin CustomerID eliminados: {no_customer}")

    # 6. Eliminar registros sin Description
    no_description = df["Description"].isnull().sum()
    df = df[df["Description"].notnull()]
    print(f"Registros sin Description eliminados: {no_description}")

    # --- Guardar dataset limpio ---
    clean_path = Path("data/clean/online_retail_clean.csv")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)

    print(f"✅ Dataset limpio guardado en: {clean_path}")
    print(f"Dimensiones finales: {df.shape}")

if __name__ == "__main__":
    clean_online_retail()

#La explicación de cada paso de limpieza se encuentra argumentada en el 
#notebook "eda and cleaning.ipynb" 

