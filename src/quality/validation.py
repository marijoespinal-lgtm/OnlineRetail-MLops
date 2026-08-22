import pandas as pd

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Reglas de validación automática para asegurar calidad de datos.
    """
    # 0. Crear una copia y asegurar la conversión a tipo Datetime
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")

    # 1. Tasa de duplicados < 5%
    dup_rate = df.duplicated().mean()
    print(f"Tasa de duplicados: {dup_rate:.2%}")
    assert dup_rate < 0.05, f"❌ Duplicados excesivos: {dup_rate:.2%}"

    # 2. No debe haber CustomerID nulos
    null_customers = df["CustomerID"].isnull().sum()
    print(f"CustomerID nulos: {null_customers}")
    assert df["CustomerID"].notnull().all(), "❌ Hay CustomerID nulos"

    # 3. Fechas dentro del rango esperado (2009–2011)
    min_date, max_date = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    print(f"Rango de fechas: {min_date} → {max_date}")
    assert pd.Timestamp("2009-01-01") <= min_date, "❌ Fechas demasiado antiguas"
    assert max_date <= pd.Timestamp("2011-12-31 23:59:59"), "❌ Fechas fuera de rango"

    # 4. Precios positivos y razonables
    print(f"Precio máximo: {df['UnitPrice'].max()}")
    assert (df["UnitPrice"] > 0).all(), "❌ Hay precios <= 0"
    assert df["UnitPrice"].max() < 10000, "❌ Precio fuera de rango razonable"

    # 5. Cantidades positivas y razonables
    print(f"Cantidad máxima: {df['Quantity'].max()}")
    assert (df["Quantity"] > 0).all(), "❌ Hay cantidades <= 0"
    assert df["Quantity"].max() < 10000, "❌ Cantidad fuera de rango razonable"

    print("✅ Todas las validaciones pasaron correctamente")


if __name__ == "__main__":
    # Cargar CSV sin parse_dates en read_csv
    df = pd.read_csv("data/clean/online_retail_clean.csv")
    validate_dataset(df)

##Tasa de duplicados: 0.00%
CustomerID nulos: 0
Rango de fechas: 2010-01-12 08:26:00 → 2011-12-10 17:19:00
Precio máximo: 908.16
Cantidad máxima: 992
✅ Todas las validaciones pasaron correctamente










