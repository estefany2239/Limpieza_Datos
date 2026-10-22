import unicodedata
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def quitar_tildes(texto):
    """Quita tildes de un texto (á -> a, ñ -> n, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(texto))
        if unicodedata.category(c) != "Mn"
    )


# 1. EXPLORACIÓN INICIAL
df = pd.read_csv("datos_para_limpiar.csv")

print(df.head())          # primeras 5 filas
df.info()                 # tipos de datos
print(df.isnull().sum())  # nulos por columna

# 2. ELIMINAR COLUMNAS INNECESARIAS (ID y Notas)
df = df.drop(columns=["ID", "Notas"])

# 3. NORMALIZAR NOMBRES DE COLUMNAS
#    minúsculas, sin espacios y sin tildes
df.columns = [
    quitar_tildes(c).strip().lower().replace(" ", "_") for c in df.columns
]
print(df.columns.tolist())

# 4. CORREGIR ERRORES DE ESCRITURA EN 'ciudad'
#    Quita espacios, unifica mayúsculas/minúsculas y tildes
mapa_ciudades = {
    "bogota": "Bogotá",
    "medellin": "Medellín",
    "cali": "Cali",
}
ciudad_clave = df["ciudad"].apply(
    lambda x: quitar_tildes(x).strip().lower() if pd.notnull(x) else x
)
# Los valores que no coinciden con el mapa se conservan limpios
df["ciudad"] = ciudad_clave.map(mapa_ciudades).fillna(df["ciudad"].str.strip())

# 5. TRATAMIENTO DE VALORES NULOS
# Antes de calcular la media, 'edad' y 'salario' deben ser numéricas
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["salario"] = pd.to_numeric(df["salario"], errors="coerce")

# edad: media calculada solo con edades válidas (0 a 120)
edades_validas = df.loc[df["edad"].between(0, 120), "edad"]
df["edad"] = df["edad"].fillna(edades_validas.mean())

# salario: mediana
df["salario"] = df["salario"].fillna(df["salario"].median())

# ciudad y genero: moda
df["ciudad"] = df["ciudad"].fillna(df["ciudad"].mode()[0])
df["genero"] = df["genero"].fillna(df["genero"].mode()[0])

# 6. ASEGURAR TIPOS DE DATOS
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# 7. FILTRAR DATOS FUERA DE RANGO
df = df[df["edad"].between(0, 120)]
df = df[df["salario"] <= 1_000_000_000]

# 8. VARIABLE ORDINAL 'nivel' -> numérica
#    bajo=1, medio=2, alto=3; n/a o vacíos quedan como nulos
df["nivel"] = (
    df["nivel"].astype(str).str.strip().str.lower()
    .map({"bajo": 1, "medio": 2, "alto": 3})
)

# 9. CODIFICACIÓN DE CATEGÓRICAS (genero, ciudad)
#    drop_first=True evita la multicolinealidad
df = pd.get_dummies(df, columns=["genero", "ciudad"], drop_first=True)

# 10. ESCALADO DE 'edad' E 'ingresos' ENTRE 0 Y 1
# Se guardan gastos e ingresos originales para calcular utilidad con valores reales
ingresos_originales = df["ingresos"].copy()

scaler = MinMaxScaler()
df[["edad", "ingresos"]] = scaler.fit_transform(df[["edad", "ingresos"]])

# 11. ELIMINAR DUPLICADOS
df = df.drop_duplicates()
ingresos_originales = ingresos_originales.loc[df.index]

# 12. NUEVA VARIABLE: utilidad = ingresos - gastos
#     (con ingresos en su escala original, no la escalada)
df["utilidad"] = ingresos_originales - df["gastos"]

# 13. GUARDAR EL DATASET LIMPIO
df.to_csv("datos_limpios.csv", index=False)
print("Archivo datos_limpios.csv generado:", df.shape)