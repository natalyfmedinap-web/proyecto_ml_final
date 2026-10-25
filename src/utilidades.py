"""Caché de resultados: evita recalcular experimentos lentos al volver a correr un notebook."""
from pathlib import Path
import pandas as pd


def cache_csv(ruta, funcion, recalcular=False, **kw):
    """Si `ruta` existe la lee; si no, ejecuta `funcion` y guarda el DataFrame resultante."""
    ruta = Path(ruta)
    if ruta.exists() and not recalcular:
        return pd.read_csv(ruta, index_col=0)
    df = funcion(**kw)
    ruta.parent.mkdir(exist_ok=True)
    df.to_csv(ruta)
    return df
