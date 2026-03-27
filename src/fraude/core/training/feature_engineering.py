import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def haversine_np(lon1, lat1, lon2, lat2):
    """Calcula distancia en km entre dos puntos geográficos (fórmula Haversine)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica feature engineering sobre un DataFrame extraído de la BD.
    Devuelve el mismo DataFrame con columnas adicionales: age, hour, distance_km.
    La columna trans_date_trans_time se convierte a datetime in-place.
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]                   = pd.to_datetime(df["dob"])
    df["age"]         = df.apply(
        lambda r: relativedelta(r["trans_date_trans_time"], r["dob"]).years, axis=1
    )
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["distance_km"] = haversine_np(df["long"], df["lat"], df["merch_long"], df["merch_lat"])
    return df
