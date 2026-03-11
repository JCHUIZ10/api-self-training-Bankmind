import pandas as pd
import numpy as np

def get_psi(df) -> dict:
    features_criticas = __get_features_criticas(df)
    baseline_json = __generate_feature_baseline(df, features_criticas)
    cleaned = __clean_inf(baseline_json)

    return cleaned # type: ignore


def __clean_inf(obj):
    if isinstance(obj, dict):
        return {k: __clean_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [__clean_inf(v) for v in obj]
    elif obj == float("inf") or obj == float("-inf"):
        return None  # o puedes poner un número muy grande
    else:
        return obj


def __get_features_criticas(df):
    return df.columns.tolist()


def __generate_feature_baseline(df, features_to_monitor):
    """
    Genera el baseline de distribución para monitoreo PSI.
    
    Args:
        df: DataFrame con los datos de entrenamiento (X_train).
        features_to_monitor: Lista de columnas a monitorear (ej. ['lag1', 'lag5', 'tendencia_lags'])
    
    Returns:
        baseline_metadata: Diccionario con bins, distribución esperada y estadísticas descriptivas.
    """
    baseline_metadata = { "features": {} }

    for feature in features_to_monitor:
        
        series = df[feature].dropna()
        null_pct = float(df[feature].isna().mean())

        # --- 1. Calcular deciles y bins ---
        quantiles = np.linspace(0, 1, 11)
        bins = np.quantile(series, quantiles)
        bins = np.unique(bins)

        # Validación: si hay muy poca varianza, la feature no es monitoreable con PSI
        if len(bins) < 3:
            baseline_metadata["features"][feature] = {
                "bins": None,
                "expected_pct": None,
                "stats": {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "null_pct": null_pct,
                    "n_samples": int(series.count())
                },
                "warning": "Insuficiente_Varianza - feature no apta para PSI"
            }
            continue

        # Ajustar extremos para cubrir cualquier valor futuro en producción
        bins[0]  = -np.inf
        bins[-1] =  np.inf

        # --- 2. Calcular distribución esperada ---
        counts, _ = np.histogram(series, bins=bins)
        
        # Validación: evitar divisiones por cero si algún bin queda vacío
        total = counts.sum()
        if total == 0: expected_pct = [0.0] * len(counts)
        else: expected_pct = (counts / total).tolist()

        # --- 3. Guardar estadísticas descriptivas ---
        # Útiles para detectar shifts que el PSI solo no captura
        stats = {
            "mean"     : float(series.mean()),
            "std"      : float(series.std()),
            "median"   : float(series.median()),
            "null_pct" : null_pct,
            "n_samples": int(series.count())
        }

        # --- 4. Consolidar en el metadata ---
        baseline_metadata["features"][feature] = {
            "bins"        : bins.tolist(),
            "expected_pct": expected_pct,
            "stats"       : stats
        }

    return baseline_metadata["features"]