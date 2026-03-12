# src/retiro_atm/model_evaluator.py
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from retiro_atm.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluación de modelos de predicción de retiros ATM."""

    @staticmethod
    def evaluar_modelo(modelo, X_test, y_test) -> Dict[str, float]:
        """
        Evalúa el modelo en escala real (invirtiendo log1p).

        Args:
            modelo: Modelo XGBRegressor entrenado.
            X_test: Features de test.
            y_test: Target real (sin transformar).

        Returns:
            Dict con MAE, MAPE y RMSE.
        """
        y_pred_log = modelo.predict(X_test)
        y_pred = DataPreprocessor.invertir_transformacion(y_pred_log)

        # Protección contra valores ínfimos
        y_real = np.maximum(y_test.values, 1e-6)

        mae = mean_absolute_error(y_real, y_pred)
        mape = mean_absolute_percentage_error(y_real, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))

        logger.info(f"📊 Evaluación: MAE={mae:.2f}, MAPE={mape:.4f}, RMSE={rmse:.2f}")
        return {"mae": mae, "mape": mape, "rmse": rmse}

    @staticmethod
    def obtener_importancia_features(modelo, features: list) -> Dict[str, float]:
        """
        Calcula la importancia relativa (gain) de cada feature.

        Returns:
            Dict feature→porcentaje de importancia.
        """
        booster = modelo.get_booster()
        scores = booster.get_score(importance_type="gain")

        importancias = pd.DataFrame({
            "feature": scores.keys(),
            "gain": scores.values(),
        })
        importancias["porcentaje"] = (
            100 * importancias["gain"] / importancias["gain"].sum()
        )
        importancias = importancias.sort_values("porcentaje", ascending=False)

        return importancias.set_index("feature")["porcentaje"].to_dict() # type: ignore

    @staticmethod
    def calcular_intervalo_confianza(
        X_test, y_test, modelo, nivel_confianza: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calcula intervalo de confianza de los residuos del modelo.

        Args:
            X_test: Features de test.
            y_test: Target real (sin transformar).
            modelo: Modelo entrenado.
            nivel_confianza: Nivel de confianza (default 95%).

        Returns:
            Dict con lower_bound, upper_bound, media, sigma, margin_error, t_crit.
        """
        y_pred_log = modelo.predict(X_test)
        y_pred = DataPreprocessor.invertir_transformacion(y_pred_log)

        residuos = y_test.values - y_pred
        n = len(residuos)
        sigma = float(residuos.std(ddof=1))
        t_crit = float(stats.t.ppf(1 - (1 - nivel_confianza) / 2, df=n - 1))
        media = float(residuos.mean())
        margin_error = t_crit * sigma

        logger.info(
            f"📐 IC ({nivel_confianza*100}%): "
            f"[{media - margin_error:.2f}, {media + margin_error:.2f}], "
            f"σ={sigma:.2f}, t_crit={t_crit:.4f}"
        )

        return {
            "lower_bound": media - margin_error,
            "upper_bound": media + margin_error,
            "media_residuos": media,
            "sigma": sigma,
            "margin_error": margin_error,
            "confidence_level": nivel_confianza * 100,
            "t_crit": t_crit,
        }

    @staticmethod
    def evaluar_cambio_significativo(
        mape_prod: float, mape_beta: float, tolerancia: float = 0.05,
    ) -> bool:
        """
        Determina si el nuevo modelo supera significativamente al champion.
        El MAPE del challenger debe ser menor que mape_prod * (1 - tolerancia).

        Returns:
            True si el challenger es significativamente mejor.
        """
        umbral = mape_prod * (1 - tolerancia)
        es_mejor = mape_beta < umbral

        if es_mejor:
            mejora = (mape_prod - mape_beta) / mape_prod * 100
            logger.info(
                f"🎉 Challenger es mejor: MAPE {mape_beta:.4f} < {umbral:.4f} "
                f"(mejora {mejora:.1f}%)"
            )
        else:
            logger.info(
                f"⚠️ Challenger NO supera al champion: "
                f"MAPE {mape_beta:.4f} >= {umbral:.4f}"
            )
        return es_mejor
