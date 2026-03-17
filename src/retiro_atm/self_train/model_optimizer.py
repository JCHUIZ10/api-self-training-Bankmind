# src/retiro_atm/model_optimizer.py
import logging
from typing import Tuple

import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimización de hiperparámetros con Optuna y entrenamiento de XGBoost."""

    @staticmethod
    def weighted_mape_objective(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Función de pérdida asimétrica personalizada.
        Penaliza más la sobreestimación (pred > true) que la subestimación.

        Returns:
            Tupla (gradiente, hessiano) para XGBoost.
        """
        residual = y_true - y_pred
        grad = np.where(residual < 0, -2.0 * residual, -0.5 * residual)
        hess = np.where(residual < 0, 2.0, 0.5)
        return grad, hess

    @classmethod
    def optimizar_hiperparametros(
        cls,
        X_train, y_train_log,
        n_trials: int = 100,
        n_splits: int = 5,
    ) -> optuna.Study:
        """
        Ejecuta la búsqueda de hiperparámetros con Optuna usando
        validación cruzada temporal (TimeSeriesSplit).

        Args:
            X_train: Features de entrenamiento.
            y_train_log: Target transformada (log1p).
            n_trials: Número de trials de Optuna.
            n_splits: Número de splits temporales para CV.

        Returns:
            Estudio de Optuna con los mejores parámetros.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            param = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "objective": cls.weighted_mape_objective,
                "random_state": 42,
                "eval_metric": "mae",
            }

            scores = []
            tscv = TimeSeriesSplit(n_splits=n_splits)

            for train_idx, val_idx in tscv.split(X_train):
                X_t = X_train.iloc[train_idx]
                X_v = X_train.iloc[val_idx]
                y_t = y_train_log.iloc[train_idx].values.flatten()
                y_v = y_train_log.iloc[val_idx].values.flatten()

                model = xgb.XGBRegressor(**param, early_stopping_rounds=50)
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

                preds_log = model.predict(X_v)
                preds_real = np.expm1(preds_log)
                y_v_real = np.maximum(np.expm1(y_v), 1e-6)

                error = mean_absolute_percentage_error(y_v_real, preds_real)
                scores.append(error)

            return np.mean(scores)

        logger.info(f"🔍 Iniciando Optuna ({n_trials} trials, {n_splits} splits)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials) # type: ignore

        logger.info(
            f"✅ Mejor MAPE CV: {study.best_value:.4f} | "
            f"Params: {study.best_params}"
        )
        return study

    @staticmethod
    def entrenar_modelo_final(
        best_params: dict,
        X_train, y_train_log,
        X_val, y_val_log,
        features: list,
    ) -> xgb.XGBRegressor:
        """
        Entrena el modelo final con los mejores hiperparámetros de Optuna,
        usando early stopping contra el set de validación.

        Args:
            best_params: Hiperparámetros óptimos.
            X_train: Features de entrenamiento final.
            y_train_log: Target transformada de entrenamiento.
            X_val: Features de validación.
            y_val_log: Target transformada de validación.
            features: Lista de nombres de features para el booster.

        Returns:
            Modelo XGBRegressor entrenado.
        """
        logger.info("🏗️ Entrenando modelo final con mejores parámetros...")

        model = xgb.XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            early_stopping_rounds=100,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False,
        )

        # Asignar nombres de features al booster
        booster = model.get_booster()
        booster.feature_names = features

        logger.info(
            f"✅ Modelo entrenado. Best iteration: {model.best_iteration}, "
            f"Best score: {model.best_score:.4f}"
        )
        return model
