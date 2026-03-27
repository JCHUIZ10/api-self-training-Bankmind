import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class XGBObjective:
    """Función objetivo inyectable para Optuna."""
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, w_train: np.ndarray) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.w_train = w_train

    def __call__(self, trial: optuna.Trial) -> float:
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 15, 35),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**param)
        model.fit(self.X_train, self.y_train, sample_weight=self.w_train)
        preds = model.predict(self.X_test)
        return f1_score(self.y_test, preds)


def train_isolation_forest(X_recent_scaled: pd.DataFrame):
    if_model = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
    if_model.fit(X_recent_scaled)
    return if_model


def optimize_and_train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, w_train: np.ndarray, X_test: pd.DataFrame, y_test: pd.Series, optuna_trials: int):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = XGBObjective(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, w_train=w_train)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)
    
    best_params = study.best_params
    best_f1 = study.best_value
    
    xgb_model = XGBClassifier(**best_params, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    
    return xgb_model, best_params, best_f1, study.best_trial.number
