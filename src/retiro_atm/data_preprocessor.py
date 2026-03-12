# src/retiro_atm/data_preprocessor.py
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Constantes del dominio ATM
FEATURES = [
    "dia_semana", "ubicacion",
    "lag_1", "lag_5", "lag_11", "tendencia_lags",
    "ratio_finde_vs_semana", "retiros_finde_anterior",
    "retiros_domingo_anterior", "domingo_bajo",
    "caida_reciente", "ambiente",
]
TARGET = "retiro"


@dataclass
class DataSplit:
    """Contenedor para un split de datos con sus componentes X, Y y Y transformada."""
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    y_log: pd.Series


@dataclass
class PreparedData:
    """Resultado completo del preprocesamiento."""
    df_original: pd.DataFrame
    train: DataSplit
    test: DataSplit
    # Validación interna (sub-split del train)
    train_final: DataSplit
    val: DataSplit


class DataPreprocessor:
    """Maneja la partición temporal y transformación del dataset ATM."""

    @staticmethod
    def particionar_dataset(
        df: pd.DataFrame, dias_particion: int = 60
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Particiona el dataset cronológicamente.
        Los últimos `dias_particion` días van a test, el resto a train.

        Args:
            df: DataFrame con columna 'fecha_transaccion'.
            dias_particion: Días reservados para el segundo split.

        Returns:
            Tupla (df_train, df_test).
        """
        df = df.sort_values(["fecha_transaccion", "atm"]).reset_index(drop=True)

        fecha_max = df["fecha_transaccion"].max()
        fecha_corte = fecha_max - pd.Timedelta(days=dias_particion)

        df_train = df[df["fecha_transaccion"] <= fecha_corte].copy()
        df_test = df[df["fecha_transaccion"] > fecha_corte].copy()

        logger.info(
            f"📦 Split: Train {df_train['fecha_transaccion'].min().date()} → "
            f"{df_train['fecha_transaccion'].max().date()} ({len(df_train)} filas) | "
            f"Test {df_test['fecha_transaccion'].min().date()} → "
            f"{df_test['fecha_transaccion'].max().date()} ({len(df_test)} filas)"
        )
        return df_train, df_test

    @staticmethod
    def separar_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extrae features (X) y target (y) del DataFrame."""
        return df[FEATURES], df[TARGET]

    @staticmethod
    def aplicar_transformacion(y: pd.Series) -> pd.Series:
        """Aplica transformación logarítmica log1p para estabilizar varianza."""
        return np.log1p(y) # type: ignore

    @staticmethod
    def invertir_transformacion(y: np.ndarray) -> np.ndarray:
        """Invierte la transformación logarítmica (expm1)."""
        return np.expm1(y)

    @classmethod
    def _crear_split(cls, df: pd.DataFrame) -> DataSplit:
        """Crea un DataSplit completo a partir de un DataFrame."""
        X, y = cls.separar_xy(df)
        y_log = cls.aplicar_transformacion(y)
        return DataSplit(df=df, X=X, y=y, y_log=y_log)

    @classmethod
    def preparar_datos_completos(
        cls,
        df: pd.DataFrame,
        dias_test: int = 60,
        dias_val: int = 15,
    ) -> PreparedData:
        """
        Pipeline completo de preparación de datos:
        1. Particiona en train/test (últimos `dias_test` días → test)
        2. Sub-particiona train en train_final/val (últimos `dias_val` días → val)
        3. Separa X/Y y aplica transformación log1p a cada split

        Args:
            df: DataFrame completo con datos limpios.
            dias_test: Días para el split de test.
            dias_val: Días para el split de validación interna.

        Returns:
            PreparedData con todos los splits listos.
        """
        logger.info("🔧 Preparando datos completos...")

        # Split principal: train / test
        df_train, df_test = cls.particionar_dataset(df, dias_particion=dias_test)

        # Sub-split: train_final / validación
        df_train_final, df_val = cls.particionar_dataset(df_train, dias_particion=dias_val)

        return PreparedData(
            df_original=df,
            train=cls._crear_split(df_train),
            test=cls._crear_split(df_test),
            train_final=cls._crear_split(df_train_final),
            val=cls._crear_split(df_val),
        )
