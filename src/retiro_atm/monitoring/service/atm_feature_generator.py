import numpy as np
import pandas as pd

# Lags de transacciones
FEATURE_LAGS = [1, 5, 11]

class AtmFeatureGenerator:
    """
    Recibe un DataFrame de transacciones y produce el DataFrame
    enriquecido con features listas para el modelo.

    Uso:
        gen = AtmFeatureGenerator(df_transacciones)
        df_features = gen.calcular_features()
    """

    def __init__(self, df: pd.DataFrame):
        # Copia defensiva para no mutar el DataFrame original
        self._df = df.copy()

    # ──────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def calcular_features(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo en el orden correcto."""
        self._df = self._df.sort_values(["id_atm", "transaction_date"])

        self._agregar_features_temporales()
        self._agregar_caida_reciente()
        self._agregar_media_movil_finde()
        self._agregar_lags(FEATURE_LAGS)
        self._agregar_domingo_bajo()
        self._agregar_retiros_finde_anterior()
        self._agregar_ratio_y_tendencia()
        self._quitar_data_antigua()

        return self._df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    # FEATURES TEMPORALES
    # ──────────────────────────────────────────────────────────────
    def _agregar_features_temporales(self) -> None:
        self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])
        self._df["dia_mes"]          = self._df["transaction_date"].dt.day
        self._df["mes"]              = self._df["transaction_date"].dt.month
        self._df["diaSemana"]        = self._df["transaction_date"].dt.isocalendar().day
        self._df["es_findesemana"]   = self._df["diaSemana"].isin([6, 7]).astype(int)
        # Semana ISO (lunes como inicio)
        self._df["periodo"]          = self._df["transaction_date"].dt.to_period("W-MON")

    # ──────────────────────────────────────────────────────────────
    # CAÍDA RECIENTE (retiro < 60 % del día anterior)
    # ──────────────────────────────────────────────────────────────
    def _agregar_caida_reciente(self) -> None:
        self._df["caida_reciente"] = (
            self._df
            .groupby("id_atm")["amount"]
            .transform(lambda x: (x < x.shift(1) * 0.6).astype(int))
        )

    # ──────────────────────────────────────────────────────────────
    # MEDIA MÓVIL DE 7 SEMANAS DE FIN DE SEMANA
    # ──────────────────────────────────────────────────────────────
    def _agregar_media_movil_finde(self) -> None:
        df_finde = (
            self._df[self._df["diaSemana"].isin([6, 7])]
            .groupby(["id_atm", "periodo"])["amount"]
            .mean()
            .reset_index()
            .sort_values(["id_atm", "periodo"])
        )
        df_finde["media_movil_7_semanas"] = (
            df_finde
            .groupby("id_atm")["amount"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        )

        # Buscar la semana anterior para cada fila
        self._df["periodo_busqueda"] = (
            self._df["transaction_date"] - pd.Timedelta(days=7)
        ).dt.to_period("W-MON")

        self._df = self._df.merge(
            df_finde[["id_atm", "periodo", "media_movil_7_semanas"]].rename(
                columns={
                    "periodo": "periodo_busqueda",
                    "media_movil_7_semanas": "retiros_finde_anterior",
                }
            ),
            on=["id_atm", "periodo_busqueda"],
            how="left",
        )

        # Fallback: promedio histórico del ATM cuando no hay semana anterior
        prom_atm = self._df.groupby("id_atm")["amount"].transform("mean")
        self._df["retiros_finde_anterior"] = self._df["retiros_finde_anterior"].fillna(prom_atm)
        self._df.drop(columns=["periodo_busqueda"], inplace=True)

    # ──────────────────────────────────────────────────────────────
    # LAGS
    # ──────────────────────────────────────────────────────────────
    def _agregar_lags(self, lags: list[int]) -> None:
        for lag in lags:
            self._df[f"lag_{lag}"] = (
                self._df.groupby("id_atm")["amount"].shift(lag)
            )

    # ──────────────────────────────────────────────────────────────
    # DOMINGO BAJO (lunes con domingo anterior < mediana del ATM)
    # ──────────────────────────────────────────────────────────────
    def _agregar_domingo_bajo(self) -> None:
        mediana_por_atm = self._df.groupby("id_atm")["amount"].median()
        self._df["domingo_bajo"] = (
            (self._df["diaSemana"] == 0) &
            (self._df["lag_1"] < self._df["id_atm"].map(mediana_por_atm))
        ).astype(int)

    # ──────────────────────────────────────────────────────────────
    # RETIROS SÁBADO Y DOMINGO ANTERIOR
    # ──────────────────────────────────────────────────────────────
    def _agregar_retiros_finde_anterior(self) -> None:
        for dia, col_retiro, col_anterior in [
            (5, "retiro_sabado",  "retiros_sabado_anterior"),
            (6, "retiro_domingo", "retiros_domingo_anterior"),
        ]:
            self._df[col_retiro] = self._df["amount"].where(self._df["diaSemana"] == dia)
            self._df[col_anterior] = (
                self._df.groupby("id_atm")[col_retiro]
                .transform(lambda x: x.shift(1).ffill())
            )
        self._df.drop(columns=["retiro_sabado", "retiro_domingo"], inplace=True)

    # ──────────────────────────────────────────────────────────────
    # RATIO Y TENDENCIA
    # ──────────────────────────────────────────────────────────────
    def _agregar_ratio_y_tendencia(self) -> None:
        promedio_finde = self._df[
            ["retiros_sabado_anterior", "retiros_domingo_anterior"]
        ].mean(axis=1, skipna=True)

        promedio_semana = (
            self._df.groupby("id_atm")["amount"]
            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        )

        self._df["ratio_finde_vs_semana"] = (
            promedio_finde / promedio_semana.replace(0, np.nan)
        ).fillna(0)

        self._df["tendencia_lags"] = (
            self._df["lag_1"] - self._df["lag_11"].replace(0, np.nan)
        ).fillna(0)

    # ──────────────────────────────────────────────────────────────
    # ELIMINAR FILAS SIN HISTORIAL SUFICIENTE
    # ──────────────────────────────────────────────────────────────
    def _quitar_data_antigua(self) -> None:
        self._df = self._df[self._df["lag_11"].notnull()]
