from datetime import date
from pydantic import BaseModel, ConfigDict


class InputDataRetiroAtm(BaseModel):
    """Payload que se envía a la API de predicción."""
    model_config = ConfigDict(from_attributes=True)

    atm:                      int
    prediction_date:          date
    diaSemana:                int
    tendencia_lags:           float
    lag1:                     float
    lag5:                     float
    lag11:                    float
    caida_reciente:           int
    retiros_finde_anterior:   float
    retiros_domingo_anterior: float
    ratio_finde_vs_semana:    float
    domingo_bajo:             int
    ubicacion:                int
    ambiente:                 int
