import sys
import os
sys.path.append(os.path.abspath('src'))

from retiro_atm.self_train.training_service import ejecutar_autoentrenamiento
from retiro_atm.schemas import TrainingRequest

if __name__ == '__main__':
    request = TrainingRequest(
        optuna_trials=5,
        tolerancia_mape=0.05,
        dias_particion_test=60,
        dias_particion_val=15
    )
    ejecutar_autoentrenamiento(request)