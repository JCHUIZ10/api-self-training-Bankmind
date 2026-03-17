import sys
import os
sys.path.append(os.path.abspath('src'))

from retiro_atm.generated.synthetic_data_service import ejecutar_sync
from retiro_atm.database import get_engine

if __name__ == "__main__":
    db = get_engine()
    print("Engine obtained:", type(db))
    ejecutar_sync('2026-02-10', db)
    print("Done")
