import psycopg2
import pandas as pd
from typing import Optional, Any
import os

class RepositoryPosgrestTrain():

    def obtener_data_train(self) -> Optional[pd.DataFrame]:
        """
        Establece conexión a PostgreSQL, ejecuta una consulta SELECT
        sobre una vista materializada y devuelve el resultado como un
        DataFrame de Pandas.
        """
        conn: Optional[Any] = None
        try:
            # 1. Establecer la conexión con psycopg2
            conn = psycopg2.connect(
                host= os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT")
            )
            print("Conexión a PostgreSQL exitosa.")

            # 2. Definir la consulta SQL
            view = os.getenv("NOMBRE_DE_TU_VISTA")

            sql_query = f"SELECT * FROM {view};"
            print(f"Executing query: {sql_query}")

            # 3. Usar pandas.read_sql para ejecutar la consulta y cargar
            #    directamente los resultados al DataFrame
            df = pd.read_sql(sql_query, conn)

            print(f"Datos cargados exitosamente. Dimensiones: {df.shape}")
            return df

        except (Exception, psycopg2.Error) as error:
            print(f"Error al conectar o ejecutar la consulta: {error}")
            return None

        finally:
            # 4. Cerrar la conexión
            if conn is not None:
                conn.close()
                print("🔗 Conexión a PostgreSQL cerrada.")