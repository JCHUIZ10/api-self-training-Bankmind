FROM python:3.11-slim
WORKDIR /app

# Instalar dependencias del sistema para psycopg2 y compiladores
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
