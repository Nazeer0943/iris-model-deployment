FROM python:3.11-slim

WORKDIR /app

# Copying requirements specification and installing dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying trained model artifact and application server
COPY iris_model_artifact.pkl .
COPY main.py .

EXPOSE 8000

# Starting the Uvicorn application server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
