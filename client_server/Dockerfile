FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models ./models
COPY data ./data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]