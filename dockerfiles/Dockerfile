FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "mlops_stock_project.api.app:app", "--host", "0.0.0.0", "--port", "8000"]