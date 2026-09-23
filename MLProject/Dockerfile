FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow==2.19.0 flask

COPY . /app

EXPOSE 5000

CMD ["python", "modelling.py"]