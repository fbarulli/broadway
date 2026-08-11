FROM python:3.12-slim

RUN pip install mlflow==2.22.1

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
