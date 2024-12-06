FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/datasets

COPY . /app/

CMD ["python", "main.py"]