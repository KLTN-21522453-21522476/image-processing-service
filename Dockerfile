FROM python:3.12-slim

WORKDIR /app

COPY src/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/

EXPOSE 5002

CMD ["python", "main.py"]
