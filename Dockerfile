FROM python:3.12-slim

WORKDIR /app

# OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /app/

RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

COPY src/ /app/

EXPOSE 5002

CMD ["python", "main.py"]
