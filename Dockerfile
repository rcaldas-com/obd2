FROM python:3
LABEL maintainer="RCaldas <docker@rcaldas.com>"

COPY requirements.txt /
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get full-upgrade -y && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
ENTRYPOINT python /app/app.py

# COPY app.py /app/
