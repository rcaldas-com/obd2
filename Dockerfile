FROM python:3.9
LABEL maintainer="RCaldas <docker@rcaldas.com>"

# Update the Python image:
RUN apt update && apt upgrade -y && \
    pip install --no-cache-dir --upgrade pip

# Install Python requirements:
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    apt clean && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

# Copy code and start Web Server
WORKDIR /app
EXPOSE 5000

# Dev
ENTRYPOINT flask run --host 0.0.0.0

# Prd
# ENTRYPOINT gunicorn --reload --timeout=30 -b:5000 --access-logfile=- --error-logfile=- obd2:app
# COPY app /app
