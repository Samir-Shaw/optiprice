# 1. Base Image: Lightweight Python 3.10 Linux
FROM python:3.10-slim

# 2. Container working directory
WORKDIR /app

# 3. System packages (C++ runtime build tools for CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy full application code
COPY . .

# 6. Expose default port
EXPOSE 5000

# 7. Start container with Gunicorn web server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "app:app"]