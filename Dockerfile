FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV ENV=prod

# Expose the API port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000"] 