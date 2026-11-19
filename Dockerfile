# Use lightweight official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the AI model while building the image, so the container
# doesn't download it again every time it starts (faster cold starts).
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Copy the rest of your application code
COPY . .

# Cloud Run / Serverless platforms provide a dynamic PORT environment variable.
# We configure Uvicorn to listen on 0.0.0.0 and look up that PORT variable.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
