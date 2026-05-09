# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run Flask app
CMD ["python", "app/app.py"]
