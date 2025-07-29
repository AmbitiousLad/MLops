# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
