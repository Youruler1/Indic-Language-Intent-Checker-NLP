# Use a lightweight official Python base image
FROM python:3.11-slim

# Set environment variables to prevent interactive prompts and enable UTF-8 encoding
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Define the default command to run your app
# (Adjust 'your_script.py' to your actual entry point)
CMD ["python", "test_script.py"]
