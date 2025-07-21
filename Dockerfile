# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (for Flask app, usually 5000)
EXPOSE $PORT

# Run the app
CMD gunicorn --workers==4 --bind 0.0.0.0:$PORT
