# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Render sets PORT env var)
EXPOSE $PORT

# Command to run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]