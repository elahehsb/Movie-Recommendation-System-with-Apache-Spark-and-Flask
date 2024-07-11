# Use the official Python image
FROM python:3.9-slim

# Install dependencies
RUN pip install pyspark flask

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Expose the port for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
