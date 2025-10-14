# Use an official Python runtime as a parent image
# We use a slim-buster image for a smaller final image size
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency files first. This allows Docker to cache the layer
# if only main.py changes, speeding up subsequent builds.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes main.py, and the model file (delivery_model.pkl) 
# created by the demo code if it exists locally.
COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Command to run the application using Uvicorn
# The --host 0.0.0.0 is crucial for listening on all interfaces 
# within the container so it can be accessed externally.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]