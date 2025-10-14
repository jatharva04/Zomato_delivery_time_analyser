# Use an official Python runtime as a parent image
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for better caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# (This includes main.py and the 'models/' directory)
COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Define the command to run your app when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
