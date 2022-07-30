FROM python:3.9

WORKDIR /app

# Copy over contents from local directory to the path in Docker container
COPY . /app/

# Install python requirements from requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

WORKDIR /app/app

# Start uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]