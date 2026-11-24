FROM python:3.11-slim

WORKDIR /app

# Install libraries first, so Docker can cache this slow step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy your code and model files
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]