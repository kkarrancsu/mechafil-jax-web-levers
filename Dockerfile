FROM python:3.11-slim

# Install system dependencies (we need git)
RUN apt-get update && apt-get install -y git && apt-get clean

# Install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . /app
WORKDIR /app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "mechafil_jax_web_levers/Filecoin_CryptoEconomics.py", "--server.port=8501", "--server.enableCORS=false"]

