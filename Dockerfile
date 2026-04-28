FROM python:3.11-slim

LABEL maintainer="ShirokaitoX1412"
LABEL description="DDoS Detection & IPS System with ML"

# System deps for scapy and iptables
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    iptables \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default: launch the Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
