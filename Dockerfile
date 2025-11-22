FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY funasr_http_server.py ./
COPY model.py ./
COPY utils ./utils

# Install Python dependencies
RUN uv sync

# ModelScope will use default cache path: /root/.cache/modelscope/hub
# We will mount the volume to this path
# Models should be placed at: ./models/iic/SenseVoiceSmall/

# Create necessary directories
RUN mkdir -p /root/.cache/modelscope/hub /workspace/temp_dir

# Expose port
EXPOSE 8200

# Run the server
CMD ["uv", "run", "funasr_http_server.py", "--host", "0.0.0.0", "--port", "8200"]
