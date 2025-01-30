# Use the official Python 3.9 slim image (lightweight)
FROM python:3.9-slim

# System packages needed for scikit-image and other libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app
# Install Python dependencies
# You can install directly:
RUN pip install --no-cache-dir streamlit numpy pywavelets scikit-image pillow plotly

COPY . .

# Expose the default Streamlit port
EXPOSE 8502

# Command to run the Streamlit app on container start
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
