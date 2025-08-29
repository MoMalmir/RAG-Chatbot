# Dockerfile
FROM python:3.11-slim

# Install OS deps only if you later need them (keeping image small)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source
COPY src /app/src
COPY app.py /app/

# Optional: avoid sys.path hacks in app.py
ENV PYTHONPATH=/app/src

# Streamlit defaults for Spaces
ENV PORT=7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]
