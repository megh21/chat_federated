# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY src /app
COPY requirements.txt .


RUN pip install -r requirements.txt && pip cache purge
ARG PORT
EXPOSE ${PORT:-8000}
CMD ["streamlit", "run", "--server.port", "80", "app.py"]