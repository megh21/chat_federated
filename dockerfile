# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY src /app
COPY requirements.txt .


RUN pip install -r requirements.txt && pip cache purge
ARG PORT
EXPOSE ${PORT:-80}
CMD ["streamlit", "run", "--server.port", "${PORT:-8000}", "app.py"]
# CMD ["sh", "-c", "python azure_setup.py & streamlit run --server.port ${PORT:-80} app.py"]