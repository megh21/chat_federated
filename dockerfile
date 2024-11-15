FROM python:3.10-slim

#copy the folder src to the container
COPY src /src

WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY databases databases

EXPOSE 8000



CMD ["streamlit", "run", "src/app.py", "--server.port=8000", "--server.address=0.0.0.0"]
