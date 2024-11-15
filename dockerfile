FROM python:3.10-slim

#copy the folder src to the container
COPY src /src

WORKDIR /src

COPY requirements.in requirements.in
RUN pip install -r requirements.in
COPY databases databases
COPY .env .env
COPY .streamlit .streamlit
EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_ENV=production

# Command to run Streamlit
# CMD ["streamlit", "run", "src/app.py"]
CMD ["tail", "-f", "/dev/null"]