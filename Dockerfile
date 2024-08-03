FROM python:3.11-slim

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app

CMD python3 -m streamlit Upload_Paper.py --server.headless true