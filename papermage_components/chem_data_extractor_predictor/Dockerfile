FROM python:3.8-slim

RUN pip install --upgrade pip

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN cde data download

COPY . /app

CMD ["python", "-m", "fastapi", "run"]