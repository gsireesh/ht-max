FROM python:3.7-bullseye

RUN pip install --upgrade pip

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN cde data download

COPY . /app

CMD ["uvicorn", "chem_data_extractor_service:app", "--host", "0.0.0.0"]