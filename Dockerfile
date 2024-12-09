# syntax=docker.io/docker/dockerfile:1.7-labs
FROM python:3.10-bullseye AS build

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir 'papermage[dev,predictors,visualizers]@git+https://github.com/gsireesh/papermage.git@ad_hoc_fixes'

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

FROM python:3.10-slim-bullseye AS final

RUN apt-get -qqy update \
  && apt-get install -qy poppler-utils \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip

COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --exclude=data . /app
WORKDIR /app
RUN mkdir -p data/uploaded_papers && mkdir data/processed_papers

CMD ["python", "-m", "streamlit", "run", "Upload_Paper.py",  "--server.headless", "true"]
