FROM python:3.10-bullseye AS build

RUN pip install --upgrade pip
RUN mkdir /app
WORKDIR /app


RUN pip install 'papermage[dev,predictors,visualizers]@git+https://github.com/gsireesh/papermage.git@ad_hoc_fixes'

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

FROM python:3.10-slim-bullseye AS final

RUN apt-get update && apt-get install -y poppler-utils
RUN pip install --upgrade pip

COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . /app
WORKDIR /app
RUN rm -rf data/
RUN mkdir -p data/uploaded_papers
RUN mkdir - data/processed_papers

CMD ["python", "-m", "streamlit", "run", "Upload_Paper.py",  "--server.headless", "true"]
