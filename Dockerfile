FROM python:3.10-bullseye

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y poppler-utils

RUN mkdir /app
WORKDIR /app


RUN pip install 'papermage[dev,predictors,visualizers]@git+https://github.com/gsireesh/papermage.git@ad_hoc_fixes'

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . /app
RUN rm -rf /app/papermage
RUN rm -rf data/
RUN mkdir -p data/uploaded_papers
RUN mkdir - data/parsed_papers

CMD ["python3", "-m", "streamlit", "run", "Upload_Paper.py",  "--server.headless", "true"]
