FROM python:3.7-bullseye AS build

RUN pip install --upgrade pip

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

FROM python:3.7-slim-bullseye AS final

RUN pip install --upgrade pip

# copy env
COPY --from=build /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages

# copy chemdataextractor executable
COPY --from=build /usr/local/bin/cde /usr/local/bin/cde
RUN cde data download

# this addresses a crash case in Uvicorn.
RUN pip install --upgrade 'click<8.0'
RUN pip uninstall -y uvicorn
RUN pip install --force uvicorn[standard]

COPY . /app
WORKDIR /app

CMD ["uvicorn", "chem_data_extractor_service:app", "--host", "0.0.0.0"]