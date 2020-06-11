FROM python:3.8

RUN pip install gsutil

COPY requirements.txt /blockflow/requirements.txt

RUN pip install -r /blockflow/requirements.txt

USER 1000:1000

COPY . /blockflow

WORKDIR /blockflow

