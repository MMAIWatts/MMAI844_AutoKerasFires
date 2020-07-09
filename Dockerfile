FROM gcr.io/deeplearning-platform-release/base-cu101:latest

WORKDIR /opt

COPY requirements.txt /opt/requirements.txt

RUN apt-get update && apt-get install -y git

RUN pip install -r /opt/requirements.txt
