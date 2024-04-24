FROM python:3.10-slim-buster

COPY . /name_entity_recoginition

WORKDIR /name_entity_recoginition

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "app.py"]
