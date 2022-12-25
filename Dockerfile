FROM python:3.9


WORKDIR /amazon_sentiment

COPY ./artifacts ./artifacts
COPY ./data ./data
COPY ./deploy ./deploy
COPY ./dist ./dist
COPY ./mlruns ./mlruns
COPY ./notebooks ./notebooks
COPY ./src ./src
COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH /amazon_sentiment
CMD ["python", "src/standard/main.py"]