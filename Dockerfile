FROM python:3.9.6-slim-buster

RUN pip install --upgrade pip
RUN pip install pipenv

COPY Pipfile* /app/
WORKDIR /app
RUN pipenv install --system --deploy

COPY . /app

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]