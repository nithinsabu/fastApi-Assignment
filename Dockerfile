FROM ultralytics/ultralytics:latest-cpu

WORKDIR /code

COPY ./requirements-docker.txt /code/requirements.txt

COPY ./index.html /code/index.html

RUN pip install --no-cache-dir --upgrade  -r /code/requirements.txt

COPY ./app.py /code/

EXPOSE 80

CMD ["fastapi", "run", "app.py", "--port", "80"]