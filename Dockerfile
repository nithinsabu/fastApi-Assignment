FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /code/requirements.txt

COPY ./app.py /code/


CMD ["fastapi", "run", "app.py", "--port", "80"]