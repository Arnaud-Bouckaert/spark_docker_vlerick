FROM python:3.9-slim-buster

WORKDIR /app
# Create a working directory

COPY requirements.txt /app 
# Requirements first! This is unlikely to change!
# This encloses the libraries that you use

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY streamlit_app.py /app

ENTRYPOINT ["streamlit"]
CMD ["run", "streamlit_app.py"]
