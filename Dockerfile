FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python lr_model.py
CMD ["python", "app.py" ]