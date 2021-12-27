FROM tensorflow/tensorflow
#:2.3.1
#FROM tensorflow/tensorflow:1.15.0-py3

WORKDIR /app

run apt update -y
RUN apt install git -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80
CMD [ "python", "-u", "main.py" ]
