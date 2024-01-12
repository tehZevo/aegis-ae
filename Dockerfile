FROM tensorflow/tensorflow

WORKDIR /app

run apt update -y
RUN apt install git -y

COPY requirements.txt .
RUN pip install --ignore-installed -r requirements.txt

COPY . .

EXPOSE 80
CMD [ "python", "-u", "main.py" ]
