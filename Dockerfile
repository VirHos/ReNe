FROM tensorflow/tensorflow:1.15.5-py3
COPY requirements.txt /rene/requirements.txt
RUN pip3 install -r requirements.txt