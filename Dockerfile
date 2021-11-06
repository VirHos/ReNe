FROM tensorflow/tensorflow:1.15.5-py3
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt --no-cache