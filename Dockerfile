FROM tensorflow/tensorflow:2.10.0
# OR for apple silicon, use this base image, but it's larger than python-buster + pip install tensorflow
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

WORKDIR /prod

COPY dockerfile_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY projet projet

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

CMD uvicorn projet.api.fast:app --host 0.0.0.0 --port $PORT
