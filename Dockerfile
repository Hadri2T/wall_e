FROM tensorflow/tensorflow:2.10.0
# OR for apple silicon, use this base image, but it's larger than python-buster + pip install tensorflow
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /prod

COPY dockerfile_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY projet projet

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

CMD uvicorn projet.api.fast:app --host 0.0.0.0 --port $PORT
