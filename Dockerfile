FROM phusion/baseimage:0.9.22

ENV HOME /root
ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL en_US.UTF-8
ENV EDITOR vim
ENV TERM xterm

RUN locale-gen en_US.UTF-8

RUN sed -i -e 's,http://archive.ubuntu.com,http://de.archive.ubuntu.com,g' /etc/apt/sources.list
RUN apt-get update && \
	apt-get -y install git wget curl jq && \
	apt-get upgrade -y -o Dpkg::Options::="--force-confold" && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Deactivate unused services
RUN mv /etc/service/cron /etc/service/.cron
RUN mv /etc/service/sshd /etc/service/.sshd
RUN mv /etc/service/syslog-ng /etc/service/.syslog-ng
RUN mv /etc/service/syslog-forwarder /etc/service/.syslog-forwarder
RUN chmod 444 /etc/my_init.d/00_regen_ssh_host_keys.sh

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential

WORKDIR /tmp
RUN wget --quiet https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz && \
  tar xzf protobuf-2.6.1.tar.gz && \
  mv protobuf-2.6.1 /opt/protobuf
RUN cd /opt/protobuf && \
  ./configure && \
  make && \
  make check && \
  make install && \
  ldconfig
RUN  protoc --version \

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda info -a

RUN conda install -y scipy
RUN pip install tensorflow pillow lxml jupyter matplotlib protobuf

RUN git clone https://github.com/tensorflow/models.git /opt/tensorflow-models
WORKDIR /opt/tensorflow-models
RUN pip install -e .
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH $PYTHONPATH:/opt/tensorflow-models:/opt/tensorflow-models/slim
RUN python object_detection/builders/model_builder_test.py

WORKDIR /opt/tensorflow-models-object_detection
RUN wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz && \
    tar xzf ssd_mobilenet_v1_coco_11_06_2017.tar.gz && \
    rm ssd_mobilenet_v1_coco_11_06_2017.tar.gz
WORKDIR /app
RUN pip install flask
ADD docker/service/ /etc/service/

