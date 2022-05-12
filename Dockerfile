FROM nvidia/cuda:10.0-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y libcudnn7 curl less
RUN apt-get install -y python3.7 python3-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.7 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py
ADD bluebenchmark_requirements.txt /tmp
RUN pip install -r /tmp/bluebenchmark_requirements.txt
ADD tensorflow-gpu_requirements.txt /tmp
RUN pip install -r /tmp/tensorflow-gpu_requirements.txt
