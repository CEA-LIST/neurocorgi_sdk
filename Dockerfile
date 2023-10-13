FROM nvcr.io/nvidia/pytorch:22.04-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update   && \
    apt-get install -y  \
        gnuplot         \
        libopencv-dev

WORKDIR /opt

# Install required Python libraries for SDK
RUN pip install onnxsim onnxruntime

# Get N2D2 from Github
RUN git clone --recursive --depth 1 https://github.com/CEA-LIST/N2D2.git
ENV N2D2_ROOT=/opt/N2D2

WORKDIR $N2D2_ROOT

# Compile N2D2 and install it in the Python libs
RUN pip install .

WORKDIR /opt

# Get NeuroCorgi SDK from Github
RUN git clone https://github.com/CEA-LIST/neurocorgi_sdk.git
WORKDIR /opt/neurocorgi_sdk

# Install the SDK in the Python libs
RUN pip install .

# Because there will be a dynamic link issue with openmpi
# Run this command to fix it
# source: https://github.com/horovod/horovod/issues/2187
RUN echo 'ldconfig' >> /root/.bashrc


WORKDIR /workspace

