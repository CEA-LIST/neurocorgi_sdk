# NeuroCorgi SDK, CeCILL-C license
# Builds neurocorgi/neurocorgi_sdk:latest image on DockerHub https://hub.docker.com/r/neurocorgi/neurocorgi_sdk
# Image is CUDA-optimized for NeuroCorgi applications for single/multi-GPU training and inference

# Use PyTorch image from https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:23.03-py3
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install linux packages
RUN apt update \
    && apt install --no-install-recommends -y git tar zip curl htop screen libpython3-dev python3-pip

# Create working directory
WORKDIR /usr/src/neurocorgi_sdk

# Get NeuroCorgi SDK from Github
RUN git clone https://github.com/CEA-LIST/neurocorgi_sdk.git /usr/src/neurocorgi_sdk

# Install SDK and required libraries 
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache .

# Creates a symbolic link to make 'python' point to 'python3'
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set environment variables
ENV OMP_NUM_THREADS=1


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=neurocorgi/neurocorgi_sdk:latest && sudo docker build -f docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run with access to all GPUs
# t=neurocorgi/neurocorgi_sdk:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with access to GPUs 2 and 3 (inside container CUDA devices will appear as 0 and 1)
# t=neurocorgi/neurocorgi_sdk:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus '"device=2,3"' $t

# Pull and Run with local directory access (i.e., local volume mounted)
# t=neurocorgi/neurocorgi_sdk:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t
