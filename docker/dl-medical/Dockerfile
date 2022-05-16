FROM nvcr.io/nvidia/pytorch:21.06-py3

WORKDIR /workspace/Project/Nodule_Detection

# Install Git
RUN apt-get update
RUN apt-get install git

# Install detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install opencv dependencies
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt