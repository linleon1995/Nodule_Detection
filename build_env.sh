env_name=dl-medical

# docker pull nvcr.io/nvidia/pytorch:21.06-py3
# docker build -t $env_name -f docker/dl-medical/Dockerfile .

project=${PWD##*/}
# docker run -it --gpus all -v "/Nodule_Detection":"/workspace/Project/Nodule_Detection" dl-medical
# docker run -it --gpus all --name $env_name --rm -v "/$pwd":"/workspace/Project/$pwd" $env_name
docker run -it --name $env_name --rm -v $(pwd):/workspace/$project dl-medical
# docker run -it --gpus all --name $env_name --rm  $env_name
