# Use the NVIDIA CUDA runtime as a parent image, available at:
# https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=11.2
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# COPY . /app
COPY requirements.txt /app/

# Install Python3 and pip3
RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --no-cache-dir -r requirements.txt

# Set the default TensorFlow mode (cpu vs gpu) to install
ARG TENSORFLOW_PKG=tensorflow-cpu==2.12.0
# Install TensorFlow based on the build argument (which will overide the above if present)
RUN pip3 install ${TENSORFLOW_PKG}

COPY *.py /app/

# Make port 80 available to the world outside this container (if needed)
# EXPOSE 80

# Run the script when the container launches
CMD ["python3", "testing.py"]

