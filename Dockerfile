# CUDA 12.6 + CuDNN 9.3 required for tensorflow==2.18. Check TF release notes if updating TF version.
# https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=12.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y python3-pip git cuda-nvvm-12-6 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

COPY *.py /app/

COPY aws/job-support-common/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
