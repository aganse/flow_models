# CUDA 12.3 required for tensorflow>=2.15. Check TF release notes if updating TF version.
# https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=12.3
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

COPY *.py /app/

COPY sagemaker-support/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
