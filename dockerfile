FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
	python3-venv \
    python3-pip \
    git

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install cuDF
RUN pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com

RUN pip install git+https://github.com/huggingface/transformers.git

# Install sentence-transformers
RUN pip install sentence-transformers

RUN pip install xformers

# Install cuml for UMAP
RUN pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.12.*"

RUN pip install grpcio protobuf zstandard

COPY ./app /app

WORKDIR /app

CMD ["python", "server.py"]
