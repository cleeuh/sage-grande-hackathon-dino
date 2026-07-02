FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app/work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV TORCH_HOME=/root/.cache/torch
ENV PYTHONWARNINGS="ignore::UserWarning"
