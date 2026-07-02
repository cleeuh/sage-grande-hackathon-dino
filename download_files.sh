#!/bin/bash

cd "$(dirname "$0")"

mkdir -p ./model/weights
mkdir -p ./data/waste

cd ./model && git clone https://github.com/facebookresearch/dinov2
cd ./weights && curl -L -O https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

cd ../../data

curl -L -O https://www.huntsvilleal.gov/wp-content/uploads/2026/01/Dog-Park-Graphic.png
curl -L -o ./waste/realwaste.zip\
  https://www.kaggle.com/api/v1/datasets/download/joebeachcapital/realwaste
cd ./waste && unzip realwaste.zip