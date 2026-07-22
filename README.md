# DINOv2 PyTorch Demos

A progression of three small [DINOv2](https://github.com/facebookresearch/dinov2)
demos, from "does it load" to a multi-class classifier.

| #   | Demo                                                     | What it adds                                                             |
| --- | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| 0   | [Basic DINOv2 Test](0-basic-dino-test)                   | Load backbone, extract features. No training.                            |
| 1   | [Semantic Similarity](1-dino-semantic-sim)               | PCA visualization of semantic similarity of patch features. No training. |
| 2   | [DINOv2 + MLP Classification](2-dino-mlp-classification) | Training a MLP layer on 3 classes of waste product.                      |

## Pre-requisites

- Linux preferably with GPU support. For hackathon Thor units will be accessible
- Docker

## Setup environment

For this project, we are using docker as our learning environment. use the `./run.sh` (if running on the Sage Thors, use `run_thor.sh`) command to run the script to build and run the docker container. The files in this directory are mounted inside the docker container. This script can be ran on x86 or arm based devices. GPU acceleration is enabled as long as system supports it.

After we are inside the containerized environment, we can run `./download_files.sh` to get our dataset, and the DinoV2 model with weights.

In the event there is a permission error, try `chmod +x ./run.sh`. Same for download_files.sh `chmod +x ./download_files.sh`

## 0-basic-dino-test

`cd 0-basic-dino-test && python3 dinov2_demo.py`

This is to demonstrate running an image through dinov2. DinoV2 acts as a feature extractor and returns an embedding represeting the semantic meaning of the image. Additionally patch level embeddings are also returned.

## 1-dino-semantic-sim

`cd 1-dino-semantic-sim && python3 dinov2_patch_semantic_pca.py`

This is to demonstrate the semantic similarity between patches using PCA. This demonstrates that the model captures the features in the image well.

## 2-dino-mlp-classification

`cd 2-dino-mlp-classification && python3 train.py`

Using the [realwaste dataset](https://www.kaggle.com/datasets/joebeachcapital/realwaste), we are able to train a MLP head attached to a frozen dinov2 backbone to classify an image into 3 different classes. After training the weights and model metadata are saved in `head.pt` and we can run `python3 predict.py` or `python3 predict.py ./test/napkin.jpg` to test against a never seen before image.
