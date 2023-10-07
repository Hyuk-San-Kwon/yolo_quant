#!/bin/bash


#docker run --gpus all -it --rm -v $(pwd):/torch2trt torch2trt:21-06 


# docker run --gpus all -it --rm -v $(pwd):/opt/project torch2trt:21-06


docker run --gpus all -it --rm -v $(pwd):/opt/project grad:0.01