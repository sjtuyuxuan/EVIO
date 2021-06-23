#!/bin/bash

docker rm -f `docker ps -a | grep slam | cut -d " " -f 1`
docker run --name=slam -d -it \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -e DISPLAY=unix$DISPLAY \
   -e GDK_SCALE \
   --gpus=all \
   -e GDK_DPI_SCALE \
   -p 11311:11311 \
   -v $PWD/../../..:/develop slam_baseline:latest roscore
