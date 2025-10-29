#!/bin/bash
xhost +

# Ensure persistent bash history file exists on host
touch docker/eternal_history.txt

docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/project \
 -v $(pwd)/docker/eternal_history.txt:/root/.bash_eternal_history \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /tmp:/tmp \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 --privileged \
 mujoco-env:latest