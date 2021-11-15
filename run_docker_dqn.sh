#!/bin/bash
docker run --runtime=nvidia -it \
-d \
-v /mnt/bigdisk2/pontesmo/dqn_sacred:/workspace \
--memory=100G \
--memory-reservation=64G \
--cpus=16.0 \
--name mateus_dqn mateus-dqn-env
