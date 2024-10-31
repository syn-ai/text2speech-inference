#!/bin/bash

source .venv/bin/activate

export CUDA_VISIBLE_DEVICE=3

docker compose up
