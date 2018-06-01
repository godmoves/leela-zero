#!/bin/bash

if [ -e leelaz-model-0.meta ]
then
  echo "LZ model exists"
else
  cd ../training/tf/
  echo "Downloading LZ network"
  # Always choose best network
  curl -o network.gz zero.sjeng.org/best-network
  gzip -d -f network.gz
  ./net_to_model.py network
  mv leelaz* ../../scripts/
  rm network checkpoint
  rm -rf __pycache__
  rm -rf leelalogs
  echo "LZ model ready"
fi
