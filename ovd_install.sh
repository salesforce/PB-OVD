#!/bin/bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 python=3.6 -c pytorch -y

for line in $(cat requirements.txt)
do
  pip install $line
done