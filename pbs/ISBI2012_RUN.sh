#!/bin/bash

cd /gdata1/limx/mx_new/unet/unet_k5_ablation/lambda_2_5 &&\
python train.py --cfg lib/ISBI2012.yaml
