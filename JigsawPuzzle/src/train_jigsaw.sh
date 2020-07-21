#!/usr/bin/env bash
#
# train_jigssaw.sh

IMAGENET_FOLD=path_to_ILSVRC2012_img

GPU=${1} # gpu used
CHECKPOINTS_FOLD=${2} #path_to_output_folder

#python3 train.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10

python3 train.py ${IMAGENET_FOLD} --classes=1000 --batch 128 --lr=0.001 --cores=10
