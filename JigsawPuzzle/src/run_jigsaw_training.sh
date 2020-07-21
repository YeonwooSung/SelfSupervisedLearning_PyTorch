IMAGENET_FOLD='imagenet'

python JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --batch 128 --lr=0.001 --cores=3
