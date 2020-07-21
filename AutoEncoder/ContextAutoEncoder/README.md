# Context AutoEncoder

_Context Encoders: Feature Learning by Inpainting_

## Authors

Deepak Pathak, Phillip Krähenbühl, Jeff Donahue, Trevor Darrell, Alexei A. Efros

## Abstract

We present an unsupervised visual feature learning algorithm driven by context-based pixel prediction. By analogy with auto-encoders, we propose Context Encoders -- a convolutional neural network trained to generate the contents of an arbitrary image region conditioned on its surroundings. In order to succeed at this task, context encoders need to both understand the content of the entire image, as well as produce a plausible hypothesis for the missing part(s). When training context encoders, we have experimented with both a standard pixel-wise reconstruction loss, as well as a reconstruction plus an adversarial loss. The latter produces much sharper results because it can better handle multiple modes in the output. We found that a context encoder learns a representation that captures not just appearance but also the semantics of visual structures. We quantitatively demonstrate the effectiveness of our learned features for CNN pre-training on classification, detection, and segmentation tasks. Furthermore, context encoders can be used for semantic inpainting tasks, either stand-alone or as initialization for non-parametric methods.

[[project webpage]](https://people.eecs.berkeley.edu/~pathak/context_encoder/)

## Dataset

Put your images under dataset/train,all images should under subdirectory:

    dataset/train/subdirectory1/some_images

    dataset/train/subdirectory2/some_images

    ...

### Paris StreetView Dataset

For Google Policy, Paris StreetView Dataset is not public data. If you need this dataset for research using please contact with [pathak22](https://github.com/pathak22). Please read [this github issue](https://github.com/pathak22/context-encoder/issues/24) for more information about how to download Paris StreetView Dataset.

You could also use the [The Paris Dataset](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) to train the model.

## Training

After preparing the dataset, you could train the network by using the train.py file. Following is the example instruction of training the Context AutoEncoder.

```bash
python3 train.py --cuda --wtl2 0.999 --niter 200
```

## Pretrained model

This repository provides the pretrained model, which is trained with the Paris StreetView Dataset. You could test the pretained model by using the following insturctions.

```bash
# Inpainting a batch iamges
$ python3 test.py --netG model/netG_streetview.pth --dataroot dataset/val --batchSize 100

# Inpainting one image 
$ python3 test_one.py --netG model/netG_streetview.pth --test_image result/test/cropped/065_im.png
```
