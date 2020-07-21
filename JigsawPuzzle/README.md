# JigsawPuzzle

_Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles_

## Authors

Mehdi Noroozi, Paolo Favaro

## Abstract

In this paper we study the problem of image representation learning without human annotation. By following the principles of self-supervision, we build a convolutional neural network (CNN) that can be trained to solve Jigsaw puzzles as a pretext task, which requires no manual labeling, and then later repurposed to solve object classification and detection. To maintain the compatibility across tasks we introduce the context-free network (CFN), a siamese-ennead CNN. The CFN takes image tiles as input and explicitly limits the receptive field (or context) of its early processing units to one tile at a time. We show that the CFN includes fewer parameters than AlexNet while preserving the same semantic learning capabilities. By training the CFN to solve Jigsaw puzzles, we learn both a feature mapping of object parts as well as their correct spatial arrangement. Our experimental evaluations show that the learned features capture semantically relevant content. Our proposed method for learning visual representations outperforms state of the art methods in several transfer learning benchmarks.

[[paper]](https://arxiv.org/abs/1603.09246)

## How to run

### Prepare Dataset

Basically, ImageNet dataset is used for training this model. Thus, please download the ImageNet, and move the new folders into {repository root}/imagenet/all and run "python3 imagenet_train_test_split.py".

### Training Network

Fill the path information in run_jigsaw_training.sh. IMAGENET_FOLD needs to point to the folder containing ImageNet.

```
./run_jigsaw_training.sh [GPU_ID]
```

or call the python script

```
python3 JigsawTrain.py [*path_to_imagenet*] --checkpoint [*path_checkpoints_and_logs*] --gpu [*GPU_ID*] --batch [*batch_size*]
```

By default the network uses 1000 permutations with maximum hamming distance selected using select_permutations.py.

To change the file name loaded for the permutations, open the file Dataset/JigsawLoader.py and change the permutation file in the method retrieve_permutations.

### Details

    - The input of the network should be 64x64, but it is resized to 75x75, otherwise the output of conv5 is 2x2 instead of 3x3 like the official architecture

    - Jigsaw trained using the approach of the paper: SGD, LRN layers, 70 epochs

    - Implemented data augmentation to discourage learning shortcuts: spatial jittering, normalize each patch indipendently, color jittering, 30% black&white image

### Results

TODO

## References

- [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)

- [JigsawPuzzlePytorch](https://github.com/bbrattoli/JigsawPuzzlePytorch)
