# Denoising AutoEncoder

_Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion_

## Authors

Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol

## Abstract

We explore an original strategy for building deep networks, based on stacking layers of denoising
autoencoders which are trained locally to denoise corrupted versions of their inputs. The resulting
algorithm is a straightforward variation on the stacking of ordinary autoencoders. It is however
shown on a benchmark of classification problems to yield significantly lower classification error,
thus bridging the performance gap with deep belief networks (DBN), and in several cases surpassing it. Higher level representations learnt in this purely unsupervised fashion also help boost the
performance of subsequent SVM classifiers. Qualitative experiments show that, contrary to ordinary autoencoders, denoising autoencoders are able to learn Gabor-like edge detectors from natural
image patches and larger stroke detectors from digit images. This work clearly establishes the value
of using a denoising criterion as a tractable unsupervised objective to guide the learning of useful
higher level representations.

[[paper]](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

## Implementation

- Two kinds of noise were introduced to the standard MNIST dataset: Gaussian and speckle, to help generalization.

- The autoencoder architecture consists of two parts: encoder and decoder. Each part consists of 3 Linear layers with ReLU activations. The last activation layer is Sigmoid.

- The training was done for 120 epochs.

- Visualizations have been included in the notebook.

- Used Google's Colaboratory with GPU enabled.

## References

- [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

- [Denoising-Autoencoder-in-Pytorch](https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch)
