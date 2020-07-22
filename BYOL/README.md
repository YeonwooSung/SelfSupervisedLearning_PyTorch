# Bootstrap Your Own Latent

_Bootstrap Your Own Latent (BYOL)_

## Authors

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko

## Abstract

We introduce Bootstrap Your Own Latent (BYOL), a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network. While state-of-the art methods intrinsically rely on negative pairs, BYOL achieves a new state of the art without them. BYOL reaches 74.3% top-1 classification accuracy on ImageNet using the standard linear evaluation protocol with a ResNet-50 architecture and 79.6% with a larger ResNet. We show that BYOL performs on par or better than the current state of the art on both transfer and semi-supervised benchmarks.

[[paper]](https://arxiv.org/abs/2006.07733)

## Implementation

You could find my implementation [here](./src/byol.py).

I declare that my implementation is based on [lucidrains's implementation](https://github.com/lucidrains). What I did is simply modifying his implementation to work more efficiently.

Also by running the following command, you would be able to demonstarte the BYOL with resnet-50 model.

```bash
$ cd src
$ python3 main.py
```

To apply BYOL to other models, simply plugin your neural network, specifying (1) the image dimensions as well as (2) the name (or index) of the hidden layer, whose output is used as the latent representation used for self-supervised training.

### References

[1] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko. [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)

[2] lucidrains. [byol-pytorch](https://github.com/lucidrains/byol-pytorch)

[3] Hoseong Lee. [Bootstrap Your Own Latent： A New Approach to Self-Supervised Learning 리뷰](https://hoya012.github.io/blog/byol/?fbclid=IwAR21KdQGj50JoKHEBQqma5RC-D6VERVzVpO1uVVaqMxGMnnsh9qPsMpK_cE)
