# Third-Party Notices

This repository does not grant rights in third-party datasets, images, pretrained model weights, or software dependencies.

## MNIST

The MNIST dataset is obtained by the experiment notebooks through `torchvision`. The qualitative-input archive contains selected MNIST-derived source and reconstruction tiles needed to reproduce the paper grids. Those tiles are excluded from this repository's MIT and CC BY 4.0 license grants. Consult the [MNIST source page](http://yann.lecun.com/exdb/mnist/) and the applicable distribution terms before reuse.

## CelebA

CelebA is provided for non-commercial research use subject to its own terms. The qualitative-input archives contain selected CelebA-derived source and reconstruction tiles needed to reproduce the paper grids. Those tiles are excluded from this repository's MIT and CC BY 4.0 license grants and must not be treated as independently licensed face images. Consult the [official CelebA project page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) before reuse.

## VGGFace2, MAAD-Face, and pretrained face models

The current notebooks use downloaded VGGFace2 training images and metadata with released MAAD-Face Smiling annotations. These data and annotations are not redistributed. Consult the [VGGFace2 project](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) and [MAAD-Face authors' repository](https://github.com/pterhoer/MAAD-Face) for their respective distribution and reuse terms.

The pretrained face-auditor weights are not redistributed. Current notebooks 10–13 use `facenet-pytorch`'s InceptionResnetV1 weights pretrained on CASIA-WebFace; historical notebooks use VGGFace2-pretrained weights. The current notebooks download and verify the official model asset. Consult [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) and the original [CASIA-WebFace paper](https://arxiv.org/abs/1411.7923) for model and dataset provenance. These weights and facial datasets are excluded from the repository's license grants.

## Software dependencies

Third-party Python and system packages remain subject to their own licenses. The MIT License in this repository covers only the original code and notebooks authored for this reproducibility package.
