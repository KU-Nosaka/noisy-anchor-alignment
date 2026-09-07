# Third-Party Notices

This repository does not grant rights in third-party datasets, images, pretrained model weights, or software dependencies.

## MNIST

The MNIST dataset is obtained by the experiment notebooks through `torchvision`. The qualitative-input archive contains selected MNIST-derived source and reconstruction tiles needed to reproduce the paper grids. Those tiles are excluded from this repository's MIT and CC BY 4.0 license grants. Consult the [MNIST source page](http://yann.lecun.com/exdb/mnist/) and the applicable distribution terms before reuse.

## CelebA

CelebA is provided for non-commercial research use subject to its own terms. The qualitative-input archives contain selected CelebA-derived source and reconstruction tiles needed to reproduce the paper grids. Those tiles are excluded from this repository's MIT and CC BY 4.0 license grants and must not be treated as independently licensed face images. Consult the [official CelebA project page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) before reuse.

## LFWA

The current LFWA experiment obtains face images and the original 40-attribute annotations from their source distributions. They are not included in `current/`. Neither source images nor generated reconstructions receive an MIT or CC BY 4.0 grant from this repository. Users must obtain the source data under its applicable terms before running the experiment or regenerating the qualitative gallery.

## VGGFace2 and pretrained face model

The pretrained face-auditor weights are not redistributed. The notebooks use `facenet-pytorch`'s InceptionResnetV1 weights pretrained on VGGFace2. Consult the [VGGFace2 project](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) and [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) for their respective terms.

## Software dependencies

Third-party Python and system packages remain subject to their own licenses. The MIT License in this repository covers only the original code and notebooks authored for this reproducibility package.

