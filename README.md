# Dataset: histology landmarks

[![Build Status](https://travis-ci.com/Borda/dataset-histology-landmarks.svg?token=HksCAm7DV2pJNEbsGJH2&branch=master)](https://travis-ci.com/Borda/dataset-histology-landmarks)

**Dataset: landmarks for registration of [histology images](http://cmp.felk.cvut.cz/~borovji3/?page=dataset)**

The dataset consists of 2D histological microscopy tissue slices, stained with different stains. The main challenges for these images are the following: very large image size, appearance differences, and lack of distinctive appearance objects. Our dataset contains 108 image pars and manually placed landmarks for registration quality evaluation.

![reconstruction](figures/images-landmarks.jpg)

The image part of the dataset are available [here](http://cmp.felk.cvut.cz/~borovji3/?page=dataset). **Note** that the accompanied landmarks are the initial from a single a user and the precise landmarks should be obtain by fusion of several users even you can help and improve the annotations.


## Landmarks

The landmarks have standard [ImageJ](https://imagej.net/Welcome) structure and coordinate frame. For handling this landmarks we provide a simple macros for [import](annotations/multiPointSet_import.ijm) and [export](annotations/multiPointSet_export.ijm).
