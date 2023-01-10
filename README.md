# DL Systems Final Project
[Official cite](https://dlsyscourse.org/)  

## Description
Our team is developing a diffusion model, a recent state-of-the-art approach to image generation problem.
The basic idea is the same as in other generative models: they all convert noise from some simple distribution to a data sample.  
We’re applying our model on a Landscapes Dataset, which is taken from [Kaggle](https://www.kaggle.com/datasets/arnaud58/landscape-pictures).  

At the moment we’ve implemented all components of UNet architecture, which are ConvTranspose and the basic version of MaxPool layer, with no padding and stride equals to the kernel size.  
In addition, we’ve written the DiffusionModel, which is responsible for the forward and backward diffusion process.

## TODO List
1. Give a brief description of the project in README.md
2. Memory usage optimization
    1. module.parameters(): List ==> Generator
    2. analyze all operations in `ops.py`
3. Implement modified summation and maximization
4. Documentation + type annotation with `mypy`
5. Modified Maxpooling layer
6. Benchmarks for speed comparison against pytorch
7. Add illustrations in a final report
8. Train a model
9. Deploy a model