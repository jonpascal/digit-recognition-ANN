# Digit Recognition ANN

## Motivation
This is my first attempt at creating an artificial neural netwotk. 

In this project I will attempt to design and train an artifical neural network (ANN) that recognises hand written digits. 

This is part of a project in the UVP course at the [Faculty of Mathematics and Physics](https://www.fmf.uni-lj.si/sl/), University of Ljubljana. 

Everything in the repository is written in English as I am more familiar with the language in the context of programming and neural networks.

## Overview
The [network.py](network.py) file contains a Network class that represents a neural network. The network can be initialised with a list [784, ..., 10] that represents the number of neurons per layer. The Network class also contains a SGD (stochastic gradinet descent) method for learning. 

The [data_loader.py](data_loader.py) file contains functions for reading data files and plotting ceratin properties of the network. The data reading functions include a function for reading training and testing idx3-ubyte and idx1-ubyte files and a function for reading end encoding the images that the user inputs.

The [trained_networks.py](trained_networks.py) file contains a script for training various neural networks while varying certain hyper parameters. The trained network's weights and biases are saved in the [model_prameters.pkl](model_parameters.pkl) in a form of a dictionary. 

The [data_presentation.ipynb](data_presentation.ipynb) file is a Jupyter notebook that gives a rough overview of the whole project.

## How to use
For Jupyter notebook to work properly download at least:
- network.py
- data_loader.py
- model_parameters.pkl
- data_presentation.ipynb 

Store all the files in the same folder. Inside that folder you should also create a floder named 'input_images', for the last part of the Jupyter notebook to work.

Other requirements include:
- pyhton 
- numpy 
- matplotlib.pyplot 
- Pillow 

## Data
Labeled data of handwritten digits for training the network was gathered from the [MNIST database](http://yann.lecun.com/exdb/mnist/). 

All the instructions for proccesing the data are available at the link provided above. 