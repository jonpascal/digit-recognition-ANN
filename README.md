# Digit Recognition ANN

## Motivation
This is my first attempt at creating an artificial neural netwotk. 

In this project I will attempt to design and train an artifical neural network (ANN) that recognises hand written digits. 

This is part of a project in the UVP course at the Faculty of Mathematics and Physics, University of Ljubljana. 

Everything in the repository is written English as I am more familiar with the language in terms of programming and neural networks.

## Overview
The network.py contains a Network class that represents a neural network. 

The data_loader.py contains a function that reads the idx3-ubyte and idx1-ubyte training and validating data, a function that reads input images, and a function thatplots network accuracy. 

The trained_networks.py and model_prameters.pkl files contain the script for training various models, and the weights and biases of those models.

The data_presentation.ipynb file is a Jupyter notebook file that represents a part of data analysis. Further description can be found in there.

## How to use
For Jupyter notebook to work properly download at least:
- network.py
- data_loader.py
- model_parameters.pkl
- data_presentation.ipynb 

Have them all in the same folder. Also, inside the same folder, create a folder named 'input_images'. Then follow the instructions in the Jupyter notebook.