# Setup

- Create a virtual environment / conda environment with a recent python version
- Install the required packages using pip install -r requirements.txt
- **You are not allowed to use any other packages!**

# Introduction
- We provide you with a simple framework to build, train and test your first neural network
- We have already taken care of all the functionalities needed to train and test your neural network
- It is your task to build up a neural network from scratch and making sure that parameter initialization, 
  forward propagation, backward propagation and parameter updates work 

# Getting started
- You can find abstract classes of neural network modules within abstract_nn_modules.py
- Inheriting from these abstract classes, you need to implement single neural network modules inside nn_modules.py that 
  will build up your complete neural network model
- Within nn_model.py, you will need to implement general functionalities of your overall neural network model


- In experiment.py, you can find all functionalities needed to train and test your neural network
- It all comes together in main.py where we set the needed parameters, build the 
  neural network, start the training, save the trained model and evaluate it on the test dataset 

# Tips
- Have a deep understanding of the theory behind neural networks, you will need it! 
- Work with breakpoints! Check the output of your variables and the dimensions of your vectors and matrices 
- If the batch case is too difficult to start with, try the implementation with single data points first
