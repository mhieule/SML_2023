import numpy as np

from abstract_nn_modules import NNModule, NNModuleParameterFree, LossModule


class LinearModule(NNModule):
    # Module implements a linear layer

    def __init__(self, number_input_neurons, number_output_neurons):
        self.number_input_neurons = number_input_neurons
        self.number_output_neurons = number_output_neurons

        # Internal parameter
        self.W = None
        self.b = None

        # Cache input for backprop
        self.input_cache = None

    # ToDo: Implement all abstract methods of NNModule
    #  Use the classical Xavier Glorot parameter initialization
    def fprop(self, input):
        # Fix input to be 2D so all matmul for fprop and bprop works
        if len(input.shape) == 1:
            input = input[None,:]
        self.input_cache = input
        return input @ self.W + self.b

    def bprop(self, output_gradient):
        return output_gradient @ self.W.T

    def initialize_parameters(self):
        bound = np.sqrt(6 / (self.number_input_neurons + 1 + self.number_output_neurons))
        W = np.random.uniform(-bound, bound, size=(self.number_input_neurons+1,self.number_output_neurons))
        self.W = W[:-1,:]
        self.b = W[-1]

    def get_batch_gradient_parameters(self, output_gradient):
        dW = (self.input_cache.T @ output_gradient) / self.input_cache.shape[0]
        db = np.mean(output_gradient, axis=0)
        return dW, db

    def apply_parameter_updates(self, batch_gradient_parameters, update_function):
        dW, db = batch_gradient_parameters
        self.W = update_function(self.W, dW)
        self.b = update_function(self.b, db)



class SoftMaxModule(NNModuleParameterFree):
    # Module implements the SoftMax activation function

    def __init__(self):
        # cache output for bprop
        self.output_cache = None

    # ToDo: Implement fprop and bprop of NNModule
    def fprop(self, input):
        z = input - np.max(input, axis=1)[:,None]
        ez = np.exp(z)
        self.output_cache = ez / np.sum(ez,axis=1)[:,None]
        return self.output_cache

    def bprop(self, output_gradient):
        return output_gradient * self.output_cache * (1 - self.output_cache)


class CrossEntropyLoss(LossModule):
    # Module implements the CrossEntropy loss function

    def __init__(self):
        # cache input for bprop
        self.input_cache = None

    # ToDo: Implement all abstract methods of LossModule
    def calculate_loss(self, input):
        self.input_cache = input
        return -np.sum(self.targets * np.log(input),axis=1)

    def calculate_loss_gradient(self):
        return self.input_cache - self.targets


class TanhModule(NNModuleParameterFree):
    # Module implements the Tanh activation function

    def __init__(self):
        # Cache output for bprop
        self.output_cache = None

    # ToDo: Implement fprop and bprop of NNModule
    def fprop(self, input):
        self.output_cache = np.tanh(input)
        return self.output_cache

    def bprop(self, output_gradient):
        return output_gradient * (1 - self.output_cache ** 2)
