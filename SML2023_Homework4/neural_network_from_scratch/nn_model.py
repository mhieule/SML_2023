import numpy as np

import nn_modules


class NNModel:
    # Model provides the needed functionalities to train and evaluate a generic stack of NNModules

    def __init__(self, module_specifications):
        self.nn_modules = []
        self.bprop_buffer = dict()

        # Build module stack for the given module specifications
        for (Module, args) in module_specifications:
            if not issubclass(Module, nn_modules.NNModule):
                raise Exception('Not a valid NNModule')

            nn_module = Module(**args)
            self.nn_modules.append(nn_module)

    def initialize_neural_network(self):
        """
        Initialize the internal parameters of the neural network
        :return:
        """
        # ToDo
        for module in self.nn_modules:
            module.initialize_parameters()

    def fprop(self, input):
        """
        Run the forward pass through the entire module stack
        :param input: data input
        :return: output of neural network
        """
        # ToDo
        res = input
        for module in self.nn_modules:
            res = module.fprop(res)
        return res



    def bprop(self, output_gradient):
        """
        Backpropagate the gradients through the complete module stack
        :param output_gradient: Gradient with respect to the neural network output
        :return: -
        """
        # ToDo
        dX = output_gradient
        for i, module in enumerate(reversed(self.nn_modules)):
            if issubclass(type(module), nn_modules.LinearModule):
                dW, db = module.get_batch_gradient_parameters(dX)
                self.bprop_buffer[i] = [dW,db]
            dX = module.bprop(dX)


    def update_internal_parameters(self, update_function):
        """
        Update all the internal parameters inside the neural network
        :param update_function: The function to update the internal parameters
        :return: -
        """
        # ToDo
        for i, module in enumerate(reversed(self.nn_modules)):
            if issubclass(type(module), nn_modules.LinearModule):
                module.apply_parameter_updates(self.bprop_buffer[i], update_function)
